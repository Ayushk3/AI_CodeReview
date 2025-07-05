# ai_code_reviewer/models/graph_representation.py

import ast
import javalang
import networkx as nx
import torch
from torch_geometric.data import Data
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

# Import ASTParser to re-use its parsing and string conversion capabilities
from ai_code_reviewer.data.processors.ast_parser import ASTParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGraphConverter:
    """
    Converts code (and its AST representation) into a graph format suitable for GNNs
    using PyTorch Geometric. Supports Python and Java.
    """

    def __init__(self):
        self.ast_parser = ASTParser()
        self.node_type_map: Dict[str, int] = {} # Maps string node type to integer ID
        self._next_node_type_id = 0
        logger.info("CodeGraphConverter initialized.")

    def _get_node_type_id(self, node_type_name: str) -> int:
        """Assigns a unique integer ID to each encountered node type."""
        if node_type_name not in self.node_type_map:
            self.node_type_map[node_type_name] = self._next_node_type_id
            self._next_node_type_id += 1
        return self.node_type_map[node_type_name]

    def _convert_python_ast_to_graph(self, py_ast_object: ast.AST) -> Optional[Data]:
        """
        Converts a Python AST object into a PyTorch Geometric Data graph.
        Nodes are AST nodes, edges are parent-child relationships.
        """
        if not py_ast_object:
            logger.warning("Empty Python AST object provided for graph conversion.")
            return None

        graph = nx.DiGraph()
        node_id_counter = 0
        node_to_id: Dict[ast.AST, int] = {}
        node_features: List[List[int]] = [] # Will store [node_type_id] for each node

        # Step 1: Add all AST nodes as graph nodes and collect features
        for node in ast.walk(py_ast_object):
            node_id_counter += 1 # Pre-increment to start from 1, then map to 0-indexed in list
            node_to_id[node] = node_id_counter - 1 # Store 0-indexed ID
            node_type_name = type(node).__name__
            node_type_id = self._get_node_type_id(node_type_name)
            node_features.append([node_type_id])
            graph.add_node(node_to_id[node], type=node_type_name)

        # Step 2: Add parent-child edges
        edge_indices: List[List[int]] = []
        for node in ast.walk(py_ast_object):
            for child_name, child_value in ast.iter_fields(node):
                if isinstance(child_value, ast.AST):
                    # Edge from parent to child
                    edge_indices.append([node_to_id[node], node_to_id[child_value]])
                elif isinstance(child_value, list):
                    for item in child_value:
                        if isinstance(item, ast.AST):
                            # Edge from parent to each child in list
                            edge_indices.append([node_to_id[node], node_to_id[item]])
        
        # Step 3: Add sequential (next-statement) edges for bodies
        # This is a heuristic and can be refined
        for node in ast.walk(py_ast_object):
            if hasattr(node, 'body') and isinstance(node.body, list):
                for i in range(len(node.body) - 1):
                    stmt1 = node.body[i]
                    stmt2 = node.body[i+1]
                    if isinstance(stmt1, ast.AST) and isinstance(stmt2, ast.AST):
                        edge_indices.append([node_to_id[stmt1], node_to_id[stmt2]])
                        # Optional: also add edge from previous statement to the next's first token/node
                        # (More advanced, skip for Phase 1 basic)

        if not node_features:
            logger.warning("No nodes extracted from Python AST.")
            return None

        x = torch.tensor(node_features, dtype=torch.long) # Node features (e.g., node type IDs)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def _walk_javalang_ast(self, node):
        """Recursively yields (path, node) for all nodes in the javalang AST."""
        stack = [([], node)]
        while stack:
            path, current = stack.pop()
            yield path, current
            if isinstance(current, javalang.tree.Node) and current.children is not None:
                for child in current.children:
                    if child is None:
                        continue
                    if isinstance(child, tuple) and len(child) == 2:
                        child_name, child_value = child
                        if child_value is None:
                            continue
                        if isinstance(child_value, javalang.tree.Node):
                            stack.append((path + [child_name], child_value))
                        elif isinstance(child_value, list):
                            for idx, item in enumerate(child_value):
                                if isinstance(item, javalang.tree.Node):
                                    stack.append((path + [child_name, str(idx)], item))

    def _convert_java_ast_to_graph(self, java_ast_object: javalang.tree.CompilationUnit) -> Optional[Data]:
        """
        Converts a Java AST object (from javalang) into a PyTorch Geometric Data graph.
        Nodes are AST nodes, edges are parent-child relationships.
        """
        if not java_ast_object:
            logger.warning("Empty Java AST object provided for graph conversion.")
            return None

        graph = nx.DiGraph()
        node_id_counter = 0
        node_to_id: Dict[Any, int] = {} # javalang nodes are not hashable by default, use id()
        node_features: List[List[int]] = []

        # Step 1: Add all AST nodes as graph nodes and collect features
        for path, node in self._walk_javalang_ast(java_ast_object):
            node_id_counter += 1
            node_to_id[id(node)] = node_id_counter - 1 # Use object ID for dictionary key
            node_type_name = type(node).__name__
            node_type_id = self._get_node_type_id(node_type_name)
            node_features.append([node_type_id])
            graph.add_node(node_to_id[id(node)], type=node_type_name)

        # Step 2: Add parent-child edges
        edge_indices: List[List[int]] = []
        for path, node in self._walk_javalang_ast(java_ast_object):
            if isinstance(node, javalang.tree.Node) and node.children is not None:
                for child in node.children:
                    if child is None:
                        continue
                    if isinstance(child, tuple) and len(child) == 2:
                        child_name, child_value = child
                        if child_value is None:
                            continue
                        if isinstance(child_value, javalang.tree.Node):
                            edge_indices.append([node_to_id[id(node)], node_to_id[id(child_value)]])
                        elif isinstance(child_value, list):
                            for item in child_value:
                                if isinstance(item, javalang.tree.Node):
                                    edge_indices.append([node_to_id[id(node)], node_to_id[id(item)]])
        
        # Step 3: Add sequential (next-statement) edges for bodies
        # javalang doesn't have a simple 'body' attribute like Python's AST.
        # We need to manually identify common sequential constructs like BlockStatements, MethodDeclarations.
        # This is more complex and less generic for Java. For Phase 1, we will primarily rely on AST structure.
        # Example for method body:
        for path, node in self._walk_javalang_ast(java_ast_object):
            if isinstance(node, javalang.tree.BlockStatement) and node.statements:
                for i in range(len(node.statements) - 1):
                    stmt1 = node.statements[i]
                    stmt2 = node.statements[i+1]
                    if isinstance(stmt1, javalang.tree.Node) and isinstance(stmt2, javalang.tree.Node):
                        edge_indices.append([node_to_id[id(stmt1)], node_to_id[id(stmt2)]])
            # Add for MethodDeclarations body directly if it's not a BlockStatement
            elif isinstance(node, javalang.tree.MethodDeclaration) and node.body and isinstance(node.body, javalang.tree.BlockStatement) and node.body.statements:
                for i in range(len(node.body.statements) - 1):
                    stmt1 = node.body.statements[i]
                    stmt2 = node.body.statements[i+1]
                    if isinstance(stmt1, javalang.tree.Node) and isinstance(stmt2, javalang.tree.Node):
                        edge_indices.append([node_to_id[id(stmt1)], node_to_id[id(stmt2)]])


        if not node_features:
            logger.warning("No nodes extracted from Java AST.")
            return None

        x = torch.tensor(node_features, dtype=torch.long)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def convert_to_graph(self, code_entry: Dict[str, Any]) -> Optional[Data]:
        """
        Converts a single code entry (with AST object/string) into a PyTorch Geometric Data object.

        Args:
            code_entry (Dict[str, Any]): A dictionary containing 'language', 'ast_object'
                                         (or 'ast_str' from which it can be re-parsed), and 'label'.

        Returns:
            Optional[Data]: A PyTorch Geometric Data object, or None if conversion fails.
        """
        language = code_entry.get("language")
        ast_obj = code_entry.get("ast_object")
        ast_str = code_entry.get("ast_str")
        label = code_entry.get("label")

        if not language:
            logger.error("Missing 'language' in code entry for graph conversion.")
            return None

        if ast_obj is None and ast_str is None:
            logger.error(f"Missing 'ast_object' or 'ast_str' in code entry for {code_entry.get('filepath')}.")
            return None

        # Re-parse AST from string if object not present (e.g., loaded from JSON)
        if ast_obj is None and ast_str is not None:
            code = code_entry.get("code")
            ast_obj = self.ast_parser.parse_code(code, language)
            if ast_obj is None:
                logger.error(f"Failed to re-parse AST from string for {code_entry.get('filepath')}.")
                return None
            else:
                code_entry["ast_object"] = ast_obj # Update entry for potential future use

        graph_data: Optional[Data] = None
        if language.lower() == "python":
            graph_data = self._convert_python_ast_to_graph(ast_obj)
        elif language.lower() == "java":
            graph_data = self._convert_java_ast_to_graph(ast_obj)
        else:
            logger.warning(f"Unsupported language '{language}' for graph conversion.")
            return None
        
        if graph_data:
            # Add label to the graph data object
            graph_data.y = torch.tensor([label], dtype=torch.long)
        
        return graph_data

# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the CodeGraphConverter class.
    """
    logger.info("Running unit tests for CodeGraphConverter...")
    converter = CodeGraphConverter()
    ast_parser = ASTParser() # Need an ASTParser instance for the test setup

    # Test 1: Python Code Conversion
    python_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

x = 10
y = factorial(x)
"""
    python_ast_obj = ast_parser.parse_code(python_code, "python")
    python_entry = {"code": python_code, "language": "python", "label": 1, "ast_object": python_ast_obj}
    py_graph = converter.convert_to_graph(python_entry)

    assert py_graph is not None, "Test 1 (Python): Graph conversion failed."
    assert py_graph.x.shape[0] > 0, "Test 1 (Python): No nodes in graph."
    assert py_graph.edge_index.shape[1] > 0, "Test 1 (Python): No edges in graph."
    assert py_graph.y.item() == 1, "Test 1 (Python): Label mismatch."
    logger.info(f"Test 1 (Python Code Conversion): PASSED. Nodes: {py_graph.num_nodes}, Edges: {py_graph.num_edges}")

    # Test 2: Java Code Conversion
    java_code = """
public class Calculator {
    public int add(int a, int b) {
        int sum = a + b;
        return sum;
    }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        int result = calc.add(5, 3);
        System.out.println(result);
    }
}
"""
    java_ast_obj = ast_parser.parse_code(java_code, "java")
    java_entry = {"code": java_code, "language": "java", "label": 0, "ast_object": java_ast_obj}
    java_graph = converter.convert_to_graph(java_entry)

    assert java_graph is not None, "Test 2 (Java): Graph conversion failed."
    assert java_graph.x.shape[0] > 0, "Test 2 (Java): No nodes in graph."
    assert java_graph.edge_index.shape[1] > 0, "Test 2 (Java): No edges in graph."
    assert java_graph.y.item() == 0, "Test 2 (Java): Label mismatch."
    logger.info(f"Test 2 (Java Code Conversion): PASSED. Nodes: {java_graph.num_nodes}, Edges: {java_graph.num_edges}")

    # Test 3: Empty Code/AST
    empty_py_entry = {"code": "", "language": "python", "label": 0, "ast_object": None}
    empty_graph = converter.convert_to_graph(empty_py_entry)
    assert empty_graph is None, "Test 3 (Empty Code): Should return None for empty AST."
    logger.info("Test 3 (Empty Code/AST): PASSED")

    # Test 4: Missing Language/Label
    missing_lang_entry = {"code": "def test(): pass", "label": 0, "ast_object": python_ast_obj}
    missing_graph = converter.convert_to_graph(missing_lang_entry)
    assert missing_graph is None, "Test 4 (Missing Language): Should return None."
    logger.info("Test 4 (Missing Language/Label): PASSED (missing label handled by `if not label` logic or default to 0)")


    logger.info("All CodeGraphConverter unit tests completed.")

# --- Performance Profiling (Conceptual) ---
def profile_graph_converter_performance(num_samples: int = 100):
    """
    Profiles the performance of the CodeGraphConverter.
    """
    logger.info(f"\nStarting CodeGraphConverter performance profiling for {num_samples} samples...")
    converter = CodeGraphConverter()
    ast_parser = ASTParser()

    python_code = """
def complex_logic(data_list):
    total = 0
    for item in data_list:
        if item % 2 == 0:
            total += item * 2
        else:
            total += item / 2
    return total

def helper_func(x, y):
    return x + y
"""
    java_code = """
public class ComplexProcessor {
    public double process(List<Integer> data) {
        double sum = 0;
        for (Integer item : data) {
            if (item % 2 == 0) {
                sum += item * 2;
            } else {
                sum += item / 2.0;
            }
        }
        return sum;
    }

    private int anotherHelper(int a, int b) {
        return a * b;
    }
}
"""
    sample_entries: List[Dict[str, Any]] = []
    for i in range(num_samples):
        if i % 2 == 0:
            ast_obj = ast_parser.parse_code(python_code, "python")
            sample_entries.append({"code": python_code, "language": "python", "label": i % 2, "ast_object": ast_obj})
        else:
            ast_obj = ast_parser.parse_code(java_code, "java")
            sample_entries.append({"code": java_code, "language": "java", "label": i % 2, "ast_object": ast_obj})
    
    start_time = datetime.now()
    converted_count = 0
    total_nodes = 0
    total_edges = 0

    for entry in sample_entries:
        graph = converter.convert_to_graph(entry)
        if graph:
            converted_count += 1
            total_nodes += graph.num_nodes
            total_edges += graph.num_edges
    
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for CodeGraphConverter:")
    logger.info(f"  Total samples attempted: {num_samples}")
    logger.info(f"  Successfully converted to graph: {converted_count}")
    logger.info(f"  Total duration: {duration}")
    if converted_count > 0:
        logger.info(f"  Average time per graph conversion: {duration / converted_count}")
        logger.info(f"  Average nodes per graph: {total_nodes / converted_count}")
        logger.info(f"  Average edges per graph: {total_edges / converted_count}")
    else:
        logger.info("  No successful conversions for average calculation.")


# --- Usage Example ---
if __name__ == "__main__":
    from datetime import datetime
    import os
    import shutil

    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Usage with a Sample Dataset File ---
    logger.info("\n--- Demonstrating CodeGraphConverter with sample data ---")

    # This demonstration assumes `data/datasets/test_dataset.json` exists
    # from a previous run of `dataset_builder.py`.
    # If not, we will create a small dummy file.
    sample_dataset_path = "data/datasets/test_dataset.json"
    
    if not os.path.exists(sample_dataset_path):
        logger.warning(f"Sample dataset file not found at {sample_dataset_path}.")
        logger.info("Creating a dummy test_dataset.json for demonstration...")
        
        dummy_output_dir = "data/datasets"
        os.makedirs(dummy_output_dir, exist_ok=True)

        dummy_data = [
            {"code": "def process(val):\n    if val > 0:\n        return val * 2\n    else:\n        return 0", "language": "python", "label": 1, "filepath": "dummy_py.py", "ast_str": ""},
            {"code": "public class MyMath {\n    public int subtract(int a, int b) {\n        return a - b;\n    }\n}", "language": "java", "label": 0, "filepath": "dummy_java.java", "ast_str": ""},
        ]
        # Re-parse ASTs for the dummy data to ensure they are available
        ast_parser_demo = ASTParser()
        for entry in dummy_data:
            ast_obj = ast_parser_demo.parse_code(entry["code"], entry["language"])
            entry["ast_object"] = ast_obj # Add the AST object directly
            entry["ast_str"] = ast_parser_demo.ast_to_string(ast_obj, entry["language"])

        with open(sample_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=4)
        logger.info(f"Dummy dataset created at {sample_dataset_path}")

    # Load data from the sample dataset
    data_to_convert: List[Dict[str, Any]] = []
    try:
        with open(sample_dataset_path, 'r', encoding='utf-8') as f:
            data_to_convert = json.load(f)
        logger.info(f"Loaded {len(data_to_convert)} entries from {sample_dataset_path} for conversion.")
    except Exception as e:
        logger.error(f"Failed to load sample dataset: {e}")

    converter_instance = CodeGraphConverter()
    converted_graphs: List[Data] = []

    for i, entry in enumerate(data_to_convert):
        logger.info(f"Converting entry {i+1}/{len(data_to_convert)} ({entry.get('filepath', 'unknown')})...")
        graph = converter_instance.convert_to_graph(entry)
        if graph:
            converted_graphs.append(graph)
            logger.info(f"  - Converted! Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, Label: {graph.y.item()}")
            # Print a snippet of node features and edge index for inspection
            if graph.num_nodes > 0:
                logger.debug(f"    Nodes (first 5): {graph.x[:5].tolist()}")
            if graph.num_edges > 0:
                logger.debug(f"    Edges (first 5): {graph.edge_index[:, :5].tolist()}")
        else:
            logger.warning(f"  - Failed to convert entry {i+1}.")

    logger.info(f"Total graphs successfully converted: {len(converted_graphs)}")

    # --- Performance Profiling ---
    profile_graph_converter_performance(num_samples=50) # Use a moderate number for demo

    # Clean up dummy dataset if it was created during this run
    if "dummy_" in sample_dataset_path: # Simple check if it's our dummy file
        shutil.rmtree(os.path.dirname(sample_dataset_path), ignore_errors=True)