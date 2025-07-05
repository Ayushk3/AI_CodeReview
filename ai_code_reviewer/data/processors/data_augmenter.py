# ai_code_reviewer/data/processors/data_augmenter.py

import ast
import javalang
import random
import string
import logging
import json
import os
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeDataAugmenter:
    """
    Performs data augmentation on Python and Java code by applying various
    code mutation techniques while attempting to preserve semantic correctness.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initializes the CodeDataAugmenter.

        Args:
            random_seed (Optional[int]): Seed for random number generator for reproducibility.
        """
        if random_seed is not None:
            random.seed(random_seed)
        logger.info(f"CodeDataAugmenter initialized with random seed: {random_seed}")

    def _generate_random_name(self, prefix: str = "", length: int = 8) -> str:
        """Generates a random string suitable for a variable or function name."""
        return prefix + ''.join(random.choices(string.ascii_lowercase, k=length))

    # --- Python Augmentations ---
    class PythonMutator(ast.NodeTransformer):
        """AST NodeTransformer for Python code mutations."""
        def __init__(self, augmenter_instance, prob_rename: float, prob_noop: float):
            self.augmenter = augmenter_instance
            self.prob_rename = prob_rename
            self.prob_noop = prob_noop
            self.name_map = {} # Original name -> New name
            self.current_scope_names = set() # Names defined in current scope
            self.parent_scopes = []

        def visit(self, node):
            # Track scope for name renaming to avoid collisions
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                self.parent_scopes.append(self.current_scope_names)
                self.current_scope_names = set()

            # Process children first
            new_node = super().visit(node)

            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                self.current_scope_names = self.parent_scopes.pop()
                self.name_map = {name: new_name for name, new_name in self.name_map.items() if new_name not in self.current_scope_names}

            return new_node

        def visit_Name(self, node: ast.Name) -> ast.Name:
            """Rename variables if they are not built-in or already renamed."""
            if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and random.random() < self.prob_rename:
                original_name = node.id
                if original_name not in self.name_map and not original_name.startswith('__') and not original_name.startswith('self.'): # Avoid magic methods and self
                    new_name = self.augmenter._generate_random_name(prefix="aug_")
                    # Ensure new name is not already in the current scope
                    while new_name in self.current_scope_names:
                        new_name = self.augmenter._generate_random_name(prefix="aug_")
                    self.name_map[original_name] = new_name
                    self.current_scope_names.add(new_name)
                    logger.debug(f"Renamed Python variable: {original_name} -> {new_name}")
                if original_name in self.name_map:
                    node.id = self.name_map[original_name]
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            """Rename function definitions."""
            if random.random() < self.prob_rename:
                original_name = node.name
                if original_name not in self.name_map and not original_name.startswith('__'):
                    new_name = self.augmenter._generate_random_name(prefix="aug_func_")
                    while new_name in self.current_scope_names:
                        new_name = self.augmenter._generate_random_name(prefix="aug_func_")
                    self.name_map[original_name] = new_name
                    self.current_scope_names.add(new_name)
                    node.name = new_name
                    logger.debug(f"Renamed Python function: {original_name} -> {new_name}")
            self.generic_visit(node) # Visit children
            return node
        
        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            """Rename class definitions."""
            if random.random() < self.prob_rename:
                original_name = node.name
                if original_name not in self.name_map and not original_name.startswith('__'):
                    new_name = self.augmenter._generate_random_name(prefix="AugClass")
                    while new_name in self.current_scope_names:
                        new_name = self.augmenter._generate_random_name(prefix="AugClass")
                    self.name_map[original_name] = new_name
                    self.current_scope_names.add(new_name)
                    node.name = new_name
                    logger.debug(f"Renamed Python class: {original_name} -> {new_name}")
            self.generic_visit(node) # Visit children
            return node

        def visit_Expr(self, node: ast.Expr) -> Any:
            """Add no-op statements (e.g., empty pass statements, dummy print calls)."""
            self.generic_visit(node)
            if isinstance(node.value, ast.Str) and node.value.s == 'pass': # Don't add to existing 'pass'
                return node
            if random.random() < self.prob_noop:
                # Insert a dummy 'pass' statement or a dummy assignment before existing statement
                # This is tricky without modifying the list of statements in a body.
                # Simpler: replace a node with a list of nodes, or add to a body's list.
                # For `visit_Expr`, we can just replace it with `pass` sometimes or return it.
                # More effective for no-ops is to insert into `body` of `FunctionDef`, `If`, etc.
                # For now, let's just sometimes return a Pass as a simple NoOp.
                # A better approach would be to override `visit_list` of statements.
                # For demonstration, we'll implement a simpler approach below.
                pass
            return node
        
        # A more robust way to insert no-ops: modify body lists.
        # This requires traversing children and potentially replacing the parent's list.
        # For simplicity, we'll add `Pass` statements at a fixed probability in `transform_python_code` function directly,
        # by walking the AST and adding nodes to body lists where appropriate, rather than using NodeTransformer for this specific NoOp.

    class PythonASTUnparser(ast.NodeVisitor):
        """
        A very basic AST unparser for Python.
        NOTE: This is NOT a full-fledged unparser. It's for conceptual demonstration.
        The `ast` module provides `ast.unparse` (Python 3.9+) or `astor` library for better unparsing.
        """
        def __init__(self):
            self.indent_level = 0
            self.code_lines = []

        def generic_visit(self, node):
            # Default visit method to print node type and recurse
            # self.code_lines.append(f"{'  ' * self.indent_level}{type(node).__name__}")
            # self.indent_level += 1
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)
            # self.indent_level -= 1

        def visit_Module(self, node):
            for stmt in node.body:
                self.visit(stmt)

        def visit_FunctionDef(self, node):
            self.code_lines.append(f"{'  ' * self.indent_level}def {node.name}(" + ", ".join(a.arg for a in node.args.args) + "):")
            self.indent_level += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent_level -= 1

        def visit_Assign(self, node):
            target = self.visit(node.targets[0]) if node.targets else '?'
            value = self.visit(node.value)
            self.code_lines.append(f"{'  ' * self.indent_level}{target} = {value}")

        def visit_Name(self, node):
            return node.id

        def visit_Constant(self, node):
            return repr(node.value) # Handles various literals

        def visit_Call(self, node):
            func = self.visit(node.func)
            args = [self.visit(arg) for arg in node.args]
            return f"{func}({', '.join(args)})"
        
        def visit_Return(self, node):
            value = self.visit(node.value) if node.value else ""
            self.code_lines.append(f"{'  ' * self.indent_level}return {value}")

        def visit_BinOp(self, node):
            left = self.visit(node.left)
            op = type(node.op).__name__
            right = self.visit(node.right)
            return f"{left} {op} {right}" # Very simplified op

        def get_code(self):
            return "\n".join(self.code_lines)

    def _unparse_python_ast(self, tree: ast.AST) -> str:
        """
        Unparses a Python AST back into code string.
        Uses ast.unparse for Python 3.9+, falls back to basic visitor.
        """
        if hasattr(ast, 'unparse'):
            return ast.unparse(tree)
        else:
            logger.warning("Using basic custom AST unparser. For better results, use Python 3.9+ or astor library.")
            unparser = self.PythonASTUnparser()
            unparser.visit(tree)
            return unparser.get_code()


    def transform_python_code(self, code: str, prob_rename: float = 0.5, prob_noop: float = 0.3) -> Optional[str]:
        """
        Applies augmentation transformations to Python code.

        Args:
            code (str): The original Python source code.
            prob_rename (float): Probability of renaming a variable/function.
            prob_noop (float): Probability of inserting a no-op statement.

        Returns:
            Optional[str]: The augmented code string, or None if parsing fails.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code for augmentation: {e}")
            return None

        # Apply renaming transformations
        mutator = self.PythonMutator(self, prob_rename, prob_noop)
        transformed_tree = mutator.visit(tree)
        ast.fix_missing_locations(transformed_tree) # Fix line numbers, etc.

        # Apply no-op insertions (more directly in the AST body lists)
        # This part is more robustly handled outside NodeTransformer for insertion.
        for node in ast.walk(transformed_tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While)) and hasattr(node, 'body'):
                # Iterate in reverse to avoid index issues when inserting
                for i in reversed(range(len(node.body))):
                    if random.random() < prob_noop:
                        # Insert a 'pass' statement
                        new_pass_node = ast.Pass()
                        # Copy original line/col info if available (for debugging, though not crucial for Pass)
                        if hasattr(node.body[i], 'lineno'):
                            new_pass_node.lineno = node.body[i].lineno
                            new_pass_node.col_offset = node.body[i].col_offset
                        node.body.insert(i, new_pass_node)
                        logger.debug(f"Inserted no-op (pass) in Python code.")
            # For class definition body:
            elif isinstance(node, ast.ClassDef) and hasattr(node, 'body'):
                 for i in reversed(range(len(node.body))):
                    if random.random() < prob_noop:
                        new_pass_node = ast.Pass()
                        if hasattr(node.body[i], 'lineno'):
                            new_pass_node.lineno = node.body[i].lineno
                            new_pass_node.col_offset = node.body[i].col_offset
                        node.body.insert(i, new_pass_node)
                        logger.debug(f"Inserted no-op (pass) in Python class body.")

        try:
            # Check if the transformed tree is valid before unparsing
            # No easy way to validate semantics without running code, but syntax can be re-parsed
            _ = ast.parse(self._unparse_python_ast(transformed_tree))
            return self._unparse_python_ast(transformed_tree)
        except SyntaxError as e:
            logger.warning(f"Generated invalid Python code after augmentation: {e}. Returning original.")
            return code
        except Exception as e:
            logger.error(f"Error during Python AST unparsing or validation: {e}. Returning original.")
            return code

    # --- Java Augmentations ---
    # javalang does not have built-in unparsing or easy AST manipulation like Python's ast.NodeTransformer.
    # We will need to implement a custom visitor for traversal and string replacement or manual tree building.
    # This will be more challenging and less robust than Python's.
    
    # Simple placeholder for Java augmentation for Phase 1.
    # Realistically, this would require a custom AST visitor/rewriter or a library like JavaParser.
    def transform_java_code(self, code: str, prob_rename: float = 0.5, prob_noop: float = 0.3) -> Optional[str]:
        """
        Applies placeholder augmentation transformations to Java code.
        (Note: Full Java AST manipulation for augmentation is complex and requires
        a robust unparser/rewriter, which is not readily available in javalang.
        This is a conceptual implementation for Phase 1.)

        Args:
            code (str): The original Java source code.
            prob_rename (float): Probability of renaming (conceptual).
            prob_noop (float): Probability of inserting a no-op (conceptual).

        Returns:
            Optional[str]: The (potentially) augmented code string, or None if parsing fails.
        """
        try:
            tree = javalang.parse.parse(code)
        except (javalang.tokenizer.LexerError, javalang.parser.JavaSyntaxError) as e:
            logger.warning(f"Failed to parse Java code for augmentation: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Java parsing for augmentation: {e}")
            return None

        augmented_code_lines = code.splitlines()
        
        # Conceptual renaming: Find simple variable/method names and replace. (Highly unsafe and basic without AST)
        # For simplicity, will not implement robust renaming via string manipulation.
        # This needs AST traversal as done for Python.
        
        # Conceptual no-op: Insert dummy comments or empty lines
        for i in range(len(augmented_code_lines)):
            if random.random() < prob_noop * 0.1: # Reduce probability for simpler no-ops
                # Insert an empty line or a comment
                augmented_code_lines.insert(i, "// No-op comment added by augmenter")
                logger.debug(f"Inserted no-op comment in Java code at line {i+1}.")
                break # Just one insertion per file for simple demo

        augmented_code = "\n".join(augmented_code_lines)

        # Basic validation: try to parse again
        try:
            javalang.parse.parse(augmented_code)
            return augmented_code
        except (javalang.tokenizer.LexerError, javalang.parser.JavaSyntaxError) as e:
            logger.warning(f"Generated invalid Java code after augmentation: {e}. Returning original.")
            return code
        except Exception as e:
            logger.error(f"Error during Java re-parsing validation after augmentation: {e}. Returning original.")
            return code


    def augment_code_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Applies augmentation to a single code entry from the dataset.
        Augmentation is applied to the 'code' field of the entry.

        Args:
            entry (Dict[str, Any]): A dictionary containing code, language, etc.
                                    Expected to have 'code' and 'language' keys.

        Returns:
            Optional[Dict[str, Any]]: A new dictionary with augmented code and updated
                                      AST string representation, or None if augmentation fails.
        """
        code = entry.get("code")
        language = entry.get("language")

        if not code or not language:
            logger.warning(f"Skipping augmentation for entry due to missing 'code' or 'language'. Entry: {entry.get('filepath')}")
            return None

        augmented_code: Optional[str] = None
        if language.lower() == "python":
            augmented_code = self.transform_python_code(code)
        elif language.lower() == "java":
            augmented_code = self.transform_java_code(code)
        else:
            logger.warning(f"Unsupported language for augmentation: {language}. Skipping.")
            return None

        if augmented_code:
            augmented_entry = entry.copy()
            augmented_entry["code"] = augmented_code
            # Re-parse AST for the augmented code
            # Note: This requires an ASTParser instance. For modularity, this might be better
            # handled in the DatasetBuilder that orchestrates all processors.
            # For this standalone file, we'll indicate that AST needs updating.
            augmented_entry["ast_str"] = None # Indicate that AST needs to be re-parsed
            augmented_entry["ast_object"] = None # Clear old AST object
            augmented_entry["is_augmented"] = True # Mark as augmented
            logger.debug(f"Successfully augmented code for {entry.get('filepath')}.")
            return augmented_entry
        else:
            logger.warning(f"Failed to augment code for {entry.get('filepath')}. Returning original (or None if original was invalid).")
            return None

# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the CodeDataAugmenter class.
    """
    logger.info("Running unit tests for CodeDataAugmenter...")
    augmenter = CodeDataAugmenter(random_seed=42)

    # Test Python Augmentation - Valid Code
    python_test_code = """
def calculate_sum(a, b):
    x = a + b
    # This is a comment
    return x
"""
    augmented_python = augmenter.transform_python_code(python_test_code, prob_rename=1.0, prob_noop=1.0)
    assert augmented_python is not None, "Test Python Augmentation (Valid): Failed to augment."
    assert "aug_func_" in augmented_python or "aug_" in augmented_python, "Test Python Augmentation (Valid): No renaming applied."
    assert "pass" in augmented_python, "Test Python Augmentation (Valid): No no-op applied."
    logger.info("Test Python Augmentation (Valid): PASSED")
    # logger.debug(f"Augmented Python Code:\n{augmented_python}")

    # Test Python Augmentation - Invalid Code
    python_invalid_code = "def test_func(x): return x +"
    augmented_python_invalid = augmenter.transform_python_code(python_invalid_code)
    assert augmented_python_invalid == python_invalid_code or augmented_python_invalid is None, "Test Python Augmentation (Invalid): Did not return original or None for invalid code."
    logger.info("Test Python Augmentation (Invalid): PASSED (expected to return original or None)")

    # Test Java Augmentation - Valid Code (conceptual)
    java_test_code = """
public class MyService {
    public int process(int val) {
        int result = val * 2;
        return result;
    }
}
"""
    augmented_java = augmenter.transform_java_code(java_test_code, prob_noop=1.0)
    assert augmented_java is not None, "Test Java Augmentation (Valid): Failed to augment."
    assert "// No-op comment added by augmenter" in augmented_java, "Test Java Augmentation (Valid): No no-op comment applied."
    logger.info("Test Java Augmentation (Valid): PASSED (conceptual)")
    # logger.debug(f"Augmented Java Code:\n{augmented_java}")

    # Test augment_code_entry
    sample_entry_py = {
        "code": "def hello(): print('world')",
        "language": "python",
        "filepath": "test_py.py",
        "parsing_successful": True,
        "label": 0
    }
    augmented_entry = augmenter.augment_code_entry(sample_entry_py)
    assert augmented_entry is not None, "Test augment_code_entry: Failed to augment entry."
    assert augmented_entry["is_augmented"] == True, "Test augment_code_entry: is_augmented flag not set."
    assert augmented_entry["ast_str"] is None, "Test augment_code_entry: AST string not cleared."
    assert "aug_" in augmented_entry["code"] or "pass" in augmented_entry["code"], "Test augment_code_entry: No Python mutation detected."
    logger.info("Test augment_code_entry (Python): PASSED")

    sample_entry_unsupported = {
        "code": "some rust code;",
        "language": "rust",
        "filepath": "test.rs",
        "parsing_successful": True,
        "label": 0
    }
    augmented_unsupported = augmenter.augment_code_entry(sample_entry_unsupported)
    assert augmented_unsupported is None, "Test augment_code_entry (Unsupported): Did not return None for unsupported language."
    logger.info("Test augment_code_entry (Unsupported Language): PASSED")

    logger.info("All CodeDataAugmenter unit tests completed.")

# --- Performance Profiling (Conceptual) ---
def profile_data_augmenter_performance(num_samples: int = 1000):
    """
    Profiles the performance of the CodeDataAugmenter.
    """
    logger.info(f"\nStarting CodeDataAugmenter performance profiling for {num_samples} samples...")
    augmenter = CodeDataAugmenter(random_seed=42)

    python_code = """
def example_function(input_data):
    intermediate_result = input_data * 2
    if intermediate_result > 100:
        final_result = intermediate_result - 50
    else:
        final_result = intermediate_result + 10
    return final_result
"""
    java_code = """
public class DataProcessor {
    public int processData(int data) {
        int tempResult = data * 2;
        if (tempResult > 100) {
            int finalResult = tempResult - 50;
            return finalResult;
        } else {
            int finalResult = tempResult + 10;
            return finalResult;
        }
    }
}
"""
    start_time = datetime.now()
    augmented_count = 0

    for i in range(num_samples):
        if i % 2 == 0: # Half Python, half Java
            augmented = augmenter.transform_python_code(python_code)
        else:
            augmented = augmenter.transform_java_code(java_code)
        
        if augmented:
            augmented_count += 1
    
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for CodeDataAugmenter:")
    logger.info(f"  Total attempts: {num_samples}")
    logger.info(f"  Successfully augmented: {augmented_count}")
    logger.info(f"  Total duration: {duration}")
    if augmented_count > 0:
        logger.info(f"  Average time per successful augmentation: {duration / augmented_count}")
    else:
        logger.info("  No successful augmentations to calculate average time.")


# --- Usage Example ---
if __name__ == "__main__":
    from datetime import datetime
    import shutil # For cleanup
    
    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Usage with Sample Data (from DataBalancer output) ---
    logger.info("\n--- Demonstrating Code Data Augmentation with sample data ---")

    # This part assumes you've run the DataBalancer and have a JSON file ready.
    input_file = "scraped_code_samples/balanced_bug_fix_data.json"
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}. Please run github_scraper.py, ast_parser.py, and data_balancer.py first.")
        # Create a dummy data for demonstration if file not found
        sample_data_for_augmentation = [
            {"code": "def my_func(a):\n    b = a + 1\n    return b", "language": "python", "filepath": "py_file1.py", "label": 1, "parsing_successful": True, "commit_hash": "abc"},
            {"code": "class MyClass {\n    int value = 10;\n    public void doWork() {}\n}", "language": "java", "filepath": "JavaFile1.java", "label": 0, "parsing_successful": True, "commit_hash": "def"},
            {"code": "def another_func(x, y):\n    result = x * y\n    print(result)\n    return result", "language": "python", "filepath": "py_file2.py", "label": 1, "parsing_successful": True, "commit_hash": "ghi"},
            {"code": "void test() { int i = 0; while(i<5) { i++; } }", "language": "java", "filepath": "JavaFile2.java", "label": 0, "parsing_successful": True, "commit_hash": "jkl"},
        ]
        logger.info("Using dummy data for augmentation demonstration.")
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            sample_data_for_augmentation = json.load(f)
        logger.info(f"Loaded {len(sample_data_for_augmentation)} entries from {input_file} for augmentation.")

    augmenter_instance = CodeDataAugmenter(random_seed=123)
    augmented_data: List[Dict[str, Any]] = []
    
    original_count = len(sample_data_for_augmentation)
    augmentation_factor = 2 # Augment each original sample to get 2 new samples

    logger.info(f"Attempting to augment each of {original_count} samples {augmentation_factor} times.")

    for original_entry in sample_data_for_augmentation:
        augmented_data.append(original_entry) # Keep the original
        for _ in range(augmentation_factor):
            augmented_entry = augmenter_instance.augment_code_entry(original_entry)
            if augmented_entry:
                augmented_data.append(augmented_entry)
            else:
                logger.warning(f"Skipped augmentation for an entry due to internal failure.")

    logger.info(f"Finished augmentation. Original samples: {original_count}, Total samples after augmentation: {len(augmented_data)}")

    output_augmented_filename = "scraped_code_samples/augmented_bug_fix_data.json"
    
    # Save the augmented data. Note: 'ast_object' will not be present, 'ast_str' might be None for newly augmented.
    # The DatasetBuilder will re-parse ASTs for these if needed.
    dumpable_augmented_data = []
    for entry in augmented_data:
        copy_entry = entry.copy()
        copy_entry.pop("ast_object", None) # Ensure AST objects are not saved as they are not JSON serializable
        dumpable_augmented_data.append(copy_entry)

    with open(output_augmented_filename, 'w', encoding='utf-8') as f:
        json.dump(dumpable_augmented_data, f, indent=4)
    logger.info(f"Augmented data saved to: {output_augmented_filename}")

    # --- Performance Profiling ---
    profile_data_augmenter_performance(num_samples=500) # Reduced for quick demo
    
    # Clean up dummy files/directories created during demonstration if not needed
    # shutil.rmtree("scraped_code_samples", ignore_errors=True)