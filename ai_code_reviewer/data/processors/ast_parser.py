# ai_code_reviewer/data/processors/ast_parser.py

import ast
import javalang
import logging
from typing import Optional, Dict, Any, Union
import json
import collections.abc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, collections.abc.Set):
            return list(obj)
        return super().default(obj)

class ASTParser:
    """
    Parses Python and Java code into Abstract Syntax Trees (ASTs).
    Uses Python's built-in 'ast' module for Python code and 'javalang' for Java code.
    """

    def parse_python_code(self, code: str) -> Optional[ast.AST]:
        """
        Parses a Python code string into its Abstract Syntax Tree.

        Args:
            code (str): The Python source code as a string.

        Returns:
            Optional[ast.AST]: The AST object if parsing is successful, None otherwise.
        """
        try:
            tree = ast.parse(code)
            logger.debug("Successfully parsed Python code into AST.")
            return tree
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python code during AST parsing: {e}")
            logger.debug(f"Problematic Python code snippet:\n{code[:200]}...") # Log beginning of problematic code
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Python AST parsing: {e}")
            return None

    def parse_java_code(self, code: str) -> Optional[javalang.tree.CompilationUnit]:
        """
        Parses a Java code string into its Abstract Syntax Tree.

        Args:
            code (str): The Java source code as a string.

        Returns:
            Optional[javalang.tree.CompilationUnit]: The AST object (CompilationUnit)
                                                    if parsing is successful, None otherwise.
        """
        try:
            # Javalang requires tokenizing first
            tokens = javalang.tokenizer.tokenize(code)
            tree = javalang.parser.parse(tokens)
            logger.debug("Successfully parsed Java code into AST.")
            return tree
        except javalang.tokenizer.LexerError as e:
            logger.warning(f"Lexer error in Java code during AST parsing: {e}")
            logger.debug(f"Problematic Java code snippet:\n{code[:200]}...")
            return None
        except javalang.parser.JavaSyntaxError as e:
            logger.warning(f"Syntax error in Java code during AST parsing: {e}")
            logger.debug(f"Problematic Java code snippet:\n{code[:200]}...")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Java AST parsing: {e}")
            return None

    def parse_code(self, code: str, language: str) -> Optional[Any]:
        """
        Parses code based on the specified language.

        Args:
            code (str): The source code as a string.
            language (str): The programming language ("python" or "java").

        Returns:
            Optional[Any]: The AST object (either ast.AST or javalang.tree.CompilationUnit)
                           if parsing is successful, None otherwise.
        """
        language_lower = language.lower()
        if language_lower == "python":
            return self.parse_python_code(code)
        elif language_lower == "java":
            return self.parse_java_code(code)
        else:
            logger.warning(f"Unsupported language for AST parsing: {language}. Skipping.")
            return None

    def ast_to_string(self, tree: Any, language: str) -> Optional[str]:
        """
        Converts an AST object into a JSON-serializable string representation.
        For Python, uses ast.dump. For Java, it converts to a dict then JSON string.

        Args:
            tree (Any): The AST object (ast.AST or javalang.tree.CompilationUnit).
            language (str): The programming language ("python" or "java").

        Returns:
            Optional[str]: A JSON string representation of the AST, or None if conversion fails.
        """
        language_lower = language.lower()
        if tree is None:
            return None

        try:
            if language_lower == "python":
                # Use ast.dump for a compact, readable representation of the AST structure
                return ast.dump(tree, indent=4)
            elif language_lower == "java":
                # For Java, convert the javalang AST to a serializable dictionary, then dump to JSON string
                return json.dumps(self._javalang_ast_to_dict(tree), indent=4, cls=SetEncoder)
            else:
                logger.warning(f"Unsupported language for AST to string conversion: {language}. Returning None.")
                return None
        except Exception as e:
            logger.error(f"Error converting AST to string for {language}: {e}", exc_info=True) # Added exc_info
            return None

    def _javalang_ast_to_dict(self, node: Any) -> Any:
        """
        Recursively converts a javalang AST node (or list of nodes) into a dictionary
        that is JSON serializable. This handles nested nodes and special javalang objects.
        """
        if node is None:
            return None
        
        # Check for primitive Python types that are JSON serializable
        if not isinstance(node, (javalang.tree.Node, list, dict)):
            return node # Return as is for int, str, bool, float, None, etc.

        # Robust recursive conversion for all types
        if isinstance(node, dict):
            return {k: self._javalang_ast_to_dict(v) for k, v in node.items()}
        if isinstance(node, set):
            return [self._javalang_ast_to_dict(item) for item in list(node)]
        if isinstance(node, list):
            return [self._javalang_ast_to_dict(item) for item in node]
        if isinstance(node, javalang.tree.Node):
            node_dict = {
                "node_type": node.__class__.__name__
            }
            for field in node.attrs:
                value = getattr(node, field, None)
                if value is not None:
                    node_dict[field] = self._javalang_ast_to_dict(value)
            return node_dict
        return str(node)


# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the ASTParser class.
    """
    logger.info("Running unit tests for ASTParser...")
    parser = ASTParser()

    # Test Python parsing - valid code
    python_code_valid = "def func(x):\n    return x + 1"
    python_ast = parser.parse_python_code(python_code_valid)
    assert python_ast is not None, "Test Python Valid: Failed to parse valid Python code."
    assert isinstance(python_ast, ast.AST), "Test Python Valid: Returned object is not an AST."
    logger.info("Test Python Valid: PASSED")

    # Test Python parsing - invalid code
    python_code_invalid = "def func(x):\n    return x +"
    python_ast_invalid = parser.parse_python_code(python_code_invalid)
    assert python_ast_invalid is None, "Test Python Invalid: Did not return None for invalid Python code."
    logger.info("Test Python Invalid: PASSED")

    # Test Java parsing - valid code
    java_code_valid = """
    public class MyClass {
        public int add(int a, int b) {
            return a + b;
        }
    }
    """
    java_ast = parser.parse_java_code(java_code_valid)
    assert java_ast is not None, "Test Java Valid: Failed to parse valid Java code."
    assert isinstance(java_ast, javalang.tree.CompilationUnit), "Test Java Valid: Returned object is not a CompilationUnit."
    logger.info("Test Java Valid: PASSED")

    # Test Java parsing - invalid code
    java_code_invalid = """
    public class MyClass {
        public int add(int a, int b)
            return a + b;
        }
    """ # Missing bracket and semicolon
    java_ast_invalid = parser.parse_java_code(java_code_invalid)
    assert java_ast_invalid is None, "Test Java Invalid: Did not return None for invalid Java code."
    logger.info("Test Java Invalid: PASSED")

    # Test parse_code method
    parsed_python = parser.parse_code(python_code_valid, "python")
    assert parsed_python is not None and isinstance(parsed_python, ast.AST), "Test parse_code Python: Failed."
    parsed_java = parser.parse_code(java_code_valid, "java")
    assert parsed_java is not None and isinstance(parsed_java, javalang.tree.CompilationUnit), "Test parse_code Java: Failed."
    parsed_unsupported = parser.parse_code("some code", "rust")
    assert parsed_unsupported is None, "Test parse_code Unsupported: Failed."
    logger.info("Test parse_code method: PASSED")

    # Test ast_to_string (CRITICAL CHECK HERE)
    python_ast_str = parser.ast_to_string(python_ast, "python")
    assert isinstance(python_ast_str, str) and "FunctionDef" in python_ast_str, "Test ast_to_string Python: Failed."
    
    # Ensure Java AST to string also produces valid JSON string
    java_ast_str = parser.ast_to_string(java_ast, "java")
    assert isinstance(java_ast_str, str), "Test ast_to_string Java: Not a string."
    try:
        json_parsed_java_ast = json.loads(java_ast_str)
        assert isinstance(json_parsed_java_ast, dict), "Test ast_to_string Java: Not valid JSON dict."
        assert "node_type" in json_parsed_java_ast and "CompilationUnit" in json_parsed_java_ast["node_type"], "Test ast_to_string Java: Missing expected node_type."
        logger.info("Test ast_to_string Java: PASSED (produced valid JSON string).")
    except json.JSONDecodeError as e:
        logger.error(f"Test ast_to_string Java: FAILED (produced invalid JSON string): {e}")
        assert False, f"Test ast_to_string Java: Produced invalid JSON: {java_ast_str[:100]}..."

    logger.info("All ASTParser unit tests completed.")


# --- Performance Profiling (Conceptual) ---
def profile_ast_parser_performance(num_iterations: int = 1000):
    """
    Profiles the performance of the ASTParser with sample code.
    """
    logger.info(f"\nStarting ASTParser performance profiling for {num_iterations} iterations...")
    parser = ASTParser()

    python_code = "def foo(a, b):\n    if a > b:\n        return a - b\n    else:\n        return b - a\n"
    java_code = """
    public class PerformanceTest {
        public int calculate(int x, int y) {
            if (x > y) {
                return x - y;
            } else {
                return y - x;
            }
        }
    }
    """

    start_time = datetime.now()
    parsed_python_count = 0
    parsed_java_count = 0

    for _ in range(num_iterations):
        if parser.parse_python_code(python_code):
            parsed_python_count += 1
        if parser.parse_java_code(java_code):
            parsed_java_count += 1

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for ASTParser:")
    logger.info(f"  Total Python parses: {parsed_python_count}/{num_iterations}")
    logger.info(f"  Total Java parses: {parsed_java_count}/{num_iterations}")
    logger.info(f"  Total duration: {duration}")
    if (parsed_python_count + parsed_java_count) > 0:
        logger.info(f"  Average time per parse (approx): {duration / (parsed_python_count + parsed_java_count)}")
    else:
        logger.info("  No successful parses to calculate average time.")


# --- Usage Example ---
if __name__ == "__main__":
    from datetime import datetime
    import json
    import os

    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Usage with Scraped Data ---
    logger.info("\n--- Demonstrating AST Parsing with sample scraped data ---")

    # Assuming 'bug_fix_pairs.json' exists from the GitHubScraper output
    input_file = "scraped_code_samples/bug_fix_pairs.json"
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}. Please run github_scraper.py first.")
        # Create a dummy data for demonstration if file not found
        sample_data = [
            {
                "repo_url": "dummy_repo",
                "commit_hash": "dummy_hash_py",
                "filepath": "buggy_py.py",
                "before_code": "def old_func(x):\n    return x - 1",
                "after_code": "def old_func(x):\n    return x + 1",
                "commit_message": "Fix: typo",
                "language": "python",
                "commit_time": "2024-01-01T00:00:00"
            },
            {
                "repo_url": "dummy_repo",
                "commit_hash": "dummy_hash_java",
                "filepath": "BuggyJava.java",
                "before_code": "public class Buggy {\n    public void doSomething() {\n        int x = 5\n    }\n}",
                "after_code": "public class Buggy {\n    public void doSomething() {\n        int x = 5;\n    }\n}",
                "commit_message": "Fix: missing semicolon",
                "language": "java",
                "commit_time": "2024-01-02T00:00:00"
            },
            {
                "repo_url": "dummy_repo",
                "commit_hash": "dummy_hash_invalid_py",
                "filepath": "invalid_py.py",
                "before_code": "def bad_func(y):\n    print('hello'", # Missing closing parenthesis
                "after_code": "def bad_func(y):\n    print('hello')",
                "commit_message": "Fix: bad syntax",
                "language": "python",
                "commit_time": "2024-01-03T00:00:00"
            }
        ]
        logger.info("Using dummy data for demonstration.")
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        logger.info(f"Loaded {len(sample_data)} entries from {input_file}")

    parser = ASTParser()
    processed_data = []

    for entry in sample_data:
        language = entry.get("language")
        before_code = entry.get("before_code")
        after_code = entry.get("after_code")
        filepath = entry.get("filepath")
        commit_hash = entry.get("commit_hash")

        if not all([language, before_code, after_code]):
            logger.warning(f"Skipping entry due to missing data: {entry.get('filepath')}")
            continue

        logger.info(f"Processing {filepath} ({language}) from commit {commit_hash[:7]}...")

        before_ast = parser.parse_code(before_code, language)
        after_ast = parser.parse_code(after_code, language)

        # Store string representation of ASTs for easier debugging/storage.
        # For actual GNNs, you'd work with the AST objects directly.
        entry["before_ast_str"] = parser.ast_to_string(before_ast, language)
        entry["after_ast_str"] = parser.ast_to_string(after_ast, language)
        entry["before_ast_object"] = before_ast # Keep the object if needed downstream
        entry["after_ast_object"] = after_ast # Keep the object if needed downstream

        if before_ast is None or after_ast is None:
            logger.warning(f"Failed to parse AST for {filepath} in commit {commit_hash[:7]}.")
            entry["parsing_successful"] = False
        else:
            entry["parsing_successful"] = True
            logger.info(f"Successfully parsed ASTs for {filepath}.")
            # logger.debug(f"Before AST:\n{entry['before_ast_str']}")
            # logger.debug(f"After AST:\n{entry['after_ast_str']}")

        processed_data.append(entry)

    output_ast_filename = "scraped_code_samples/bug_fix_pairs_with_asts.json"
    # Note: Cannot directly JSON dump AST objects, only their string representations.
    # So, we'll remove the actual AST objects before dumping.
    dumpable_data = []
    for entry in processed_data:
        copy_entry = entry.copy()
        copy_entry.pop("before_ast_object", None)
        copy_entry.pop("after_ast_object", None)
        dumpable_data.append(copy_entry)

    os.makedirs(os.path.dirname(output_ast_filename), exist_ok=True) # Ensure dir exists
    with open(output_ast_filename, 'w', encoding='utf-8') as f:
        json.dump(dumpable_data, f, indent=4)
    logger.info(f"Processed data with AST string representations saved to: {output_ast_filename}")
    logger.info(f"Example of first successful entry's AST (if available):")
    for entry in processed_data:
        if entry["parsing_successful"]:
            logger.info(f"  File: {entry['filepath']}")
            logger.info(f"  Language: {entry['language']}")
            # Print only first 500 chars to avoid very long output
            logger.info(f"  Before AST Sample:\n{str(entry['before_ast_str'])[:500]}...") 
            logger.info(f"  After AST Sample:\n{str(entry['after_ast_str'])[:500]}...")
            break
    else:
        logger.warning("No successful AST parsing examples found in the collected data.")

    # --- Performance Profiling ---
    profile_ast_parser_performance(num_iterations=200) # Reduced iterations for quick test
