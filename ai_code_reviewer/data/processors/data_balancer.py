# ai_code_reviewer/data/processors/data_balancer.py

import random
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataBalancer:
    """
    Balances a dataset to address class imbalance, particularly for bug-fix pairs
    where 'before_code' can be seen as positive (buggy) and 'after_code' as negative (fixed).
    """

    def __init__(self):
        """
        Initializes the DataBalancer.
        """
        logger.info("DataBalancer initialized.")

    def _prepare_binary_classification_data(self, bug_fix_pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepares data for binary classification (buggy vs. fixed) from bug-fix pairs.
        Each 'before_code' is treated as a positive (buggy) example.
        Each 'after_code' is treated as a negative (fixed) example.

        Args:
            bug_fix_pairs (List[Dict[str, Any]]): List of bug-fix commit data,
                                                  each containing 'before_code' and 'after_code'.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing two lists:
                                                                - positive_samples (buggy code with label 1)
                                                                - negative_samples (fixed code with label 0)
        """
        positive_samples = [] # Buggy code
        negative_samples = [] # Fixed code

        for pair in bug_fix_pairs:
            # Check if parsing was successful for both before and after ASTs
            # We are assuming 'parsing_successful' field from ASTParser output
            if not pair.get("parsing_successful", True):
                logger.debug(f"Skipping entry {pair.get('filepath', 'unknown')} due to AST parsing failure.")
                continue

            # Create a positive example from 'before_code'
            positive_example = pair.copy()
            positive_example["code"] = pair["before_code"]
            positive_example["ast_str"] = pair.get("before_ast_str")
            positive_example["ast_object"] = pair.get("before_ast_object")
            positive_example["label"] = 1 # 1 for buggy/vulnerable
            positive_samples.append(positive_example)

            # Create a negative example from 'after_code'
            negative_example = pair.copy()
            negative_example["code"] = pair["after_code"]
            negative_example["ast_str"] = pair.get("after_ast_str")
            negative_example["ast_object"] = pair.get("after_ast_object")
            negative_example["label"] = 0 # 0 for fixed/clean
            negative_samples.append(negative_example)
        
        logger.info(f"Prepared {len(positive_samples)} positive (buggy) samples and {len(negative_samples)} negative (fixed) samples.")
        return positive_samples, negative_samples

    def undersample_majority_class(
        self,
        positive_samples: List[Dict[str, Any]],
        negative_samples: List[Dict[str, Any]],
        random_state: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs random undersampling on the majority class to match the minority class count.
        Assumes positive_samples are the minority (buggy) and negative_samples are the majority (clean).
        If positive_samples are majority, it will undersample positive_samples.

        Args:
            positive_samples (List[Dict[str, Any]]): List of positive examples.
            negative_samples (List[Dict[str, Any]]): List of negative examples.
            random_state (Optional[int]): Seed for random number generator for reproducibility.

        Returns:
            List[Dict[str, Any]]: A new list of balanced samples.
        """
        if random_state is not None:
            random.seed(random_state)

        num_positive = len(positive_samples)
        num_negative = len(negative_samples)

        logger.info(f"Initial counts: Positive={num_positive}, Negative={num_negative}")

        if num_positive == 0 or num_negative == 0:
            logger.warning("One of the classes is empty. Cannot balance.")
            return positive_samples + negative_samples

        if num_positive == num_negative:
            logger.info("Classes are already balanced. No undersampling needed.")
            return positive_samples + negative_samples

        balanced_samples: List[Dict[str, Any]] = []

        if num_positive < num_negative:
            # Positive is minority, undersample negative
            sampled_negative = random.sample(negative_samples, num_positive)
            balanced_samples = positive_samples + sampled_negative
            logger.info(f"Undersampled negative class from {num_negative} to {num_positive}.")
        else:
            # Negative is minority (unlikely for bug-fix pairs, but handle generally)
            sampled_positive = random.sample(positive_samples, num_negative)
            balanced_samples = sampled_positive + negative_samples
            logger.info(f"Undersampled positive class from {num_positive} to {num_negative}.")
        
        random.shuffle(balanced_samples) # Shuffle the combined list
        logger.info(f"Dataset balanced. Total samples: {len(balanced_samples)}. New distribution: Positive={len([s for s in balanced_samples if s['label'] == 1])}, Negative={len([s for s in balanced_samples if s['label'] == 0])}")
        return balanced_samples

# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the DataBalancer class.
    """
    logger.info("Running unit tests for DataBalancer...")
    balancer = DataBalancer()

    # Test 1: Balanced input
    balanced_input = [
        {"before_code": "b1", "after_code": "a1", "parsing_successful": True},
        {"before_code": "b2", "after_code": "a2", "parsing_successful": True},
        {"before_code": "b3", "after_code": "a3", "parsing_successful": True},
    ]
    pos_samples, neg_samples = balancer._prepare_binary_classification_data(balanced_input)
    assert len(pos_samples) == 3 and len(neg_samples) == 3, "Test 1 Prepare: Incorrect counts."
    
    balanced_output = balancer.undersample_majority_class(pos_samples, neg_samples, random_state=42)
    assert len(balanced_output) == 6, "Test 1 Undersample: Incorrect total count."
    assert sum(1 for s in balanced_output if s['label'] == 1) == 3, "Test 1 Undersample: Positive count incorrect."
    assert sum(1 for s in balanced_output if s['label'] == 0) == 3, "Test 1 Undersample: Negative count incorrect."
    logger.info("Test 1 (Balanced input): PASSED")

    # Test 2: Imbalanced input (Negative majority)
    imbalanced_input_neg_majority = [
        {"before_code": "b1", "after_code": "a1", "parsing_successful": True},
        {"before_code": "b2", "after_code": "a2", "parsing_successful": True},
        {"before_code": "b3", "after_code": "a3", "parsing_successful": True},
        {"before_code": "b4", "after_code": "a4", "parsing_successful": True},
        {"before_code": "b5", "after_code": "a5", "parsing_successful": True},
    ]
    # Simulate more negative samples, e.g., if we added separate "clean" files
    # For now, let's just make the 'after_code' duplicates to simulate imbalance.
    # In a real scenario, you'd feed distinct negative samples.
    # For this test, let's manually create imbalance beyond the 1:1 of bug-fix pairs.
    pos_samples_imbalanced = [{"code": f"b{i}", "label": 1, "parsing_successful": True} for i in range(5)]
    neg_samples_imbalanced = [{"code": f"a{i}", "label": 0, "parsing_successful": True} for i in range(15)] # 15 negative, 5 positive

    assert len(pos_samples_imbalanced) == 5 and len(neg_samples_imbalanced) == 15, "Test 2 Prepare: Incorrect initial counts."
    
    balanced_output_imbalanced = balancer.undersample_majority_class(pos_samples_imbalanced, neg_samples_imbalanced, random_state=42)
    assert len(balanced_output_imbalanced) == 10, f"Test 2 Undersample: Expected 10 samples, got {len(balanced_output_imbalanced)}"
    assert sum(1 for s in balanced_output_imbalanced if s['label'] == 1) == 5, "Test 2 Undersample: Positive count incorrect."
    assert sum(1 for s in balanced_output_imbalanced if s['label'] == 0) == 5, "Test 2 Undersample: Negative count incorrect."
    logger.info("Test 2 (Imbalanced input - Negative majority): PASSED")

    # Test 3: Imbalanced input (Positive majority - unlikely but for robustness)
    pos_samples_imbalanced_p = [{"code": f"b{i}", "label": 1, "parsing_successful": True} for i in range(10)]
    neg_samples_imbalanced_p = [{"code": f"a{i}", "label": 0, "parsing_successful": True} for i in range(3)]
    
    balanced_output_imbalanced_p = balancer.undersample_majority_class(pos_samples_imbalanced_p, neg_samples_imbalanced_p, random_state=42)
    assert len(balanced_output_imbalanced_p) == 6, f"Test 3 Undersample: Expected 6 samples, got {len(balanced_output_imbalanced_p)}"
    assert sum(1 for s in balanced_output_imbalanced_p if s['label'] == 1) == 3, "Test 3 Undersample: Positive count incorrect."
    assert sum(1 for s in balanced_output_imbalanced_p if s['label'] == 0) == 3, "Test 3 Undersample: Negative count incorrect."
    logger.info("Test 3 (Imbalanced input - Positive majority): PASSED")

    # Test 4: Empty input
    pos_empty, neg_empty = balancer._prepare_binary_classification_data([])
    assert len(pos_empty) == 0 and len(neg_empty) == 0, "Test 4 Prepare: Non-empty output for empty input."
    balanced_empty = balancer.undersample_majority_class(pos_empty, neg_empty)
    assert len(balanced_empty) == 0, "Test 4 Undersample: Non-empty output for empty input."
    logger.info("Test 4 (Empty input): PASSED")

    # Test 5: Handling parsing failures
    data_with_failures = [
        {"before_code": "b1", "after_code": "a1", "parsing_successful": True},
        {"before_code": "b2", "after_code": "a2", "parsing_successful": False}, # This one should be skipped
        {"before_code": "b3", "after_code": "a3", "parsing_successful": True},
    ]
    pos_filtered, neg_filtered = balancer._prepare_binary_classification_data(data_with_failures)
    assert len(pos_filtered) == 2 and len(neg_filtered) == 2, "Test 5 Prepare: Failed to filter parsing failures."
    logger.info("Test 5 (Handling parsing failures): PASSED")

    logger.info("All DataBalancer unit tests completed.")

# --- Performance Profiling (Conceptual) ---
def profile_data_balancer_performance(num_samples: int = 10000, imbalance_ratio: float = 0.1):
    """
    Profiles the performance of the DataBalancer.

    Args:
        num_samples (int): Total number of bug-fix pairs to simulate.
        imbalance_ratio (float): Ratio of positive samples to negative samples
                                 (e.g., 0.1 means 10% positive, 90% negative of the total).
    """
    logger.info(f"\nStarting DataBalancer performance profiling with {num_samples} samples...")
    balancer = DataBalancer()

    # Simulate highly imbalanced input data
    num_positive_pairs = int(num_samples * imbalance_ratio)
    num_negative_pairs = num_samples - num_positive_pairs

    # Create dummy bug_fix_pairs for profiling
    dummy_bug_fix_pairs = []
    for i in range(num_positive_pairs):
        dummy_bug_fix_pairs.append({
            "before_code": f"buggy_code_{i}",
            "after_code": f"fixed_code_{i}",
            "language": "python",
            "parsing_successful": True,
            "before_ast_str": "...", # Simplified
            "after_ast_str": "..." # Simplified
        })
    # Add extra "clean" files to create stronger imbalance
    # (assuming 'after_code' generates fixed, but we also have truly independent clean code)
    # For this profiling, we'll just use the _prepare method's 1:1 from bug_fix_pairs
    # and then manually create an imbalance for the undersample function itself.

    pos_samples_profiling = [{"code": f"b{i}", "label": 1, "parsing_successful": True} for i in range(int(num_samples * imbalance_ratio))]
    neg_samples_profiling = [{"code": f"a{i}", "label": 0, "parsing_successful": True} for i in range(num_samples)] # Total negatives

    start_time = datetime.now()
    
    # First, prepare data (simulated with the full initial set if we didn't mock)
    # In a real scenario, this step is sequential from AST parsing
    # Here, we directly call the undersample function with pre-imbalanced lists
    balanced_data = balancer.undersample_majority_class(pos_samples_profiling, neg_samples_profiling, random_state=42)

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for DataBalancer:")
    logger.info(f"  Initial Positive samples: {len(pos_samples_profiling)}")
    logger.info(f"  Initial Negative samples: {len(neg_samples_profiling)}")
    logger.info(f"  Final total balanced samples: {len(balanced_data)}")
    logger.info(f"  Final positive samples: {sum(1 for s in balanced_data if s['label'] == 1)}")
    logger.info(f"  Final negative samples: {sum(1 for s in balanced_data if s['label'] == 0)}")
    logger.info(f"  Total duration: {duration}")
    if len(balanced_data) > 0:
        logger.info(f"  Average time per sample (approx): {duration / len(balanced_data)}")
    else:
        logger.info("  No samples to calculate average time.")


# --- Usage Example ---
if __name__ == "__main__":
    from datetime import datetime

    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Usage with Sample Data ---
    logger.info("\n--- Demonstrating Data Balancing with sample processed data ---")

    # Load previously processed data from AST parsing (if available)
    input_file = "scraped_code_samples/bug_fix_pairs_with_asts.json"
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}. Please run github_scraper.py and ast_parser.py first.")
        # Create a dummy data for demonstration if file not found
        sample_data_for_balancing = [
            # 5 'buggy' (before) and 5 'fixed' (after) pairs (1:1 ratio)
            {"repo_url": "dummy", "commit_hash": "c1", "filepath": "f1.py", "before_code": "b1", "after_code": "a1", "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""},
            {"repo_url": "dummy", "commit_hash": "c2", "filepath": "f2.py", "before_code": "b2", "after_code": "a2", "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""},
            {"repo_url": "dummy", "commit_hash": "c3", "filepath": "f3.py", "before_code": "b3", "after_code": "a3", "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""},
            {"repo_url": "dummy", "commit_hash": "c4", "filepath": "f4.py", "before_code": "b4", "after_code": "a4", "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""},
            {"repo_url": "dummy", "commit_hash": "c5", "filepath": "f5.py", "before_code": "b5", "after_code": "a5", "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""},
        ]
        # Add extra "negative" examples to simulate imbalance (e.g., from truly clean codebases)
        # For a robust dataset, these would come from scraping non-bugfix commits or known clean projects.
        for i in range(10): # Add 10 extra negative samples
             sample_data_for_balancing.append(
                {"repo_url": "dummy_clean", "commit_hash": f"clean_c{i}", "filepath": f"clean_f{i}.py",
                 "before_code": f"clean_code_{i}", "after_code": f"clean_code_{i}", # 'after_code' serves as the "clean" reference
                 "language": "python", "parsing_successful": True, "before_ast_str": "", "after_ast_str": ""}
            )
        # Note: The _prepare_binary_classification_data assumes 'before'/'after' pairs.
        # To truly simulate independent clean files, you'd feed them into `_prepare_binary_classification_data`
        # as separate "negative-only" entries. For this demo, we will use the generated `pos_samples` and `neg_samples`
        # from a mix, and then explicitly pass them to `undersample_majority_class`.
        logger.info("Using dummy data for demonstration, simulating an imbalance.")
        # Re-creating the simulation logic to be clear for demo:
        # 5 positive (from before_code) + 5 negative (from after_code of bug-fix pairs)
        # + 10 additional "clean" negatives (from "clean" codebases/files)
        demo_bug_fix_pairs = [
            {"before_code": "b1", "after_code": "a1", "language": "python", "parsing_successful": True},
            {"before_code": "b2", "after_code": "a2", "language": "python", "parsing_successful": True},
            {"before_code": "b3", "after_code": "a3", "language": "python", "parsing_successful": True},
        ]
        dummy_positive_samples, dummy_negative_samples_from_fixes = balancer._prepare_binary_classification_data(demo_bug_fix_pairs)
        
        # Add more negative samples to simulate imbalance from external clean code
        for i in range(len(dummy_positive_samples) * 3): # 3x more negatives
            dummy_negative_samples_from_fixes.append({"code": f"clean_extra_{i}", "label": 0, "language": "python", "parsing_successful": True, "ast_str": ""})
        
        input_pos_count = len(dummy_positive_samples)
        input_neg_count = len(dummy_negative_samples_from_fixes)
        logger.info(f"Demonstration input: {input_pos_count} positive, {input_neg_count} negative samples.")
        
        initial_positive_samples = dummy_positive_samples
        initial_negative_samples = dummy_negative_samples_from_fixes

    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        logger.info(f"Loaded {len(raw_data)} raw entries from {input_file} for balancing.")
        # Prepare data into positive/negative lists. This assumes each 'before' is positive and each 'after' is negative.
        initial_positive_samples, initial_negative_samples = balancer._prepare_binary_classification_data(raw_data)


    balancer_instance = DataBalancer()
    balanced_dataset = balancer_instance.undersample_majority_class(
        initial_positive_samples,
        initial_negative_samples,
        random_state=42 # For reproducible results
    )

    output_balanced_filename = "scraped_code_samples/balanced_bug_fix_data.json"
    # Remove AST objects before saving to JSON as they are not directly serializable
    dumpable_balanced_data = []
    for entry in balanced_dataset:
        copy_entry = entry.copy()
        copy_entry.pop("ast_object", None)
        dumpable_balanced_data.append(copy_entry)

    with open(output_balanced_filename, 'w', encoding='utf-8') as f:
        json.dump(dumpable_balanced_data, f, indent=4)
    logger.info(f"Balanced data saved to: {output_balanced_filename}")

    # --- Performance Profiling ---
    profile_data_balancer_performance(num_samples=5000, imbalance_ratio=0.2) # Simulate with more samples