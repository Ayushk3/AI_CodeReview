# ai_code_reviewer/data/datasets/dataset_builder.py

import json
import os
import logging
import random
from typing import List, Dict, Any, Tuple, Optional

# Import components from other modules
from ai_code_reviewer.data.processors.ast_parser import ASTParser
from ai_code_reviewer.data.processors.data_balancer import DataBalancer
from ai_code_reviewer.data.processors.data_augmenter import CodeDataAugmenter
from ai_code_reviewer.data.collectors.cve_integrator import CVEIntegrator # For potential metadata enrichment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Orchestrates the data processing pipeline to build a final dataset
    for bug and vulnerability detection. It combines scraping, AST parsing,
    balancing, augmentation, and data splitting.
    """

    def __init__(
        self,
        raw_data_path: str = "scraped_code_samples/bug_fix_pairs.json",
        cve_data_path: str = "collected_cve_data/recent_cves.json",
        output_dir: str = "data/datasets",
        random_seed: Optional[int] = None
    ):
        """
        Initializes the DatasetBuilder with paths and processing parameters.

        Args:
            raw_data_path (str): Path to the initial scraped bug-fix pairs JSON file.
            cve_data_path (str): Path to the collected CVE data JSON file.
            output_dir (str): Directory where the final datasets (train/val/test) will be saved.
            random_seed (Optional[int]): Seed for reproducibility of random operations (splitting, augmentation).
        """
        self.raw_data_path = raw_data_path
        self.cve_data_path = cve_data_path
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.ast_parser = ASTParser()
        self.data_balancer = DataBalancer()
        self.code_augmenter = CodeDataAugmenter(random_seed=random_seed) # Pass seed to augmenter
        
        if random_seed is not None:
            random.seed(random_seed)
        
        logger.info(f"DatasetBuilder initialized. Output directory: {self.output_dir}")

    def _load_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Loads data from a JSON file."""
        if not os.path.exists(filepath):
            logger.error(f"Required input file not found: {filepath}")
            return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entries from {filepath}.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return []

    def _enrich_with_cve_info(self, dataset: List[Dict[str, Any]], cve_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        (Conceptual) Enriches code entries with CVE information based on heuristic matching (e.g., CVE ID in commit message).
        This is a basic placeholder; robust CVE-to-code linking is very complex.

        Args:
            dataset (List[Dict[str, Any]]): The list of code entries (e.g., bug-fix pairs).
            cve_data (List[Dict[str, Any]]): The list of collected CVE records.

        Returns:
            List[Dict[str, Any]]: The dataset with potentially enriched CVE information.
        """
        if not cve_data:
            logger.info("No CVE data provided for enrichment.")
            return dataset

        cve_map = {cve['id'].lower(): cve for cve in cve_data if 'id' in cve}
        enriched_dataset = []

        for entry in dataset:
            commit_message = entry.get("commit_message", "").lower()
            found_cves = []
            for cve_id_full, cve_record in cve_map.items():
                # Simple check: does commit message contain the CVE ID?
                # More advanced: NLP to match description, CWEs, etc.
                if cve_id_full in commit_message:
                    # Append relevant CVE info, not the full massive record
                    found_cves.append({
                        "cve_id": cve_record.get('id'),
                        "description": cve_record.get('descriptions', [{'value': ''}])[0].get('value'),
                        "cvss_v3_base_score": cve_record.get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseScore'),
                        "cwe_ids": [weakness.get('weakness').get('id') for weakness in cve_record.get('weaknesses', []) if weakness.get('weakness')]
                    })
            if found_cves:
                entry["associated_cves"] = found_cves
                logger.debug(f"Associated CVEs to commit {entry.get('commit_hash', '')[:7]}")
            enriched_dataset.append(entry)
        
        logger.info(f"Attempted CVE enrichment for {len(enriched_dataset)} entries.")
        return enriched_dataset

    def build_dataset(
        self,
        apply_balancing: bool = True,
        augmentation_factor: int = 1, # 1 means keep original, 2 means original + 1 augmented copy
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Builds the final train, validation, and test datasets.

        Args:
            apply_balancing (bool): Whether to apply undersampling for class balancing.
            augmentation_factor (int): How many augmented copies to create for each original sample (0 for no augmentation).
                                       If > 0, original sample is always kept, and factor-1 new augmented samples are added.
            train_ratio (float): Proportion of data for the training set.
            val_ratio (float): Proportion of data for the validation set.
            test_ratio (float): Proportion of data for the test set.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            (train_dataset, val_dataset, test_dataset)
        """
        if not (0 < train_ratio + val_ratio + test_ratio <= 1.0):
            raise ValueError("Train, validation, and test ratios must sum up to <= 1.0 and be positive.")
        
        logger.info("Starting dataset build process...")
        raw_data = self._load_data(self.raw_data_path)
        if not raw_data:
            logger.error("No raw data loaded. Cannot build dataset.")
            return [], [], []

        cve_data = self._load_data(self.cve_data_path)
        processed_data = self._enrich_with_cve_info(raw_data, cve_data)
        
        # Step 1: Prepare binary classification data (buggy vs. fixed/clean)
        # Each 'before_code' becomes a positive sample (label 1)
        # Each 'after_code' becomes a negative sample (label 0)
        # This is where we might also integrate truly "clean" external code as negative samples
        # if such data was scraped and provided separately. For now, it's 1:1 from bug_fix_pairs.
        positive_samples, negative_samples = self.data_balancer._prepare_binary_classification_data(processed_data)

        # Combine all samples for subsequent processing steps
        all_samples = positive_samples + negative_samples
        logger.info(f"Initial total samples after preparation: {len(all_samples)}")

        # Step 2: Augmentation
        if augmentation_factor > 0:
            logger.info(f"Applying data augmentation with factor {augmentation_factor}...")
            augmented_dataset_list: List[Dict[str, Any]] = []
            
            # Augment original samples. It's often beneficial to augment the minority class more.
            # For simplicity here, we apply augmentation to all samples if requested.
            # A more sophisticated approach would target 'label=1' samples for more augmentation.
            
            for entry in all_samples:
                augmented_dataset_list.append(entry) # Always keep original
                for _ in range(augmentation_factor -1): # Create (factor - 1) new augmented copies
                    augmented_entry = self.code_augmenter.augment_code_entry(entry)
                    if augmented_entry:
                        augmented_dataset_list.append(augmented_entry)
            all_samples = augmented_dataset_list
            logger.info(f"Total samples after augmentation: {len(all_samples)}")
            
            # Re-parse ASTs for all samples, especially augmented ones
            logger.info("Re-parsing ASTs for all samples (including augmented ones)...")
            re_parsed_samples = []
            for entry in all_samples:
                if entry.get("ast_object") is None or entry.get("ast_str") is None or entry.get("is_augmented", False):
                    # Re-parse if AST is missing or it's an augmented entry
                    code = entry.get("code")
                    language = entry.get("language")
                    if code and language:
                        ast_obj = self.ast_parser.parse_code(code, language)
                        entry["ast_object"] = ast_obj
                        entry["ast_str"] = self.ast_parser.ast_to_string(ast_obj, language)
                        if ast_obj is None:
                            entry["parsing_successful"] = False
                            logger.warning(f"Failed to re-parse AST for augmented/original entry: {entry.get('filepath')}. Skipping this entry.")
                        else:
                            entry["parsing_successful"] = True
                    else:
                        entry["parsing_successful"] = False
                        logger.warning(f"Missing code/language for re-parsing entry: {entry.get('filepath')}. Skipping this entry.")
                re_parsed_samples.append(entry)
            all_samples = [s for s in re_parsed_samples if s.get("parsing_successful", False)]
            logger.info(f"Total samples after AST re-parsing and filtering invalid: {len(all_samples)}")


        # Step 3: Balancing (applied AFTER augmentation to ensure final class balance)
        final_positive_samples = [s for s in all_samples if s['label'] == 1]
        final_negative_samples = [s for s in all_samples if s['label'] == 0]

        if apply_balancing:
            logger.info("Applying data balancing (undersampling)...")
            balanced_samples = self.data_balancer.undersample_majority_class(
                final_positive_samples, final_negative_samples, random_state=random.randint(0, 10000) # Use a fresh random seed for this step
            )
        else:
            logger.info("Skipping data balancing.")
            balanced_samples = final_positive_samples + final_negative_samples
            random.shuffle(balanced_samples) # Ensure randomness even if not balancing

        logger.info(f"Final total samples after balancing: {len(balanced_samples)}")

        # Step 4: Split the dataset
        random.shuffle(balanced_samples) # Shuffle before splitting to ensure random distribution
        
        total_samples = len(balanced_samples)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        train_dataset = balanced_samples[:train_end]
        val_dataset = balanced_samples[train_end:val_end]
        test_dataset = balanced_samples[val_end:]

        logger.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        # Step 5: Save datasets
        self._save_dataset(train_dataset, "train_dataset.json")
        self._save_dataset(val_dataset, "val_dataset.json")
        self._save_dataset(test_dataset, "test_dataset.json")

        logger.info("Dataset build process completed successfully.")
        return train_dataset, val_dataset, test_dataset

    def _save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Saves a dataset list to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Remove AST objects before saving as they are not JSON serializable
        dumpable_dataset = []
        for entry in dataset:
            copy_entry = entry.copy()
            copy_entry.pop("ast_object", None)
            dumpable_dataset.append(copy_entry)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dumpable_dataset, f, indent=4)
            logger.info(f"Saved {len(dataset)} entries to {filepath}")
        except Exception as e:
            logger.error(f"Error saving dataset to {filepath}: {e}")

# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the DatasetBuilder class.
    This will require dummy input files to simulate the pipeline.
    """
    logger.info("Running unit tests for DatasetBuilder...")
    
    # Create dummy input files
    dummy_raw_data_dir = "test_raw_data"
    os.makedirs(dummy_raw_data_dir, exist_ok=True)
    dummy_cve_data_dir = "test_cve_data"
    os.makedirs(dummy_cve_data_dir, exist_ok=True)
    dummy_output_dir = "test_datasets"
    
    # Dummy bug_fix_pairs data (simulates output from GitHubScraper/ASTParser)
    dummy_bug_fix_pairs = [
        {"repo_url": "repo1", "commit_hash": "c1", "filepath": "f1.py",
         "before_code": "def old_py(x): return x - 1", "after_code": "def new_py(x): return x + 1",
         "language": "python", "commit_message": "Fix for CVE-2023-1234", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "repo1", "commit_hash": "c2", "filepath": "f2.java",
         "before_code": "public class Old { int x = 0; }", "after_code": "public class New { int x = 1; }",
         "language": "java", "commit_message": "Bugfix something", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "repo2", "commit_hash": "c3", "filepath": "f3.py",
         "before_code": "def vulnerable_code(): pass", "after_code": "def secure_code(): pass",
         "language": "python", "commit_message": "Security fix CVE-2023-5678", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "repo3", "commit_hash": "c4", "filepath": "f4.py",
         "before_code": "def other_bug(y): y /= 0", "after_code": "def other_bug(y): y = 1",
         "language": "python", "commit_message": "Another bug fix", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
         # Simulate some entries that might fail AST parsing to ensure filtering
        {"repo_url": "repo4", "commit_hash": "c5", "filepath": "f5.py",
         "before_code": "def syntax_err(:", "after_code": "def syntax_err():",
         "language": "python", "commit_message": "Syntax fix", "parsing_successful": False,
         "before_ast_str": None, "after_ast_str": None}
    ]
    dummy_raw_filepath = os.path.join(dummy_raw_data_dir, "bug_fix_pairs.json")
    with open(dummy_raw_filepath, 'w', encoding='utf-8') as f:
        json.dump(dummy_bug_fix_pairs, f, indent=4)

    # Dummy CVE data
    dummy_cves = [
        {"id": "CVE-2023-1234", "descriptions": [{"value": "A critical vulnerability in X library."}], "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]}, "weaknesses": [{"weakness": {"id": "CWE-100"}}]},
        {"id": "CVE-2023-5678", "descriptions": [{"value": "Another security flaw in Y library."}], "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]}, "weaknesses": [{"weakness": {"id": "CWE-200"}}]}
    ]
    dummy_cve_filepath = os.path.join(dummy_cve_data_dir, "recent_cves.json")
    with open(dummy_cve_filepath, 'w', encoding='utf-8') as f:
        json.dump(dummy_cves, f, indent=4)

    try:
        # Test 1: Basic build without augmentation and balancing
        builder = DatasetBuilder(
            raw_data_path=dummy_raw_filepath,
            cve_data_path=dummy_cve_filepath,
            output_dir=dummy_output_dir,
            random_seed=42
        )
        train_ds, val_ds, test_ds = builder.build_dataset(
            apply_balancing=False, augmentation_factor=1,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        total_entries = len(dummy_bug_fix_pairs) - 1 # One entry is skipped due to parsing_successful=False
        expected_total_prepared = (total_entries) * 2 # Each bug_fix_pair becomes 2 (before/after)
        
        # Check initial preparation and filtering
        initial_pos, initial_neg = builder.data_balancer._prepare_binary_classification_data(dummy_bug_fix_pairs)
        assert len(initial_pos) == total_entries, f"Test 1 Prepare: Expected {total_entries} positive, got {len(initial_pos)}"
        assert len(initial_neg) == total_entries, f"Test 1 Prepare: Expected {total_entries} negative, got {len(initial_neg)}"

        assert len(train_ds) + len(val_ds) + len(test_ds) == expected_total_prepared, "Test 1: Total samples mismatch."
        assert any("associated_cves" in entry for entry in train_ds), "Test 1: CVE enrichment not applied."
        assert all("ast_str" in entry and entry["ast_str"] is not None for entry in train_ds), "Test 1: AST strings missing or None."
        logger.info("Test 1 (Basic build): PASSED")

        # Test 2: Build with augmentation (should increase dataset size)
        builder_aug = DatasetBuilder(
            raw_data_path=dummy_raw_filepath,
            cve_data_path=dummy_cve_filepath,
            output_dir=dummy_output_dir,
            random_seed=42
        )
        _, _, _ = builder_aug.build_dataset(
            apply_balancing=False, augmentation_factor=2, # Each original becomes 1 (orig) + 1 (aug) = 2
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        # Expected total is (total_entries * 2) * augmentation_factor
        expected_augmented_total = expected_total_prepared * 2
        # Need to re-load to get the actual count from the builder's state
        final_train_ds = builder_aug._load_data(os.path.join(dummy_output_dir, "train_dataset.json"))
        final_val_ds = builder_aug._load_data(os.path.join(dummy_output_dir, "val_dataset.json"))
        final_test_ds = builder_aug._load_data(os.path.join(dummy_output_dir, "test_dataset.json"))

        assert len(final_train_ds) + len(final_val_ds) + len(final_test_ds) >= expected_augmented_total * 0.9, "Test 2: Augmentation did not significantly increase size."
        assert any(entry.get("is_augmented") for entry in final_train_ds), "Test 2: No augmented samples found."
        logger.info("Test 2 (With augmentation): PASSED")

        # Test 3: Build with balancing (should make positive/negative counts equal)
        builder_bal = DatasetBuilder(
            raw_data_path=dummy_raw_filepath,
            cve_data_path=dummy_cve_filepath,
            output_dir=dummy_output_dir,
            random_seed=42
        )
        train_ds_bal, val_ds_bal, test_ds_bal = builder_bal.build_dataset(
            apply_balancing=True, augmentation_factor=1,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Count positive/negative in training set
        train_pos_count = sum(1 for s in train_ds_bal if s['label'] == 1)
        train_neg_count = sum(1 for s in train_ds_bal if s['label'] == 0)
        # Undersampling should make them equal
        assert train_pos_count == train_neg_count, f"Test 3: Training set not balanced. Pos={train_pos_count}, Neg={train_neg_count}"
        logger.info("Test 3 (With balancing): PASSED")

    except Exception as e:
        logger.error(f"Unit Test FAILED: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        shutil.rmtree(dummy_raw_data_dir, ignore_errors=True)
        shutil.rmtree(dummy_cve_data_dir, ignore_errors=True)
        shutil.rmtree(dummy_output_dir, ignore_errors=True)
        logger.info("Unit test cleanup complete.")

# --- Performance Profiling (Conceptual) ---
def profile_dataset_builder_performance(num_raw_entries: int = 500, augmentation_factor: int = 2):
    """
    Profiles the performance of the DatasetBuilder.
    This will simulate the full pipeline.
    """
    logger.info(f"\nStarting DatasetBuilder performance profiling with {num_raw_entries} raw entries and augmentation factor {augmentation_factor}...")
    
    # Create dummy raw data on the fly for profiling
    profiling_raw_data = []
    for i in range(num_raw_entries):
        lang = "python" if i % 2 == 0 else "java"
        profiling_raw_data.append({
            "repo_url": f"repo_prof{i//10}", "commit_hash": f"prof_c{i}", "filepath": f"prof_f{i}.{lang.replace('python','py')}",
            "before_code": "def func_a(): return 1 + 1" if lang == "python" else "public class A { int x = 1; }",
            "after_code": "def func_a(): return 2 + 2" if lang == "python" else "public class A { int x = 2; }",
            "language": lang, "commit_message": f"Fixing bug {i}", "parsing_successful": True,
            "before_ast_str": "...", "after_ast_str": "..."
        })
    
    # Save dummy raw data
    profiling_raw_data_dir = "temp_profiling_raw_data"
    os.makedirs(profiling_raw_data_dir, exist_ok=True)
    profiling_raw_filepath = os.path.join(profiling_raw_data_dir, "profiling_bug_fix_pairs.json")
    with open(profiling_raw_filepath, 'w', encoding='utf-8') as f:
        json.dump(profiling_raw_data, f, indent=4)

    # Empty CVE data for profiling simplicity
    profiling_cve_data_dir = "temp_profiling_cve_data"
    os.makedirs(profiling_cve_data_dir, exist_ok=True)
    profiling_cve_filepath = os.path.join(profiling_cve_data_dir, "profiling_cves.json")
    with open(profiling_cve_filepath, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=4) # Empty CVE for faster profiling

    profiling_output_dir = "temp_profiling_datasets"

    builder = DatasetBuilder(
        raw_data_path=profiling_raw_filepath,
        cve_data_path=profiling_cve_filepath,
        output_dir=profiling_output_dir,
        random_seed=42
    )

    start_time = datetime.now()
    
    try:
        train_ds, val_ds, test_ds = builder.build_dataset(
            apply_balancing=True, augmentation_factor=augmentation_factor,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
    except Exception as e:
        logger.error(f"Error during DatasetBuilder profiling: {e}")
        train_ds, val_ds, test_ds = [], [], []
    finally:
        # Clean up profiling files/directories
        shutil.rmtree(profiling_raw_data_dir, ignore_errors=True)
        shutil.rmtree(profiling_cve_data_dir, ignore_errors=True)
        shutil.rmtree(profiling_output_dir, ignore_errors=True)

    end_time = datetime.now()
    duration = end_time - start_time

    total_final_samples = len(train_ds) + len(val_ds) + len(test_ds)

    logger.info(f"Profiling complete for DatasetBuilder:")
    logger.info(f"  Initial raw entries: {num_raw_entries}")
    logger.info(f"  Augmentation factor: {augmentation_factor}")
    logger.info(f"  Total samples in final dataset: {total_final_samples}")
    logger.info(f"  Total duration: {duration}")
    if total_final_samples > 0:
        logger.info(f"  Average time per final sample: {duration / total_final_samples}")
    else:
        logger.info("  No samples generated for average time calculation.")


# --- Usage Example ---
if __name__ == "__main__":
    import shutil
    from datetime import datetime

    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Full Dataset Build Process ---
    logger.info("\n--- Demonstrating Full Dataset Build Process ---")

    # IMPORTANT: Ensure 'scraped_code_samples/bug_fix_pairs.json'
    # and 'collected_cve_data/recent_cves.json' exist by running:
    # 1. python data/collectors/github_scraper.py
    # 2. python data/collectors/cve_integrator.py (Optional, but good for enrichment)
    #    (The ast_parser.py and data_balancer.py outputs are often intermediate and passed internally)

    # Let's create a simplified 'bug_fix_pairs_with_asts.json' here for a self-contained demo
    # in case previous steps weren't fully run or you want to start fresh.
    demo_scraped_data_dir = "scraped_code_samples"
    os.makedirs(demo_scraped_data_dir, exist_ok=True)
    
    demo_raw_data_for_builder = [
        {"repo_url": "demo_repo", "commit_hash": "c_py_1", "filepath": "file_py_1.py",
         "before_code": "def old_py(x):\n    return x - 1 #bug", "after_code": "def new_py(x):\n    return x + 1",
         "language": "python", "commit_message": "Fix py logic", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "demo_repo", "commit_hash": "c_java_1", "filepath": "file_java_1.java",
         "before_code": "public class Buggy {\n    int a = 0;\n}", "after_code": "public class Buggy {\n    int a = 1;\n}",
         "language": "java", "commit_message": "Fix java null ptr", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "demo_repo", "commit_hash": "c_py_2", "filepath": "file_py_2.py",
         "before_code": "def calc(v):\n    res = v*v #bug", "after_code": "def calc(v):\n    res = v**2",
         "language": "python", "commit_message": "Fix math error", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "demo_repo", "commit_hash": "c_py_3", "filepath": "file_py_3.py",
         "before_code": "def security_vuln(data):\n    eval(data) #CVE-2023-9999", "after_code": "def security_vuln(data):\n    print(data)",
         "language": "python", "commit_message": "Security fix for CVE-2023-9999", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        # Add more entries to create an imbalance initially for demonstration of balancing
        {"repo_url": "demo_repo", "commit_hash": "c_java_2", "filepath": "file_java_2.java",
         "before_code": "class A { void foo() {} }", "after_code": "class A { void foo() { System.out.println(\"hello\"); } }",
         "language": "java", "commit_message": "Adding log", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
        {"repo_url": "demo_repo", "commit_hash": "c_py_4", "filepath": "file_py_4.py",
         "before_code": "x = 10", "after_code": "x = 20",
         "language": "python", "commit_message": "Update constant", "parsing_successful": True,
         "before_ast_str": "...", "after_ast_str": "..."},
    ]
    demo_raw_data_path = os.path.join(demo_scraped_data_dir, "bug_fix_pairs.json")
    with open(demo_raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(demo_raw_data_for_builder, f, indent=4)
    logger.info(f"Created dummy raw data at {demo_raw_data_path}")

    # Create dummy CVE data for enrichment demo
    demo_cve_data_dir = "collected_cve_data"
    os.makedirs(demo_cve_data_dir, exist_ok=True)
    demo_cve_data = [{"id": "CVE-2023-9999", "descriptions": [{"value": "Arbitrary code execution."}]}]
    demo_cve_data_path = os.path.join(demo_cve_data_dir, "recent_cves.json")
    with open(demo_cve_data_path, 'w', encoding='utf-8') as f:
        json.dump(demo_cve_data, f, indent=4)
    logger.info(f"Created dummy CVE data at {demo_cve_data_path}")

    dataset_output_dir = "data/datasets" # This is the standard output for builder

    builder = DatasetBuilder(
        raw_data_path=demo_raw_data_path,
        cve_data_path=demo_cve_data_path,
        output_dir=dataset_output_dir,
        random_seed=42 # For reproducibility
    )

    train_set, val_set, test_set = builder.build_dataset(
        apply_balancing=True,      # Apply undersampling
        augmentation_factor=2,     # Original + 1 augmented copy per original sample
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    logger.info(f"\nFinal Dataset Sizes:")
    logger.info(f"  Train: {len(train_set)} samples")
    logger.info(f"  Validation: {len(val_set)} samples")
    logger.info(f"  Test: {len(test_set)} samples")

    # Verify a sample from the train set (look for augmentation/labels)
    if train_set:
        sample = train_set[0]
        logger.info(f"\nSample from Train Set:")
        logger.info(f"  File: {sample.get('filepath')}")
        logger.info(f"  Language: {sample.get('language')}")
        logger.info(f"  Label: {sample.get('label')} (1=buggy/vulnerable, 0=fixed/clean)")
        logger.info(f"  Is Augmented: {sample.get('is_augmented', False)}")
        if sample.get('associated_cves'):
            logger.info(f"  Associated CVEs: {sample.get('associated_cves')}")
        # logger.debug(f"  Code: \n{sample.get('code')}")
        # logger.debug(f"  AST String: \n{sample.get('ast_str')}")
    
    # --- Performance Profiling ---
    profile_dataset_builder_performance(num_raw_entries=200, augmentation_factor=2)

    # Clean up dummy files/directories created during demonstration if not needed
    shutil.rmtree(demo_scraped_data_dir, ignore_errors=True)
    shutil.rmtree(demo_cve_data_dir, ignore_errors=True)
    # Keep the actual 'data/datasets' directory if you want to use the generated data
    # shutil.rmtree(dataset_output_dir, ignore_errors=True)