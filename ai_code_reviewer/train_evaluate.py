# ai_code_reviewer/train_evaluate.py

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
import logging
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm # For progress bars
import pickle # Added for saving/loading node_type_map
import shutil # Added for directory operations
from typing import List, Dict, Any, Tuple # Added for type hints

# Import components from other modules (using absolute imports)
from ai_code_reviewer.models.gnn_model import CodeGNNModel
from ai_code_reviewer.models.graph_representation import CodeGraphConverter
from ai_code_reviewer.data.datasets.dataset_builder import DatasetBuilder
from ai_code_reviewer.data.processors.ast_parser import ASTParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_and_convert_dataset(filepath: str, converter: CodeGraphConverter) -> List[Data]:
    """
    Loads a dataset from a JSON file and converts each entry into a PyTorch Geometric Data object.
    """
    logger.info(f"Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"Dataset file not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} raw entries. Converting to graphs...")

    graph_data_list: List[Data] = []
    # Re-use existing AST object if present, otherwise converter will re-parse from ast_str/code
    for i, entry in enumerate(tqdm(raw_data, desc="Converting to graphs")):
        # For entries loaded from JSON, 'ast_object' won't be there, so it needs re-parsing
        # The converter handles this by trying 'ast_str' or 'code'
        graph = converter.convert_to_graph(entry)
        if graph:
            graph_data_list.append(graph)
        else:
            logger.warning(f"Skipping entry {i+1} from {filepath} due to failed graph conversion.")
    
    logger.info(f"Successfully converted {len(graph_data_list)} entries to graphs from {filepath}.")
    return graph_data_list

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Ensure data.y is float and has shape [batch_size, 1]
        loss = criterion(out, data.y.float().view(-1, 1)) 
        loss.backward()
        optimizer.step()
        # total_loss accumulates sum of loss * num_graphs in batch
        total_loss += loss.item() * data.num_graphs 
    
    # Calculate average loss per graph in the dataset
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.float().view(-1, 1))
        total_loss += loss.item() * data.num_graphs

        # Apply sigmoid to logits to get probabilities, then threshold for binary prediction
        preds = (torch.sigmoid(out) > 0.5).long().squeeze(-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(data.y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def train_and_evaluate(
    train_data_path: str = "data/datasets/train_dataset.json",
    val_data_path: str = "data/datasets/val_dataset.json",
    test_data_path: str = "data/datasets/test_dataset.json",
    model_save_path: str = "trained_models/best_gnn_model.pth",
    # num_node_types will be determined dynamically from the converter, 
    # but kept as a parameter for initial model instantiation flexibility
    num_node_types_initial_estimate: int = 200, 
    embedding_dim: int = 64,
    hidden_channels: int = 128,
    num_gnn_layers: int = 2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    random_seed: int = 42
):
    """
    Main function to train and evaluate the GNN model.
    """
    set_seed(random_seed)

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize graph converter (will manage node type IDs as it processes data)
    graph_converter = CodeGraphConverter()

    # Load and convert datasets. The converter's node_type_map will be populated here.
    train_graphs = load_and_convert_dataset(train_data_path, graph_converter)
    val_graphs = load_and_convert_dataset(val_data_path, graph_converter)
    test_graphs = load_and_convert_dataset(test_data_path, graph_converter)

    if not train_graphs or not val_graphs or not test_graphs:
        logger.error("One or more datasets are empty after graph conversion. Aborting training.")
        return

    # Update num_node_types based on actual types seen by converter
    actual_num_node_types = len(graph_converter.node_type_map)
    if actual_num_node_types == 0:
        logger.error("No node types identified by converter. Cannot initialize model. Aborting.")
        return
    logger.info(f"Detected {actual_num_node_types} unique AST node types for model initialization.")
    
    # Save the node_type_map for consistent inference later
    node_type_map_path = os.path.join(os.path.dirname(model_save_path), "node_type_map.pkl")
    try:
        with open(node_type_map_path, 'wb') as f:
            pickle.dump(graph_converter.node_type_map, f)
        logger.info(f"Saved node_type_map to {node_type_map_path}")
    except Exception as e:
        logger.error(f"Failed to save node_type_map: {e}")


    # Create DataLoaders
    # Note: PyTorch Geometric's DataLoader handles batching of disparate graph sizes
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = CodeGNNModel(
        num_node_types=actual_num_node_types, # Use the dynamically determined number of node types
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_gnn_layers=num_gnn_layers,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and BCE for numerical stability
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting training...")
    best_val_f1 = -1.0 # Track best F1-score for model saving
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch:03d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}, "
                    f"Val Prec: {val_precision:.4f}, "
                    f"Val Rec: {val_recall:.4f}, "
                    f"Val F1: {val_f1:.4f}")
        
        # Save best model based on validation F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path} with Val F1: {best_val_f1:.4f}")

    logger.info("Training finished. Evaluating on test set...")

    # Load best model for testing
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logger.info(f"Loaded best model from {model_save_path} for final testing.")
    else:
        logger.warning(f"Best model not found at {model_save_path}. Using the last epoch's model for testing.")

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)

    logger.info(f"Test Results: "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Acc: {test_accuracy:.4f}, "
                f"Test Prec: {test_precision:.4f}, "
                f"Test Rec: {test_recall:.4f}, "
                f"Test F1: {test_f1:.4f}")

    # Calculate and log confusion matrix for test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            preds = (torch.sigmoid(out) > 0.5).long().squeeze(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nTest Confusion Matrix:\n{cm}")
    # cm[0,0] = True Negatives (Correctly predicted non-buggy)
    # cm[0,1] = False Positives (Incorrectly predicted buggy)
    # cm[1,0] = False Negatives (Incorrectly predicted non-buggy)
    # cm[1,1] = True Positives (Correctly predicted buggy)
    logger.info(f"True Positives (TP): {cm[1,1]}, False Positives (FP): {cm[0,1]}")
    logger.info(f"True Negatives (TN): {cm[0,0]}, False Negatives (FN): {cm[1,0]}")


# --- Unit Tests (Conceptual, as full training is long) ---
def run_conceptual_unit_tests():
    logger.info("Running conceptual unit tests for train_evaluate.py components...")

    # Test 1: Data loading and conversion
    # This requires dummy JSON files to be present
    dummy_dataset_dir = "test_dummy_data_for_train_eval"
    os.makedirs(dummy_dataset_dir, exist_ok=True)
    
    # Create dummy data for testing
    ast_parser_temp = ASTParser() # Instantiate ASTParser
    
    # Prepare dummy data by parsing code and converting ASTs to string representation
    dummy_data_raw = [
        {"code": "def buggy_func(x): return x / 0", "language": "python", "label": 1, "filepath": "test1.py"},
        {"code": "def fixed_func(x): return x + 1", "language": "python", "label": 0, "filepath": "test2.py"},
        {"code": "public class Bug {\n    public void method() {\n        int a = null;\n    }\n}", "language": "java", "label": 1, "filepath": "test3.java"},
        {"code": "public class Clean {\n    public void method() {\n        int a = 0;\n    }\n}", "language": "java", "label": 0, "filepath": "test4.java"},
    ]

    # Create a new list to hold JSON-serializable data
    dummy_data_for_json = []
    for entry_raw in dummy_data_raw:
        # Pass the language to parse_code
        ast_obj = ast_parser_temp.parse_code(entry_raw["code"], entry_raw["language"])
        
        # Create a new dictionary that is JSON-serializable
        entry_for_json = entry_raw.copy() # Start with basic fields
        
        # Pass the language to ast_to_string
        entry_for_json["ast_str"] = ast_parser_temp.ast_to_string(ast_obj, entry_raw["language"]) if ast_obj else ""
        
        dummy_data_for_json.append(entry_for_json)


    dummy_train_path = os.path.join(dummy_dataset_dir, "train_dataset.json")
    dummy_val_path = os.path.join(dummy_dataset_dir, "val_dataset.json")
    dummy_test_path = os.path.join(dummy_dataset_dir, "test_dataset.json")

    # Now dump the JSON-serializable list
    with open(dummy_train_path, 'w', encoding='utf-8') as f: json.dump(dummy_data_for_json, f)
    with open(dummy_val_path, 'w', encoding='utf-8') as f: json.dump(dummy_data_for_json, f)
    with open(dummy_test_path, 'w', encoding='utf-8') as f: json.dump(dummy_data_for_json, f)

    converter = CodeGraphConverter()
    train_graphs = load_and_convert_dataset(dummy_train_path, converter)
    assert len(train_graphs) == len(dummy_data_for_json), "Test 1: Incorrect number of graphs loaded/converted."
    assert all(isinstance(g, Data) for g in train_graphs), "Test 1: Not all converted objects are Data instances."
    logger.info("Test 1 (Data loading and conversion): PASSED.")

    # Test 2: Model forward pass (already covered in gnn_model.py unit tests, but sanity check)
    dummy_num_node_types = len(converter.node_type_map) if converter.node_type_map else 10 # Ensure non-zero
    model = CodeGNNModel(num_node_types=dummy_num_node_types).to('cpu') # Use CPU for quick test
    dummy_loader = DataLoader(train_graphs, batch_size=2)
    batch = next(iter(dummy_loader))
    try:
        output = model(batch)
        assert output.shape[0] == batch.num_graphs and output.shape[1] == 1, "Test 2: Model output shape incorrect."
        logger.info("Test 2 (Model forward pass with DataLoader batch): PASSED.")
    except Exception as e:
        logger.error(f"Test 2 (Model forward pass): FAILED with error {e}")

    # Clean up dummy data
    shutil.rmtree(dummy_dataset_dir, ignore_errors=True)
    logger.info("Conceptual unit tests completed.")

# --- Main Execution ---
if __name__ == "__main__":
    # It's highly recommended to run dataset_builder.py first to generate the datasets
    # If the datasets don't exist, we'll give a warning and create a very small dummy one.
    
    train_path = "data/datasets/train_dataset.json"
    val_path = "data/datasets/val_dataset.json"
    test_path = "data/datasets/test_dataset.json"

    # Check if dataset files exist. If not, generate a tiny one for demonstration.
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        logger.warning(f"Dataset files not found ({train_path}, etc.). Creating a minimal dummy dataset for demonstration.")
        # Create a tiny dummy dataset for the demo if none exists
        demo_scraped_data_dir = "scraped_code_samples_demo" # Use a temp dir to avoid conflicts
        os.makedirs(demo_scraped_data_dir, exist_ok=True)
        demo_raw_data_path = os.path.join(demo_scraped_data_dir, "bug_fix_pairs.json")

        dummy_bug_fix_pairs_for_builder = [
            {"repo_url": "demo_repo", "commit_hash": "c_py_1", "filepath": "file_py_1.py",
             "before_code": "def old_py(x):\n    return x - 1 #bug", "after_code": "def new_py(x):\n    return x + 1",
             "language": "python", "commit_message": "Fix py logic", "parsing_successful": True,
             "before_ast_str": "...", "after_ast_str": "..."},
            {"repo_url": "demo_repo", "commit_hash": "c_java_1", "filepath": "file_java_1.java",
             "before_code": "public class Buggy {\n    public void method() {\n        int a = 0;\n    }\n}",
             "after_code": "public class Buggy {\n    public void method() {\n        int a = 1;\n    }\n}",
             "language": "java", "commit_message": "Fix java null ptr", "parsing_successful": True,
             "before_ast_str": "...", "after_ast_str": "..."},
             {"repo_url": "demo_repo", "commit_hash": "c_py_2", "filepath": "file_py_2.py",
             "before_code": "def vulnerable_code(): eval('1')", "after_code": "def secure_code(): pass",
             "language": "python", "commit_message": "Security fix CVE-2023-9999", "parsing_successful": True,
             "before_ast_str": "...", "after_ast_str": "..."},
             {"repo_url": "demo_repo", "commit_hash": "c_java_2", "filepath": "file_java_2.java",
             # FIX: Added 'public' to class and method for stricter javalang parsing
             "before_code": "public class A { public void problematic() {} }",
             "after_code": "public class A { public void clean() { System.out.println(\"ok\"); } }",
             "language": "java", "commit_message": "Refactor code", "parsing_successful": True,
             "before_ast_str": "...", "after_ast_str": "..."},
        ]
        with open(demo_raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_bug_fix_pairs_for_builder, f, indent=4)
        
        # No CVE data needed for minimal demo
        dummy_cve_data_dir = "collected_cve_data_demo"
        os.makedirs(dummy_cve_data_dir, exist_ok=True)
        dummy_cve_data_path = os.path.join(dummy_cve_data_dir, "recent_cves.json")
        with open(dummy_cve_data_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4) # Empty CVE for faster profiling

        builder = DatasetBuilder(
            raw_data_path=demo_raw_data_path,
            cve_data_path=dummy_cve_data_path,
            output_dir="data/datasets", # Output to standard dir
            random_seed=42
        )
        builder.build_dataset(apply_balancing=True, augmentation_factor=1,train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # Clean up dummy raw data generated by the builder call
        shutil.rmtree(demo_scraped_data_dir, ignore_errors=True)
        shutil.rmtree(dummy_cve_data_dir, ignore_errors=True)
        logger.info("Tiny dummy dataset successfully generated for training demo.")


    # Run conceptual unit tests for train_evaluate.py parts
    run_conceptual_unit_tests()

    # --- Main Training and Evaluation Call ---
    logger.info("\n--- Starting Main Training and Evaluation Process ---")
    try:
        train_and_evaluate(
            train_data_path=train_path,
            val_data_path=val_path,
            test_data_path=test_path,
            model_save_path="trained_models/best_gnn_model.pth",
            num_node_types_initial_estimate=300, # A generous estimate. Will be auto-adjusted.
            embedding_dim=128,
            hidden_channels=256,
            num_gnn_layers=3,
            dropout_rate=0.4,
            learning_rate=0.0005,
            epochs=20, # Reduced epochs for faster demo. Increase for real training.
            batch_size=64,
            random_seed=42
        )
    except Exception as e:
        logger.error(f"An error occurred during main training and evaluation: {e}", exc_info=True)
    
    logger.info("\n--- Training and Evaluation Process Completed ---")

    # Final cleanup of the generated datasets if they were temporary for this run
    # (Uncomment if you want to clean up `data/datasets` after every run)
    # shutil.rmtree("data/datasets", ignore_errors=True)
    # shutil.rmtree("trained_models", ignore_errors=True)
