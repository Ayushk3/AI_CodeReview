# ai_code_reviewer/predict.py

import torch
import torch.nn as nn
import os
import json
import logging
import pickle # For loading node_type_map

# Import components from other modules
from ai_code_reviewer.models.gnn_model import CodeGNNModel
from ai_code_reviewer.models.graph_representation import CodeGraphConverter
from ai_code_reviewer.data.processors.ast_parser import ASTParser # Need ASTParser for raw code parsing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodePredictor:
    """
    Handles loading the trained GNN model and making predictions on new code snippets.
    """
    def __init__(
        self,
        model_path: str = "trained_models/best_gnn_model.pth",
        node_type_map_path: str = "trained_models/node_type_map.pkl",
        embedding_dim: int = 128, # Must match training config
        hidden_channels: int = 256, # Must match training config
        num_gnn_layers: int = 3, # Must match training config
        dropout_rate: float = 0.4 # Not strictly needed for inference but good to keep consistent
    ):
        """
        Initializes the CodePredictor by loading the model and necessary components.

        Args:
            model_path (str): Path to the saved PyTorch model state_dict.
            node_type_map_path (str): Path to the saved node_type_map.
            embedding_dim (int): Dimension used for node embeddings during training.
            hidden_channels (int): Number of hidden channels used in GNN layers during training.
            num_gnn_layers (int): Number of GNN layers used during training.
            dropout_rate (float): Dropout rate used during training (affects model architecture).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Predictor using device: {self.device}")

        # Load node_type_map first
        self.node_type_map = self._load_node_type_map(node_type_map_path)
        if not self.node_type_map:
            raise RuntimeError(f"Failed to load node_type_map from {node_type_map_path}. Cannot initialize predictor.")
        
        self.num_node_types = len(self.node_type_map)
        if self.num_node_types == 0:
            raise RuntimeError("node_type_map is empty. Cannot initialize predictor.")

        logger.info(f"Loaded {self.num_node_types} unique node types.")

        # Initialize graph converter with the loaded node_type_map
        self.graph_converter = CodeGraphConverter()
        # Explicitly set the node_type_map in the converter to ensure consistency
        self.graph_converter.node_type_map = self.node_type_map
        self.graph_converter._next_node_type_id = self.num_node_types # Ensure it continues from correct ID if new types encountered (though shouldn't for inference)

        # Initialize AST parser
        self.ast_parser = ASTParser()

        # Initialize and load the model
        self.model = CodeGNNModel(
            num_node_types=self.num_node_types,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout_rate=dropout_rate # Dropout is inactive in eval mode, but needed for model structure
        ).to(self.device)

        self._load_model(model_path)
        self.model.eval() # Set model to evaluation mode (disables dropout, batch norm etc.)
        logger.info("CodePredictor initialized successfully.")

    def _load_node_type_map(self, path: str) -> dict:
        """Loads the node_type_map from a pickle file."""
        if not os.path.exists(path):
            logger.error(f"node_type_map file not found at {path}")
            return {}
        try:
            with open(path, 'rb') as f:
                node_map = pickle.load(f)
            logger.info(f"Successfully loaded node_type_map from {path}")
            return node_map
        except Exception as e:
            logger.error(f"Error loading node_type_map from {path}: {e}")
            return {}

    def _load_model(self, path: str):
        """Loads the trained model state dictionary."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}")

    @torch.no_grad() # No gradient calculations during inference
    def predict(self, code_snippet: str, language: str) -> Tuple[str, float]:
        """
        Makes a prediction (buggy/clean) for a given code snippet.

        Args:
            code_snippet (str): The raw code string to analyze.
            language (str): The programming language of the code (e.g., "python", "java").

        Returns:
            Tuple[str, float]: A tuple containing the prediction label ("buggy/vulnerable" or "clean/fixed")
                               and the confidence score (probability).
        """
        logger.info(f"Predicting for a {language} code snippet (length {len(code_snippet)})...")

        # 1. Parse AST
        ast_obj = self.ast_parser.parse_code(code_snippet, language)
        if ast_obj is None:
            logger.error(f"Failed to parse AST for the provided {language} code snippet.")
            return "Error: Could not parse code", 0.0

        # 2. Convert to graph
        # For prediction, we create a dummy entry to match the expected input format for convert_to_graph
        dummy_entry = {
            "code": code_snippet,
            "language": language,
            "label": 0, # Dummy label, not used for inference
            "ast_object": ast_obj # Pass the actual AST object
        }
        graph_data = self.graph_converter.convert_to_graph(dummy_entry)
        
        if graph_data is None:
            logger.error("Failed to convert code snippet to graph.")
            return "Error: Could not convert to graph", 0.0

        # PyG expects a batch, even for a single graph.
        # DataLoader handles this, but for single prediction, we can manually add batch dim.
        # The .batch attribute is automatically handled if you create a PyG Data object and then directly use it.
        # But if you process multiple graphs, you should combine them into a single Batch object
        # which `DataLoader` does for you. For single instance, it's simpler:
        
        # Ensure x is 2D, even if it's (N, 1) for node type IDs
        if graph_data.x.dim() == 1:
            graph_data.x = graph_data.x.unsqueeze(-1)

        # Move data to the correct device
        graph_data = graph_data.to(self.device)

        # Manually add batch information if not using DataLoader for single inference
        # This is for global_mean_pool to work correctly.
        # It creates a batch tensor where all nodes belong to graph 0.
        graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)

        # 3. Inference
        output_logits = self.model(graph_data)

        # 4. Post-processing
        # Apply sigmoid to convert logits to probabilities
        probability = torch.sigmoid(output_logits).item() # .item() to get Python float
        
        # Threshold for binary prediction
        prediction_label = "buggy/vulnerable" if probability > 0.5 else "clean/fixed"

        logger.info(f"Prediction: {prediction_label}, Confidence: {probability:.4f}")
        return prediction_label, probability

# --- Usage Example / Main Script Entry Point ---
if __name__ == "__main__":
    # Define paths to the trained model and node_type_map
    MODEL_PATH = "trained_models/best_gnn_model.pth"
    NODE_TYPE_MAP_PATH = "trained_models/node_type_map.pkl"

    # These hyperparameters MUST match the ones used during training in train_evaluate.py
    # If you change them here, make sure they align with how the model was trained.
    EMBEDDING_DIM = 128
    HIDDEN_CHANNELS = 256
    NUM_GNN_LAYERS = 3
    DROPOUT_RATE = 0.4 # This impacts the model's structure.

    # Ensure trained_models directory exists (and contains the .pth and .pkl files)
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        logger.error(f"Directory {os.path.dirname(MODEL_PATH)} not found. "
                     "Please run train_evaluate.py first to train a model and save artifacts.")
        exit() # Exit if models are not present

    try:
        # Initialize the predictor
        predictor = CodePredictor(
            model_path=MODEL_PATH,
            node_type_map_path=NODE_TYPE_MAP_PATH,
            embedding_dim=EMBEDDING_DIM,
            hidden_channels=HIDDEN_CHANNELS,
            num_gnn_layers=NUM_GNN_LAYERS,
            dropout_rate=DROPOUT_RATE
        )

        logger.info("\n--- Demonstrating Code Prediction ---")

        # Example 1: A potentially buggy Python snippet
        python_buggy_code = """
def divide_by_zero(a, b):
    # Potential bug: division by zero if b is 0
    return a / b

x = 10
y = divide_by_zero(x, 0)
print(y)
"""
        logger.info("\nAnalyzing Python Buggy Code:")
        prediction, confidence = predictor.predict(python_buggy_code, "python")
        print(f"Result: {prediction}, Confidence: {confidence:.4f}")

        # Example 2: A clean Python snippet
        python_clean_code = """
def safe_divide(a, b):
    if b == 0:
        return 0 # Handle division by zero gracefully
    return a / b

x = 10
y = safe_divide(x, 2)
print(y)
"""
        logger.info("\nAnalyzing Python Clean Code:")
        prediction, confidence = predictor.predict(python_clean_code, "python")
        print(f"Result: {prediction}, Confidence: {confidence:.4f}")

        # Example 3: A potentially vulnerable Java snippet
        java_vulnerable_code = """
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class FileCreator {
    public void createFile(String filename, String content) throws IOException {
        // CWE-73: External Control of File Name or Path
        // Filename is directly used without sanitization
        FileOutputStream fos = new FileOutputStream(new File(filename));
        fos.write(content.getBytes());
        fos.close();
    }
}
"""
        logger.info("\nAnalyzing Java Vulnerable Code:")
        prediction, confidence = predictor.predict(java_vulnerable_code, "java")
        print(f"Result: {prediction}, Confidence: {confidence:.4f}")

        # Example 4: A clean Java snippet
        java_clean_code = """
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class SafeFileCreator {
    public void createSafeFile(String filename, String content) throws IOException {
        // Sanitize filename to prevent directory traversal
        String safeFilename = filename.replaceAll("[^a-zA-Z0-9.-]", "_");
        FileOutputStream fos = new FileOutputStream(new File("/tmp/" + safeFilename));
        fos.write(content.getBytes());
        fos.close();
    }
}
"""
        logger.info("\nAnalyzing Java Clean Code:")
        prediction, confidence = predictor.predict(java_clean_code, "java")
        print(f"Result: {prediction}, Confidence: {confidence:.4f}")

    except FileNotFoundError as e:
        logger.error(f"Deployment setup failed: {e}. Please ensure `train_evaluate.py` has been run successfully.")
    except RuntimeError as e:
        logger.error(f"Model/Predictor initialization failed: {e}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)