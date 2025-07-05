# ai_code_reviewer/models/gnn_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGNNModel(nn.Module):
    """
    A Graph Neural Network model for code classification (e.g., bug/vulnerability detection).
    It uses GCN layers to learn from code graph representations (ASTs).
    """

    def __init__(
        self,
        num_node_types: int,      # Number of unique AST node types (from CodeGraphConverter)
        embedding_dim: int = 64,  # Dimension of node embeddings
        hidden_channels: int = 128, # Number of hidden units in GNN layers
        num_gnn_layers: int = 2,  # Number of GNN convolutional layers
        dropout_rate: float = 0.5 # Dropout rate
    ):
        """
        Initializes the CodeGNNModel.

        Args:
            num_node_types (int): The total number of unique AST node types identified by CodeGraphConverter.
            embedding_dim (int): The dimension of the learned embeddings for each node type.
            hidden_channels (int): The number of output channels for each GNN layer.
            num_gnn_layers (int): The number of stacked GNN convolutional layers.
            dropout_rate (float): The dropout rate to apply for regularization.
        """
        super().__init__()
        logger.info(f"Initializing CodeGNNModel with: num_node_types={num_node_types}, embedding_dim={embedding_dim}, hidden_channels={hidden_channels}, num_gnn_layers={num_gnn_layers}, dropout_rate={dropout_rate}")

        # 1. Node Embedding Layer
        # Converts categorical node types (integers) into dense vectors.
        self.node_embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=embedding_dim)

        # 2. GNN Layers (Graph Convolutional Networks)
        self.gnn_layers = nn.ModuleList()
        # Input to first GNN layer is embedding_dim
        self.gnn_layers.append(GCNConv(embedding_dim, hidden_channels))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(dropout_rate)

        # 3. Classifier Head (MLP for graph-level prediction)
        # Input to the first linear layer is hidden_channels (from pooling the GNN output)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels // 2, 1) # Output a single logit for binary classification
        )

        logger.info("CodeGNNModel initialized successfully.")

    def forward(self, data: Data) -> torch.Tensor:
        """
        Performs a forward pass through the GNN model.

        Args:
            data (Data): A PyTorch Geometric Data object containing:
                         - data.x: Node feature matrix (node type IDs).
                         - data.edge_index: Graph connectivity in COO format.
                         - data.batch: Batch vector, which maps each node to its respective graph in the batch.

        Returns:
            torch.Tensor: Logits for binary classification (shape: [batch_size, 1]).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Node Embedding
        # x is typically long tensor of node type IDs [num_nodes_in_batch, 1]
        # Squeeze to [num_nodes_in_batch]
        x = self.node_embedding(x.squeeze(-1)) # Output: [num_nodes_in_batch, embedding_dim]

        # 2. GNN Layers
        for i, conv_layer in enumerate(self.gnn_layers):
            x = conv_layer(x, edge_index)
            # Apply ReLU after all but the last GNN layer (or before pooling)
            if i < len(self.gnn_layers) -1 :
                 x = x.relu()
            # Or always apply ReLU and then dropout:
            # x = x.relu()
            # x = self.dropout(x) # Apply dropout after activation

        # Apply dropout to the final node representations before pooling
        x = self.dropout(x)

        # 3. Global Pooling
        # Aggregates node features into a single graph-level feature vector.
        # global_mean_pool returns [batch_size, hidden_channels]
        graph_representation = global_mean_pool(x, batch)

        # 4. Classifier Head
        # Output is [batch_size, 1] (logits)
        logits = self.classifier(graph_representation)

        return logits

# --- Unit Tests ---
def run_unit_tests():
    """
    Runs unit tests for the CodeGNNModel.
    """
    logger.info("Running unit tests for CodeGNNModel...")

    # Define dummy parameters for the model
    NUM_NODE_TYPES = 50 # Example: 50 unique AST node types
    EMBEDDING_DIM = 32
    HIDDEN_CHANNELS = 64
    NUM_GNN_LAYERS = 2
    DROPOUT_RATE = 0.5

    # Create a dummy model instance
    try:
        model = CodeGNNModel(
            num_node_types=NUM_NODE_TYPES,
            embedding_dim=EMBEDDING_DIM,
            hidden_channels=HIDDEN_CHANNELS,
            num_gnn_layers=NUM_GNN_LAYERS,
            dropout_rate=DROPOUT_RATE
        )
        logger.info("Model instantiation PASSED.")
    except Exception as e:
        logger.error(f"Model instantiation FAILED: {e}", exc_info=True)
        return

    # Create dummy graph data for testing forward pass
    # Simulate a batch of 3 graphs
    # Graph 1: 5 nodes, 8 edges
    # Graph 2: 7 nodes, 10 edges
    # Graph 3: 4 nodes, 6 edges

    # Node features (node type IDs) for all nodes in the batch
    # Each node needs a feature. Let's make it a 2D tensor where 2nd dim is 1 (for node ID)
    x_batch = torch.randint(0, NUM_NODE_TYPES, (5 + 7 + 4, 1), dtype=torch.long) 
    
    # Edge index for all graphs combined (COO format)
    edge_index_g1 = torch.tensor([[0, 1, 1, 2, 3, 4, 0, 2], [1, 0, 2, 3, 0, 3, 4, 4]], dtype=torch.long)
    edge_index_g2 = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 6, 0], [1, 0, 2, 3, 4, 5, 0, 6, 1, 3]], dtype=torch.long) + 5 # Offset for graph 2 nodes
    edge_index_g3 = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]], dtype=torch.long) + (5 + 7) # Offset for graph 3 nodes

    edge_index_batch = torch.cat([edge_index_g1, edge_index_g2, edge_index_g3], dim=1)

    # Batch vector: maps each node to its respective graph
    batch_g1 = torch.zeros(5, dtype=torch.long)
    batch_g2 = torch.ones(7, dtype=torch.long)
    batch_g3 = torch.full((4,), 2, dtype=torch.long)
    batch_vector = torch.cat([batch_g1, batch_g2, batch_g3], dim=0)

    # Create a dummy Data object
    dummy_data = Data(x=x_batch, edge_index=edge_index_batch, batch=batch_vector)
    dummy_data.num_graphs = 3 # Manually set for clarity

    logger.info(f"Dummy Data: x.shape={dummy_data.x.shape}, edge_index.shape={dummy_data.edge_index.shape}, batch.shape={dummy_data.batch.shape}")

    # Test forward pass
    try:
        model.eval() # Set to evaluation mode for consistent output (dropout off)
        output_logits = model(dummy_data)
        
        assert output_logits.shape == (dummy_data.num_graphs, 1), \
            f"Test forward: Output shape mismatch. Expected ({dummy_data.num_graphs}, 1), got {output_logits.shape}"
        
        logger.info(f"Model output logits (first 5): {output_logits[:5].tolist()}")
        logger.info("Forward pass PASSED: Output shape is correct.")
        
        # Test output range (logits can be anything, but after sigmoid would be 0-1)
        sigmoid_output = torch.sigmoid(output_logits)
        assert torch.all(sigmoid_output >= 0) and torch.all(sigmoid_output <= 1), \
            "Test forward: Sigmoid output not within [0, 1] range."
        logger.info("Forward pass PASSED: Sigmoid output range is correct.")

    except Exception as e:
        logger.error(f"Forward pass FAILED: {e}", exc_info=True)
        return

    logger.info("All CodeGNNModel unit tests completed.")

# --- Performance Profiling (Conceptual) ---
def profile_gnn_model_performance(num_graphs: int = 100, avg_nodes: int = 50, avg_edges_per_node: int = 3):
    """
    Profiles the performance of the CodeGNNModel forward pass.
    """
    logger.info(f"\nStarting CodeGNNModel performance profiling with {num_graphs} graphs...")
    
    # Dummy model setup
    NUM_NODE_TYPES = 100
    EMBEDDING_DIM = 128
    HIDDEN_CHANNELS = 256
    NUM_GNN_LAYERS = 3
    model = CodeGNNModel(NUM_NODE_TYPES, EMBEDDING_DIM, HIDDEN_CHANNELS, NUM_GNN_LAYERS)
    model.eval() # Eval mode for consistent timing (no dropout)

    # Generate dummy graph data for profiling
    list_of_data_objects = []
    total_nodes = 0
    total_edges = 0
    for i in range(num_graphs):
        num_nodes = random.randint(avg_nodes - 10, avg_nodes + 10) # Vary node count
        num_edges = num_nodes * avg_edges_per_node # More edges than nodes typical for code graphs

        x = torch.randint(0, NUM_NODE_TYPES, (num_nodes, 1), dtype=torch.long)
        
        # Generate random edges for a connected graph (simple approximation)
        # Avoid self-loops for GCNConv if not explicitly handled
        row = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Ensure no self-loops for GCNConv if not configured for it
        # edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        list_of_data_objects.append(Data(x=x, edge_index=edge_index, y=torch.tensor([i % 2], dtype=torch.long)))
        total_nodes += num_nodes
        total_edges += edge_index.shape[1]
    
    logger.info(f"Generated {len(list_of_data_objects)} dummy graphs.")
    logger.info(f"Total dummy nodes: {total_nodes}, Total dummy edges: {total_edges}")

    # Create a PyTorch Geometric Batch object
    from torch_geometric.data import Batch
    batch_data = Batch.from_data_list(list_of_data_objects)
    
    logger.info(f"Batch data created. Total nodes: {batch_data.num_nodes}, Total edges: {batch_data.num_edges}")

    start_time = datetime.now()
    
    with torch.no_grad(): # No gradient calculation needed for profiling
        _ = model(batch_data)
    
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Profiling complete for CodeGNNModel forward pass:")
    logger.info(f"  Total graphs in batch: {num_graphs}")
    logger.info(f"  Total duration for batch inference: {duration}")
    logger.info(f"  Average time per graph: {duration / num_graphs}")


# --- Usage Example ---
if __name__ == "__main__":
    from datetime import datetime
    import random

    # Run unit tests
    run_unit_tests()

    # --- Demonstrate Model Instantiation and Dummy Forward Pass ---
    logger.info("\n--- Demonstrating CodeGNNModel Instantiation and Dummy Forward Pass ---")

    # In a real scenario, `num_node_types` would come from `CodeGraphConverter`
    # after processing the entire dataset. For this demo, let's pick a reasonable number.
    demo_num_node_types = 150 # Max ID from converter + 1
    demo_embedding_dim = 128
    demo_hidden_channels = 256
    demo_num_gnn_layers = 3
    demo_dropout_rate = 0.3

    demo_model = CodeGNNModel(
        num_node_types=demo_num_node_types,
        embedding_dim=demo_embedding_dim,
        hidden_channels=demo_hidden_channels,
        num_gnn_layers=demo_num_gnn_layers,
        dropout_rate=demo_dropout_rate
    )

    logger.info(f"Demo Model Architecture:\n{demo_model}")
    total_params = sum(p.numel() for p in demo_model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    # Create a single dummy graph for a quick forward pass demo
    # Simulate a graph with 20 nodes and 30 edges
    dummy_x = torch.randint(0, demo_num_node_types, (20, 1), dtype=torch.long)
    dummy_edge_index = torch.randint(0, 20, (2, 30), dtype=torch.long)
    dummy_batch = torch.zeros(20, dtype=torch.long) # Single graph batch
    dummy_y = torch.tensor([1], dtype=torch.long) # Example label

    single_graph_data = Data(x=dummy_x, edge_index=dummy_edge_index, batch=dummy_batch, y=dummy_y)

    logger.info("\nPerforming a dummy forward pass with a single graph...")
    demo_model.eval() # Set to evaluation mode
    with torch.no_grad():
        output = demo_model(single_graph_data)
        predicted_prob = torch.sigmoid(output).item()
    
    logger.info(f"Dummy graph has {single_graph_data.num_nodes} nodes and {single_graph_data.num_edges} edges.")
    logger.info(f"Raw output logit: {output.item():.4f}")
    logger.info(f"Predicted probability (after sigmoid): {predicted_prob:.4f}")
    logger.info(f"Actual label: {single_graph_data.y.item()}")


    # --- Performance Profiling ---
    profile_gnn_model_performance(num_graphs=50, avg_nodes=40, avg_edges_per_node=2) # Reduced for quick demo