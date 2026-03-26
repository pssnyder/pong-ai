"""
Model Insights Analyzer
Tools for understanding neural network decisions and behavior

Usage:
    from model_insights import ModelInsights
    
    insights = ModelInsights('models/pong_learning_ai.pkl')
    
    # Analyze decision for a game state
    insights.analyze_decision(state_vector)
    
    # Feature importance
    importance = insights.get_feature_importance()
    
    # Weight evolution (if multiple checkpoints)
    insights.compare_checkpoints(['model_100.pkl', 'model_500.pkl'])
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json


class ModelInsights:
    """
    Analyzes neural network model for insights into decision-making
    """
    
    def __init__(self, model_path=None):
        """
        Initialize model insights analyzer
        
        Args:
            model_path: Path to pickled model file
        """
        self.model_path = model_path
        self.model_data = None
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model from file"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.W1 = self.model_data['W1']
        self.W2 = self.model_data['W2']
        self.b1 = self.model_data['b1']
        self.b2 = self.model_data['b2']
        
        print(f"✅ Loaded model from: {model_path}")
        print(f"   Architecture: {self.W1.shape[0]} → {self.W1.shape[1]} → {self.W2.shape[1]}")
        print(f"   Games played: {self.model_data.get('games_played', 'N/A')}")
        print(f"   Win rate: {self.model_data.get('win_count', 0) / max(self.model_data.get('games_played', 1), 1) * 100:.1f}%")
    
    def forward_with_activations(self, state_vector):
        """
        Forward pass through network, returning all intermediate values
        
        Args:
            state_vector: Input state (9 features)
            
        Returns:
            dict with z1, a1, z2, q_values, best_action
        """
        if self.W1 is None:
            raise ValueError("No model loaded")
        
        X = np.array(state_vector).reshape(1, -1)
        
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        
        # Best action
        best_action = np.argmax(z2)
        
        return {
            'input': X,
            'z1': z1,           # Pre-activation hidden layer
            'a1': a1,           # Post-activation hidden layer
            'z2': z2,           # Output layer logits
            'q_values': z2[0],  # Q-values for each action
            'best_action': best_action,
            'best_action_name': ['DOWN', 'STAY', 'UP'][best_action]
        }
    
    def analyze_decision(self, state_vector, verbose=True):
        """
        Detailed analysis of network's decision for a given state
        
        Args:
            state_vector: Game state (9 features)
            verbose: Print detailed analysis
            
        Returns:
            dict: Analysis results
        """
        result = self.forward_with_activations(state_vector)
        
        # Calculate additional metrics
        q_values = result['q_values']
        q_diff = q_values - np.min(q_values)
        q_softmax = np.exp(q_values) / np.sum(np.exp(q_values))
        
        # Hidden layer analysis
        activations = result['a1'][0]
        active_neurons = np.sum(activations > 0)
        max_activation = np.max(activations)
        
        analysis = {
            'best_action': result['best_action_name'],
            'q_values': q_values,
            'q_diff': q_diff,
            'q_confidence': q_softmax,
            'active_neurons': active_neurons,
            'max_activation': max_activation,
            'activation_sparsity': active_neurons / len(activations)
        }
        
        if verbose:
            self._print_decision_analysis(state_vector, analysis)
        
        return analysis
    
    def _print_decision_analysis(self, state_vector, analysis):
        """Print formatted decision analysis"""
        feature_names = ['ball_x', 'ball_y', 'ball_dx', 'ball_dy', 'ball_spin',
                        'my_paddle_y', 'my_paddle_vel', 'opp_y', 'opp_vel']
        
        print("\n" + "=" * 70)
        print("🧠 NEURAL NETWORK DECISION ANALYSIS")
        print("=" * 70)
        
        print("\n📥 Input State:")
        for name, value in zip(feature_names, state_vector):
            print(f"   {name:15s}: {value:+.4f}")
        
        print(f"\n🎯 Decision: {analysis['best_action']}")
        
        print("\n📊 Q-Values (Action Quality):")
        actions = ['DOWN', 'STAY', 'UP']
        for action, q_val, conf in zip(actions, analysis['q_values'], analysis['q_confidence']):
            marker = "👉" if action == analysis['best_action'] else "  "
            bar = "█" * int(conf * 50)
            print(f"   {marker} {action:5s}: {q_val:+.4f} | {conf*100:5.1f}% {bar}")
        
        print(f"\n🧠 Hidden Layer Activity:")
        print(f"   Active neurons: {analysis['active_neurons']} / {len(analysis['q_values'])} ({analysis['activation_sparsity']*100:.1f}%)")
        print(f"   Max activation: {analysis['max_activation']:.4f}")
        
        print("=" * 70 + "\n")
    
    def get_feature_importance(self):
        """
        Calculate feature importance based on weight magnitudes
        
        Returns:
            dict: Feature names and their importance scores
        """
        if self.W1 is None:
            raise ValueError("No model loaded")
        
        # Sum of absolute weights from each input feature
        importance = np.sum(np.abs(self.W1), axis=1)
        
        feature_names = ['ball_x', 'ball_y', 'ball_dx', 'ball_dy', 'ball_spin',
                        'my_paddle_y', 'my_paddle_vel', 'opp_y', 'opp_vel']
        
        results = {name: score for name, score in zip(feature_names, importance)}
        
        # Sort by importance
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results
    
    def print_feature_importance(self):
        """Print formatted feature importance"""
        importance = self.get_feature_importance()
        
        print("\n" + "=" * 70)
        print("📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        print("\nWhich input features influence the network's decisions most?\n")
        
        max_importance = max(importance.values())
        
        for i, (feature, score) in enumerate(importance.items(), 1):
            normalized = score / max_importance
            bar = "█" * int(normalized * 40)
            print(f"{i}. {feature:15s}: {score:.2f} {bar}")
        
        print("\n" + "=" * 70 + "\n")
    
    def visualize_weights(self, save_path='model_weights_visualization.png'):
        """Create comprehensive weight visualization"""
        if self.W1 is None:
            raise ValueError("No model loaded")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Neural Network Weight Analysis', fontsize=16, fontweight='bold')
        
        # 1. W1 Heatmap
        ax = axes[0, 0]
        im1 = ax.imshow(self.W1, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(self.W1)), vmax=np.max(np.abs(self.W1)))
        ax.set_title('Input Layer Weights (W1)')
        ax.set_xlabel('Hidden Neurons')
        ax.set_ylabel('Input Features')
        ax.set_yticks(range(9))
        ax.set_yticklabels(['ball_x', 'ball_y', 'ball_dx', 'ball_dy', 'ball_spin',
                           'my_y', 'my_vel', 'opp_y', 'opp_vel'], fontsize=8)
        plt.colorbar(im1, ax=ax)
        
        # 2. W1 Distribution
        ax = axes[0, 1]
        ax.hist(self.W1.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title('W1 Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.grid(alpha=0.3)
        
        # 3. W2 Heatmap
        ax = axes[1, 0]
        im2 = ax.imshow(self.W2, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(self.W2)), vmax=np.max(np.abs(self.W2)))
        ax.set_title('Hidden Layer Weights (W2)')
        ax.set_xlabel('Output Actions')
        ax.set_ylabel('Hidden Neurons')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['DOWN', 'STAY', 'UP'])
        plt.colorbar(im2, ax=ax)
        
        # 4. Feature Importance
        ax = axes[1, 1]
        importance = self.get_feature_importance()
        features = list(importance.keys())
        values = list(importance.values())
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        ax.barh(features, values, color=colors, edgecolor='black')
        ax.set_title('Feature Importance (Weight Magnitudes)')
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
        plt.show()
    
    def compare_q_values_batch(self, state_vectors, labels=None):
        """
        Compare Q-values across multiple states
        
        Args:
            state_vectors: List of state vectors
            labels: Optional labels for each state
            
        Returns:
            DataFrame with Q-values for each state
        """
        if labels is None:
            labels = [f"State {i+1}" for i in range(len(state_vectors))]
        
        results = []
        for state, label in zip(state_vectors, labels):
            forward = self.forward_with_activations(state)
            results.append({
                'label': label,
                'DOWN': forward['q_values'][0],
                'STAY': forward['q_values'][1],
                'UP': forward['q_values'][2],
                'best_action': forward['best_action_name']
            })
        
        return results
    
    def sensitivity_analysis(self, base_state, feature_idx, feature_name=None, steps=20):
        """
        Analyze how sensitive the decision is to changes in one feature
        
        Args:
            base_state: Base state vector (9 features)
            feature_idx: Index of feature to vary (0-8)
            feature_name: Optional name for the feature
            steps: Number of steps to test
            
        Returns:
            dict: Feature values and corresponding Q-values
        """
        if feature_name is None:
            feature_names = ['ball_x', 'ball_y', 'ball_dx', 'ball_dy', 'ball_spin',
                           'my_paddle_y', 'my_paddle_vel', 'opp_y', 'opp_vel']
            feature_name = feature_names[feature_idx]
        
        feature_values = np.linspace(-1, 1, steps)
        q_down = []
        q_stay = []
        q_up = []
        best_actions = []
        
        for val in feature_values:
            test_state = base_state.copy()
            test_state[feature_idx] = val
            
            forward = self.forward_with_activations(test_state)
            q_values = forward['q_values']
            
            q_down.append(q_values[0])
            q_stay.append(q_values[1])
            q_up.append(q_values[2])
            best_actions.append(forward['best_action'])
        
        return {
            'feature_name': feature_name,
            'feature_values': feature_values,
            'q_down': np.array(q_down),
            'q_stay': np.array(q_stay),
            'q_up': np.array(q_up),
            'best_actions': np.array(best_actions)
        }
    
    def plot_sensitivity(self, base_state, feature_idx, feature_name=None):
        """Plot sensitivity analysis"""
        result = self.sensitivity_analysis(base_state, feature_idx, feature_name)
        
        plt.figure(figsize=(10, 6))
        plt.plot(result['feature_values'], result['q_down'], label='DOWN', linewidth=2)
        plt.plot(result['feature_values'], result['q_stay'], label='STAY', linewidth=2)
        plt.plot(result['feature_values'], result['q_up'], label='UP', linewidth=2)
        
        plt.xlabel(f"{result['feature_name']} Value", fontsize=12)
        plt.ylabel('Q-Value', fontsize=12)
        plt.title(f"Sensitivity Analysis: {result['feature_name']}", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Model Insights Analyzer - Demo\n")
    
    # Check if model exists
    model_path = 'models/pong_learning_ai.pkl'
    
    if Path(model_path).exists():
        print(f"Loading model from: {model_path}\n")
        insights = ModelInsights(model_path)
        
        # Feature importance
        insights.print_feature_importance()
        
        # Example state analysis
        print("\nAnalyzing a sample game state...\n")
        sample_state = [0.5, 0.2, -0.8, 0.3, 0.1, -0.3, 0.0, 0.4, 0.2]
        insights.analyze_decision(sample_state, verbose=True)
        
        # Visualize weights
        insights.visualize_weights()
        
    else:
        print(f"❌ Model not found at: {model_path}")
        print("Train a model first using: python training/train_ai.py")
