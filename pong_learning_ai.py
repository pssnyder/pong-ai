"""
Learning AI for Pong using Neural Networks
Learns to play by experience (reinforcement learning)
"""

import numpy as np
import random
from collections import deque
from pong_engine import Action
import pickle
import os


class NeuralNetwork:
    """Simple neural network for Pong AI with support for advanced physics"""
    
    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        """
        Initialize neural network with random weights
        
        Args:
            input_size: Number of input features (game state + physics)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of possible actions (UP, DOWN, STAY)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization for better learning
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input state (normalized)
            
        Returns:
            Action probabilities/Q-values
        """
        # Hidden layer with ReLU activation
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        
        return self.z2
    
    def predict(self, state):
        """Predict best action for given state"""
        X = np.array(state).reshape(1, -1)
        output = self.forward(X)
        return np.argmax(output)
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()


class LearningAI:
    """
    AI that learns to play Pong through Deep Q-Learning
    
    This AI starts knowing nothing and learns by:
    1. Playing games against the physics expert
    2. Remembering successful and failed actions
    3. Discovering that spin and curve can beat straight predictions
    4. Gradually improving to exploit the expert's weaknesses
    
    Key Learning Goal:
    - Expert uses perfect straight-line physics
    - This AI must learn to use spin/curve to trick the expert
    """
    
    def __init__(self, side='A', learning_rate=0.001, discount_factor=0.95):
        """
        Initialize Learning AI
        
        Args:
            side: Which paddle ('A' or 'B')
            learning_rate: How fast the AI learns (0.0 to 1.0)
            discount_factor: How much to value future rewards
        """
        self.side = side
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Neural networks (10 inputs: ball x/y/dx/dy/spin, paddle positions/velocities)
        self.network = NeuralNetwork(input_size=10, hidden_size=32, output_size=3)
        self.target_network = NeuralNetwork(input_size=10, hidden_size=32, output_size=3)
        self.target_network.copy_weights_from(self.network)
        
        # Experience replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration parameters (epsilon-greedy)
        self.epsilon = 1.0  # Start with 100% random actions
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9995  # Decay rate per game
        
        # Training stats
        self.games_played = 0
        self.total_rewards = 0
        self.win_count = 0
        self.loss_count = 0
        
        # For tracking learning progress
        self.reward_history = []
        
    def get_state_vector(self, state):
        """
        Convert game state to neural network input (includes physics features)
        
        Args:
            state: Game state dict from engine
            
        Returns:
            numpy array of normalized features
            
        Features (10 total):
        - Ball position (x, y)
        - Ball velocity (dx, dy)
        - Ball spin (rotational speed - KEY for beating expert!)
        - Own paddle (y position, y velocity)
        - Opponent paddle (y position, y velocity)
        - Distance to ball
        """
        # Get our paddle data
        if self.side == 'A':
            my_paddle_y = state['paddle_a_y']
            my_paddle_vel = state.get('paddle_a_vel', 0)
            opp_paddle_y = state['paddle_b_y']
            opp_paddle_vel = state.get('paddle_b_vel', 0)
        else:
            my_paddle_y = state['paddle_b_y']
            my_paddle_vel = state.get('paddle_b_vel', 0)
            opp_paddle_y = state['paddle_a_y']
            opp_paddle_vel = state.get('paddle_a_vel', 0)
        
        # Calculate additional useful features
        ball_distance = abs(state['ball_x'])  # How close is ball
        
        features = [
            state['ball_x'],          # Ball X position (-1 to 1)
            state['ball_y'],          # Ball Y position
            state['ball_dx'],         # Ball X velocity
            state['ball_dy'],         # Ball Y velocity
            state.get('ball_spin', 0), # Ball spin (KEY FEATURE!)
            my_paddle_y,              # My paddle position
            my_paddle_vel,            # My paddle velocity (for creating spin!)
            opp_paddle_y,             # Opponent paddle position
            opp_paddle_vel,           # Opponent paddle velocity
            ball_distance,            # Distance to ball
        ]
        
        return np.array(features, dtype=np.float32)
    
    def decide_action(self, state, training=True):
        """
        Decide action using epsilon-greedy strategy
        
        Args:
            state: Game state from engine
            training: If True, use exploration; if False, use best action
            
        Returns:
            Action enum
        """
        state_vector = self.get_state_vector(state)
        
        # Epsilon-greedy: explore vs exploit
        if training and random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, 2)
        else:
            # Exploit: use network's best prediction
            action_idx = self.network.predict(state_vector)
        
        # Convert index to Action enum (0=DOWN, 1=STAY, 2=UP)
        return [Action.DOWN, Action.STAY, Action.UP][action_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether game ended
        """
        state_vector = self.get_state_vector(state)
        next_state_vector = self.get_state_vector(next_state)
        action_idx = [Action.DOWN, Action.STAY, Action.UP].index(action)
        
        self.memory.append((state_vector, action_idx, reward, next_state_vector, done))
    
    def train_on_batch(self):
        """
        Train the network on a batch of experiences
        This is where the actual learning happens!
        """
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough data yet
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            # Get current Q-values
            current_q = self.network.forward(state.reshape(1, -1))[0]
            
            # Calculate target Q-value
            if done:
                target_q = reward
            else:
                # Q-learning: reward + discounted max future Q-value
                next_q = self.target_network.forward(next_state.reshape(1, -1))[0]
                target_q = reward + self.discount_factor * np.max(next_q)
            
            # Update only the Q-value for the action taken
            target = current_q.copy()
            target[action] = target_q
            
            # Calculate loss
            loss = (current_q[action] - target_q) ** 2
            total_loss += loss
            
            # Backpropagation (simplified gradient descent)
            # This is a simplified version - in practice you'd use proper backprop
            self._simple_update(state, target)
        
        return total_loss / self.batch_size
    
    def _simple_update(self, state, target):
        """Simplified weight update using gradient descent"""
        # Forward pass
        X = state.reshape(1, -1)
        output = self.network.forward(X)
        
        # Output layer gradient
        d_output = 2 * (output - target.reshape(1, -1))
        
        # Update output layer
        self.network.W2 -= self.learning_rate * (self.network.a1.T @ d_output)
        self.network.b2 -= self.learning_rate * d_output
        
        # Hidden layer gradient
        d_hidden = (d_output @ self.network.W2.T) * (self.network.a1 > 0)
        
        # Update hidden layer
        self.network.W1 -= self.learning_rate * (X.T @ d_hidden)
        self.network.b1 -= self.learning_rate * d_hidden
    
    def end_game(self, final_score_self, final_score_opponent):
        """
        Called when a game ends to update statistics
        
        Args:
            final_score_self: This AI's final score
            final_score_opponent: Opponent's final score
        """
        self.games_played += 1
        
        if final_score_self > final_score_opponent:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        if self.games_played % 10 == 0:
            self.target_network.copy_weights_from(self.network)
        
        # Track progress
        win_rate = self.win_count / self.games_played if self.games_played > 0 else 0
        self.reward_history.append(win_rate)
    
    def save(self, filepath):
        """Save the AI's brain to a file"""
        data = {
            'W1': self.network.W1,
            'b1': self.network.b1,
            'W2': self.network.W2,
            'b2': self.network.b2,
            'games_played': self.games_played,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'epsilon': self.epsilon
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load a previously trained AI brain"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.network.W1 = data['W1']
        self.network.b1 = data['b1']
        self.network.W2 = data['W2']
        self.network.b2 = data['b2']
        self.games_played = data['games_played']
        self.win_count = data['win_count']
        self.loss_count = data['loss_count']
        self.epsilon = data.get('epsilon', 0.1)
        
        # Update target network
        self.target_network.copy_weights_from(self.network)
        
        return True
    
    def get_stats(self):
        """Get training statistics"""
        win_rate = self.win_count / self.games_played if self.games_played > 0 else 0
        
        return {
            'games_played': self.games_played,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': win_rate,
            'epsilon': self.epsilon,
            'exploration': f"{self.epsilon * 100:.1f}%"
        }
    
    def reset(self):
        """Reset for new game (but keep learning)"""
        pass  # Learning AI doesn't need per-game reset
