"""
Learning AI for Pong using Neural Networks
Learns to play by experience (reinforcement learning) with Curriculum Learning
"""

import numpy as np
import random
from collections import deque
from pong_engine import Action
import pickle
import os


# ============================================================================
#                    CURRICULUM LEARNING SYSTEM
# ============================================================================
# Staged training progression: Observation → Prediction → Exploitation

class TrainingPhase:
    """
    Defines a training phase with specific behavior constraints
    
    Curriculum Learning Philosophy:
    - Phase 1 (Beginner): Learn to OBSERVE and TRACK the ball
      * High decision_interval = more time measuring ball flight
      * High action_commitment = less micro-management
      * Focus: Build basic trajectory understanding
      
    - Phase 2 (Intermediate): Learn to PREDICT trajectories  
      * Medium intervals = balance measurement and action
      * Focus: Improve interception accuracy
      
    - Phase 3 (Advanced): Learn to EXPLOIT physics (spin/curve)
      * Low intervals = freedom to experiment with rapid movements
      * Focus: Discover advanced tactics like late-hit spin
    """
    
    def __init__(self, name, min_games, max_games, 
                 decision_interval, action_commitment_frames,
                 max_velocity, kp, kd, description):
        self.name = name
        self.min_games = min_games  # Games when this phase starts
        self.max_games = max_games  # Games when this phase ends (None = infinite)
        self.decision_interval = decision_interval
        self.action_commitment_frames = action_commitment_frames
        self.max_velocity = max_velocity
        self.kp = kp
        self.kd = kd
        self.description = description
    
    def is_active(self, games_played):
        """Check if this phase is currently active"""
        if games_played < self.min_games:
            return False
        if self.max_games is None:
            return True
        return games_played < self.max_games


class CurriculumConfig:
    """
    Configurable curriculum learning progression
    
    Adjusts AI behavior based on training progress to prevent
    premature optimization and encourage proper skill building.
    """
    
    def __init__(self, phases=None):
        if phases is None:
            # Default 3-phase curriculum
            self.phases = [
                # PHASE 1: BEGINNER - Observation & Tracking (0-500 games)
                # Goal: Learn ball exists, learn to follow it, build position sense
                TrainingPhase(
                    name="Phase 1: Observation & Tracking",
                    min_games=0,
                    max_games=500,
                    decision_interval=8,        # VERY patient - 8 frames between decisions
                    action_commitment_frames=6,  # Long commitment - reduced micro-management
                    max_velocity=35.0,          # Limited speed - smooth tracking
                    kp=0.25,                    # Gentle response
                    kd=0.60,                    # Strong damping - smooth tracking
                    description="Learning to observe ball trajectory and basic tracking"
                ),
                
                # PHASE 2: INTERMEDIATE - Trajectory Prediction (500-2000 games)
                # Goal: Improve interception, understand bounces, build prediction
                TrainingPhase(
                    name="Phase 2: Trajectory Prediction",
                    min_games=500,
                    max_games=2000,
                    decision_interval=5,        # Moderate patience - more responsive
                    action_commitment_frames=4,  # Medium commitment
                    max_velocity=40.0,          # Increased speed capability
                    kp=0.32,                    # Moderate response
                    kd=0.45,                    # Moderate damping
                    description="Building trajectory prediction and interception skills"
                ),
                
                # PHASE 3: ADVANCED - Physics Exploitation (2000+ games)
                # Goal: Experiment with spin, late hits, advanced tactics
                TrainingPhase(
                    name="Phase 3: Advanced Physics",
                    min_games=2000,
                    max_games=None,  # No upper limit
                    decision_interval=3,        # More freedom - can experiment
                    action_commitment_frames=2,  # Quick adjustments allowed
                    max_velocity=45.0,          # Full speed capability
                    kp=0.38,                    # Stronger response
                    kd=0.35,                    # Less damping - more agile
                    description="Exploring advanced physics: spin, curve, late-hit tactics"
                )
            ]
        else:
            self.phases = phases
    
    def get_active_phase(self, games_played):
        """Get the currently active training phase"""
        for phase in self.phases:
            if phase.is_active(games_played):
                return phase
        # Fallback to last phase
        return self.phases[-1]
    
    def get_config_for_games(self, games_played):
        """Get PaddleControlConfig for current training progress"""
        phase = self.get_active_phase(games_played)
        
        return PaddleControlConfig(
            decision_interval=phase.decision_interval,
            action_commitment_frames=phase.action_commitment_frames,
            max_velocity=phase.max_velocity,
            kp=phase.kp,
            kd=phase.kd,
            # Keep other params consistent
            acceleration=8.0,
            deceleration_factor=0.6,
            use_pid=True,
            ki=0.008,
            dead_zone_threshold=8.0,
            velocity_threshold=0.8,
            action_threshold_multiplier=0.25
        )


# ============================================================================
#                    PADDLE CONTROL CONFIGURATION
# ============================================================================
# Configure paddle physics and decision timing here for easy tuning

class PaddleControlConfig:
    """
    Configuration for Learning AI paddle control and decision timing
    
    Tuning Guide:
    - Increase decision_interval for more patient/strategic play (less jittery)
    - Decrease decision_interval for more reactive/instinctive play
    - Adjust max_velocity to limit paddle speed (prevents overshooting)
    - Tune PID parameters to reduce oscillation:
        * Higher kp = stronger response to position error
        * Higher ki = corrects persistent offset errors
        * Higher kd = dampens oscillations (resistance to change)
    - Increase action_commitment to make AI stick with decisions longer
    """
    
    def __init__(self,
                 # Decision Timing Parameters
                 decision_interval=3,           # How many frames between AI decisions (measure phase)
                 action_commitment_frames=2,    # Min frames to commit to an action before changing
                 
                 # Paddle Physics Parameters  
                 max_velocity=45.0,             # Maximum paddle velocity (pixels/frame, default: 45)
                 acceleration=8.0,              # Paddle acceleration rate (default: 8)
                 deceleration_factor=0.6,       # How quickly to slow down when stopping (0-1)
                 
                 # PID Controller Parameters (smooth movement, reduces oscillation)
                 use_pid=True,                  # Enable PID-based smooth movement
                 kp=0.4,                        # Proportional gain (strength of response)
                 ki=0.008,                      # Integral gain (correct persistent errors)
                 kd=0.3,                        # Derivative gain (dampen oscillations)
                 
                 # Dead Zones (prevents micro-adjustments)
                 dead_zone_threshold=8.0,       # Stop moving if within this distance of target
                 velocity_threshold=0.8,        # Minimum velocity to register movement
                 
                 # Action Conversion
                 action_threshold_multiplier=0.25  # Fraction of max speed needed to commit to UP/DOWN
                ):
        
        self.decision_interval = decision_interval
        self.action_commitment_frames = action_commitment_frames
        
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.deceleration_factor = deceleration_factor
        
        self.use_pid = use_pid
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.dead_zone_threshold = dead_zone_threshold
        self.velocity_threshold = velocity_threshold
        self.action_threshold_multiplier = action_threshold_multiplier
        self.max_integral = 100.0  # Prevent integral windup


# ============================================================================
#                    LEARNING AI PRESETS
# ============================================================================
# Pre-configured control profiles for different play styles

class ControlPresets:
    """Pre-configured control profiles for easy selection"""
    
    @staticmethod
    def reactive():
        """Fast, reactive play - responds quickly but may be jittery"""
        return PaddleControlConfig(
            decision_interval=1,
            action_commitment_frames=1,
            max_velocity=50.0,
            kp=0.5,
            kd=0.2
        )
    
    @staticmethod
    def balanced():
        """Balanced play - good mix of speed and stability"""
        return PaddleControlConfig(
            decision_interval=3,
            action_commitment_frames=2,
            max_velocity=45.0,
            kp=0.4,
            kd=0.3
        )
    
    @staticmethod
    def patient():
        """Patient, strategic play - smooth but slower to react"""
        return PaddleControlConfig(
            decision_interval=5,
            action_commitment_frames=3,
            max_velocity=40.0,
            kp=0.35,
            kd=0.4
        )
    
    @staticmethod
    def ultra_smooth():
        """Ultra-smooth movement - minimal jitter, best for visual quality"""
        return PaddleControlConfig(
            decision_interval=4,
            action_commitment_frames=4,
            max_velocity=38.0,
            kp=0.3,
            ki=0.01,
            kd=0.5,
            deceleration_factor=0.7
        )


# ============================================================================


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
    AI that learns to play Pong through Deep Q-Learning with Curriculum Learning
    
    This AI starts knowing nothing and learns by:
    1. PHASE 1: Observing ball movement and learning to track (games 0-500)
    2. PHASE 2: Building trajectory prediction skills (games 500-2000)
    3. PHASE 3: Discovering advanced physics exploitation (games 2000+)
    
    Key Learning Goal:
    - Expert uses perfect straight-line physics
    - This AI must learn to use spin/curve to trick the expert
    - Curriculum ensures proper skill progression without premature optimization
    """
    
    def __init__(self, side='A', learning_rate=0.001, discount_factor=0.95, 
                 paddle_config=None, use_curriculum=True, curriculum_config=None):
        """
        Initialize Learning AI
        
        Args:
            side: Which paddle ('A' or 'B')
            learning_rate: How fast the AI learns (0.0 to 1.0)
            discount_factor: How much to value future rewards
            paddle_config: PaddleControlConfig instance (overrides curriculum if provided)
            use_curriculum: Enable curriculum learning (progressive skill building)
            curriculum_config: CurriculumConfig instance (or None for default)
        """
        self.side = side
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Curriculum learning system
        self.use_curriculum = use_curriculum
        self.curriculum = curriculum_config if curriculum_config else CurriculumConfig()
        
        # Paddle control configuration
        # If using curriculum, this will be updated each game based on progress
        if paddle_config is not None:
            self.paddle_config = paddle_config
            self.use_curriculum = False  # Explicit config overrides curriculum
        elif use_curriculum:
            self.paddle_config = self.curriculum.get_config_for_games(0)
        else:
            self.paddle_config = ControlPresets.balanced()
        
        # Paddle physics state (for smooth PID-based movement)
        self.current_velocity = 0.0
        self.target_position = 0.0
        self.last_error = 0.0
        self.integral_error = 0.0
        
        # Decision timing state
        self.frames_since_decision = 0
        self.frames_since_action_change = 0
        self.last_action = Action.STAY
        self.last_paddle_y = 0.0
        
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
        self.tie_count = 0  # Track ties separately
        
        # For tracking learning progress
        self.reward_history = []
        
        # Curriculum phase tracking
        self.current_phase = None
        self.phase_transition_games = []  # Track when phases changed
        
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
        Decide action using epsilon-greedy strategy with decision throttling
        
        Features:
        - Decision throttling: Only makes new decisions every N frames (reduces jitter)
        - Action commitment: Sticks with actions for minimum duration
        - PID-based smooth movement: Converts discrete actions to smooth velocity
        - Curriculum learning: Adjusts behavior complexity based on training phase
        
        Args:
            state: Game state from engine
            training: If True, use exploration; if False, use best action
            
        Returns:
            Action enum
        """
        # Get current paddle position
        if self.side == 'A':
            current_paddle_y = state['paddle_a_y']
        else:
            current_paddle_y = state['paddle_b_y']
        
        self.last_paddle_y = current_paddle_y
        
        # Track frames for decision timing
        self.frames_since_decision += 1
        self.frames_since_action_change += 1
        
        # DECISION THROTTLING: Only make new decisions at intervals
        # This is the "measure -> calculate -> act" cycle spacing
        # In early training (Phase 1), this is LONG (more measurement)
        # In late training (Phase 3), this is SHORT (more action freedom)
        if self.frames_since_decision >= self.paddle_config.decision_interval:
            self.frames_since_decision = 0
            
            # Get state vector for neural network
            state_vector = self.get_state_vector(state)
            
            # Epsilon-greedy: explore vs exploit
            if training and random.random() < self.epsilon:
                # Explore: random action
                action_idx = random.randint(0, 2)
            else:
                # Exploit: use network's best prediction
                action_idx = self.network.predict(state_vector)
            
            # Convert index to Action enum (0=DOWN, 1=STAY, 2=UP)
            desired_action = [Action.DOWN, Action.STAY, Action.UP][action_idx]
            
            # ACTION COMMITMENT: Only change action if enough time has passed
            # Prevents rapid action flickering
            if (self.frames_since_action_change >= self.paddle_config.action_commitment_frames or
                desired_action == self.last_action):
                self.last_action = desired_action
                self.frames_since_action_change = 0
        
        # SMOOTH MOVEMENT: Apply PID-based or direct movement
        if self.paddle_config.use_pid:
            # PID mode: smooth, physics-based movement with velocity control
            return self._apply_smooth_movement(current_paddle_y, self.last_action)
        else:
            # Direct mode: immediate response (legacy behavior)
            return self.last_action
    
    def _apply_smooth_movement(self, current_y, desired_action):
        """
        Apply smooth PID-based movement instead of instant action
        
        This converts discrete actions (UP/DOWN/STAY) into smooth velocity-based movement
        that reduces jitter and oscillation.
        
        Args:
            current_y: Current paddle Y position
            desired_action: The action the AI wants to take
            
        Returns:
            Action enum after PID smoothing
        """
        config = self.paddle_config
        
        # Determine target position based on desired action
        # The AI's action becomes a target direction rather than instant movement
        if desired_action == Action.UP:
            # Want to move up - set target above current position
            self.target_position = current_y + 50
        elif desired_action == Action.DOWN:
            # Want to move down - set target below current position
            self.target_position = current_y - 50
        else:
            # Want to stay - target is current position
            self.target_position = current_y
        
        # Clamp target to valid range
        self.target_position = max(-230, min(230, self.target_position))
        
        # Calculate error (distance to target)
        error = self.target_position - current_y
        
        # DEAD ZONE: If close enough to target, decelerate and stop
        if abs(error) < config.dead_zone_threshold:
            self.current_velocity *= config.deceleration_factor
            self.integral_error = 0
            
            if abs(self.current_velocity) < config.velocity_threshold:
                self.current_velocity = 0
                return Action.STAY
        else:
            # PID CONTROLLER: Calculate smooth velocity
            
            # P: Proportional - move toward target
            p_term = config.kp * error
            
            # I: Integral - accumulate error over time (corrects persistent offset)
            self.integral_error += error
            self.integral_error = max(-config.max_integral, 
                                     min(config.max_integral, self.integral_error))
            i_term = config.ki * self.integral_error
            
            # D: Derivative - resist changes (dampens oscillation)
            d_term = config.kd * (error - self.last_error)
            
            # Calculate desired velocity
            desired_velocity = p_term + i_term + d_term
            
            # Smooth velocity changes (low-pass filter)
            alpha = 0.7
            self.current_velocity = (alpha * desired_velocity) + ((1 - alpha) * self.current_velocity)
            
            # Acceleration limiting (prevents instant speed changes)
            max_accel_change = config.acceleration
            velocity_change = self.current_velocity - self.current_velocity
            if abs(velocity_change) > max_accel_change:
                sign = 1 if velocity_change > 0 else -1
                self.current_velocity = self.current_velocity + (sign * max_accel_change)
            
            # Clamp to max velocity
            self.current_velocity = max(-config.max_velocity, 
                                       min(config.max_velocity, self.current_velocity))
        
        # Store error for next derivative calculation
        self.last_error = error
        
        # VELOCITY TO ACTION CONVERSION
        # Convert smooth velocity back to discrete action
        threshold = config.max_velocity * config.action_threshold_multiplier
        
        if self.current_velocity > threshold:
            return Action.UP
        elif self.current_velocity < -threshold:
            return Action.DOWN
        else:
            return Action.STAY
    
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
        """Simplified weight update using gradient descent with gradient clipping"""
        # Forward pass
        X = state.reshape(1, -1)
        output = self.network.forward(X)
        
        # Output layer gradient
        d_output = 2 * (output - target.reshape(1, -1))
        
        # Clip gradients to prevent explosion (gradient clipping)
        max_grad = 10.0
        d_output = np.clip(d_output, -max_grad, max_grad)
        
        # Hidden layer gradient
        d_hidden = (d_output @ self.network.W2.T) * (self.network.a1 > 0)
        d_hidden = np.clip(d_hidden, -max_grad, max_grad)
        
        # Calculate weight gradients
        grad_W2 = self.network.a1.T @ d_output
        grad_b2 = d_output
        grad_W1 = X.T @ d_hidden
        grad_b1 = d_hidden
        
        # Clip weight gradients
        grad_W2 = np.clip(grad_W2, -max_grad, max_grad)
        grad_W1 = np.clip(grad_W1, -max_grad, max_grad)
        
        # Update weights
        self.network.W2 -= self.learning_rate * grad_W2
        self.network.b2 -= self.learning_rate * grad_b2
        self.network.W1 -= self.learning_rate * grad_W1
        self.network.b1 -= self.learning_rate * grad_b1
        
        # Check for NaN/Inf and reset if necessary
        if np.any(np.isnan(self.network.W1)) or np.any(np.isinf(self.network.W1)):
            print("⚠️ NaN/Inf detected in W1, resetting weights...")
            self.network.W1 = np.random.randn(self.network.input_size, self.network.hidden_size) * np.sqrt(2.0 / self.network.input_size)
        if np.any(np.isnan(self.network.W2)) or np.any(np.isinf(self.network.W2)):
            print("⚠️ NaN/Inf detected in W2, resetting weights...")
            self.network.W2 = np.random.randn(self.network.hidden_size, self.network.output_size) * np.sqrt(2.0 / self.network.hidden_size)
    
    def end_game(self, final_score_self, final_score_opponent):
        """
        Called when a game ends to update statistics and curriculum progression
        
        Args:
            final_score_self: This AI's final score
            final_score_opponent: Opponent's final score
        """
        self.games_played += 1
        
        # Update win/loss stats
        if final_score_self > final_score_opponent:
            self.win_count += 1
            self.total_rewards += 1
        elif final_score_self < final_score_opponent:
            self.loss_count += 1
            self.total_rewards -= 1
        else:
            self.tie_count += 1
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically (every 10 games)
        if self.games_played % 10 == 0:
            self.target_network.copy_weights_from(self.network)
        
        # CURRICULUM PROGRESSION: Update paddle config based on training progress
        if self.use_curriculum:
            new_phase = self.curriculum.get_active_phase(self.games_played)
            
            # Check for phase transition
            if self.current_phase is None or new_phase.name != self.current_phase.name:
                old_phase_name = self.current_phase.name if self.current_phase else "None"
                self.current_phase = new_phase
                self.phase_transition_games.append(self.games_played)
                
                # Update paddle configuration for new phase
                self.paddle_config = self.curriculum.get_config_for_games(self.games_played)
                
                # Log phase transition
                print(f"\n{'='*75}")
                print(f"📚 CURRICULUM PHASE TRANSITION (Game {self.games_played})")
                print(f"{'='*75}")
                print(f"Old Phase: {old_phase_name}")
                print(f"New Phase: {new_phase.name}")
                print(f"Description: {new_phase.description}")
                print(f"")
                print(f"Updated Behavior:")
                print(f"  Decision Interval: {new_phase.decision_interval} frames (measure)")
                print(f"  Action Commitment: {new_phase.action_commitment_frames} frames (act)")
                print(f"  Max Velocity: {new_phase.max_velocity:.1f}")
                print(f"  PID: Kp={new_phase.kp:.2f}, Kd={new_phase.kd:.2f}")
                print(f"{'='*75}\n")
        
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
            'tie_count': self.tie_count,
            'epsilon': self.epsilon,
            'use_curriculum': self.use_curriculum,
            'current_phase_name': self.current_phase.name if self.current_phase else None,
            'phase_transition_games': self.phase_transition_games,
            # Save paddle config as dict for compatibility
            'paddle_config': {
                'decision_interval': self.paddle_config.decision_interval,
                'action_commitment_frames': self.paddle_config.action_commitment_frames,
                'max_velocity': self.paddle_config.max_velocity,
                'acceleration': self.paddle_config.acceleration,
                'deceleration_factor': self.paddle_config.deceleration_factor,
                'use_pid': self.paddle_config.use_pid,
                'kp': self.paddle_config.kp,
                'ki': self.paddle_config.ki,
                'kd': self.paddle_config.kd,
                'dead_zone_threshold': self.paddle_config.dead_zone_threshold,
                'velocity_threshold': self.paddle_config.velocity_threshold,
                'action_threshold_multiplier': self.paddle_config.action_threshold_multiplier
            }
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
        self.tie_count = data.get('tie_count', 0)   # Default to 0 for old models
        self.epsilon = data.get('epsilon', 0.1)
        
        # Load curriculum data if available
        self.use_curriculum = data.get('use_curriculum', self.use_curriculum)
        current_phase_name = data.get('current_phase_name')
        self.phase_transition_games = data.get('phase_transition_games', [])
        
        # Restore current phase
        if current_phase_name and self.use_curriculum:
            for phase in self.curriculum.phases:
                if phase.name == current_phase_name:
                    self.current_phase = phase
                    break
        
        # Load or update paddle config
        if 'paddle_config' in data and not self.use_curriculum:
            # Explicit config stored, use it
            config_dict = data['paddle_config']
            self.paddle_config = PaddleControlConfig(**config_dict)
        elif self.use_curriculum:
            # Using curriculum, update config based on current progress
            self.paddle_config = self.curriculum.get_config_for_games(self.games_played)
            if self.current_phase is None:
                self.current_phase = self.curriculum.get_active_phase(self.games_played)
        # else: keep the config passed to __init__
        
        # Update target network
        self.target_network.copy_weights_from(self.network)
        
        return True
    
    def get_stats(self):
        """Get training statistics with curriculum information"""
        win_rate = self.win_count / self.games_played if self.games_played > 0 else 0
        loss_rate = self.loss_count / self.games_played if self.games_played > 0 else 0
        tie_rate = self.tie_count / self.games_played if self.games_played > 0 else 0
        
        stats = {
            'games_played': self.games_played,
            'wins': self.win_count,
            'losses': self.loss_count,
            'ties': self.tie_count,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'tie_rate': tie_rate,
            'epsilon': self.epsilon,
            'exploration': f"{self.epsilon * 100:.1f}%"
        }
        
        # Add curriculum info if active
        if self.use_curriculum and self.current_phase:
            stats['curriculum'] = {
                'enabled': True,
                'current_phase': self.current_phase.name,
                'phase_description': self.current_phase.description,
                'decision_interval': self.paddle_config.decision_interval,
                'action_commitment': self.paddle_config.action_commitment_frames,
                'max_velocity': self.paddle_config.max_velocity
            }
        else:
            stats['curriculum'] = {'enabled': False}
        
        return stats
    
    def reset(self):
        """Reset paddle physics state for new game (but keep learning progress)"""
        # Reset paddle control state
        self.current_velocity = 0.0
        self.target_position = 0.0
        self.last_error = 0.0
        self.integral_error = 0.0
        self.frames_since_decision = 0
        self.frames_since_action_change = 0
        self.last_action = Action.STAY
        self.last_paddle_y = 0.0
    
    def get_info(self):
        """Get information about this AI for display"""
        config = self.paddle_config
        info = {
            'type': 'Deep Q-Learning Neural Network with Curriculum Learning',
            'side': self.side,
            'description': f"Learning AI (Games: {self.games_played}, Exploration: {self.epsilon*100:.1f}%)",
            'paddle_control': {
                'decision_interval': config.decision_interval,
                'action_commitment': config.action_commitment_frames,
                'max_velocity': config.max_velocity,
                'use_pid': config.use_pid,
                'pid_params': f"Kp={config.kp:.2f}, Ki={config.ki:.3f}, Kd={config.kd:.2f}"
            },
            'learning_params': {
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            }
        }
        
        # Add curriculum info
        if self.use_curriculum and self.current_phase:
            info['curriculum'] = {
                'enabled': True,
                'current_phase': self.current_phase.name,
                'phase_description': self.current_phase.description
            }
        else:
            info['curriculum'] = {'enabled': False}
        
        return info
