"""
Physics-Perfect Expert System for Pong
Uses exact physics calculations to predict ball trajectory
WARNING: Ignores spin and curve - only uses straight-line vectors!
"""

from pong_engine import Action
import math


class PhysicsExpertAI:
    """
    Physics-perfect AI that calculates exact ball trajectories
    
    Strategy:
    - Calculates EXACT pixel where ball will arrive using physics
    - Uses only straight-line vector mathematics (ignores spin/curve)
    - Has paddle speed limits (slightly slower than needed to be beatable)
    - Perfect prediction but exploitable by advanced tactics
    
    This creates an interesting dynamic:
    - Expert is "perfect" at basic physics
    - Learning AI must discover spin/curve to beat it
    - As learning AI gets better, it exploits the expert's blind spots
    """
    
    def __init__(self, side='B', paddle_speed_multiplier=0.9, reaction_frames=0, difficulty=None):
        """
        Initialize Physics Expert AI
        
        Args:
            side: Which paddle this AI controls ('A' or 'B')
            paddle_speed_multiplier: Speed multiplier (0.9 = slightly slower than perfect)
            reaction_frames: Frames of delay before reacting (0 = instant)
            difficulty: DifficultyLevel object for tournament scaling (optional)
        """
        self.side = side
        self.paddle_x = -350 if side == 'A' else 350
        self.paddle_speed_multiplier = paddle_speed_multiplier
        self.reaction_frames = reaction_frames
        self.difficulty = difficulty
        
        # Distance interrupt - freeze when ball is this close
        self.freeze_distance_before = 150  # Freeze when ball approaching within this distance
        self.freeze_distance_after = 100   # Stay frozen after ball passes until it's this far away
        self.is_frozen = False  # Track if currently in freeze state
        self.last_ball_x = 0  # Track ball position to detect pass-through
        
        # State tracking
        self.frames_since_decision = 0
        self.target_y = 0
        self.current_velocity = 0
        self.max_velocity = 50  # Maximum paddle velocity per frame
        self.acceleration = 5  # Paddle acceleration
        
        # Calculation frequency throttling
        self.calculation_interval = 4  # Only recalculate every N frames (reduces jitter)
        self.frames_since_calculation = 0
        self.cached_target_y = 0
        
        # PID Controller base parameters (can be scaled by difficulty)
        base_kp = 0.35  # Proportional gain
        base_ki = 0.005  # Integral gain
        base_kd = 0.25  # Derivative gain
        
        # Apply difficulty-based PID scaling if difficulty provided
        if self.difficulty:
            self.kp, self.ki, self.kd = self.difficulty.get_pid_params(base_kp, base_ki, base_kd)
        else:
            self.kp = base_kp
            self.ki = base_ki
            self.kd = base_kd
        
        self.last_error = 0
        self.integral_error = 0
        self.max_integral = 100  # Prevent integral windup
        
        # Physics calculation cache
        self.prediction_cache = None
        self.last_ball_state = None
    
    def decide_action(self, state):
        """
        Decide action using perfect physics prediction
        
        Args:
            state: Game state dictionary from engine
            
        Returns:
            Action: UP, DOWN, or STAY
        """
        # Reaction delay
        self.frames_since_decision += 1
        if self.frames_since_decision < self.reaction_frames:
            return self._apply_current_velocity()
        
        # Get raw physics state
        raw = state['raw']
        ball_x = raw['ball_x']
        ball_y = raw['ball_y']
        ball_dx = raw['ball_dx']
        ball_dy = raw['ball_dy']
        paddle_speed = raw['paddle_speed']
        
        # Get our paddle position
        if self.side == 'A':
            paddle_y = raw['paddle_a_y']
        else:
            paddle_y = raw['paddle_b_y']
        
        # EXTENDED FREEZE ZONE - Prevents movement before AND after impact
        # This eliminates spin control by ensuring paddle is stationary during entire collision
        ball_distance = abs(ball_x - self.paddle_x)
        ball_approaching = (self.side == 'A' and ball_dx < 0) or (self.side == 'B' and ball_dx > 0)
        ball_moving_away = (self.side == 'A' and ball_dx > 0) or (self.side == 'B' and ball_dx < 0)
        
        # Enter freeze when ball gets close while approaching
        if ball_approaching and ball_distance < self.freeze_distance_before:
            self.is_frozen = True
        
        # Stay frozen even after ball passes, until it's far enough away
        # This is the key: prevents immediate movement after contact
        if self.is_frozen:
            if ball_moving_away and ball_distance > self.freeze_distance_after:
                # Ball has moved far enough away, release freeze
                self.is_frozen = False
            else:
                # Still in freeze zone - no movement allowed
                self.current_velocity = 0
                self.integral_error = 0
                self.last_error = 0
                self.last_ball_x = ball_x
                return Action.STAY
        
        # Store ball position for next frame
        self.last_ball_x = ball_x
        
        # THROTTLE CALCULATIONS - Only recalculate target every N frames to reduce jitter
        self.frames_since_calculation += 1
        if self.frames_since_calculation >= self.calculation_interval:
            self.frames_since_calculation = 0
            
            # Check if ball is moving away - if so, drift to center instead of tracking
            ball_moving_away = (self.side == 'A' and ball_dx > 0) or (self.side == 'B' and ball_dx < 0)
            
            if ball_moving_away:
                # When ball moves away, slowly return to center
                # But STOP when already near center to avoid jitter
                if abs(paddle_y) < 10:  # Already at center
                    self.cached_target_y = paddle_y  # Stay put
                else:
                    self.cached_target_y = 0  # Move to center
            else:
                # Ball approaching - calculate perfect intercept
                self.cached_target_y = self._calculate_perfect_intercept(
                    ball_x, ball_y, ball_dx, ball_dy
                )
        
        # Use cached target between calculations
        self.target_y = self.cached_target_y
        
        # Apply speed multiplier limitation
        effective_speed = paddle_speed * self.paddle_speed_multiplier
        
        # Calculate desired action with acceleration
        action = self._calculate_action_with_physics(paddle_y, self.target_y, effective_speed)
        
        return action
    
    def _calculate_perfect_intercept(self, ball_x, ball_y, ball_dx, ball_dy):
        """
        Calculate EXACT position where ball will cross our paddle line
        Uses pure physics - straight line vectors only, NO spin/curve
        
        This is the "perfect but limited" part:
        - Perfect prediction for straight-line physics
        - Completely ignores spin and curve effects
        - Learning AI can exploit this by using spin/curve
        
        NOTE: Ball direction check removed - handled in decide_action now
        """
        
        # Calculate frames until ball reaches our paddle line
        distance_to_paddle = abs(self.paddle_x - ball_x)
        if abs(ball_dx) < 0.01:
            return ball_y  # Ball not moving horizontally, stay with current position
        
        frames_to_arrival = distance_to_paddle / abs(ball_dx)
        
        # Predict Y position using straight-line physics only
        # NOTE: This ignores spin and curve! That's the exploit!
        predicted_y = ball_y + (ball_dy * frames_to_arrival)
        
        # Account for wall bounces (ball bounces at y = ±290)
        # Simulate each bounce
        bounces = 0
        current_dy = ball_dy
        current_y = ball_y
        
        for frame in range(int(frames_to_arrival)):
            current_y += current_dy
            
            # Check for wall bounce
            if current_y > 290:
                current_y = 290 - (current_y - 290)
                current_dy *= -1
                bounces += 1
            elif current_y < -290:
                current_y = -290 + (-290 - current_y)
                current_dy *= -1
                bounces += 1
        
        # Handle remaining fractional frame
        remaining = frames_to_arrival - int(frames_to_arrival)
        current_y += current_dy * remaining
        
        # Final bounds check
        current_y = max(-290, min(290, current_y))
        
        return current_y
    
    def _calculate_action_with_physics(self, current_y, target_y, max_speed):
        """
        Calculate action using PID controller for smooth, oscillation-free movement
        
        PID Controller:
        - P (Proportional): Move based on distance to target
        - I (Integral): Correct for persistent offset errors
        - D (Derivative): Dampen oscillation by resisting rapid changes
        
        This creates smooth, professional-looking paddle movement
        """
        # Calculate error (distance to target)
        error = target_y - current_y
        
        # Dead zone - close enough, stop moving
        # Larger dead zone when at center to prevent jitter
        dead_zone_threshold = 15 if abs(target_y) < 10 else 5
        
        if abs(error) < dead_zone_threshold:
            self.current_velocity *= 0.5  # Quick deceleration
            self.integral_error = 0  # Reset integral when at target
            
            if abs(self.current_velocity) < 0.5:
                self.current_velocity = 0
                return Action.STAY
        else:
            # P: Proportional term - move toward target
            p_term = self.kp * error
            
            # I: Integral term - accumulate error over time
            self.integral_error += error
            # Prevent integral windup
            self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))
            i_term = self.ki * self.integral_error
            
            # D: Derivative term - resist changes (dampen oscillation)
            d_term = self.kd * (error - self.last_error)
            
            # Calculate desired velocity using PID
            desired_velocity = p_term + i_term + d_term
            
            # Smooth velocity changes (low-pass filter)
            alpha = 0.7  # Smoothing factor (0 = no change, 1 = instant change)
            self.current_velocity = (alpha * desired_velocity) + ((1 - alpha) * self.current_velocity)
            
            # Clamp to max velocity
            self.current_velocity = max(-self.max_velocity, min(self.max_velocity, self.current_velocity))
        
        # Store error for next derivative calculation
        self.last_error = error
        
        # Convert velocity to discrete action
        # Use hysteresis to prevent action flickering
        threshold = max_speed * 0.3
        
        if self.current_velocity > threshold:
            return Action.UP
        elif self.current_velocity < -threshold:
            return Action.DOWN
        else:
            return Action.STAY
    
    def _apply_current_velocity(self):
        """Apply current velocity as action (for reaction delay)"""
        if abs(self.current_velocity) < 0.5:
            return Action.STAY
        elif self.current_velocity > 0:
            return Action.UP
        else:
            return Action.DOWN
    
    def reset(self):
        """Reset AI state for new game"""
        self.frames_since_decision = 0
        self.target_y = 0
        self.current_velocity = 0
        self.last_error = 0
        self.integral_error = 0
        self.is_frozen = False
        self.last_ball_x = 0
        self.frames_since_calculation = 0
        self.cached_target_y = 0
        self.prediction_cache = None
        self.last_ball_state = None
    
    def get_info(self):
        """Get information about this AI for display"""
        return {
            'type': 'Physics Expert System',
            'side': self.side,
            'speed_multiplier': self.paddle_speed_multiplier,
            'reaction_frames': self.reaction_frames,
            'freeze_before': self.freeze_distance_before,
            'freeze_after': self.freeze_distance_after,
            'description': f"Perfect Physics AI (Speed: {self.paddle_speed_multiplier:.0%}, Freeze: {self.freeze_distance_before}px→{self.freeze_distance_after}px)",
            'strategy': 'Extended Freeze Zone - No movement during entire collision window',
            'weakness': 'Ignores spin and curve - exploitable by advanced tactics!'
        }


# Difficulty presets
class PerfectPhysicsAI(PhysicsExpertAI):
    """Perfect physics AI - nearly unbeatable without spin/curve"""
    def __init__(self, side='B'):
        super().__init__(side=side, paddle_speed_multiplier=1.0, reaction_frames=0)


class FastPhysicsAI(PhysicsExpertAI):
    """Fast physics AI - very challenging"""
    def __init__(self, side='B'):
        super().__init__(side=side, paddle_speed_multiplier=0.95, reaction_frames=1)


class MediumPhysicsAI(PhysicsExpertAI):
    """Medium physics AI - balanced opponent"""
    def __init__(self, side='B'):
        super().__init__(side=side, paddle_speed_multiplier=0.85, reaction_frames=2)


class SlowPhysicsAI(PhysicsExpertAI):
    """Slow physics AI - easier to beat"""
    def __init__(self, side='B'):
        super().__init__(side=side, paddle_speed_multiplier=0.7, reaction_frames=3)


def create_physics_expert(difficulty='perfect', side='B'):
    """
    Create physics expert AI with specified difficulty
    
    Args:
        difficulty: 'perfect', 'fast', 'medium', or 'slow'
        side: 'A' or 'B'
    
    Returns:
        PhysicsExpertAI instance
    """
    if difficulty == 'perfect':
        return PerfectPhysicsAI(side)
    elif difficulty == 'fast':
        return FastPhysicsAI(side)
    elif difficulty == 'medium':
        return MediumPhysicsAI(side)
    elif difficulty == 'slow':
        return SlowPhysicsAI(side)
    else:
        return MediumPhysicsAI(side)
