"""
Pong Game Engine
Provides a clean API for AI agents to play Pong with advanced physics
"""

import turtle
import time
import math
from enum import Enum


class Action(Enum):
    """Possible actions for a paddle"""
    UP = 1
    DOWN = -1
    STAY = 0


class DifficultyLevel:
    """
    Progressive difficulty system that affects game mechanics
    
    As difficulty increases:
    - Paddle speed DECREASES (harder to reach ball)
    - Max ball speed INCREASES (ball moves faster)
    - Ball speed variation increases (more unpredictable)
    
    This creates natural skill ceilings and prevents infinite perfect games
    """
    
    def __init__(self, level=1):
        """
        Initialize difficulty level
        
        Args:
            level: Difficulty level (1-10+)
                  1 = Easy, 5 = Medium, 10 = Hard, 15+ = Expert
        """
        self.level = max(1, level)
        
        # Calculate difficulty-scaled parameters
        # Paddle speed: 100% at level 1, down to 40% at level 10
        self.paddle_speed_multiplier = max(0.4, 1.0 - (self.level - 1) * 0.06)
        
        # Max ball speed: 1.0x at level 1, up to 3.0x at level 10
        self.max_ball_speed_multiplier = 1.0 + (self.level - 1) * 0.2
        
        # Ball speed variation on impact (randomness factor)
        self.ball_speed_variation = 0.1 + (self.level - 1) * 0.02  # 10% to 28% at level 10
        
        # Impact velocity transfer (how much paddle velocity affects ball)
        self.impact_velocity_transfer = 0.3 + (self.level - 1) * 0.02  # 30% to 48% at level 10
    
    def get_paddle_speed(self, base_speed):
        """Get difficulty-adjusted paddle speed"""
        return base_speed * self.paddle_speed_multiplier
    
    def get_max_ball_speed(self, base_speed):
        """Get difficulty-adjusted max ball speed"""
        return base_speed * self.max_ball_speed_multiplier
    
    def get_description(self):
        """Get human-readable difficulty description"""
        if self.level <= 3:
            return f"Level {self.level}: Easy"
        elif self.level <= 6:
            return f"Level {self.level}: Medium"
        elif self.level <= 9:
            return f"Level {self.level}: Hard"
        else:
            return f"Level {self.level}: Expert"
    
    def __str__(self):
        return f"Difficulty {self.get_description()} (Paddle: {self.paddle_speed_multiplier:.0%}, Ball: {self.max_ball_speed_multiplier:.1f}x)"


class PongEngine:
    """
    Core Pong game engine with AI-friendly interface and advanced physics
    
    Features:
    - Ball spin mechanics (affects bounce angle)
    - Air curve (ball can curve mid-flight based on spin)
    - Frame-perfect physics simulation
    - Paddle impact velocity affects ball trajectory
    """
    
    def __init__(self, width=800, height=600, ball_speed=1.0, paddle_speed=25, 
                 visible=True, enable_spin=True, enable_curve=True, frame_rate=200,
                 difficulty_level=1, auto_progress=False, points_per_level=5):
        """
        Initialize Pong game engine
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            ball_speed: Base ball movement speed
            paddle_speed: Base paddle speed
            visible: Whether to show the game window (False for training)
            enable_spin: Enable spin mechanics
            enable_curve: Enable ball curve based on spin
            frame_rate: Target frame rate (frames per second) for physics calculations
            difficulty_level: Starting difficulty level (1-10+)
            auto_progress: Automatically increase difficulty as game progresses
            points_per_level: Points needed to advance difficulty (if auto_progress=True)
        """
        self.width = width
        self.height = height
        self.base_ball_speed = ball_speed
        self.base_paddle_speed = paddle_speed
        self.visible = visible
        self.enable_spin = enable_spin
        self.enable_curve = enable_curve
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / frame_rate  # Time per frame in seconds
        
        # Progressive difficulty system
        self.difficulty = DifficultyLevel(difficulty_level)
        self.auto_progress = auto_progress
        self.points_per_level = points_per_level
        self.points_at_last_level = 0
        
        # Apply difficulty scaling
        self.ball_speed = self.difficulty.get_max_ball_speed(self.base_ball_speed)
        self.paddle_speed = self.difficulty.get_paddle_speed(self.base_paddle_speed)
        
        # Game state
        self.score_a = 0
        self.score_b = 0
        self.game_over = False
        self.frame_count = 0
        
        # Telemetry system for debugging and analysis
        self.telemetry_enabled = False
        self.telemetry_data = {
            'frame': [],
            'time': [],
            'ball_x': [],
            'ball_y': [],
            'ball_dx': [],
            'ball_dy': [],
            'ball_spin': [],
            'paddle_a_y': [],
            'paddle_b_y': [],
            'paddle_a_velocity': [],
            'paddle_b_velocity': [],
            'paddle_a_target': [],  # Will be set by AI if available
            'paddle_b_target': [],
            'score_a': [],
            'score_b': [],
            'events': []  # Special events (scoring, bounces, etc.)
        }
        self.current_paddle_a_target = None
        self.current_paddle_b_target = None
        
        # Physics constants
        self.spin_decay = 0.98  # Spin decreases over time
        self.max_spin = 50.0  # Max spin speed
        self.curve_factor = 0.015  # How much spin affects trajectory
        self.spin_bounce_factor = 0.3  # How much spin affects bounce angle
        
        # Initialize graphics if visible
        if self.visible:
            self._init_graphics()
        else:
            # Headless mode - just track positions
            self.padA_y = 0
            self.padB_y = 0
            self.padA_velocity = 0  # Track paddle velocity for spin calculation
            self.padB_velocity = 0
            self.ball_x = 0
            self.ball_y = 0  
            self.ball_dx = ball_speed
            self.ball_dy = ball_speed
            self.ball_spin = 0.0  # Rotation speed (affects curve and bounce)
    
    def _init_graphics(self):
        """Initialize turtle graphics"""
        self.win = turtle.Screen()
        self.win.title('Pong AI - Advanced Physics')
        self.win.bgcolor('green')
        self.win.setup(width=self.width, height=self.height)
        self.win.tracer(0)
        
        # Draw field
        self._draw_field()
        
        # Paddle A (left)
        self.padA = turtle.Turtle()
        self.padA.speed(0)
        self.padA.shape('square')
        self.padA.shapesize(stretch_wid=6, stretch_len=1)
        self.padA.color('white')
        self.padA.penup()
        self.padA.goto(-350, 0)
        self.padA_prev_y = 0  # Track for velocity calculation
        self.padA_velocity = 0
        
        # Paddle B (right)
        self.padB = turtle.Turtle()
        self.padB.speed(0)
        self.padB.shape('square')
        self.padB.shapesize(stretch_wid=6, stretch_len=1)
        self.padB.color('white')
        self.padB.penup()
        self.padB.goto(350, 0)
        self.padB_prev_y = 0
        self.padB_velocity = 0
        
        # Ball
        self.ball = turtle.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('white')
        self.ball.penup()
        self.ball.goto(0, 0)
        self.ball.dx = self.ball_speed
        self.ball.dy = self.ball_speed
        self.ball.spin = 0.0  # Ball spin (rotation speed)
        
        # Score display
        self.pen = turtle.Turtle()
        self.pen.speed(0)
        self.pen.color('white')
        self.pen.penup()
        self.pen.hideturtle()
        self.pen.goto(0, 250)
        
        # Physics info display
        self.physics_pen = turtle.Turtle()
        self.physics_pen.speed(0)
        self.physics_pen.color('yellow')
        self.physics_pen.penup()
        self.physics_pen.hideturtle()
        self.physics_pen.goto(0, -270)
        
        self._update_score()
        self._update_physics_display()
    
    def _draw_field(self):
        """Draw the playing field"""
        draw = turtle.Turtle()
        draw.penup()
        draw.speed(0)
        draw.color('white')
        draw.hideturtle()
        draw.goto(-390, 295)
        draw.pendown()
        for i in range(2):
            draw.forward(770)
            draw.right(90)
            draw.forward(580)
            draw.right(90)
        draw.goto(0, 295)
        draw.right(90)
        draw.goto(0, -285)
        draw.penup()
        draw.goto(-50, 0)
        draw.pendown()
        draw.circle(50)
    
    def _update_score(self):
        """Update score display"""
        if self.visible:
            self.pen.clear()
            physics_mode = []
            if self.enable_spin:
                physics_mode.append("SPIN")
            if self.enable_curve:
                physics_mode.append("CURVE")
            mode_str = "+".join(physics_mode) if physics_mode else "BASIC"
            
            # Include difficulty level in display
            score_text = f"AI A: {self.score_a}    AI B: {self.score_b}    [{mode_str}]"
            if self.auto_progress:
                score_text += f" LVL {self.difficulty.level}"
            
            self.pen.write(score_text, align='center', font=('Courier', 20, 'normal'))
    
    def _update_physics_display(self):
        """Update physics information display"""
        if self.visible and (self.enable_spin or self.enable_curve):
            self.physics_pen.clear()
            if hasattr(self, 'ball'):
                spin_value = getattr(self.ball, 'spin', 0)
            else:
                spin_value = getattr(self, 'ball_spin', 0)
            
            self.physics_pen.write(f"Spin: {spin_value:+.1f} | Frame: {self.frame_count}", 
                                  align='center', font=('Courier', 12, 'normal'))
    
    def get_state(self):
        """
        Get current game state for AI decision making
        
        Returns:
            dict: Game state with normalized values and physics data
        """
        if self.visible:
            ball_x = self.ball.xcor()
            ball_y = self.ball.ycor()
            ball_dx = self.ball.dx
            ball_dy = self.ball.dy
            ball_spin = self.ball.spin if hasattr(self.ball, 'spin') else 0
            padA_y = self.padA.ycor()
            padB_y = self.padB.ycor()
            padA_vel = self.padA_velocity
            padB_vel = self.padB_velocity
        else:
            ball_x = self.ball_x
            ball_y = self.ball_y
            ball_dx = self.ball_dx
            ball_dy = self.ball_dy
            ball_spin = self.ball_spin
            padA_y = self.padA_y
            padB_y = self.padB_y
            padA_vel = self.padA_velocity
            padB_vel = self.padB_velocity
        
        # Return state with both raw and normalized values
        state = {
            # Normalized values (-1 to 1) for neural networks
            'ball_x': ball_x / 390,
            'ball_y': ball_y / 290,
            'ball_dx': ball_dx / (self.ball_speed * 2),  # Normalize velocity
            'ball_dy': ball_dy / (self.ball_speed * 2),
            'ball_spin': ball_spin / self.max_spin if self.enable_spin else 0,
            'paddle_a_y': padA_y / 290,
            'paddle_b_y': padB_y / 290,
            'paddle_a_vel': padA_vel / self.paddle_speed,
            'paddle_b_vel': padB_vel / self.paddle_speed,
            'score_a': self.score_a,
            'score_b': self.score_b,
            
            # Raw values for physics calculations
            'raw': {
                'ball_x': ball_x,
                'ball_y': ball_y,
                'ball_dx': ball_dx,
                'ball_dy': ball_dy,
                'ball_spin': ball_spin,
                'paddle_a_y': padA_y,
                'paddle_b_y': padB_y,
                'paddle_a_vel': padA_vel,
                'paddle_b_vel': padB_vel,
                'ball_speed': self.ball_speed,
                'paddle_speed': self.paddle_speed,
                'enable_spin': self.enable_spin,
                'enable_curve': self.enable_curve,
                'frame_rate': self.frame_rate,
                'frame_count': self.frame_count
            }
        }
        
        return state
    
    def step(self, action_a, action_b):
        """
        Execute one game step with given actions
        
        Args:
            action_a: Action for paddle A (Action enum or int)
            action_b: Action for paddle B (Action enum or int)
            
        Returns:
            tuple: (state, reward_a, reward_b, done)
        """
        # Convert to Action enum if needed
        if isinstance(action_a, int):
            action_a = Action(action_a)
        if isinstance(action_b, int):
            action_b = Action(action_b)
        
        # Move paddles and calculate velocities
        self._move_paddle('A', action_a)
        self._move_paddle('B', action_b)
        
        # Move ball and check collisions
        reward_a, reward_b = self._update_ball()
        
        # Update frame counter
        self.frame_count += 1
        
        # Collect telemetry if enabled
        self._collect_telemetry()
        
        # Update graphics if visible
        if self.visible:
            self.win.update()
            self._update_physics_display()
            time.sleep(0.005)
        
        return self.get_state(), reward_a, reward_b, self.game_over
    
    def _move_paddle(self, paddle, action):
        """Move a paddle based on action and track velocity"""
        if action == Action.STAY:
            if paddle == 'A':
                self.padA_velocity = 0
            else:
                self.padB_velocity = 0
            return
        
        movement = self.paddle_speed * action.value
        
        if self.visible:
            paddle_obj = self.padA if paddle == 'A' else self.padB
            old_y = paddle_obj.ycor()
            new_y = old_y + movement
            # Keep paddle in bounds (290 is top, -290 is bottom, paddle is 60 tall)
            new_y = max(-230, min(230, new_y))
            paddle_obj.sety(new_y)
            
            # Calculate actual velocity (actual movement / frame_time)
            actual_movement = new_y - old_y
            velocity = actual_movement / self.frame_time if self.frame_time > 0 else 0
            
            if paddle == 'A':
                self.padA_velocity = velocity
            else:
                self.padB_velocity = velocity
        else:
            if paddle == 'A':
                old_y = self.padA_y
                self.padA_y += movement
                self.padA_y = max(-230, min(230, self.padA_y))
                self.padA_velocity = (self.padA_y - old_y) / self.frame_time if self.frame_time > 0 else 0
            else:
                old_y = self.padB_y
                self.padB_y += movement
                self.padB_y = max(-230, min(230, self.padB_y))
                self.padB_velocity = (self.padB_y - old_y) / self.frame_time if self.frame_time > 0 else 0
    
    def _update_ball(self):
        """
        Update ball position with advanced physics including spin and curve
        Returns (reward_a, reward_b)
        """
        reward_a = 0
        reward_b = 0
        
        if self.visible:
            # Get current ball state
            ball_x = self.ball.xcor()
            ball_y = self.ball.ycor()
            ball_dx = self.ball.dx
            ball_dy = self.ball.dy
            ball_spin = self.ball.spin if hasattr(self.ball, 'spin') else 0
            
            # Apply curve based on spin (Magnus effect)
            if self.enable_curve and ball_spin != 0:
                # Spin creates perpendicular force
                curve_dy = ball_spin * self.curve_factor
                ball_dy += curve_dy
            
            # Move ball
            new_x = ball_x + ball_dx
            new_y = ball_y + ball_dy
            
            # Apply spin decay
            if self.enable_spin:
                ball_spin *= self.spin_decay
                self.ball.spin = ball_spin
            
            # Top/bottom wall collision
            if new_y > 290 or new_y < -290:
                ball_dy *= -1
                new_y = 290 if new_y > 290 else -290
                # Spin is reversed on wall bounce
                if self.enable_spin:
                    ball_spin *= -0.5
                    self.ball.spin = ball_spin
            
            # Right boundary (Paddle A scores)
            if new_x > 390:
                self.ball.goto(0, 0)
                self.ball.dx = self.ball_speed
                self.ball.dy = self.ball_speed
                self.ball.spin = 0
                self.score_a += 1
                reward_a = 1
                reward_b = -1
                self._update_score()
                self._check_difficulty_progression()  # Check for level up
            
            # Left boundary (Paddle B scores)
            elif new_x < -390:
                self.ball.goto(0, 0)
                self.ball.dx = self.ball_speed
                self.ball.dy = self.ball_speed
                self.ball.spin = 0
                self.score_b += 1
                reward_a = -1
                reward_b = 1
                self._update_score()
                self._check_difficulty_progression()  # Check for level up
            
            # Paddle B collision (right paddle)
            elif (new_x > 340 and new_x < 350 and 
                  new_y < self.padB.ycor() + 60 and new_y > self.padB.ycor() - 60):
                new_x = 340
                
                # Calculate impact offset from paddle center
                impact_offset = new_y - self.padB.ycor()
                
                # Dynamic ball speed based on impact physics
                ball_dx, ball_dy = self._calculate_impact_ball_speed(
                    ball_dx, ball_dy, self.padB_velocity, impact_offset
                )
                
                # Apply spin ONLY from paddle velocity (not impact position)
                # Off-center hits affect angle via ball_dy calculation, not spin
                if self.enable_spin:
                    # Spin comes ONLY from paddle lateral velocity
                    ball_spin = self.padB_velocity * 0.15  # Only paddle movement creates spin
                    
                    # Clamp spin to max
                    ball_spin = max(-self.max_spin, min(self.max_spin, ball_spin))
                    self.ball.spin = ball_spin
                
                reward_b = 0.1  # Small reward for hitting ball
            
            # Paddle A collision (left paddle)
            elif (new_x < -340 and new_x > -350 and 
                  new_y < self.padA.ycor() + 60 and new_y > self.padA.ycor() - 60):
                new_x = -340
                
                # Calculate impact offset from paddle center
                impact_offset = new_y - self.padA.ycor()
                
                # Dynamic ball speed based on impact physics
                ball_dx, ball_dy = self._calculate_impact_ball_speed(
                    ball_dx, ball_dy, self.padA_velocity, impact_offset
                )
                
                # Apply spin ONLY from paddle velocity (not impact position)
                if self.enable_spin:
                    # Spin comes ONLY from paddle lateral velocity
                    ball_spin = self.padA_velocity * 0.15  # Only paddle movement creates spin
                    ball_spin = max(-self.max_spin, min(self.max_spin, ball_spin))
                    self.ball.spin = ball_spin
                
                reward_a = 0.1
            
            # Update ball position and velocity
            self.ball.setx(new_x)
            self.ball.sety(new_y)
            self.ball.dx = ball_dx
            self.ball.dy = ball_dy
        
        else:
            # Headless mode - same physics
            # Apply curve
            if self.enable_curve and self.ball_spin != 0:
                curve_dy = self.ball_spin * self.curve_factor
                self.ball_dy += curve_dy
            
            # Move ball
            self.ball_x += self.ball_dx
            self.ball_y += self.ball_dy
            
            # Spin decay
            if self.enable_spin:
                self.ball_spin *= self.spin_decay
            
            # Top/bottom collision
            if self.ball_y > 290 or self.ball_y < -290:
                self.ball_dy *= -1
                self.ball_y = 290 if self.ball_y > 290 else -290
                if self.enable_spin:
                    self.ball_spin *= -0.5
            
            # Scoring
            if self.ball_x > 390:
                self.ball_x = 0
                self.ball_y = 0
                self.ball_dx = self.ball_speed
                self.ball_dy = self.ball_speed
                self.ball_spin = 0
                self.score_a += 1
                reward_a = 1
                reward_b = -1
            elif self.ball_x < -390:
                self.ball_x = 0
                self.ball_y = 0
                self.ball_dx = self.ball_speed
                self.ball_dy = self.ball_speed
                self.ball_spin = 0
                self.score_b += 1
                reward_a = -1
                reward_b = 1
            
            # Paddle collisions with spin and dynamic ball speed
            elif (self.ball_x > 340 and self.ball_x < 350 and 
                  self.ball_y < self.padB_y + 60 and self.ball_y > self.padB_y - 60):
                self.ball_x = 340
                
                # Calculate impact offset
                impact_offset = self.ball_y - self.padB_y
                
                # Dynamic ball speed
                self.ball_dx, self.ball_dy = self._calculate_impact_ball_speed(
                    self.ball_dx, self.ball_dy, self.padB_velocity, impact_offset
                )
                
                # Spin ONLY from paddle velocity
                if self.enable_spin:
                    self.ball_spin = self.padB_velocity * 0.15
                    self.ball_spin = max(-self.max_spin, min(self.max_spin, self.ball_spin))
                
                reward_b = 0.1
                
            elif (self.ball_x < -340 and self.ball_x > -350 and 
                  self.ball_y < self.padA_y + 60 and self.ball_y > self.padA_y - 60):
                self.ball_x = -340
                
                # Calculate impact offset
                impact_offset = self.ball_y - self.padA_y
                
                # Dynamic ball speed
                self.ball_dx, self.ball_dy = self._calculate_impact_ball_speed(
                    self.ball_dx, self.ball_dy, self.padA_velocity, impact_offset
                )
                
                # Spin ONLY from paddle velocity
                if self.enable_spin:
                    self.ball_spin = self.padA_velocity * 0.15
                    self.ball_spin = max(-self.max_spin, min(self.max_spin, self.ball_spin))
                
                reward_a = 0.1
        
        return reward_a, reward_b
    
    def reset(self):
        """Reset the game to initial state"""
        self.score_a = 0
        self.score_b = 0
        self.game_over = False
        self.frame_count = 0
        
        if self.visible:
            self.padA.goto(-350, 0)
            self.padB.goto(350, 0)
            self.padA_prev_y = 0
            self.padB_prev_y = 0
            self.padA_velocity = 0
            self.padB_velocity = 0
            
            self.ball.goto(0, 0)
            self.ball.dx = self.ball_speed
            self.ball.dy = self.ball_speed
            self.ball.spin = 0.0
            
            self._update_score()
            self._update_physics_display()
        else:
            self.padA_y = 0
            self.padB_y = 0
            self.padA_velocity = 0
            self.padB_velocity = 0
            self.ball_x = 0
            self.ball_y = 0
            self.ball_dx = self.ball_speed
            self.ball_dy = self.ball_speed
            self.ball_spin = 0.0
        
        return self.get_state()
    
    def enable_telemetry(self):
        """Enable telemetry data collection"""
        self.telemetry_enabled = True
        print("📊 Telemetry collection enabled")
    
    def disable_telemetry(self):
        """Disable telemetry data collection"""
        self.telemetry_enabled = False
        print("📊 Telemetry collection disabled")
    
    def clear_telemetry(self):
        """Clear all collected telemetry data"""
        for key in self.telemetry_data:
            if isinstance(self.telemetry_data[key], list):
                self.telemetry_data[key].clear()
        print("📊 Telemetry data cleared")
    
    def _collect_telemetry(self, event=None):
        """Collect current frame data for telemetry"""
        if not self.telemetry_enabled:
            return
        
        # Get current state
        if self.visible:
            ball_x = self.ball.xcor()
            ball_y = self.ball.ycor()
            ball_dx = self.ball.dx
            ball_dy = self.ball.dy
            ball_spin = self.ball.spin if hasattr(self.ball, 'spin') else 0
            paddle_a_y = self.padA.ycor()
            paddle_b_y = self.padB.ycor()
            paddle_a_velocity = self.padA_velocity
            paddle_b_velocity = self.padB_velocity
        else:
            ball_x = self.ball_x
            ball_y = self.ball_y
            ball_dx = self.ball_dx
            ball_dy = self.ball_dy
            ball_spin = self.ball_spin
            paddle_a_y = self.padA_y
            paddle_b_y = self.padB_y
            paddle_a_velocity = self.padA_velocity
            paddle_b_velocity = self.padB_velocity
        
        # Collect data
        self.telemetry_data['frame'].append(self.frame_count)
        self.telemetry_data['time'].append(self.frame_count * self.frame_time)
        self.telemetry_data['ball_x'].append(ball_x)
        self.telemetry_data['ball_y'].append(ball_y)
        self.telemetry_data['ball_dx'].append(ball_dx)
        self.telemetry_data['ball_dy'].append(ball_dy)
        self.telemetry_data['ball_spin'].append(ball_spin)
        self.telemetry_data['paddle_a_y'].append(paddle_a_y)
        self.telemetry_data['paddle_b_y'].append(paddle_b_y)
        self.telemetry_data['paddle_a_velocity'].append(paddle_a_velocity)
        self.telemetry_data['paddle_b_velocity'].append(paddle_b_velocity)
        self.telemetry_data['paddle_a_target'].append(self.current_paddle_a_target or paddle_a_y)
        self.telemetry_data['paddle_b_target'].append(self.current_paddle_b_target or paddle_b_y)
        self.telemetry_data['score_a'].append(self.score_a)
        self.telemetry_data['score_b'].append(self.score_b)
        
        # Record event if provided
        if event:
            self.telemetry_data['events'].append((self.frame_count, event))
    
    def get_telemetry_data(self):
        """Get all collected telemetry data"""
        return self.telemetry_data.copy()
    
    def export_telemetry(self, filename='telemetry.json'):
        """Export telemetry data to JSON file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.telemetry_data, f, indent=2)
        print(f"📊 Telemetry exported to {filename}")
    
    def set_paddle_targets(self, target_a=None, target_b=None):
        """Set target positions for telemetry (called by AIs)"""
        if target_a is not None:
            self.current_paddle_a_target = target_a
        if target_b is not None:
            self.current_paddle_b_target = target_b
    
    def close(self):
        """Close the game window"""
        if self.visible:
            try:
                self.win.bye()
            except:
                pass
    
    def _check_difficulty_progression(self):
        """Check if difficulty should increase based on total score"""
        if not self.auto_progress:
            return False
        
        total_score = self.score_a + self.score_b
        points_since_level = total_score - self.points_at_last_level
        
        if points_since_level >= self.points_per_level:
            self._increase_difficulty()
            self.points_at_last_level = total_score
            return True
        
        return False
    
    def _increase_difficulty(self):
        """Increase difficulty level and update game parameters"""
        old_level = self.difficulty.level
        self.difficulty = DifficultyLevel(old_level + 1)
        
        # Update game parameters based on new difficulty
        self.paddle_speed = self.difficulty.get_paddle_speed(self.base_paddle_speed)
        self.ball_speed = self.difficulty.get_max_ball_speed(self.base_ball_speed)
        
        if self.visible:
            print(f"\n🔥 DIFFICULTY INCREASED! {self.difficulty}")
            print(f"   Paddle Speed: {self.paddle_speed:.1f} ({self.difficulty.paddle_speed_multiplier:.0%})")
            print(f"   Max Ball Speed: {self.ball_speed:.1f} ({self.difficulty.max_ball_speed_multiplier:.1f}x)\n")
    
    def set_difficulty(self, level):
        """Manually set difficulty level"""
        self.difficulty = DifficultyLevel(level)
        self.paddle_speed = self.difficulty.get_paddle_speed(self.base_paddle_speed)
        self.ball_speed = self.difficulty.get_max_ball_speed(self.base_ball_speed)
        print(f"Difficulty set to: {self.difficulty}")
    
    def _calculate_impact_ball_speed(self, current_speed_x, current_speed_y, paddle_velocity, impact_offset):
        """
        Calculate new ball speed based on paddle impact physics
        
        Args:
            current_speed_x: Current ball horizontal velocity
            current_speed_y: Current ball vertical velocity
            paddle_velocity: Velocity of paddle at impact
            impact_offset: Distance from paddle center (-60 to +60)
            
        Returns:
            tuple: (new_speed_x, new_speed_y) with momentum preserved but variation applied
        """
        import random
        
        # Calculate current ball speed magnitude
        current_magnitude = math.sqrt(current_speed_x**2 + current_speed_y**2)
        
        # Base momentum preservation - maintain most of the speed
        base_momentum = 0.9  # 90% of current speed is preserved
        
        # Add velocity transfer from paddle (if moving in direction of ball)
        velocity_transfer = paddle_velocity * self.difficulty.impact_velocity_transfer
        
        # Calculate speed variation (randomness within difficulty range)
        speed_variation = random.uniform(-self.difficulty.ball_speed_variation, 
                                        self.difficulty.ball_speed_variation)
        
        # New magnitude combines momentum, velocity transfer, and variation
        new_magnitude = (current_magnitude * base_momentum) + abs(velocity_transfer) + (current_magnitude * speed_variation)
        
        # Clamp to min/max speeds (ensure ball never stops or goes too fast)
        min_speed = self.base_ball_speed * 0.5  # Never slower than 50% base
        max_speed = self.difficulty.get_max_ball_speed(self.base_ball_speed)
        new_magnitude = max(min_speed, min(max_speed, new_magnitude))
        
        # Apply to X direction (maintain direction, modify magnitude)
        new_speed_x = math.copysign(new_magnitude, -current_speed_x)  # Reverse X direction (bounce)
        
        # Y direction is modified by impact offset and paddle velocity
        angle_change = impact_offset * 0.05  # Off-center hits change angle
        new_speed_y = current_speed_y + angle_change + (velocity_transfer * 0.3)
        
        # Normalize to maintain overall speed
        actual_magnitude = math.sqrt(new_speed_x**2 + new_speed_y**2)
        if actual_magnitude > 0:
            scale = new_magnitude / actual_magnitude
            new_speed_x *= scale
            new_speed_y *= scale
        
        return new_speed_x, new_speed_y
