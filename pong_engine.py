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
                 visible=True, enable_spin=True, enable_curve=True, frame_rate=200):
        """
        Initialize Pong game engine
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            ball_speed: Ball movement speed multiplier
            paddle_speed: How far paddle moves per action
            visible: Whether to show the game window (False for training)
            enable_spin: Enable spin mechanics
            enable_curve: Enable ball curve based on spin
            frame_rate: Target frame rate (frames per second) for physics calculations
        """
        self.width = width
        self.height = height
        self.ball_speed = ball_speed
        self.paddle_speed = paddle_speed
        self.visible = visible
        self.enable_spin = enable_spin
        self.enable_curve = enable_curve
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / frame_rate  # Time per frame in seconds
        
        # Game state
        self.score_a = 0
        self.score_b = 0
        self.game_over = False
        self.frame_count = 0
        
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
            self.pen.write(f"AI A: {self.score_a}    AI B: {self.score_b}    [{mode_str}]", 
                          align='center', font=('Courier', 20, 'normal'))
    
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
            
            # Paddle B collision (right paddle)
            elif (new_x > 340 and new_x < 350 and 
                  new_y < self.padB.ycor() + 60 and new_y > self.padB.ycor() - 60):
                new_x = 340
                ball_dx *= -1
                
                # Apply spin based on paddle velocity and impact position
                if self.enable_spin:
                    # Paddle velocity creates spin
                    ball_spin = self.padB_velocity * 0.1
                    
                    # Off-center hits create additional spin
                    impact_offset = new_y - self.padB.ycor()
                    ball_spin += impact_offset * 0.3
                    
                    # Clamp spin to max
                    ball_spin = max(-self.max_spin, min(self.max_spin, ball_spin))
                    self.ball.spin = ball_spin
                
                # Angle variation based on impact point
                impact_offset = new_y - self.padB.ycor()
                ball_dy += impact_offset * 0.05
                
                reward_b = 0.1  # Small reward for hitting ball
            
            # Paddle A collision (left paddle)
            elif (new_x < -340 and new_x > -350 and 
                  new_y < self.padA.ycor() + 60 and new_y > self.padA.ycor() - 60):
                new_x = -340
                ball_dx *= -1
                
                # Apply spin based on paddle velocity and impact position
                if self.enable_spin:
                    ball_spin = self.padA_velocity * 0.1
                    impact_offset = new_y - self.padA.ycor()
                    ball_spin += impact_offset * 0.3
                    ball_spin = max(-self.max_spin, min(self.max_spin, ball_spin))
                    self.ball.spin = ball_spin
                
                # Angle variation
                impact_offset = new_y - self.padA.ycor()
                ball_dy += impact_offset * 0.05
                
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
            
            # Paddle collisions with spin
            elif (self.ball_x > 340 and self.ball_x < 350 and 
                  self.ball_y < self.padB_y + 60 and self.ball_y > self.padB_y - 60):
                self.ball_x = 340
                self.ball_dx *= -1
                
                if self.enable_spin:
                    self.ball_spin = self.padB_velocity * 0.1
                    impact_offset = self.ball_y - self.padB_y
                    self.ball_spin += impact_offset * 0.3
                    self.ball_spin = max(-self.max_spin, min(self.max_spin, self.ball_spin))
                
                impact_offset = self.ball_y - self.padB_y
                self.ball_dy += impact_offset * 0.05
                reward_b = 0.1
                
            elif (self.ball_x < -340 and self.ball_x > -350 and 
                  self.ball_y < self.padA_y + 60 and self.ball_y > self.padA_y - 60):
                self.ball_x = -340
                self.ball_dx *= -1
                
                if self.enable_spin:
                    self.ball_spin = self.padA_velocity * 0.1
                    impact_offset = self.ball_y - self.padA_y
                    self.ball_spin += impact_offset * 0.3
                    self.ball_spin = max(-self.max_spin, min(self.max_spin, self.ball_spin))
                
                impact_offset = self.ball_y - self.padA_y
                self.ball_dy += impact_offset * 0.05
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
    
    def close(self):
        """Close the game window"""
        if self.visible:
            try:
                self.win.bye()
            except:
                pass
