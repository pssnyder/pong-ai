"""
Real-time Telemetry Monitor for Pong AI
Displays live graphs of game metrics for debugging and analysis
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import os


class TelemetryMonitor:
    """
    Real-time monitoring tool for Pong AI telemetry
    
    Displays:
    - Ball trajectory (X-Y plot)
    - Paddle positions vs ball position
    - Paddle tracking (targets vs actual)
    - Ball velocity and spin
    - Oscillation analysis
    """
    
    def __init__(self, telemetry_file='telemetry.json', refresh_interval=100):
        """
        Initialize telemetry monitor
        
        Args:
            telemetry_file: Path to telemetry JSON file
            refresh_interval: Update interval in milliseconds
        """
        self.telemetry_file = telemetry_file
        self.refresh_interval = refresh_interval
        self.data = None
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🏓 Pong AI Telemetry Monitor', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Subplots
        self.ax_trajectory = self.fig.add_subplot(gs[0:2, 0])
        self.ax_positions = self.fig.add_subplot(gs[0, 1:])
        self.ax_tracking_a = self.fig.add_subplot(gs[1, 1])
        self.ax_tracking_b = self.fig.add_subplot(gs[1, 2])
        self.ax_velocity = self.fig.add_subplot(gs[2, 0])
        self.ax_spin = self.fig.add_subplot(gs[2, 1])
        self.ax_oscillation = self.fig.add_subplot(gs[2, 2])
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup plot formatting"""
        # Ball trajectory
        self.ax_trajectory.set_title('Ball Trajectory (2D)', fontweight='bold')
        self.ax_trajectory.set_xlabel('X Position')
        self.ax_trajectory.set_ylabel('Y Position')
        self.ax_trajectory.set_xlim(-400, 400)
        self.ax_trajectory.set_ylim(-300, 300)
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.axvline(x=-350, color='blue', linestyle='--', alpha=0.5, label='Paddle A')
        self.ax_trajectory.axvline(x=350, color='red', linestyle='--', alpha=0.5, label='Paddle B')
        self.ax_trajectory.axhline(y=290, color='gray', linestyle='-', alpha=0.3)
        self.ax_trajectory.axhline(y=-290, color='gray', linestyle='-', alpha=0.3)
        
        # Paddle positions
        self.ax_positions.set_title('Ball & Paddle Positions Over Time', fontweight='bold')
        self.ax_positions.set_xlabel('Frame')
        self.ax_positions.set_ylabel('Position')
        self.ax_positions.grid(True, alpha=0.3)
        
        # Tracking A
        self.ax_tracking_a.set_title('Paddle A: Target vs Actual', fontweight='bold')
        self.ax_tracking_a.set_xlabel('Frame')
        self.ax_tracking_a.set_ylabel('Y Position')
        self.ax_tracking_a.grid(True, alpha=0.3)
        
        # Tracking B
        self.ax_tracking_b.set_title('Paddle B: Target vs Actual', fontweight='bold')
        self.ax_tracking_b.set_xlabel('Frame')
        self.ax_tracking_b.set_ylabel('Y Position')
        self.ax_tracking_b.grid(True, alpha=0.3)
        
        # Velocity
        self.ax_velocity.set_title('Ball Velocity Components', fontweight='bold')
        self.ax_velocity.set_xlabel('Frame')
        self.ax_velocity.set_ylabel('Velocity')
        self.ax_velocity.grid(True, alpha=0.3)
        
        # Spin
        self.ax_spin.set_title('Ball Spin', fontweight='bold')
        self.ax_spin.set_xlabel('Frame')
        self.ax_spin.set_ylabel('Spin Rate')
        self.ax_spin.grid(True, alpha=0.3)
        
        # Oscillation analysis
        self.ax_oscillation.set_title('Paddle Oscillation (Velocity)', fontweight='bold')
        self.ax_oscillation.set_xlabel('Frame')
        self.ax_oscillation.set_ylabel('Velocity')
        self.ax_oscillation.grid(True, alpha=0.3)
    
    def load_data(self):
        """Load telemetry data from file"""
        if not os.path.exists(self.telemetry_file):
            return False
        
        try:
            with open(self.telemetry_file, 'r') as f:
                self.data = json.load(f)
            return True
        except:
            return False
    
    def update_plots(self, frame_num=None):
        """Update all plots with latest data"""
        if not self.load_data():
            return
        
        if not self.data or len(self.data.get('frame', [])) == 0:
            return
        
        # Clear all axes
        for ax in [self.ax_trajectory, self.ax_positions, self.ax_tracking_a, 
                   self.ax_tracking_b, self.ax_velocity, self.ax_spin, self.ax_oscillation]:
            ax.clear()
        
        self._setup_plots()
        
        # Extract data
        frames = np.array(self.data['frame'])
        ball_x = np.array(self.data['ball_x'])
        ball_y = np.array(self.data['ball_y'])
        ball_dx = np.array(self.data['ball_dx'])
        ball_dy = np.array(self.data['ball_dy'])
        ball_spin = np.array(self.data['ball_spin'])
        paddle_a_y = np.array(self.data['paddle_a_y'])
        paddle_b_y = np.array(self.data['paddle_b_y'])
        paddle_a_target = np.array(self.data['paddle_a_target'])
        paddle_b_target = np.array(self.data['paddle_b_target'])
        paddle_a_vel = np.array(self.data['paddle_a_velocity'])
        paddle_b_vel = np.array(self.data['paddle_b_velocity'])
        
        # Plot ball trajectory
        self.ax_trajectory.plot(ball_x, ball_y, 'g-', alpha=0.6, linewidth=1, label='Ball Path')
        self.ax_trajectory.scatter(ball_x[-1], ball_y[-1], c='green', s=100, marker='o', 
                                   edgecolors='darkgreen', linewidths=2, label='Current')
        self.ax_trajectory.legend()
        
        # Plot positions over time
        window = min(500, len(frames))  # Show last 500 frames
        start_idx = max(0, len(frames) - window)
        
        self.ax_positions.plot(frames[start_idx:], ball_y[start_idx:], 'g-', 
                              label='Ball Y', linewidth=2)
        self.ax_positions.plot(frames[start_idx:], paddle_a_y[start_idx:], 'b-', 
                              label='Paddle A', linewidth=1.5)
        self.ax_positions.plot(frames[start_idx:], paddle_b_y[start_idx:], 'r-', 
                              label='Paddle B', linewidth=1.5)
        self.ax_positions.legend()
        
        # Plot tracking A
        self.ax_tracking_a.plot(frames[start_idx:], paddle_a_y[start_idx:], 'b-', 
                               label='Actual', linewidth=2)
        self.ax_tracking_a.plot(frames[start_idx:], paddle_a_target[start_idx:], 'b--', 
                               label='Target', linewidth=1, alpha=0.7)
        # Highlight tracking error
        error_a = np.abs(paddle_a_y[start_idx:] - paddle_a_target[start_idx:])
        self.ax_tracking_a.fill_between(frames[start_idx:], paddle_a_y[start_idx:] - error_a, 
                                        paddle_a_y[start_idx:] + error_a, alpha=0.2, color='red')
        self.ax_tracking_a.legend()
        
        # Plot tracking B
        self.ax_tracking_b.plot(frames[start_idx:], paddle_b_y[start_idx:], 'r-', 
                               label='Actual', linewidth=2)
        self.ax_tracking_b.plot(frames[start_idx:], paddle_b_target[start_idx:], 'r--', 
                               label='Target', linewidth=1, alpha=0.7)
        error_b = np.abs(paddle_b_y[start_idx:] - paddle_b_target[start_idx:])
        self.ax_tracking_b.fill_between(frames[start_idx:], paddle_b_y[start_idx:] - error_b, 
                                        paddle_b_y[start_idx:] + error_b, alpha=0.2, color='red')
        self.ax_tracking_b.legend()
        
        # Plot velocity
        self.ax_velocity.plot(frames[start_idx:], ball_dx[start_idx:], 'b-', 
                             label='Velocity X', linewidth=1.5)
        self.ax_velocity.plot(frames[start_idx:], ball_dy[start_idx:], 'r-', 
                             label='Velocity Y', linewidth=1.5)
        self.ax_velocity.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.ax_velocity.legend()
        
        # Plot spin
        self.ax_spin.plot(frames[start_idx:], ball_spin[start_idx:], 'purple', linewidth=2)
        self.ax_spin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        max_spin = np.max(np.abs(ball_spin)) if len(ball_spin) > 0 else 1
        self.ax_spin.fill_between(frames[start_idx:], 0, ball_spin[start_idx:], 
                                 alpha=0.3, color='purple')
        
        # Plot oscillation (paddle velocities)
        self.ax_oscillation.plot(frames[start_idx:], paddle_a_vel[start_idx:], 'b-', 
                                label='Paddle A Velocity', linewidth=1.5)
        self.ax_oscillation.plot(frames[start_idx:], paddle_b_vel[start_idx:], 'r-', 
                                label='Paddle B Velocity', linewidth=1.5)
        self.ax_oscillation.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.ax_oscillation.legend()
        
        # Add statistics
        if len(paddle_a_vel) > 0:
            avg_error_a = np.mean(np.abs(paddle_a_y - paddle_a_target))
            avg_error_b = np.mean(np.abs(paddle_b_y - paddle_b_target))
            oscillation_a = np.std(paddle_a_vel)
            oscillation_b = np.std(paddle_b_vel)
            
            stats_text = f"Avg Tracking Error: A={avg_error_a:.2f}  B={avg_error_b:.2f}\n"
            stats_text += f"Oscillation (std): A={oscillation_a:.2f}  B={oscillation_b:.2f}\n"
            stats_text += f"Total Frames: {len(frames)}"
            
            self.fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def start_realtime_monitor(self):
        """Start real-time monitoring (auto-refresh)"""
        print("🔍 Starting real-time telemetry monitor...")
        print(f"   Monitoring: {self.telemetry_file}")
        print(f"   Refresh: {self.refresh_interval}ms")
        print("   Close window to stop monitoring")
        
        # Create animation
        ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                     interval=self.refresh_interval, 
                                     cache_frame_data=False)
        plt.show()
    
    def show_static_analysis(self):
        """Show static analysis of telemetry file"""
        print("📊 Loading telemetry for analysis...")
        
        if not self.load_data():
            print("❌ Could not load telemetry file")
            return
        
        self.update_plots()
        plt.show()


def monitor_live_game(telemetry_file='telemetry.json', refresh_ms=100):
    """
    Monitor a live game in real-time
    
    Args:
        telemetry_file: Path to telemetry JSON file
        refresh_ms: Refresh interval in milliseconds
    """
    monitor = TelemetryMonitor(telemetry_file, refresh_ms)
    monitor.start_realtime_monitor()


def analyze_telemetry(telemetry_file='telemetry.json'):
    """
    Analyze completed telemetry data
    
    Args:
        telemetry_file: Path to telemetry JSON file
    """
    monitor = TelemetryMonitor(telemetry_file)
    monitor.show_static_analysis()


if __name__ == "__main__":
    import sys
    
    print("\n📊 Pong AI Telemetry Monitor\n")
    print("Choose mode:")
    print("  1. Real-time monitor (live game)")
    print("  2. Analyze saved telemetry")
    print()
    
    choice = input("Enter choice (1-2) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\n📍 Make sure to:")
        print("   1. Enable telemetry in your game: engine.enable_telemetry()")
        print("   2. Export telemetry periodically: engine.export_telemetry()")
        print()
        monitor_live_game(refresh_ms=500)  # 500ms refresh for live monitoring
    elif choice == "2":
        filename = input("Telemetry file [telemetry.json]: ").strip() or "telemetry.json"
        analyze_telemetry(filename)
    else:
        print("Invalid choice")
