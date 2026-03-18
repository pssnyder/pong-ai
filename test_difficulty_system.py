"""
Test and visualize the tournament difficulty system
Shows how ball speed, paddle speed, and physics change across levels 1-10
"""

from pong_engine import DifficultyLevel
import math

def test_difficulty_levels():
    """Test and display all difficulty levels"""
    print("=" * 80)
    print("PONG AI TOURNAMENT DIFFICULTY SYSTEM")
    print("=" * 80)
    print()
    
    for level in range(1, 11):
        diff = DifficultyLevel(level)
        
        print(f"{'='*80}")
        print(f"LEVEL {level}: {diff.name}")
        print(f"{'='*80}")
        print(diff.get_physics_description())
        print()
        
        # Show ball speed progression within a volley
        print(f"Ball Speed Progression (within volley):")
        base_speed = 5.0
        for hit in range(0, 11, 2):
            speed = diff.get_ball_speed_after_hit(base_speed, hit)
            increase_pct = ((speed / base_speed) - 1) * 100
            print(f"  Hit {hit:2d}: {speed:5.2f} px/s (+{increase_pct:5.1f}%)")
        print()
        
        # Show PID parameter scaling
        base_kp, base_ki, base_kd = 0.35, 0.005, 0.25
        kp, ki, kd = diff.get_pid_params(base_kp, base_ki, base_kd)
        print(f"PID Parameters (Expert AI control tuning):")
        print(f"  Kp: {kp:.3f} (base: {base_kp:.3f}, ratio: {kp/base_kp:.1%})")
        print(f"  Ki: {ki:.3f} (base: {base_ki:.3f}, ratio: {ki/base_ki:.1%})")
        print(f"  Kd: {kd:.3f} (base: {base_kd:.3f}, ratio: {kd/base_kd:.1%})")
        print()
        
        # Show paddle physics
        base_accel = 50.0
        accel = diff.get_paddle_acceleration(base_accel)
        print(f"Paddle Physics:")
        print(f"  Speed: {diff.paddle_speed_multiplier:.1%}")
        print(f"  Acceleration: {accel:.1f} px/s² (base: {base_accel:.1f}, ratio: {accel/base_accel:.1%})")
        print(f"  Mass: {diff.paddle_mass:.2f}x (inertia/overshoot factor)")
        print()
        print()

def compare_tournament_progression():
    """Show side-by-side comparison of key metrics across all levels"""
    print("\n" + "=" * 80)
    print("TOURNAMENT PROGRESSION SUMMARY")
    print("=" * 80)
    print()
    
    # Header
    print(f"{'Level':<8} {'Name':<15} {'Paddle':<10} {'Mass':<8} {'Ball/Hit':<10} {'Max Ball':<10}")
    print("-" * 80)
    
    for level in range(1, 11):
        diff = DifficultyLevel(level)
        
        # Calculate representative values
        base_speed = 5.0
        speed_after_10_hits = diff.get_ball_speed_after_hit(base_speed, 10)
        ball_factor = ((speed_after_10_hits / base_speed) - 1) * 100
        
        print(f"{level:<8} "
              f"{diff.name:<15} "
              f"{diff.paddle_speed_multiplier:>6.1%}     "
              f"{diff.paddle_mass:>4.2f}x    "
              f"+{ball_factor:>4.1f}%      "
              f"{diff.max_ball_speed_multiplier:>4.1f}x")
    
    print()
    print("Legend:")
    print("  Paddle: Paddle speed multiplier (lower = slower)")
    print("  Mass: Paddle mass (higher = more inertia/overshoot)")
    print("  Ball/Hit: Ball speed increase after 10 paddle hits")
    print("  Max Ball: Maximum ball speed cap")
    print()

def show_recommended_tournament_structure():
    """Show recommended tournament structure"""
    print("=" * 80)
    print("RECOMMENDED TOURNAMENT STRUCTURE")
    print("=" * 80)
    print()
    
    print("Best of 3 Format (Recommended for Testing):")
    print("  - Each match: First to 11 points")
    print("  - Win 2/3 games to advance to next level")
    print("  - Tournament: Progress through levels 1-10")
    print("  - Total games: ~30-60 games (if AI wins efficiently)")
    print()
    
    print("Best of 5 Format (Championship Mode):")
    print("  - Each match: First to 11 points")
    print("  - Win 3/5 games to advance to next level")
    print("  - Tournament: Progress through levels 1-10")
    print("  - Total games: ~50-100 games")
    print()
    
    print("Training Strategy:")
    print("  1. Start with basic training (no difficulty scaling)")
    print("  2. Once AI can beat Level 1 Expert ~50% of the time")
    print("  3. Begin tournament mode to test progressive difficulty")
    print("  4. Monitor win rate at each level")
    print("  5. Retrain if AI struggles at specific difficulty thresholds")
    print()
    
    print("Expected Difficulty Curve:")
    print("  Levels 1-3:  AI should win majority (amateur expert)")
    print("  Levels 4-6:  Competitive matches (professional expert)")
    print("  Levels 7-9:  Expert struggles (elite expert)")
    print("  Level 10:    Extreme challenge (master expert)")
    print()

if __name__ == "__main__":
    test_difficulty_levels()
    compare_tournament_progression()
    show_recommended_tournament_structure()
    
    print("=" * 80)
    print("Test complete! Difficulty system is configured.")
    print("=" * 80)
