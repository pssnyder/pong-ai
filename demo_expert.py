"""
Demo: Watch the Physics Expert AI in action!
This script lets you see the expert system play with perfect physics prediction
"""

from pong_engine import PongEngine, Action
from pong_expert_ai import create_physics_expert, PhysicsExpertAI
import time


def demo_expert_vs_expert(games=5, enable_spin=True, enable_curve=True):
    """
    Watch two expert AIs battle (one faster than the other)
    
    Args:
        games: Number of games to play
        enable_spin: Enable spin physics
        enable_curve: Enable ball curve
    """
    print("=" * 70)
    print("🏓 PHYSICS EXPERT AI DEMONSTRATION 🤖")
    print("=" * 70)
    
    physics_mode = []
    if enable_spin:
        physics_mode.append("SPIN")
    if enable_curve:
        physics_mode.append("CURVE")
    mode_str = " + ".join(physics_mode) if physics_mode else "BASIC"
    
    print(f"\nPhysics Mode: {mode_str}")
    print("\n🎯 PROGRESSIVE DIFFICULTY ENABLED!")
    print("   Difficulty increases every 5 points scored")
    print("   • Paddles get SLOWER (harder to react)")
    print("   • Ball gets FASTER (harder to track)")
    print("   • Watch as perfect AIs eventually fail!")
    print()
    print("\nWatching Expert AIs Battle:")
    print("  • LEFT (Paddle A): FAST Physics Expert (95% speed)")
    print("  • RIGHT (Paddle B): MEDIUM Physics Expert (85% speed)")
    print()
    print("IMPORTANT: Both AIs use PERFECT physics prediction,")
    print("           BUT they only see straight-line vectors!")
    print("           Spin and curve are INVISIBLE to them!")
    print()
    print("EXTENDED FREEZE ZONE:")
    print("  • Freeze when ball within 150px (approaching)")
    print("  • Stay frozen during impact")
    print("  • Remain frozen until ball 100px away (departing)")
    print("  → This prevents ANY spin control from paddle movement")
    print("  → Trade-off: Predictability over Perfection!")
    print()
    print("Watch how the physics creates unpredictable outcomes...")
    print("=" * 70)
    print()
    
    # Create game engine with advanced physics AND progressive difficulty
    engine = PongEngine(
        visible=True, 
        ball_speed=1.5,
        paddle_speed=25,
        enable_spin=enable_spin,
        enable_curve=enable_curve,
        frame_rate=200,
        difficulty_level=1,  # Start at level 1
        auto_progress=True,  # Auto-increase difficulty
        points_per_level=5   # Level up every 5 points
    )
    
    # Enable telemetry for debugging
    engine.enable_telemetry()
    print("📊 Telemetry enabled - data will be saved to telemetry.json")
    print("   Run 'python telemetry_monitor.py' in another terminal to watch live graphs!\n")
    
    # Create two expert AIs with different speeds
    ai_a = PhysicsExpertAI(side='A', paddle_speed_multiplier=0.95, reaction_frames=1)
    ai_b = PhysicsExpertAI(side='B', paddle_speed_multiplier=0.85, reaction_frames=2)
    
    print(f"AI A Info: {ai_a.get_info()}")
    print(f"AI B Info: {ai_b.get_info()}")
    print()
    
    try:
        for game_num in range(1, games + 1):
            print(f"\n{'='*70}")
            print(f"GAME {game_num}/{games}")
            print(f"{'='*70}\n")
            
            state = engine.reset()
            ai_a.reset()
            ai_b.reset()
            
            frame_count = 0
            max_frames = 5000  # Prevent infinite games
            
            while frame_count < max_frames:
                # Get actions from both AIs
                action_a = ai_a.decide_action(state)
                action_b = ai_b.decide_action(state)
                
                # Report target positions for telemetry
                engine.set_paddle_targets(ai_a.target_y, ai_b.target_y)
                
                # Execute game step
                state, reward_a, reward_b, done = engine.step(action_a, action_b)
                
                frame_count += 1
                
                # Export telemetry periodically for live monitoring
                if frame_count % 100 == 0:
                    engine.export_telemetry('telemetry.json')
                
                # Check for scoring
                if reward_a > 0.5 or reward_b > 0.5:
                    scorer = "AI A" if reward_a > 0.5 else "AI B"
                    print(f"  ⚡ {scorer} SCORES! (Frame {frame_count})")
                    print(f"     Score: AI A: {state['score_a']} - AI B: {state['score_b']}")
                
                # End game after first to 5
                if state['score_a'] >= 5 or state['score_b'] >= 5:
                    break
            
            # Game over
            winner = "AI A (FAST)" if state['score_a'] > state['score_b'] else "AI B (MEDIUM)"
            print(f"\n  🏆 WINNER: {winner}")
            print(f"  📊 Final Score: AI A: {state['score_a']} - AI B: {state['score_b']}")
            print(f"  ⏱️  Total Frames: {frame_count}")
            
            if game_num < games:
                print("\n  (Starting next game in 2 seconds...)")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo stopped by user")
    
    # Final telemetry export
    engine.export_telemetry('telemetry.json')
    print("📊 Final telemetry saved to telemetry.json")
    
    print("\n" + "=" * 70)
    print("Demo Complete! Press Ctrl+C to exit.")
    print("=" * 70)
    
    engine.close()


def demo_expert_perfect(points=3, enable_spin=True, enable_curve=True):
    """
    Watch a PERFECT expert (100% speed, instant reaction) vs MEDIUM expert
    Perfect expert should dominate... unless spin/curve throw it off!
    """
    print("=" * 70)
    print("🎯 PERFECT PHYSICS AI vs MEDIUM AI 🤖")
    print("=" * 70)
    print()
    print("Testing if the 'PERFECT' expert can be beaten by physics!")
    print()
    print("LEFT (A): PERFECT Expert (100% speed, 0-frame reaction)")
    print("RIGHT (B): MEDIUM Expert (85% speed, 2-frame reaction)")
    print()
    print("The perfect AI should win... but spin/curve might cause surprises!")
    print("=" * 70)
    print()
    
    engine = PongEngine(
        visible=True,
        ball_speed=1.5,
        enable_spin=enable_spin,
        enable_curve=enable_curve
    )
    
    # Perfect vs Medium
    ai_a = PhysicsExpertAI(side='A', paddle_speed_multiplier=1.0, reaction_frames=0)
    ai_b = PhysicsExpertAI(side='B', paddle_speed_multiplier=0.85, reaction_frames=2)
    
    state = engine.reset()
    ai_a.reset()
    ai_b.reset()
    
    try:
        frame = 0
        while True:
            action_a = ai_a.decide_action(state)
            action_b = ai_b.decide_action(state)
            
            state, reward_a, reward_b, done = engine.step(action_a, action_b)
            frame += 1
            
            if reward_a > 0.5 or reward_b > 0.5:
                scorer = "PERFECT" if reward_a > 0.5 else "MEDIUM"
                print(f"⚡ {scorer} scores! (Frame {frame}) | Score: {state['score_a']}-{state['score_b']}")
            
            if state['score_a'] >= points or state['score_b'] >= points:
                winner = "PERFECT AI" if state['score_a'] > state['score_b'] else "MEDIUM AI"
                print(f"\n🏆 {winner} WINS! Final: {state['score_a']}-{state['score_b']}")
                break
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
    
    engine.close()


if __name__ == "__main__":
    import sys
    
    print("\n🎮 Pong Physics Expert AI Demo\n")
    print("Choose a demo:")
    print("  1. Expert vs Expert (multiple games)")
    print("  2. Perfect vs Medium (single game)")
    print("  3. Expert vs Expert (NO spin/curve - basic physics)")
    print()
    
    choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        demo_expert_vs_expert(games=3, enable_spin=True, enable_curve=True)
    elif choice == "2":
        demo_expert_perfect(points=5, enable_spin=True, enable_curve=True)
    elif choice == "3":
        demo_expert_vs_expert(games=3, enable_spin=False, enable_curve=False)
    else:
        print("Invalid choice. Running default demo...")
        demo_expert_vs_expert(games=3, enable_spin=True, enable_curve=True)
