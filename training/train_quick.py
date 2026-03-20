"""
Quick Start Training Script
Runs Phase 1 training with sensible defaults - perfect for getting started

Usage:
  python train_quick.py              # Train 500 games
  python train_quick.py 1000         # Train 1000 games
  python train_quick.py 1000 --watch # Train 1000 games and watch progress
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_progressive import ProgressiveTrainer


def main():
    # Parse arguments
    games = 500  # Default
    watch_mode = False
    
    if len(sys.argv) > 1:
        try:
            games = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of games: {sys.argv[1]}")
            print("Usage: python train_quick.py [games] [--watch]")
            return
    
    if '--watch' in sys.argv:
        watch_mode = True
    
    # Start training
    print("="*80)
    print("QUICK START: Phase 1 Training")
    print("="*80)
    print(f"Games: {games}")
    print(f"Save Interval: Every 100 games")
    print(f"Model Output: models/progressive/phase1_final.pkl")
    print("="*80 + "\n")
    
    trainer = ProgressiveTrainer()
    ai = trainer.train_phase_1(games=games, paddle_speed_pct=100)
    
    if ai and watch_mode:
        print("\n" + "="*80)
        print("WATCH MODE: Visualizing trained AI")
        print("="*80)
        print("Starting visual game in 3 seconds...")
        
        import time
        from pong_engine import PongEngine
        from pong_expert_ai import PhysicsExpertAI
        
        time.sleep(3)
        
        # Create visible game
        engine = PongEngine(
            visible=True,
            enable_spin=False,
            enable_curve=False,
            ball_speed=1.0,
            paddle_speed=25,
            frame_rate=60  # Slower for watching
        )
        
        expert = PhysicsExpertAI(side='B')
        
        print("Playing visual game... Close window to exit")
        
        # Play one game visually
        while not engine.game_over:
            state = engine.get_state()
            ai_action = ai.decide_action(state, training=False)  # No exploration
            expert_action = expert.decide_action(state)
            engine.step(ai_action, expert_action)
        
        print(f"\nFinal Score - AI: {engine.score_a}, Expert: {engine.score_b}")
        
        if engine.score_a > engine.score_b:
            print("AI WINS! 🎉")
        else:
            print("Expert wins. AI needs more training.")
    
    print("\n" + "="*80)
    print("Training session complete!")
    print("="*80)
    print("\nNext steps:")
    print("  - Continue Phase 1: python train_quick.py 500")
    print("  - Move to Phase 2: python train_progressive.py (option 3)")
    print("  - Evaluate performance: python evaluate_ai.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
