"""
Quick test for your trained model from train_ai.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pong_engine import PongEngine
from pong_expert_ai import PhysicsExpertAI
from pong_learning_ai import LearningAI
import time

def test_trained_model(num_games=10, visual=True):
    """
    Test your trained model
    
    Args:
        num_games: Number of games to play
        visual: True to watch, False for headless speed test
    """
    # Load your trained model
    model_path = 'models/pong_learning_ai.pkl'
    
    print("="*80)
    print("🧪 TESTING YOUR TRAINED MODEL")
    print("="*80)
    
    ai = LearningAI(side='A')
    if not ai.load(model_path):
        print(f"❌ Model not found: {model_path}")
        print("\nTrain a model first with:")
        print("  python training/train_ai.py")
        return
    
    # Show model stats
    stats = ai.get_stats()
    print(f"\n📊 Loaded Model Statistics:")
    print(f"  Training Games: {stats['games_played']}")
    print(f"  Training Win Rate: {stats['win_rate']*100:.1f}% (vs Expert)")
    print(f"  Record: {stats['wins']}W-{stats['losses']}L-{stats['ties']}T")
    print(f"  Exploration Level: {stats['exploration']}")
    print("="*80)
    
    if visual:
        print("\n🎮 VISUAL TEST MODE")
        print("   Watching AI play 1 game...")
        print("   Left (Paddle A) = Learning AI (YOUR MODEL)")
        print("   Right (Paddle B) = Expert AI")
        print("\nStarting in 2 seconds...")
        time.sleep(2)
        num_games = 1
    else:
        print(f"\n⚡ BENCHMARK MODE - {num_games} games (headless)")
    
    # Create opponent
    expert = PhysicsExpertAI(side='B')
    
    # Test statistics
    wins = 0
    losses = 0
    ties = 0
    total_ai_points = 0
    total_expert_points = 0
    
    start_time = time.time()
    
    for game_num in range(1, num_games + 1):
        # Create game with SAME parameters as training
        engine = PongEngine(
            visible=visual,
            ball_speed=2.0,  # Must match training ball_speed!
            paddle_speed=25,
            enable_spin=True,
            enable_curve=True,
            frame_rate=30 if visual else 200,  # 30 FPS for visual, 200 FPS for benchmark
            winning_score=11
        )
        
        # Reset to ensure clean start
        engine.reset()
        
        # Play game
        while not engine.game_over:
            state = engine.get_state()
            
            # AI uses trained model (no exploration - pure exploitation)
            ai_action = ai.decide_action(state, training=False)
            expert_action = expert.decide_action(state)
            
            engine.step(ai_action, expert_action)
        
        # Record results
        total_ai_points += engine.score_a
        total_expert_points += engine.score_b
        
        if engine.score_a > engine.score_b:
            wins += 1
            result = "✅ WIN"
        elif engine.score_b > engine.score_a:
            losses += 1
            result = "❌ LOSS"
        else:
            ties += 1
            result = "🤝 TIE"
        
        if visual:
            print(f"\n{'='*80}")
            print(f"GAME RESULT: {result}")
            print(f"{'='*80}")
            print(f"Learning AI: {engine.score_a}")
            print(f"Expert AI:   {engine.score_b}")
        else:
            # Progress for benchmark
            if game_num % 10 == 0 or game_num == 1:
                current_wr = (wins / game_num) * 100
                print(f"Game {game_num}/{num_games}: {result} | "
                      f"Win Rate: {current_wr:.1f}% ({wins}W-{losses}L-{ties}T)")
        
        engine.close()
    
    elapsed = time.time() - start_time
    
    # Final results
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Games Played: {num_games}")
    print(f"Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
    print(f"\n🎯 Learning AI Performance (vs Expert):")
    print(f"  Wins:   {wins:3d} ({wins/num_games*100:5.1f}%)")
    print(f"  Losses: {losses:3d} ({losses/num_games*100:5.1f}%)")
    print(f"  Ties:   {ties:3d} ({ties/num_games*100:5.1f}%)")
    print(f"\n📈 Average Score Per Game:")
    print(f"  Learning AI: {total_ai_points/num_games:.1f} points")
    print(f"  Expert AI:   {total_expert_points/num_games:.1f} points")
    print(f"  Point Margin: {(total_ai_points - total_expert_points)/num_games:+.1f}")
    
    # Interpretation
    print(f"\n💡 Performance Assessment:")
    test_wr = (wins / num_games) * 100
    if test_wr >= 95:
        print("  🌟 EXCELLENT - AI has mastered the game!")
    elif test_wr >= 70:
        print("  ✅ STRONG - AI is beating the expert consistently")
    elif test_wr >= 50:
        print("  ⚖️  COMPETITIVE - AI trades wins with expert")
    elif test_wr >= 25:
        print("  📚 LEARNING - AI is improving but needs more training")
    else:
        print("  🔄 NEEDS TRAINING - Model needs more training games")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            # Benchmark mode: 100 games headless
            games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            test_trained_model(num_games=games, visual=False)
        elif sys.argv[1] == "--quick":
            # Quick visual test
            test_trained_model(num_games=1, visual=True)
        else:
            print("Usage:")
            print("  python test_my_model.py              # Watch 1 visual game")
            print("  python test_my_model.py --benchmark  # Test 100 games (headless)")
            print("  python test_my_model.py --benchmark 50  # Test 50 games")
    else:
        # Default: visual test
        test_trained_model(num_games=1, visual=True)
