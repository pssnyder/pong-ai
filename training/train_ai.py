"""
Training Script: Learning AI vs Physics Expert
Watch the neural network learn to beat the expert system!
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pong_engine import PongEngine, Action
from pong_expert_ai import PhysicsExpertAI, ExpertPresets, ExpertControlConfig
from pong_learning_ai import LearningAI, ControlPresets, PaddleControlConfig
import os
import time


# ============================================================================
#                    TRAINING CONFIGURATION
# ============================================================================
# Configure AI behaviors here for easy tuning

def get_training_configs(use_curriculum=True):
    """
    Configure both AIs for training
    
    Tuning Guide:
    - Set use_curriculum=True to enable progressive skill building (RECOMMENDED)
    - Set use_curriculum=False to use fixed configuration
    - Use ControlPresets for quick preset selection
    - Create custom PaddleControlConfig for fine-tuning
    - Adjust learning AI's decision_interval to control reactivity
    - Adjust expert AI's paddle_speed_multiplier to control difficulty
    
    Args:
        use_curriculum: Enable curriculum learning (True = staged progression)
    
    Returns:
        tuple: (learning_config, expert_config, use_curriculum)
    """
    
    # LEARNING AI CONFIGURATION
    if use_curriculum:
        # Use curriculum learning (RECOMMENDED for proper skill development)
        # AI will progress through phases:
        #   Phase 1 (0-500): Observation & Tracking
        #   Phase 2 (500-2000): Trajectory Prediction  
        #   Phase 3 (2000+): Advanced Physics Exploitation
        learning_config = None  # Will be set by curriculum system
    else:
        # Use fixed configuration (no progression)
        learning_config = ControlPresets.ultra_smooth()
        
        # OR customize for specific behavior:
        # learning_config = PaddleControlConfig(
        #     decision_interval=4,        # More patient decisions
        #     action_commitment_frames=3,  # Commits to actions longer
        #     max_velocity=42.0,          # Slightly limited speed
        #     kp=0.35,                    # Moderate response
        #     kd=0.4                      # Strong damping (smooth)
        # )
    
    # EXPERT AI CONFIGURATION
    # Use medium preset for balanced training opponent
    expert_config = ExpertPresets.medium()
    
    # OR customize difficulty:
    # expert_config = ExpertControlConfig(
    #     paddle_speed_multiplier=0.85,  # 85% speed
    #     reaction_frames=2,             # 2 frame delay
    #     calculation_interval=5,        # Recalculate less often
    #     kp=0.3,                        # Gentler response
    #     kd=0.3                         # More damping
    # )
    
    return learning_config, expert_config, use_curriculum


# ============================================================================


def visual_training(games=100, max_points=5, save_every=25, enable_spin=True, enable_curve=True):
    """
    Visual training mode - watch the AI learn in real-time!
    
    Args:
        games: Number of games to train
        max_points: Points needed to win a game
        save_every: Save model every N games
        enable_spin: Enable spin physics
        enable_curve: Enable curve physics
    """
    print("=" * 75)
    print("🎓 LEARNING AI TRAINING vs PHYSICS EXPERT 🤖")
    print("=" * 75)
    print()
    print("STRATEGY:")
    print("  🔵 LEFT (Paddle A): LEARNING AI - Neural Network (starts knowing nothing!)")
    print("  🔴 RIGHT (Paddle B): PHYSICS EXPERT - Perfect predictions (but blind to spin!)")
    print()
    print("GOAL:")
    print("  The Learning AI must discover that spinner/curve beats the expert!")
    print()
    print(f"Training for {games} games | First to {max_points} points wins each game")
    print(f"Physics: {'SPIN + CURVE' if enable_spin and enable_curve else 'BASIC'}")
    print("=" * 75)
    print()
    
    # Get AI configurations
    learning_config, expert_config, use_curriculum = get_training_configs(use_curriculum=True)
    
    # Create engine
    engine = PongEngine(
        visible=True,
        ball_speed=1.5,
        paddle_speed=25,
        enable_spin=enable_spin,
        enable_curve=enable_curve,
        frame_rate=200
    )
    
    # Create AIs with paddle control configurations
    learning_ai = LearningAI(
        side='A', 
        learning_rate=0.001, 
        discount_factor=0.95,
        paddle_config=learning_config,
        use_curriculum=use_curriculum
    )
    expert_ai = PhysicsExpertAI(
        side='B', 
        control_config=expert_config
    )
    
    # Display AI configurations
    print("🔧 TRAINING CONFIGURATION:")
    if use_curriculum:
        print(f"  Learning AI: CURRICULUM LEARNING ENABLED")
        if learning_ai.current_phase:
            phase = learning_ai.current_phase
            print(f"    Current Phase: {phase.name}")
            print(f"    Description: {phase.description}")
            print(f"    Settings: interval={phase.decision_interval}f, "
                  f"commit={phase.action_commitment_frames}f, "
                  f"vel={phase.max_velocity:.0f}, "
                  f"PID=({phase.kp:.2f}/{phase.kd:.2f})")
            print(f"    Phase progression will auto-adjust behavior at 500 and 2000 games")
    else:
        print(f"  Learning AI: FIXED CONFIGURATION")
        if learning_config:
            print(f"    Settings: interval={learning_config.decision_interval}f, "
                  f"vel={learning_config.max_velocity:.0f}, "
                  f"PID=({learning_config.kp:.2f}, {learning_config.kd:.2f})")
    print(f"  Expert AI: speed={expert_config.paddle_speed_multiplier:.0%}, "
          f"reaction={expert_config.reaction_frames}f")
    print()
    
    # Try to load existing model
    model_path = 'models/pong_learning_ai.pkl'
    if os.path.exists(model_path):
        if learning_ai.load(model_path):
            print(f"✅ Loaded existing model from: {model_path}")
            stats = learning_ai.get_stats()
            print(f"   Resuming from: {stats['games_played']} games, {stats['win_rate']*100:.1f}% Learning AI win rate")
            print()
        else:
            print(f"🆕 Starting fresh training - existing model incompatible with current code")
            print()
    else:
        print("🆕 Starting fresh training - new neural network created")
        print()
    
    try:
        for game_num in range(1, games + 1):
            print(f"\n{'─'*75}")
            print(f"🎮 GAME {game_num}/{games}")
            print(f"{'─'*75}")
            
            state = engine.reset()
            learning_ai.reset()
            expert_ai.reset()
            
            frame_count = 0
            max_frames = 5000  # Prevent infinite games
            prev_state = None
            prev_action = None
            total_reward = 0
            
            while frame_count < max_frames:
                # Get actions
                action_learning = learning_ai.decide_action(state, training=True)
                action_expert = expert_ai.decide_action(state)
                
                # Store previous state for learning
                if prev_state is not None and prev_action is not None:
                    reward = total_reward
                    learning_ai.remember(prev_state, prev_action, reward, state, False)
                
                prev_state = state
                prev_action = action_learning
                
                # Execute step
                state, reward_a, reward_b, done = engine.step(action_learning, action_expert)
                total_reward += reward_a
                
                frame_count += 1
                
                # Log scoring
                if abs(reward_a) >= 1.0:
                    scorer = "🔵 LEARNING AI" if reward_a > 0 else "🔴 EXPERT"
                    print(f"  ⚡ {scorer} scores! | Score: {state['score_a']}-{state['score_b']} | Frame: {frame_count}")
                    
                    # Show physics state if spin enabled
                    if enable_spin and 'raw' in state:
                        spin = state['raw'].get('ball_spin', 0)
                        if abs(spin) > 5:
                            print(f"     💫 Ball spin: {spin:+.1f} (Expert can't see this!)")
                
                # Check win condition
                if state['score_a'] >= max_points or state['score_b'] >= max_points:
                    break
            
            # Game ended - final learning update
            if prev_state is not None and prev_action is not None:
                final_reward = 10 if state['score_a'] > state['score_b'] else -10
                learning_ai.remember(prev_state, prev_action, final_reward, state, True)
            
            # Train on experiences
            if len(learning_ai.memory) >= learning_ai.batch_size:
                loss = learning_ai.train_on_batch()
            
            # Update game stats
            learning_ai.end_game(state['score_a'], state['score_b'])
            
            # Print game results
            winner = "🔵 LEARNING AI" if state['score_a'] > state['score_b'] else "🔴 EXPERT"
            print(f"\n  🏆 WINNER: {winner}")
            print(f"  📊 Final Score: {state['score_a']}-{state['score_b']}")
            print(f"  ⏱️  Frames: {frame_count}")
            
            # Show progress stats
            stats = learning_ai.get_stats()
            print(f"\n  📈 Learning AI Progress:")
            print(f"     Win Rate: {stats['win_rate']*100:.1f}% ({stats['wins']}W-{stats['losses']}L vs Expert)")
            print(f"     Exploration: {stats['exploration']} (randomness)")
            print(f"     Total Games: {stats['games_played']}")
            
            # Save periodically
            if game_num % save_every == 0:
                os.makedirs('models', exist_ok=True)
                learning_ai.save(model_path)
                print(f"\n  💾 Model saved! ({model_path})")
            
            # Brief pause between games
            if game_num < games:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
    
    # Final save and stats
    os.makedirs('models', exist_ok=True)
    learning_ai.save(model_path)
    
    print("\n" + "=" * 75)
    print("🎓 TRAINING SESSION COMPLETE!")
    print("=" * 75)
    
    stats = learning_ai.get_stats()
    print(f"\n📊 Final Learning AI Statistics (vs Physics Expert):")
    print(f"   Total Games: {stats['games_played']}")
    print(f"   Learning AI Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"   Record: {stats['wins']}W - {stats['losses']}L - {stats['ties']}T")
    print(f"   Final Exploration: {stats['exploration']}")
    print(f"\n💾 Model saved to: {model_path}")
    print("=" * 75)
    
    engine.close()


def fast_training(games=1000, max_points=5, save_every=100):
    """
    Fast headless training - no graphics, much faster!
    """
    print("=" * 75)
    print("⚡ FAST TRAINING MODE (HEADLESS) 🚀")
    print("=" * 75)
    print(f"\nTraining {games} games with no graphics (much faster!)")
    print("Progress will be shown every 100 games\n")
    
    # Get AI configurations  
    learning_config, expert_config, use_curriculum = get_training_configs(use_curriculum=True)
    
    # Headless engine
    engine = PongEngine(visible=False, ball_speed=2.0, enable_spin=True, enable_curve=True)
    
    # Create AIs with configurations
    learning_ai = LearningAI(
        side='A', 
        learning_rate=0.001,
        paddle_config=learning_config,
        use_curriculum=use_curriculum
    )
    expert_ai = PhysicsExpertAI(
        side='B', 
        control_config=expert_config
    )
    
    model_path = 'models/pong_learning_ai.pkl'
    if os.path.exists(model_path):
        if learning_ai.load(model_path):
            stats = learning_ai.get_stats()
            print(f"✅ Loaded existing model ({stats['games_played']} games, {stats['win_rate']*100:.1f}% Learning AI win rate)")
        else:
            print(f"🆕 Starting fresh - existing model incompatible with current code")
    else:
        print(f"🆕 Starting fresh - new neural network created")
    
    # Display configurations
    if use_curriculum:
        phase = learning_ai.current_phase
        if phase:
            print(f"🔧 Learning AI: CURRICULUM - {phase.name}")
            print(f"   interval={phase.decision_interval}f, vel={phase.max_velocity:.0f}, PID=({phase.kp:.2f}/{phase.kd:.2f})")
    elif learning_config:
        print(f"🔧 Learning AI: FIXED CONFIG")
        print(f"   interval={learning_config.decision_interval}f, "
              f"vel={learning_config.max_velocity:.0f}, PID=({learning_config.kp:.2f},{learning_config.kd:.2f})")
    print(f"🔧 Expert AI: speed={expert_config.paddle_speed_multiplier:.0%}, "
          f"reaction={expert_config.reaction_frames}f\n")
    
    start_time = time.time()
    
    try:
        for game_num in range(1, games + 1):
            state = engine.reset()
            frame_count = 0
            max_frames = 5000
            
            while frame_count < max_frames:
                action_learning = learning_ai.decide_action(state, training=True)
                action_expert = expert_ai.decide_action(state)
                
                prev_state = state
                state, reward_a, reward_b, done = engine.step(action_learning, action_expert)
                
                learning_ai.remember(prev_state, action_learning, reward_a, state, done)
                frame_count += 1
                
                if state['score_a'] >= max_points or state['score_b'] >= max_points:
                    break
            
            # Train
            if len(learning_ai.memory) >= learning_ai.batch_size:
                learning_ai.train_on_batch()
            
            learning_ai.end_game(state['score_a'], state['score_b'])
            
            # Progress report
            if game_num % 100 == 0:
                stats = learning_ai.get_stats()
                elapsed = time.time() - start_time
                games_per_sec = game_num / elapsed
                print(f"Game {game_num}/{games} | Learning AI Win Rate: {stats['win_rate']*100:.1f}% | "
                      f"Explore: {stats['exploration']} | Speed: {games_per_sec:.1f} games/sec")
            
            # Save periodically
            if game_num % save_every == 0:
                os.makedirs('models', exist_ok=True)
                learning_ai.save(model_path)
    
    except KeyboardInterrupt:
        print("\n⏸️  Training stopped")
    
    # Final save
    os.makedirs('models', exist_ok=True)
    learning_ai.save(model_path)
    
    stats = learning_ai.get_stats()
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 75)
    print("⚡ FAST TRAINING COMPLETE!")
    print("=" * 75)
    print(f"Games: {stats['games_played']} | Learning AI Win Rate: {stats['win_rate']*100:.1f}% (vs Expert)")
    print(f"Record: {stats['wins']}W-{stats['losses']}L-{stats['ties']}T")
    print(f"Total Time: {elapsed:.1f}s | Speed: {stats['games_played']/elapsed:.1f} games/sec")
    print("=" * 75)


if __name__ == "__main__":
    import sys
    
    print("\n🎮 Pong AI Training System\n")
    print("Choose training mode:")
    print("  1. Visual Training (watch it learn) - slower but fun to watch")
    print("  2. Fast Training (headless) - much faster, no graphics")
    print()
    
    choice = input("Enter choice (1-2) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        num_games = input("\nNumber of games [default: 50]: ").strip()
        num_games = int(num_games) if num_games else 50
        visual_training(games=num_games, max_points=5, save_every=10)
    elif choice == "2":
        num_games = input("\nNumber of games [default: 1000]: ").strip()
        num_games = int(num_games) if num_games else 1000
        fast_training(games=num_games, max_points=5, save_every=100)
    else:
        print("Invalid choice.")
