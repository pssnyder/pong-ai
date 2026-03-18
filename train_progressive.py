"""
Progressive Training Script for Pong Learning AI
Three-phase training curriculum to gradually introduce complexity

Phase 1: Basic Physics
- No spin, no curve physics
- Steady ball speed
- 100% paddle speed
- Goal: Learn basic positioning and trajectory prediction

Phase 2: Spin Physics  
- Enable spin and curve mechanics
- Steady ball speed
- 100% paddle speed
- Goal: Learn to handle and exploit spin/curve

Phase 3: Progressive Speed
- Spin and curve enabled
- Ball speed increases with each paddle strike
- 100% paddle speed
- Goal: Handle increasing difficulty and adapt to speed changes

Future: Can reduce paddle speed to 80% for additional challenge
"""

import os
import time
import json
from datetime import datetime
from pong_engine import PongEngine
from pong_expert_ai import PhysicsExpertAI
from pong_learning_ai import LearningAI


class ProgressiveTrainer:
    """
    Manages progressive training across multiple phases
    Tracks progress, saves checkpoints, and provides detailed statistics
    """
    
    def __init__(self, models_dir='models/progressive'):
        """
        Initialize progressive trainer
        
        Args:
            models_dir: Directory to save model checkpoints
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'phase_1': {'games': 0, 'wins': 0, 'losses': 0, 'avg_score': 0, 'completed': False},
            'phase_2': {'games': 0, 'wins': 0, 'losses': 0, 'avg_score': 0, 'completed': False},
            'phase_3': {'games': 0, 'wins': 0, 'losses': 0, 'avg_score': 0, 'completed': False}
        }
        
        # Load history if exists
        self.history_file = os.path.join(models_dir, 'training_history.json')
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
    
    def save_history(self):
        """Save training history to disk"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_phase_header(self, phase_name, description, config):
        """Print formatted phase header"""
        print("\n" + "="*80)
        print(f"PHASE: {phase_name}")
        print("="*80)
        print(f"Description: {description}")
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key:20s}: {value}")
        print("="*80 + "\n")
    
    def print_progress(self, phase, games_completed, target_games, ai_stats, 
                      recent_scores, time_elapsed):
        """Print training progress"""
        win_rate = ai_stats['win_rate'] * 100
        progress_pct = (games_completed / target_games) * 100
        
        # Calculate recent performance (last 10 games)
        recent_win_rate = 0
        if len(recent_scores) >= 10:
            recent_wins = sum(1 for ai_score, exp_score in recent_scores[-10:] if ai_score > exp_score)
            recent_win_rate = (recent_wins / 10) * 100
        
        print(f"\r[{phase}] Progress: {games_completed}/{target_games} ({progress_pct:.1f}%) | "
              f"Win Rate: {win_rate:.1f}% (Recent: {recent_win_rate:.0f}%) | "
              f"Epsilon: {ai_stats['exploration']} | "
              f"Time: {time_elapsed:.0f}s", end='', flush=True)
    
    def train_phase_1(self, games=500, save_interval=100, paddle_speed_pct=100):
        """
        Phase 1: Basic Physics Training
        No spin, no curve - learn fundamental positioning and prediction
        
        Args:
            games: Number of games to train
            save_interval: Save checkpoint every N games
            paddle_speed_pct: Paddle speed percentage (100 = full speed)
        """
        phase_name = "PHASE 1: Basic Physics"
        description = "Learning fundamental positioning and trajectory prediction"
        config = {
            'Spin Physics': 'DISABLED',
            'Curve Physics': 'DISABLED',
            'Ball Speed': 'STEADY (1.0x)',
            'Paddle Speed': f'{paddle_speed_pct}% (Expert and AI)',
            'Training Games': games,
            'Strategy': 'Master straight-line physics before complexity'
        }
        
        self.print_phase_header(phase_name, description, config)
        
        # Initialize AI
        ai = LearningAI(side='A', learning_rate=0.001, discount_factor=0.95)
        
        # Try to load previous checkpoint
        phase1_model = os.path.join(self.models_dir, 'phase1_latest.pkl')
        if os.path.exists(phase1_model):
            print(f"Loading existing Phase 1 model from {phase1_model}...")
            ai.load(phase1_model)
            print(f"Resumed training from game {ai.games_played}")
        
        # Initialize expert opponent
        expert = PhysicsExpertAI(side='B', paddle_speed_multiplier=paddle_speed_pct/100)
        
        # Track progress
        start_game = ai.games_played
        target_game = start_game + games
        recent_scores = []
        start_time = time.time()
        
        print(f"\nStarting training from game {start_game} to {target_game}...")
        print("Press Ctrl+C to stop training and save progress")
        print("(First 5 games will show detailed output, then updates every 10 games)\n")
        
        try:
            game_num = 0
            while ai.games_played < target_game:
                game_num += 1
                
                if game_num == 1:
                    print(f"Starting game loop... (target: {target_game} games)\n")
                
                # Create game engine - NO spin, NO curve, steady speed
                engine = PongEngine(
                    visible=False,
                    enable_spin=False,
                    enable_curve=False,
                    ball_speed=1.0,
                    paddle_speed=25 * (paddle_speed_pct/100),
                    frame_rate=200
                )
                
                # Play one game
                state = engine.get_state()
                prev_state = state
                frame_count = 0
                max_frames = 10000  # Prevent infinite games
                
                while not engine.game_over and frame_count < max_frames:
                    frame_count += 1
                    # AI decides action
                    ai_action = ai.decide_action(state, training=True)
                    expert_action = expert.decide_action(state)
                    
                    # Step game forward
                    engine.step(ai_action, expert_action)
                    new_state = engine.get_state()
                    
                    # Calculate reward
                    reward = 0
                    if engine.score_a > prev_state['score_a']:  # AI scored
                        reward = 1.0
                    elif engine.score_b > prev_state['score_b']:  # Expert scored
                        reward = -1.0
                    else:
                        # Small reward for tracking ball
                        ball_y = new_state['ball_y']
                        paddle_y = new_state['paddle_a_y']
                        distance = abs(ball_y - paddle_y)
                        reward = -distance * 0.001  # Penalize being far from ball
                    
                    # Remember experience
                    ai.remember(prev_state, ai_action, reward, new_state, engine.game_over)
                    
                    # Train on batch
                    if len(ai.memory) >= ai.batch_size:
                        ai.train_on_batch()
                    
                    prev_state = new_state
                    state = new_state
                
                # Game ended
                ai.end_game(engine.score_a, engine.score_b)
                recent_scores.append((engine.score_a, engine.score_b))
                
                # Print game result
                if game_num <= 5 or game_num % 10 == 0:  # First 5 games and every 10th
                    stats = ai.get_stats()
                    result = "WIN" if engine.score_a > engine.score_b else ("LOSS" if engine.score_b > engine.score_a else "TIE")
                    print(f"\nGame {ai.games_played}: AI {engine.score_a}-{engine.score_b} Expert [{result}] | "
                          f"Overall: W:{stats['wins']} L:{stats['losses']} T:{stats['ties']} ({stats['win_rate']*100:.1f}% WR)")
                
                # Update progress (in-place for other games)
                games_completed = ai.games_played - start_game
                time_elapsed = time.time() - start_time
                self.print_progress("Phase 1", games_completed, games, ai.get_stats(),
                                  recent_scores, time_elapsed)
                
                # Save checkpoint
                if ai.games_played % save_interval == 0:
                    ai.save(phase1_model)
                    checkpoint_file = os.path.join(self.models_dir, f'phase1_game{ai.games_played}.pkl')
                    ai.save(checkpoint_file)
            
            # Phase 1 complete
            print("\n\n" + "="*80)
            print("PHASE 1 COMPLETE!")
            print("="*80)
            
            # Save final model
            final_model = os.path.join(self.models_dir, 'phase1_final.pkl')
            ai.save(final_model)
            
            # Update history
            stats = ai.get_stats()
            self.history['phase_1'] = {
                'games': ai.games_played,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate'],
                'completed': True,
                'completed_at': datetime.now().isoformat()
            }
            self.save_history()
            
            # Print final stats
            print(f"\nFinal Statistics:")
            print(f"  Total Games: {ai.games_played}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Final Epsilon: {stats['epsilon']:.3f}")
            print(f"\nModel saved to: {final_model}")
            print("\nReady for Phase 2: Spin Physics Training")
            print("="*80 + "\n")
            
            return ai
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            ai.save(phase1_model)
            print(f"Progress saved to {phase1_model}")
            print(f"Games completed: {ai.games_played - start_game}/{games}")
            return ai
    
    def train_phase_2(self, games=500, save_interval=100, paddle_speed_pct=100):
        """
        Phase 2: Spin Physics Training
        Build on Phase 1 by adding spin and curve mechanics
        
        Args:
            games: Number of games to train
            save_interval: Save checkpoint every N games
            paddle_speed_pct: Paddle speed percentage
        """
        phase_name = "PHASE 2: Spin Physics"
        description = "Learning to handle and exploit spin & curve mechanics"
        config = {
            'Spin Physics': 'ENABLED',
            'Curve Physics': 'ENABLED',
            'Ball Speed': 'STEADY (1.0x)',
            'Paddle Speed': f'{paddle_speed_pct}% (Expert and AI)',
            'Training Games': games,
            'Strategy': 'Discover spin creates opportunities to beat expert'
        }
        
        self.print_phase_header(phase_name, description, config)
        
        # Load Phase 1 model
        phase1_model = os.path.join(self.models_dir, 'phase1_final.pkl')
        if not os.path.exists(phase1_model):
            print(f"ERROR: Phase 1 model not found at {phase1_model}")
            print("Please complete Phase 1 training first!")
            return None
        
        # Initialize AI and load Phase 1 knowledge
        ai = LearningAI(side='A', learning_rate=0.001, discount_factor=0.95)
        ai.load(phase1_model)
        print(f"Loaded Phase 1 model (trained on {ai.games_played} games)")
        
        # Reset epsilon for new learning phase (but not too high)
        ai.epsilon = 0.3  # Some exploration to discover spin mechanics
        print(f"Reset exploration rate to {ai.epsilon} for spin discovery\n")
        
        # Initialize expert opponent
        expert = PhysicsExpertAI(side='B', paddle_speed_multiplier=paddle_speed_pct/100)
        
        # Track progress
        start_game = ai.games_played
        target_game = start_game + games
        recent_scores = []
        start_time = time.time()
        
        print(f"Starting Phase 2 training from game {start_game} to {target_game}...")
        print("Press Ctrl+C to stop training and save progress\n")
        
        try:
            while ai.games_played < target_game:
                # Create game engine - WITH spin and curve, steady speed
                engine = PongEngine(
                    visible=False,
                    enable_spin=True,
                    enable_curve=True,
                    ball_speed=1.0,
                    paddle_speed=25 * (paddle_speed_pct/100),
                    frame_rate=200
                )
                
                # Play one game (same structure as Phase 1)
                state = engine.get_state()
                prev_state = state
                
                while not engine.game_over:
                    ai_action = ai.decide_action(state, training=True)
                    expert_action = expert.decide_action(state)
                    
                    engine.step(ai_action, expert_action)
                    new_state = engine.get_state()
                    
                    # Calculate reward (bonus for winning with spin!)
                    reward = 0
                    if engine.score_a > prev_state['score_a']:
                        # Bonus reward if ball had spin (AI learning to use spin)
                        spin_bonus = abs(new_state.get('ball_spin', 0)) * 0.01
                        reward = 1.0 + spin_bonus
                    elif engine.score_b > prev_state['score_b']:
                        reward = -1.0
                    else:
                        ball_y = new_state['ball_y']
                        paddle_y = new_state['paddle_a_y']
                        distance = abs(ball_y - paddle_y)
                        reward = -distance * 0.001
                    
                    ai.remember(prev_state, ai_action, reward, new_state, engine.game_over)
                    
                    if len(ai.memory) >= ai.batch_size:
                        ai.train_on_batch()
                    
                    prev_state = new_state
                    state = new_state
                
                ai.end_game(engine.score_a, engine.score_b)
                recent_scores.append((engine.score_a, engine.score_b))
                
                games_completed = ai.games_played - start_game
                time_elapsed = time.time() - start_time
                self.print_progress("Phase 2", games_completed, games, ai.get_stats(),
                                  recent_scores, time_elapsed)
                
                # Save checkpoint
                if (ai.games_played - start_game) % save_interval == 0:
                    phase2_model = os.path.join(self.models_dir, 'phase2_latest.pkl')
                    ai.save(phase2_model)
            
            # Phase 2 complete
            print("\n\n" + "="*80)
            print("PHASE 2 COMPLETE!")
            print("="*80)
            
            final_model = os.path.join(self.models_dir, 'phase2_final.pkl')
            ai.save(final_model)
            
            stats = ai.get_stats()
            self.history['phase_2'] = {
                'games': ai.games_played,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate'],
                'completed': True,
                'completed_at': datetime.now().isoformat()
            }
            self.save_history()
            
            print(f"\nFinal Statistics:")
            print(f"  Total Games: {ai.games_played}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Phase 2 Games: {games}")
            print(f"\nModel saved to: {final_model}")
            print("\nReady for Phase 3: Progressive Speed Training")
            print("="*80 + "\n")
            
            return ai
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            phase2_model = os.path.join(self.models_dir, 'phase2_latest.pkl')
            ai.save(phase2_model)
            print(f"Progress saved to {phase2_model}")
            return ai
    
    def train_phase_3(self, games=500, save_interval=100, paddle_speed_pct=100, 
                     speed_increment=0.02):
        """
        Phase 3: Progressive Speed Training
        Build on Phase 2 by adding progressive ball speed increases
        
        Args:
            games: Number of games to train
            save_interval: Save checkpoint every N games
            paddle_speed_pct: Paddle speed percentage
            speed_increment: Ball speed increment per paddle strike
        """
        phase_name = "PHASE 3: Progressive Speed"
        description = "Learning to adapt to progressively faster gameplay"
        config = {
            'Spin Physics': 'ENABLED',
            'Curve Physics': 'ENABLED',
            'Ball Speed': f'PROGRESSIVE (starts 1.0x, +{speed_increment} per hit)',
            'Paddle Speed': f'{paddle_speed_pct}% (Expert and AI)',
            'Training Games': games,
            'Strategy': 'Handle escalating difficulty and maintain control'
        }
        
        self.print_phase_header(phase_name, description, config)
        
        # Load Phase 2 model
        phase2_model = os.path.join(self.models_dir, 'phase2_final.pkl')
        if not os.path.exists(phase2_model):
            print(f"ERROR: Phase 2 model not found at {phase2_model}")
            print("Please complete Phase 2 training first!")
            return None
        
        # Initialize AI and load Phase 2 knowledge
        ai = LearningAI(side='A', learning_rate=0.001, discount_factor=0.95)
        ai.load(phase2_model)
        print(f"Loaded Phase 2 model (trained on {ai.games_played} games)")
        
        # Reset epsilon for final learning phase
        ai.epsilon = 0.2  # Less exploration, more exploitation of learned skills
        print(f"Reset exploration rate to {ai.epsilon} for adaptation training\n")
        
        # Initialize expert opponent
        expert = PhysicsExpertAI(side='B', paddle_speed_multiplier=paddle_speed_pct/100)
        
        # Track progress
        start_game = ai.games_played
        target_game = start_game + games
        recent_scores = []
        start_time = time.time()
        
        print(f"Starting Phase 3 training from game {start_game} to {target_game}...")
        print("Press Ctrl+C to stop training and save progress\n")
        
        try:
            while ai.games_played < target_game:
                # Create game engine - WITH progressive difficulty
                # Using auto_progress to increase difficulty every 5 points
                engine = PongEngine(
                    visible=False,
                    enable_spin=True,
                    enable_curve=True,
                    ball_speed=1.0,  # Starts at 1.0
                    paddle_speed=25 * (paddle_speed_pct/100),
                    frame_rate=200,
                    difficulty_level=1,  # Start at level 1
                    auto_progress=True,  # Enable auto difficulty progression
                    points_per_level=5   # Increase difficulty every 5 points
                )
                
                # Play one game
                state = engine.get_state()
                prev_state = state
                
                while not engine.game_over:
                    ai_action = ai.decide_action(state, training=True)
                    expert_action = expert.decide_action(state)
                    
                    engine.step(ai_action, expert_action)
                    new_state = engine.get_state()
                    
                    # Calculate reward
                    reward = 0
                    if engine.score_a > prev_state['score_a']:
                        # Extra reward for scoring at higher difficulty
                        difficulty_bonus = engine.difficulty.level * 0.1
                        spin_bonus = abs(new_state.get('ball_spin', 0)) * 0.01
                        reward = 1.0 + difficulty_bonus + spin_bonus
                    elif engine.score_b > prev_state['score_b']:
                        reward = -1.0
                    else:
                        ball_y = new_state['ball_y']
                        paddle_y = new_state['paddle_a_y']
                        distance = abs(ball_y - paddle_y)
                        reward = -distance * 0.001
                    
                    ai.remember(prev_state, ai_action, reward, new_state, engine.game_over)
                    
                    if len(ai.memory) >= ai.batch_size:
                        ai.train_on_batch()
                    
                    prev_state = new_state
                    state = new_state
                
                ai.end_game(engine.score_a, engine.score_b)
                recent_scores.append((engine.score_a, engine.score_b))
                
                games_completed = ai.games_played - start_game
                time_elapsed = time.time() - start_time
                self.print_progress("Phase 3", games_completed, games, ai.get_stats(),
                                  recent_scores, time_elapsed)
                
                # Save checkpoint
                if (ai.games_played - start_game) % save_interval == 0:
                    phase3_model = os.path.join(self.models_dir, 'phase3_latest.pkl')
                    ai.save(phase3_model)
            
            # Phase 3 complete
            print("\n\n" + "="*80)
            print("PHASE 3 COMPLETE!")
            print("="*80)
            print("ALL TRAINING PHASES FINISHED!")
            print("="*80)
            
            final_model = os.path.join(self.models_dir, 'phase3_final.pkl')
            ai.save(final_model)
            
            stats = ai.get_stats()
            self.history['phase_3'] = {
                'games': ai.games_played,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate'],
                'completed': True,
                'completed_at': datetime.now().isoformat()
            }
            self.save_history()
            
            print(f"\nFinal Statistics:")
            print(f"  Total Games: {ai.games_played}")
            print(f"  Overall Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Phase 3 Games: {games}")
            print(f"\nMaster model saved to: {final_model}")
            print("\nNext Steps:")
            print("  - Test the trained AI against expert at different speeds")
            print("  - Try reducing paddle speed to 80% for harder challenge")
            print("  - Analyze which phases contributed most to success")
            print("="*80 + "\n")
            
            return ai
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            phase3_model = os.path.join(self.models_dir, 'phase3_latest.pkl')
            ai.save(phase3_model)
            print(f"Progress saved to {phase3_model}")
            return ai


def main():
    """Main training entry point"""
    print("\n" + "="*80)
    print("PROGRESSIVE PONG AI TRAINING SYSTEM")
    print("="*80)
    print("\nThis training system uses a progressive curriculum:")
    print("  Phase 1: Master basic physics (no spin)")
    print("  Phase 2: Learn spin and curve mechanics")
    print("  Phase 3: Adapt to progressive speed increases")
    print("\nEach phase builds on the previous one's knowledge.")
    print("="*80 + "\n")
    
    trainer = ProgressiveTrainer()
    
    # Show menu
    print("Training Options:")
    print("  1. Run all three phases sequentially (recommended for first time)")
    print("  2. Train Phase 1 only (basic physics)")
    print("  3. Train Phase 2 only (requires Phase 1 complete)")
    print("  4. Train Phase 3 only (requires Phase 2 complete)")
    print("  5. Custom training session")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == '1':
        # Full training pipeline
        print("\nStarting full training pipeline...")
        print("This will train all three phases with 500 games each (1500 total)")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled.")
            return
        
        print("\nPhase 1: Basic Physics (500 games)")
        ai = trainer.train_phase_1(games=500, paddle_speed_pct=100)
        
        if ai:
            input("\nPhase 1 complete! Press Enter to continue to Phase 2...")
            print("\nPhase 2: Spin Physics (500 games)")
            ai = trainer.train_phase_2(games=500, paddle_speed_pct=100)
        
        if ai:
            input("\nPhase 2 complete! Press Enter to continue to Phase 3...")
            print("\nPhase 3: Progressive Speed (500 games)")
            ai = trainer.train_phase_3(games=500, paddle_speed_pct=100)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
    
    elif choice == '2':
        games = input("Number of games (default 500): ").strip()
        games = int(games) if games else 500
        trainer.train_phase_1(games=games, paddle_speed_pct=100)
    
    elif choice == '3':
        games = input("Number of games (default 500): ").strip()
        games = int(games) if games else 500
        trainer.train_phase_2(games=games, paddle_speed_pct=100)
    
    elif choice == '4':
        games = input("Number of games (default 500): ").strip()
        games = int(games) if games else 500
        trainer.train_phase_3(games=games, paddle_speed_pct=100)
    
    elif choice == '5':
        print("\nCustom Training:")
        phase = input("Phase (1/2/3): ").strip()
        games = int(input("Number of games: ").strip())
        paddle_speed = int(input("Paddle speed % (default 100): ").strip() or "100")
        
        if phase == '1':
            trainer.train_phase_1(games=games, paddle_speed_pct=paddle_speed)
        elif phase == '2':
            trainer.train_phase_2(games=games, paddle_speed_pct=paddle_speed)
        elif phase == '3':
            trainer.train_phase_3(games=games, paddle_speed_pct=paddle_speed)
    
    else:
        print("Invalid option.")


if __name__ == '__main__':
    main()
