"""
AI Evaluation and Testing Script
Test trained models and visualize their performance

Features:
- Load any trained model (Phase 1, 2, or 3)
- Play visual games to watch AI in action
- Run performance benchmarks
- Compare across different physics settings
"""

import os
import sys
import time
from pong_engine import PongEngine
from pong_expert_ai import PhysicsExpertAI
from pong_learning_ai import LearningAI


class AIEvaluator:
    """Evaluate and test trained AI models"""
    
    def __init__(self):
        self.models_dir = 'models/progressive'
        self.available_models = self._find_models()
    
    def _find_models(self):
        """Find all available trained models"""
        models = {}
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl') and 'phase' in filename:
                filepath = os.path.join(self.models_dir, filename)
                models[filename] = filepath
        
        return models
    
    def list_models(self):
        """Display all available models"""
        if not self.available_models:
            print("No trained models found in models/progressive/")
            print("\nTrain a model first using:")
            print("  python train_quick.py 500")
            print("  python train_progressive.py")
            return
        
        print("\nAvailable Models:")
        print("="*80)
        for i, (name, path) in enumerate(self.available_models.items(), 1):
            # Load to get stats
            ai = LearningAI(side='A')
            if ai.load(path):
                stats = ai.get_stats()
                print(f"{i}. {name:30s} - Games: {stats['games_played']:5d}, "
                      f"Win Rate: {stats['win_rate']*100:5.1f}%, "
                      f"Exploration: {stats['exploration']}")
        print("="*80)
    
    def play_visual_game(self, model_path, phase_config):
        """
        Play a visual game with the trained AI
        
        Args:
            model_path: Path to trained model
            phase_config: Dict with 'spin', 'curve', 'progressive' settings
        """
        # Load AI
        ai = LearningAI(side='A')
        if not ai.load(model_path):
            print(f"Failed to load model: {model_path}")
            return
        
        stats = ai.get_stats()
        print("\n" + "="*80)
        print(f"LOADED MODEL: {os.path.basename(model_path)}")
        print("="*80)
        print(f"Training History: {stats['games_played']} games")
        print(f"Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"Current Exploration: {stats['exploration']}")
        print("="*80)
        
        # Setup game configuration
        print("\nGame Configuration:")
        print(f"  Spin Physics: {'ENABLED' if phase_config['spin'] else 'DISABLED'}")
        print(f"  Curve Physics: {'ENABLED' if phase_config['curve'] else 'DISABLED'}")
        print(f"  Progressive Speed: {'ENABLED' if phase_config['progressive'] else 'DISABLED'}")
        print("="*80 + "\n")
        
        print("Starting game in 2 seconds...")
        time.sleep(2)
        
        # Create game
        engine = PongEngine(
            visible=True,
            enable_spin=phase_config['spin'],
            enable_curve=phase_config['curve'],
            ball_speed=1.0,
            paddle_speed=25,
            frame_rate=60,  # Slower for watching
            difficulty_level=1,
            auto_progress=phase_config['progressive'],
            points_per_level=5
        )
        
        expert = PhysicsExpertAI(side='B')
        
        print("Game started! Watch the AI play...")
        print("Left paddle (white) = Learning AI")
        print("Right paddle (white) = Expert AI")
        print("\nClose window to exit\n")
        
        # Play game
        try:
            while not engine.game_over:
                state = engine.get_state()
                ai_action = ai.decide_action(state, training=False)  # Full exploitation
                expert_action = expert.decide_action(state)
                engine.step(ai_action, expert_action)
            
            # Game ended
            print("\n" + "="*80)
            print("GAME OVER")
            print("="*80)
            print(f"Learning AI Score: {engine.score_a}")
            print(f"Expert AI Score: {engine.score_b}")
            
            if engine.score_a > engine.score_b:
                print("\n🎉 LEARNING AI WINS! 🎉")
                margin = engine.score_a - engine.score_b
                print(f"   Victory by {margin} point{'s' if margin != 1 else ''}!")
            elif engine.score_b > engine.score_a:
                print("\n❌ Expert AI wins")
                margin = engine.score_b - engine.score_a
                print(f"   AI lost by {margin} point{'s' if margin != 1 else ''}")
            else:
                print("\n🤝 TIE GAME!")
            
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGame interrupted by user")
    
    def benchmark_model(self, model_path, phase_config, num_games=100):
        """
        Run benchmark testing on a model (headless)
        
        Args:
            model_path: Path to trained model
            phase_config: Physics configuration
            num_games: Number of games to test
        """
        print("\n" + "="*80)
        print(f"BENCHMARK: {os.path.basename(model_path)}")
        print("="*80)
        
        # Load AI
        ai = LearningAI(side='A')
        if not ai.load(model_path):
            print(f"Failed to load model: {model_path}")
            return
        
        print(f"Configuration: Spin={phase_config['spin']}, "
              f"Curve={phase_config['curve']}, "
              f"Progressive={phase_config['progressive']}")
        print(f"Running {num_games} games in headless mode...\n")
        
        expert = PhysicsExpertAI(side='B')
        
        wins = 0
        losses = 0
        ties = 0
        total_ai_score = 0
        total_expert_score = 0
        
        start_time = time.time()
        
        for game_num in range(num_games):
            # Create game (headless)
            engine = PongEngine(
                visible=False,
                enable_spin=phase_config['spin'],
                enable_curve=phase_config['curve'],
                ball_speed=1.0,
                paddle_speed=25,
                frame_rate=200,
                difficulty_level=1,
                auto_progress=phase_config['progressive'],
                points_per_level=5
            )
            
            # Play game
            while not engine.game_over:
                state = engine.get_state()
                ai_action = ai.decide_action(state, training=False)
                expert_action = expert.decide_action(state)
                engine.step(ai_action, expert_action)
            
            # Record results
            total_ai_score += engine.score_a
            total_expert_score += engine.score_b
            
            if engine.score_a > engine.score_b:
                wins += 1
            elif engine.score_b > engine.score_a:
                losses += 1
            else:
                ties += 1
            
            # Progress indicator
            if (game_num + 1) % 10 == 0:
                current_win_rate = (wins / (game_num + 1)) * 100
                print(f"Progress: {game_num + 1}/{num_games} games "
                      f"(Win Rate: {current_win_rate:.1f}%)", end='\r', flush=True)
        
        elapsed = time.time() - start_time
        
        # Print results
        print("\n\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"Total Games: {num_games}")
        print(f"Time Elapsed: {elapsed:.1f} seconds ({elapsed/num_games:.2f}s per game)")
        print(f"\nResults:")
        print(f"  Wins:   {wins:4d} ({wins/num_games*100:5.1f}%)")
        print(f"  Losses: {losses:4d} ({losses/num_games*100:5.1f}%)")
        print(f"  Ties:   {ties:4d} ({ties/num_games*100:5.1f}%)")
        print(f"\nAverage Scores:")
        print(f"  AI:     {total_ai_score/num_games:.2f} points/game")
        print(f"  Expert: {total_expert_score/num_games:.2f} points/game")
        print(f"  Margin: {(total_ai_score - total_expert_score)/num_games:+.2f} points/game")
        print("="*80 + "\n")


def main():
    """Main evaluation entry point"""
    evaluator = AIEvaluator()
    
    print("\n" + "="*80)
    print("PONG AI EVALUATION & TESTING")
    print("="*80)
    
    # List available models
    evaluator.list_models()
    
    if not evaluator.available_models:
        return
    
    print("\nOptions:")
    print("  1. Play visual game (watch AI)")
    print("  2. Run benchmark test (100 games, headless)")
    print("  3. Quick test - Phase 1 model (basic physics)")
    print("  4. Quick test - Phase 2 model (with spin)")
    print("  5. Quick test - Phase 3 model (progressive)")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == '1':
        # Play visual game
        print("\nSelect model:")
        models_list = list(evaluator.available_models.items())
        for i, (name, _) in enumerate(models_list, 1):
            print(f"  {i}. {name}")
        
        model_idx = int(input("\nModel number: ").strip()) - 1
        if 0 <= model_idx < len(models_list):
            _, model_path = models_list[model_idx]
            
            # Determine phase config based on model name
            if 'phase1' in models_list[model_idx][0]:
                config = {'spin': False, 'curve': False, 'progressive': False}
            elif 'phase2' in models_list[model_idx][0]:
                config = {'spin': True, 'curve': True, 'progressive': False}
            elif 'phase3' in models_list[model_idx][0]:
                config = {'spin': True, 'curve': True, 'progressive': True}
            else:
                config = {'spin': False, 'curve': False, 'progressive': False}
            
            evaluator.play_visual_game(model_path, config)
    
    elif choice == '2':
        # Benchmark
        print("\nSelect model:")
        models_list = list(evaluator.available_models.items())
        for i, (name, _) in enumerate(models_list, 1):
            print(f"  {i}. {name}")
        
        model_idx = int(input("\nModel number: ").strip()) - 1
        num_games = int(input("Number of games (default 100): ").strip() or "100")
        
        if 0 <= model_idx < len(models_list):
            _, model_path = models_list[model_idx]
            
            # Determine config
            if 'phase1' in models_list[model_idx][0]:
                config = {'spin': False, 'curve': False, 'progressive': False}
            elif 'phase2' in models_list[model_idx][0]:
                config = {'spin': True, 'curve': True, 'progressive': False}
            elif 'phase3' in models_list[model_idx][0]:
                config = {'spin': True, 'curve': True, 'progressive': True}
            else:
                config = {'spin': False, 'curve': False, 'progressive': False}
            
            evaluator.benchmark_model(model_path, config, num_games)
    
    elif choice == '3':
        # Quick test Phase 1
        model_path = os.path.join(evaluator.models_dir, 'phase1_final.pkl')
        if os.path.exists(model_path):
            config = {'spin': False, 'curve': False, 'progressive': False}
            evaluator.play_visual_game(model_path, config)
        else:
            print("Phase 1 model not found. Train it first with: python train_quick.py")
    
    elif choice == '4':
        # Quick test Phase 2
        model_path = os.path.join(evaluator.models_dir, 'phase2_final.pkl')
        if os.path.exists(model_path):
            config = {'spin': True, 'curve': True, 'progressive': False}
            evaluator.play_visual_game(model_path, config)
        else:
            print("Phase 2 model not found. Train it first with: python train_progressive.py")
    
    elif choice == '5':
        # Quick test Phase 3
        model_path = os.path.join(evaluator.models_dir, 'phase3_final.pkl')
        if os.path.exists(model_path):
            config = {'spin': True, 'curve': True, 'progressive': True}
            evaluator.play_visual_game(model_path, config)
        else:
            print("Phase 3 model not found. Train it first with: python train_progressive.py")
    
    else:
        print("Invalid option")


if __name__ == '__main__':
    main()
