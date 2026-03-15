"""
Pong AI Game Launcher
Main entry point for playing and training Pong AI
"""

import sys
from pong_engine import PongEngine, Action
from pong_expert_ai import create_expert_ai
from pong_learning_ai import LearningAI
import time


class PongGame:
    """Main game controller for different play modes"""
    
    def __init__(self):
        self.engine = None
        self.player_a = None
        self.player_b = None
        
    def watch_ai_learn(self, games=100, save_model=True):
        """
        Watch the learning AI train against expert AI
        Visual mode - you can see it learning!
        
        Args:
            games: Number of games to play
            save_model: Whether to save the trained model
        """
        print("=" * 60)
        print("🎮 PONG AI TRAINING - VISUAL MODE 🤖")
        print("=" * 60)
        print(f"\nTraining Learning AI vs Expert AI for {games} games")
        print("Watch as the AI learns from experience!\n")
        
        # Create game engine (visible)
        self.engine = PongEngine(visible=True, ball_speed=2.0)
        
        # Create AIs
        self.player_a = LearningAI(side='A', learning_rate=0.001)
        self.player_b = create_expert_ai(difficulty='medium', side='B')
        
        # Try to load existing model
        if self.player_a.load('models/learning_ai.pkl'):
            print("✓ Loaded existing model - continuing training\n")
        else:
            print("✓ Starting fresh - new AI brain created\n")
        
        try:
            for game_num in range(1, games + 1):
                self._play_training_game(game_num, games)
                
                # Print stats every 10 games
                if game_num % 10 == 0:
                    stats = self.player_a.get_stats()
                    print(f"\n📊 Stats after {game_num} games:")
                    print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
                    print(f"   Exploration: {stats['exploration']}")
                    print(f"   Record: {stats['wins']}W - {stats['losses']}L\n")
                
                # Save model periodically
                if save_model and game_num % 50 == 0:
                    self.player_a.save('models/learning_ai.pkl')
                    print(f"💾 Model saved (Game {game_num})")
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Training interrupted by user")
        
        # Final save
        if save_model:
            self.player_a.save('models/learning_ai.pkl')
            print("\n💾 Final model saved!")
        
        # Final stats
        stats = self.player_a.get_stats()
        print("\n" + "=" * 60)
        print("🏆 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total Games: {stats['games_played']}")
        print(f"Final Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"Final Record: {stats['wins']}W - {stats['losses']}L")
        print(f"Exploration Rate: {stats['exploration']}")
        print("=" * 60)
        
        self.engine.close()
    
    def fast_train(self, games=1000, save_interval=100):
        """
        Fast training mode (no graphics) - trains much faster!
        
        Args:
            games: Number of games to play
            save_interval: Save model every N games
        """
        print("=" * 60)
        print("⚡ PONG AI FAST TRAINING MODE 🚀")
        print("=" * 60)
        print(f"\nTraining for {games} games (headless mode)")
        print("This will be much faster than visual training!\n")
        
        # Create headless engine
        self.engine = PongEngine(visible=False, ball_speed=2.0)
        
        # Create AIs
        self.player_a = LearningAI(side='A', learning_rate=0.001)
        self.player_b = create_expert_ai(difficulty='medium', side='B')
        
        # Load existing model if available
        if self.player_a.load('models/learning_ai.pkl'):
            print("✓ Loaded existing model - continuing training\n")
        else:
            print("✓ Starting fresh - new AI brain created\n")
        
        start_time = time.time()
        
        try:
            for game_num in range(1, games + 1):
                self._play_training_game(game_num, games, show_progress=True)
                
                # Save periodically
                if game_num % save_interval == 0:
                    self.player_a.save('models/learning_ai.pkl')
                    elapsed = time.time() - start_time
                    games_per_sec = game_num / elapsed
                    
                    stats = self.player_a.get_stats()
                    print(f"\n📊 Progress: {game_num}/{games} games")
                    print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
                    print(f"   Speed: {games_per_sec:.1f} games/sec")
                    print(f"   Exploration: {stats['exploration']}\n")
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Training interrupted")
        
        # Save final model
        self.player_a.save('models/learning_ai.pkl')
        
        # Final stats
        elapsed = time.time() - start_time
        stats = self.player_a.get_stats()
        
        print("\n" + "=" * 60)
        print("🏆 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total Games: {stats['games_played']}")
        print(f"Training Time: {elapsed:.1f} seconds")
        print(f"Final Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"Final Record: {stats['wins']}W - {stats['losses']}L")
        print("=" * 60)
    
    def demo_ai_battle(self, games=10):
        """
        Watch two expert AIs battle each other
        Good for testing and fun to watch!
        """
        print("=" * 60)
        print("⚔️  EXPERT AI BATTLE! ⚔️")
        print("=" * 60)
        print(f"\nWatching {games} games between expert AIs\n")
        
        self.engine = PongEngine(visible=True, ball_speed=2.0)
        
        self.player_a = create_expert_ai(difficulty='hard', side='A')
        self.player_b = create_expert_ai(difficulty='hard', side='B')
        
        wins_a = 0
        wins_b = 0
        
        for game_num in range(1, games + 1):
            print(f"Game {game_num}/{games}...", end=' ')
            
            state = self.engine.reset()
            max_steps = 2000
            
            for step in range(max_steps):
                action_a = self.player_a.decide_action(state)
                action_b = self.player_b.decide_action(state)
                
                state, _, _, done = self.engine.step(action_a, action_b)
                
                # End game if someone reaches 5 points
                if state['score_a'] >= 5 or state['score_b'] >= 5:
                    break
            
            if state['score_a'] > state['score_b']:
                wins_a += 1
                print(f"AI A wins! ({state['score_a']}-{state['score_b']})")
            else:
                wins_b += 1
                print(f"AI B wins! ({state['score_b']}-{state['score_a']})")
        
        print(f"\n🏆 Final Score: AI A {wins_a} - {wins_b} AI B")
        self.engine.close()
    
    def _play_training_game(self, game_num, total_games, show_progress=False):
        """Play one training game"""
        state = self.engine.reset()
        
        max_steps = 2000  # Prevent infinite games
        last_state = None
        last_action = None
        
        for step in range(max_steps):
            # Get actions from both AIs
            action_a = self.player_a.decide_action(state, training=True)
            action_b = self.player_b.decide_action(state)
            
            # Execute actions
            next_state, reward_a, reward_b, done = self.engine.step(action_a, action_b)
            
            # Learning AI remembers this experience
            if last_state is not None:
                self.player_a.remember(last_state, last_action, reward_a, state, False)
            
            # Train on a batch of past experiences
            if len(self.player_a.memory) > 32 and step % 4 == 0:
                self.player_a.train_on_batch()
            
            last_state = state
            last_action = action_a
            state = next_state
            
            # End game if someone reaches 5 points
            if state['score_a'] >= 5 or state['score_b'] >= 5:
                # Remember final state
                self.player_a.remember(last_state, last_action, reward_a, state, True)
                break
        
        # Game over
        self.player_a.end_game(state['score_a'], state['score_b'])
        
        if show_progress and game_num % 10 == 0:
            print(f"Game {game_num}/{total_games} - Score: {state['score_a']}-{state['score_b']}")


def print_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("🏓 PONG AI PROJECT 🤖")
    print("=" * 60)
    print("\nChoose a mode:")
    print()
    print("  1. 👁️  WATCH AI LEARN (Visual Training)")
    print("      - Watch the neural network learn in real-time")
    print("      - Slower but satisfying to watch!")
    print()
    print("  2. ⚡ FAST TRAIN (Headless Mode)")
    print("      - Train thousands of games quickly")
    print("      - No graphics, just pure learning")
    print()
    print("  3. ⚔️  EXPERT AI BATTLE")
    print("      - Watch two rule-based AIs compete")
    print("      - Great for testing")
    print()
    print("  4. 🧪 TEST TRAINED MODEL")
    print("      - Watch your trained AI vs Expert")
    print("      - See how much it learned!")
    print()
    print("  5. ❌ EXIT")
    print()
    print("=" * 60)


def main():
    """Main entry point"""
    game = PongGame()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Watch AI learn
            try:
                num_games = int(input("\nHow many games to train? (default 100): ") or "100")
            except ValueError:
                num_games = 100
            game.watch_ai_learn(games=num_games)
        
        elif choice == '2':
            # Fast train
            try:
                num_games = int(input("\nHow many games to train? (default 1000): ") or "1000")
            except ValueError:
                num_games = 1000
            game.fast_train(games=num_games)
        
        elif choice == '3':
            # Expert battle
            try:
                num_games = int(input("\nHow many games? (default 10): ") or "10")
            except ValueError:
                num_games = 10
            game.demo_ai_battle(games=num_games)
        
        elif choice == '4':
            # Test trained model
            print("\n🧪 Testing trained model...")
            game.watch_ai_learn(games=10, save_model=False)
        
        elif choice == '5':
            print("\n👋 Thanks for playing! Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
