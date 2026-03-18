"""
Diagnostic Script - Check Learning AI Model and Test Scoring
Helps debug issues with trained models and game scoring
"""

import os
import pickle
import numpy as np
from pong_learning_ai import LearningAI
from pong_engine import PongEngine, Action
from pong_expert_ai import PhysicsExpertAI


def check_model_file(filepath):
    """Inspect a saved model file"""
    print("\n" + "="*80)
    print(f"INSPECTING MODEL: {os.path.basename(filepath)}")
    print("="*80)
    
    if not os.path.exists(filepath):
        print(f"❌ Model file not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print("\n📦 Model Contents:")
        print(f"  Games Played: {data.get('games_played', 'N/A')}")
        print(f"  Wins:         {data.get('win_count', 'N/A')}")
        print(f"  Losses:       {data.get('loss_count', 'N/A')}")
        print(f"  Ties:         {data.get('tie_count', 'N/A')}")
        print(f"  Epsilon:      {data.get('epsilon', 'N/A'):.4f}")
        
        # Check network weights
        print(f"\n🧠 Neural Network Weights:")
        print(f"  W1 shape: {data['W1'].shape}")
        print(f"  W1 mean:  {np.mean(data['W1']):.6f}")
        print(f"  W1 std:   {np.std(data['W1']):.6f}")
        print(f"  W2 shape: {data['W2'].shape}")
        print(f"  W2 mean:  {np.mean(data['W2']):.6f}")
        print(f"  W2 std:   {np.std(data['W2']):.6f}")
        
        # Check for NaN or Inf
        has_nan_w1 = np.any(np.isnan(data['W1']))
        has_inf_w1 = np.any(np.isinf(data['W1']))
        has_nan_w2 = np.any(np.isnan(data['W2']))
        has_inf_w2 = np.any(np.isinf(data['W2']))
        
        if has_nan_w1 or has_inf_w1 or has_nan_w2 or has_inf_w2:
            print("\n  ⚠️  WARNING: Found NaN or Inf values in weights!")
            print(f"     W1 has NaN: {has_nan_w1}, Inf: {has_inf_w1}")
            print(f"     W2 has NaN: {has_nan_w2}, Inf: {has_inf_w2}")
            print("     Model is CORRUPTED - retrain from scratch!")
            return False
        else:
            print("  ✅ No NaN/Inf values found")
        
        # Calculate win rate
        games = data.get('games_played', 0)
        wins = data.get('win_count', 0)
        if games > 0:
            win_rate = (wins / games) * 100
            print(f"\n📊 Performance:")
            print(f"  Win Rate: {win_rate:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_ai_actions(model_path, num_samples=10):
    """Test what actions the AI produces for different game states"""
    print("\n" + "="*80)
    print("TESTING AI ACTIONS")
    print("="*80)
    
    ai = LearningAI(side='A')
    if not ai.load(model_path):
        print(f"Failed to load model: {model_path}")
        return
    
    print(f"\nGenerating {num_samples} sample game states and checking AI actions...\n")
    
    # Create a test engine
    engine = PongEngine(visible=False, enable_spin=False, enable_curve=False)
    
    action_counts = {'UP': 0, 'DOWN': 0, 'STAY': 0}
    
    for i in range(num_samples):
        state = engine.get_state()
        action = ai.decide_action(state, training=False)  # No randomness
        action_counts[action.name] += 1
        
        if i < 5:  # Show first 5
            print(f"Sample {i+1}:")
            print(f"  Ball position: ({state['ball_x']:.2f}, {state['ball_y']:.2f})")
            print(f"  AI paddle Y:   {state['paddle_a_y']:.2f}")
            print(f"  Action:        {action.name}")
            print()
        
        # Step the engine to change state
        expert = PhysicsExpertAI(side='B')
        expert_action = expert.decide_action(state)
        engine.step(action, expert_action)
    
    print(f"\n📊 Action Distribution over {num_samples} samples:")
    for action, count in action_counts.items():
        pct = (count / num_samples) * 100
        print(f"  {action:5s}: {count:3d} ({pct:5.1f}%)")
    
    if action_counts['STAY'] == num_samples:
        print("\n  ⚠️  AI ALWAYS chooses STAY - model hasn't learned yet!")
        print("     Need more training games (~100-200 minimum)")
    elif action_counts['STAY'] > num_samples * 0.8:
        print("\n  ⚠️  AI mostly chooses STAY - limited learning")
        print("     Try more training games")
    else:
        print("\n  ✅ AI shows varied actions - some learning detected!")


def test_scoring_system():
    """Test that the scoring system works correctly"""
    print("\n" + "="*80)
    print("TESTING SCORING SYSTEM")
    print("="*80)
    
    print("\nCreating test game with winning_score=3 (first to 3 wins)...")
    
    engine = PongEngine(
        visible=False,
        enable_spin=False,
        enable_curve=False,
        winning_score=3  # First to 3 points wins
    )
    
    ai = LearningAI(side='A')
    expert = PhysicsExpertAI(side='B')
    
    print("\nPlaying test game...")
    frame = 0
    max_frames = 5000
    
    while not engine.game_over and frame < max_frames:
        state = engine.get_state()
        ai_action = ai.decide_action(state, training=True)  # Random
        expert_action = expert.decide_action(state)
        engine.step(ai_action, expert_action)
        
        # Print whenever score changes
        if frame > 0 and (engine.score_a > 0 or engine.score_b > 0):
            if engine.score_a != getattr(test_scoring_system, 'last_score_a', 0) or \
               engine.score_b != getattr(test_scoring_system, 'last_score_b', 0):
                print(f"  Frame {frame}: Score is now {engine.score_a}-{engine.score_b}")
                test_scoring_system.last_score_a = engine.score_a
                test_scoring_system.last_score_b = engine.score_b
                
                if engine.game_over:
                    print(f"  Game Over detected at {engine.score_a}-{engine.score_b}")
        
        frame += 1
    
    print(f"\n✅ Game ended after {frame} frames")
    print(f"   Final Score: AI {engine.score_a} - Expert {engine.score_b}")
    print(f"   Game Over: {engine.game_over}")
    print(f"   Winning Score: {engine.winning_score}")
    
    if engine.game_over:
        if engine.score_a >= engine.winning_score or engine.score_b >= engine.winning_score:
            print("   ✅ Game ended correctly when winning score reached")
        else:
            print(f"   ❌ BUG: Game over but no one reached winning score!")
    else:
        print("   ❌ BUG: Game didn't end (hit max frames)")


def main():
    print("\n" + "="*80)
    print("PONG AI DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis tool helps diagnose issues with:")
    print("  - Saved model files (corruption, NaN values)")
    print("  - AI action decisions (why paddle doesn't move)")
    print("  - Scoring system (game-over conditions)")
    print("="*80)
    
    # Check for models
    models_dir = 'models/progressive'
    if os.path.exists(models_dir):
        print(f"\n📂 Found models directory: {models_dir}")
        
        # Check phase1_final
        phase1_model = os.path.join(models_dir, 'phase1_final.pkl')
        if os.path.exists(phase1_model):
            if check_model_file(phase1_model):
                print("\n" + "-"*80)
                test_ai_actions(phase1_model, num_samples=20)
        
        # Check phase1_latest
        phase1_latest = os.path.join(models_dir, 'phase1_latest.pkl')
        if os.path.exists(phase1_latest) and phase1_latest != phase1_model:
            if check_model_file(phase1_latest):
                print("\n" + "-"*80)
                test_ai_actions(phase1_latest, num_samples=20)
    else:
        print(f"\n📂 No models directory found at: {models_dir}")
        print("   Train a model first with: python train_quick.py 100")
    
    # Test scoring system
    print("\n" + "-"*80)
    test_scoring_system()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
