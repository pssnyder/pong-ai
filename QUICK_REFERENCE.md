# 🏓 Pong AI - Quick Reference Guide

## 🚀 Quick Start Commands

### Getting Started (First Time)
```bash
# 1. Watch expert AIs battle (no training)
python demo_expert.py

# 2. Train your first AI (500 games, ~5-10 minutes)
python train_quick.py 500

# 3. Watch your trained AI play
python evaluate_ai.py
# → Select option 3 (Quick test - Phase 1)
```

### Full Training Pipeline
```bash
# Run all 3 phases (1500 games total, ~20-30 minutes)
python train_progressive.py
# → Select option 1 (Run all phases)
```

### Continue Training
```bash
# Phase 1 (more games)
python train_progressive.py
# → Select option 2

# Phase 2 (after Phase 1 complete)
python train_progressive.py
# → Select option 3

# Phase 3 (after Phase 2 complete)
python train_progressive.py
# → Select option 4
```

### Testing & Evaluation
```bash
# Visual game (watch AI play)
python evaluate_ai.py
# → Select option 1

# Benchmark (100 games, headless)
python evaluate_ai.py
# → Select option 2
```

## 📋 Training Phases Explained

| Phase | Physics | Ball Speed | Goal | Win Rate Target |
|-------|---------|------------|------|-----------------|
| **1: Basic** | No spin/curve | Steady 1.0x | Learn positioning | 40%+ |
| **2: Spin** | Spin + curve | Steady 1.0x | Exploit spin vs expert | 50%+ |
| **3: Speed** | Spin + curve | Progressive (1.0→3.0x) | Adapt to difficulty | 55%+ |

## 🎯 Common Tasks

### Watch Expert AI Demo
```bash
python demo_expert.py
# Options:
#   1 - Expert vs Expert (with spin chaos)
#   2 - Perfect vs Medium expert
#   3 - Basic physics (no spin/curve)
```

### Train Specific Amount
```bash
# Train 1000 games and watch result
python train_quick.py 1000 --watch

# Custom training
python train_progressive.py
# → Select option 5 (Custom)
# → Enter: Phase, Games, Paddle Speed %
```

### Load & Test Model
```bash
python evaluate_ai.py
# Available models:
#   - phase1_final.pkl (basic physics)
#   - phase2_final.pkl (with spin)
#   - phase3_final.pkl (progressive)
#   - phase1_latest.pkl (resumable checkpoint)
```

## 📊 Monitor Training Progress

### Console Output Explained
```
[Phase 1] Progress: 250/500 (50.0%) | Win Rate: 38.2% (Recent: 45%) | Epsilon: 50.0% | Time: 145s
           └─────┬─────┘              └────┬────┘              └──┬──┘              └──┬─┘
              Current                  Overall                Recent             Exploration
              game out                 win rate              10 games             rate
              of target                                      win rate
```

### Understanding the Metrics

- **Progress**: Current game / Target games
- **Win Rate**: Overall percentage of wins against expert
- **Recent Win Rate**: Last 10 games (shows current form)
- **Epsilon**: Exploration rate (100% = random, 0% = learned strategy)
- **Time**: Seconds elapsed since phase started

### Good Training Signs ✅
- Win rate steadily climbing
- Epsilon decreasing over time
- Recent win rate higher than overall (improving!)
- Time per game decreasing (better strategies)

### Warning Signs ⚠️
- Win rate stuck/decreasing (may need more games)
- Recent win rate much lower than overall (getting worse)
- Win rate < 20% after 200+ games (check configuration)

## 🛠️ Configuration Tips

### Make Expert Easier/Harder
Edit `train_progressive.py` or `train_quick.py`:
```python
# Line ~100 in train_phase_X methods:
expert = PhysicsExpertAI(
    side='B', 
    paddle_speed_multiplier=0.90  # Change this:
                                  # 1.0 = 100% speed (hardest)
                                  # 0.8 = 80% speed (easier)
                                  # 0.6 = 60% speed (very easy)
)
```

### Adjust Training Speed
```python
# In train_progressive.py, adjust these in main():
games=500        # Number of games per phase
save_interval=100  # Save checkpoint every N games
```

### Change Learning Rate
Edit `pong_learning_ai.py`:
```python
# Line ~100 in __init__:
self.learning_rate = 0.001  # Increase = faster learning (less stable)
                            # Decrease = slower learning (more stable)
                            # Range: 0.0001 to 0.01
```

## 📁 File Locations

### Models Saved To
```
models/progressive/
├── phase1_final.pkl      # Completed Phase 1
├── phase1_latest.pkl     # Resume Phase 1
├── phase2_final.pkl      # Completed Phase 2
├── phase3_final.pkl      # Completed Phase 3
└── training_history.json # Stats & progress
```

### Debug Output
```
telemetry_*.json  # Created by demo_expert.py
                  # Load in telemetry_monitor.py to visualize
```

## 🎮 Keyboard Controls (Visual Mode)

When watching games:
- **No controls needed** - AIs play automatically
- **Close window** - End game
- **Ctrl+C** in terminal - Stop training / Return to menu

## 🔍 Troubleshooting

### "No trained models found"
→ Run training first: `python train_quick.py 500`

### "Phase X model not found"
→ Complete previous phase first:
  - Phase 2 needs Phase 1: `python train_progressive.py` → option 2
  - Phase 3 needs Phase 2: `python train_progressive.py` → option 3

### Training interrupted / Want to continue
→ Just run the same command again! Training auto-resumes from checkpoints

### Win rate not improving
→ Try:
  - More games (1000+ total for Phase 1)
  - Easier expert (paddle_speed_multiplier=0.8)
  - Check that correct physics enabled for phase

### Game too fast/slow to watch
→ Edit frame_rate in script:
```python
engine = PongEngine(
    frame_rate=60   # Change this: 30=slow, 60=normal, 200=fast
)
```

## 💡 Pro Tips

1. **Start with Quick Start**: Get Phase 1 done first with `train_quick.py`
2. **Watch Occasionally**: Use `--watch` flag to verify AI is learning
3. **Check Recent Win Rate**: More important than overall (shows current performance)
4. **Phase 1 is Foundation**: Spend extra time here (1000+ games OK)
5. **Phases 2 & 3**: Can be 300-500 games each (building on Phase 1)
6. **Save Checkpoints**: Training auto-saves, but stop cleanly with Ctrl+C
7. **Benchmark After Training**: Run evaluate_ai.py option 2 for true performance

## 📈 Expected Timeline

| Task | Time | Games | Result |
|------|------|-------|--------|
| Demo expert | 1 min | 1 game | Understand system |
| Phase 1 basic | 5-10 min | 500 games | ~40% win rate |
| Phase 1 extended | 10-20 min | 1000 games | ~45% win rate |
| Phase 2 spin | 5-10 min | 500 games | ~55% win rate |
| Phase 3 progressive | 5-10 min | 500 games | ~60% win rate |
| **Full pipeline** | **20-30 min** | **1500 games** | **Master AI!** |

## 🎯 Success Checklist

- [ ] Demo expert system works (`demo_expert.py`)
- [ ] Phase 1 training complete (win rate > 40%)
- [ ] Phase 2 training complete (win rate > 50%)
- [ ] Phase 3 training complete (win rate > 55%)
- [ ] Can load and watch trained AI (`evaluate_ai.py`)
- [ ] Understand what each phase teaches
- [ ] Ready to experiment with custom configurations!

---

**Need more help?** Check the full README.md for detailed explanations!
