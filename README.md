# 🏓 Pong AI - Expert System vs Neural Network with Advanced Physics

An educational AI playground where you can watch a neural network learn to beat a physics-perfect expert system!

## 🎯 The Fascinating Challenge

This isn't just another Pong game - it's a demonstration of how learning AI can discover strategies that rule-based systems miss:

- **🔴 Physics Expert (Right Paddle B)**: Uses perfect mathematical physics to calculate exact ball trajectories
  - Calculates the EXACT pixel where the ball will land
  - Uses only straight-line vector mathematics  
  - **WEAKNESS**: Completely blind to spin and curve effects!

- **🔵 Learning AI (Left Paddle A)**: Neural network that starts knowing nothing
  - Learns through trial and error
  - Can discover and exploit spin/curve mechanics
  - **GOAL**: Learn that spinning the ball beats perfect predictions!

## 🧪 The Physics System

The game includes advanced ball physics that creates a fascinating dynamic:

### Spin Mechanics
- **Paddle velocity creates spin** - Moving paddle imparts rotation
- **Off-center hits add spin** - Hitting top/bottom of paddle adds extra rotation  
- **Spin affects bounces** - Changes bounce angle when hitting paddles
- **Spin decays over time** - Gradually reduces through air resistance

### Curve Mechanics (Magnus Effect)
- **Spinning balls curve** - Rotation creates sideways force
- **Expert can't predict curves** - Only uses straight-line physics
- **Learning AI can discover this** - Neural network learns spin patterns

### Frame-Perfect Physics
- **200 FPS physics simulation** - Precise calculations
- **Exact trajectory prediction** - Expert calculates pixel-perfect paths
- **Paddle acceleration** - Realistic movement physics

## 🚀 Quick Start

### 1. Demo the Expert System

Watch the physics expert in action (perfect predictions, but exploitable!):

```bash
python demo_expert.py
```

Choose from:
- **Expert vs Expert** - Watch two perfect AIs battle (with spin creating chaos!)
- **Perfect vs Medium** - See if perfect prediction always wins
- **Basic Physics** - See how it works without spin/curve

### 2. Train the Learning AI

🎓 **Progressive Training System** - Three-phase curriculum for optimal learning:

#### Quick Start (Phase 1 Only)
```bash
python train_quick.py 500             # Train 500 games
python train_quick.py 1000 --watch    # Train and watch result
```

#### Full Training Pipeline (All 3 Phases)
```bash
python train_progressive.py
# Select option 1 to run all phases
```

**Phase 1: Basic Physics** (500 games)
- No spin, no curve - just straight-line physics
- Learn fundamental positioning and trajectory prediction
- Goal: 40%+ win rate before advancing
- Configuration: `enable_spin=False, enable_curve=False`

**Phase 2: Spin Physics** (500 games)  
- Builds on Phase 1 model
- Enables spin and curve mechanics
- Learn to handle and exploit spin to beat expert
- Configuration: `enable_spin=True, enable_curve=True`

**Phase 3: Progressive Speed** (500 games)
- Builds on Phase 2 model
- Ball gets progressively faster (increases every 5 points)
- Learn to adapt to escalating difficulty
- Configuration: `auto_progress=True, difficulty scaling`

### 3. Evaluate Trained Models

Test and visualize your trained AI:

```bash
python evaluate_ai.py
```

**Options**:
- Play visual game (watch AI in action)
- Run benchmark test (100 games, headless)
- Quick tests for each phase model

**Benchmark Results** (typical after full training):
```
Phase 1 Model:  Win Rate ~40-50%  (basic physics)
Phase 2 Model:  Win Rate ~50-60%  (with spin exploitation)
Phase 3 Model:  Win Rate ~55-65%  (adaptive to difficulty)
```

### 4. Watch Progress

**Phase 1 Training** (Basic Physics):
```
Game 1-50:   Win Rate: ~20% (Learning to hit ball)
Game 50-200: Win Rate: ~35% (Discovering positioning)
Game 200+:   Win Rate: ~45% (Consistent basic play)
```

**Phase 2 Training** (Adding Spin):
```
Game 1-50:   Win Rate: ~40% (Adapting to new physics)
Game 50-200: Win Rate: ~50% (Learning spin patterns)
Game 200+:   Win Rate: ~60% (Exploiting spin vs expert!)
```

**Phase 3 Training** (Progressive Speed):
```
Game 1-50:   Win Rate: ~50% (Handling speed changes)
Game 50-200: Win Rate: ~55% (Adaptive strategy)
Game 200+:   Win Rate: ~60%+ (Mastery across difficulties)
```

## 🎮 How It Works

### Physics Expert AI Strategy

1. **Perfect Trajectory Calculation**:
   ```python
   # Calculate EXACTLY where ball will be
   frames_to_arrival = distance_to_paddle / ball_dx
   predicted_y = ball_y + (ball_dy * frames_to_arrival)
   # Account for wall bounces...
   ```

2. **Limitation**: Only uses straight vectors - ignores spin/curve:
   ```python
   # Expert prediction: straight line
   # Actual ball path: curves due to spin!
   # → Expert misses ball!
   ```

3. **Speed Limiter**: Paddle has acceleration (not instant teleport)
   - Makes it beatable even with perfect prediction
   - More realistic physics

### Learning AI Strategy

1. **Neural Network** (10 inputs → 32 hidden → 3 outputs):
   - Inputs: ball position, velocity, **spin**, paddle positions/velocities
   - Outputs: Move UP, STAY, or DOWN

2. **Deep Q-Learning**:
   - Plays games, remembers experiences
   - Learns Q-values (action quality estimates)
   - Improves through reinforcement

3. **Key Discovery Process**:
   - Early: Random movements → low win rate
   - Middle: Learns to track ball → moderate wins
   - Late: **Discovers spin creates unpredictable bounces** → beats expert!

## 📁 Project Structure

```
pong-ai/
├── pong_engine.py          # Game engine with advanced physics
├── pong_expert_ai.py       # Physics-perfect expert system
├── pong_learning_ai.py     # Neural network AI
├── demo_expert.py          # Demo script (watch expert play)
├── train_ai.py             # Training script (train neural network)
├── models/                 # Saved trained models
│   └── pong_learning_ai.pkl
└── README.md
```

## 🧠 Educational Value

This project demonstrates:

### Classical AI (Expert System)
- ✅ Perfect from day one
- ✅ Explainable decisions
- ✅ Deterministic behavior
- ❌ Can't adapt to new patterns
- ❌ Brittle to edge cases (spin/curve)

### Modern AI (Neural Network)
- ❌ Terrible at first (random)
- ❌ "Black box" decisions
- ❌ Requires lots of training
- ✅ Discovers emergent strategies  
- ✅ Adapts to physics it wasn't told about

### Key Insight
**The Learning AI discovers that creating spin beats perfect straight-line prediction!** This mirrors how modern AI finds strategies humans never explicitly programmed.

## 📊 Monitoring Training

Watch for these indicators of learning:

1. **Win Rate Climbing**: 10% → 30% → 50%+
2. **Exploration Decreasing**: 100% → 50% → 10% randomness
3. **Spin Values**: AI starts creating consistent spin patterns
4. **Frame Efficiency**: Games end faster (better strategies)

## 🔧 Customization

### Adjust Expert Difficulty

```python
# In demo_expert.py or train_ai.py
expert_ai = PhysicsExpertAI(
    side='B',
    paddle_speed_multiplier=0.90,  # 0.7-1.0 (lower = easier)
    reaction_frames=1              # 0-5 (higher = slower reaction)
)
```

### Adjust Physics

```python
# In either script
engine = PongEngine(
    enable_spin=True,    # Paddle velocity creates spin
    enable_curve=True,   # Spin curves ball path
    ball_speed=1.5,      # Ball movement speed
    paddle_speed=25      # Paddle movement per frame
)
```

### Tune Learning AI

```python
learning_ai = LearningAI(
    learning_rate=0.001,     # How fast it learns (0.0001-0.01)
    discount_factor=0.95     # Future reward value (0.9-0.99)
)
```

## 📁 File Structure

```
pong-ai/
│
├── Core Engine & Physics
│   ├── pong_engine.py          # Game engine with advanced physics simulation
│   └── pong_game.py             # Main game launcher (legacy, use scripts below)
│
├── AI Systems
│   ├── pong_expert_ai.py        # Physics-perfect expert system (PID controller)
│   └── pong_learning_ai.py      # Neural network learning AI (Deep Q-Learning)
│
├── Training Scripts (Use These!)
│   ├── train_quick.py           # Quick start: Train Phase 1 (basic physics)
│   ├── train_progressive.py     # Full pipeline: All 3 training phases
│   └── evaluate_ai.py           # Test and visualize trained models
│
├── Demo & Debugging
│   ├── demo_expert.py           # Watch expert AIs battle (visual demo)
│   └── telemetry_monitor.py     # Real-time monitoring & debugging tool
│
├── Models & Data (Auto-created)
│   └── models/progressive/
│       ├── phase1_final.pkl     # Phase 1: Basic physics model
│       ├── phase2_final.pkl     # Phase 2: Spin physics model
│       ├── phase3_final.pkl     # Phase 3: Progressive speed model
│       └── training_history.json # Training progress tracking
│
└── Documentation
    ├── README.md                # This file
    └── QUICK_REFERENCE.md       # (Optional) Quick command reference
```

### File Descriptions

**Core Engine**:
- `pong_engine.py` - Complete game engine with spin, curve, progressive difficulty
- Features: 200 FPS physics, telemetry collection, difficulty scaling

**Expert AI**:
- `pong_expert_ai.py` - Physics-perfect trajectory prediction system
- Uses: PID controller, freeze zones, straight-line physics only
- Weakness: Blind to spin and curve effects!

**Learning AI**:
- `pong_learning_ai.py` - Deep Q-Learning neural network
- Architecture: 10 inputs → 32 hidden → 3 outputs (UP/DOWN/STAY)
- Features: Experience replay, epsilon-greedy exploration, target network

**Training System**:
- `train_quick.py` - Fastest way to get started (Phase 1 only)
- `train_progressive.py` - Complete 3-phase curriculum training
- `evaluate_ai.py` - Test trained models and run benchmarks

**Utilities**:
- `demo_expert.py` - Visual demonstration of expert systems
- `telemetry_monitor.py` - Real-time 7-graph monitoring system

## 📊 Training Data & Models

Models are automatically saved to `models/progressive/`:

**Phase 1 Models**:
- `phase1_latest.pkl` - Resumable checkpoint (saves every 100 games)
- `phase1_final.pkl` - Completed Phase 1 training
- `phase1_gameX.pkl` - Periodic snapshots at game milestones

**Phase 2 & 3 Models**: Same pattern

**Training History**:
- `training_history.json` - Win rate, games played, completion status

## 🎓 Next Steps & Enhancements

Potential improvements to explore:

1. **Advanced Training**:
   - Reduce paddle speed to 80% for harder challenge
   - Train against multiple expert difficulty levels
   - Implement curriculum learning with gradual transitions

2. **Enhanced Physics**:
   - Power shots (hold direction for stronger hits)
   - Paddle friction zones (top/bottom create more spin)
   - Ball degradation (spin decays faster)

3. **Better Learning**:
   - Larger neural network (64+ hidden neurons)
   - CNN for spatial awareness
   - Multi-agent training (AI vs AI)
   - Prioritized experience replay

4. **Analysis Tools**:
   - Plot win rate curves over training
   - Visualize learned Q-values
   - Heatmap of paddle positioning strategies
   - Spin usage analysis

## 🏆 Success Metrics

Your Learning AI is doing well when:
- ✅ **Phase 1**: Win rate > 40% (basic physics mastered)
- ✅ **Phase 2**: Win rate > 50% (spin exploitation discovered)
- ✅ **Phase 3**: Win rate > 55% (adaptive to progressive difficulty)
- ✅ Consistently creates spin (ball_spin values > 10)
- ✅ Wins increase when spin/curve enabled vs disabled
- ✅ Learns defensive positioning when ball is far away

## 📝 Technical Notes

- **Physics Frame Rate**: 200 FPS for accurate calculations
- **Training Speed**: ~5-10 games/sec visual, ~100+ games/sec headless
- **Memory**: Stores last 10,000 experiences for training
- **Model Size**: ~50KB (lightweight neural network)

## 🎯 The Core Insight

**This project demonstrates why learning systems can surpass rule-based systems**: The expert has perfect knowledge of basic physics, but the learning AI discovers emergent properties (spin/curve effects) that weren't explicitly programmed. This is the power of machine learning!

---

**Have fun watching AI learn to outsmart perfect physics!** 🎮🤖
```bash
python pong_game.py
```

### 2. Choose a mode from the menu:
- **Watch AI Learn** - Visual training (slow but satisfying!)
- **Fast Train** - Headless training (1000s of games)
- **Expert Battle** - Watch two AIs compete
- **Test Trained Model** - See your AI's progress

## 📊 Training Your AI

### Visual Training (Recommended First)
```bash
python pong_game.py
> Choose option 1
> Enter 100 games
```

Watch as the AI:
- Starts by moving randomly
- Gradually learns to track the ball
- Eventually becomes competitive!

### Fast Training (For Serious Training)
```bash
python pong_game.py
> Choose option 2
> Enter 1000 games
```

This trains much faster (no graphics overhead). Perfect for overnight training!

## 🎯 How the Learning Works

The neural network uses **Deep Q-Learning** (DQN):

1. **State** → Ball position, velocity, paddle positions
2. **Action** → Move UP, DOWN, or STAY
3. **Reward** → +1 for scoring, -1 for opponent scoring, +0.1 for hitting ball
4. **Learning** → Adjust brain to maximize future rewards

### Training Strategy
- **Epsilon-Greedy Exploration**: Starts 100% random, gradually becomes more strategic
- **Experience Replay**: Learns from past experiences stored in memory
- **Target Network**: Stabilizes learning by using a slower-updating reference

## 📈 Typical Learning Curve

```
Games    | Win Rate | Behavior
---------|----------|----------------------------------
0-100    | ~0%      | Random movement, rarely hits ball
100-500  | ~20%     | Starting to track ball movement
500-1000 | ~40%     | Decent positioning and timing
1000+    | 50-60%   | Competitive against medium AI
2000+    | 60-70%   | Can beat hard AI sometimes
```

## 🎓 Educational Value

### Learn About:
1. **Expert Systems** - How classical AI uses logic and rules
2. **Neural Networks** - How modern AI learns from data
3. **Reinforcement Learning** - Learning from rewards/punishments
4. **Exploration vs Exploitation** - The fundamental AI trade-off
5. **Deep Q-Learning** - A popular RL algorithm

### Great For:
- Students learning AI/ML concepts
- Understanding the difference between programmed vs learned behavior
- Seeing reinforcement learning in action
- Building intuition about neural networks

## 📁 Project Structure

```
pong-ai/
├── pong_engine.py        # Core game engine with clean API
├── pong_expert_ai.py     # Rule-based AI (expert system)
├── pong_learning_ai.py   # Neural network AI (DQN)
├── pong_game.py          # Main launcher with different modes
├── models/               # Saved trained models
│   └── learning_ai.pkl   # Your AI's brain (after training)
└── README.md            # This file
```

## 🔬 Experiment Ideas

1. **Compare Difficulties**
   - Train against Easy AI vs Hard AI
   - Which trains a better neural network?

2. **Tune Hyperparameters**
   - Adjust learning rate (0.001 default)
   - Change epsilon decay rate
   - Modify network architecture

3. **Track Progress**
   - Graph win rate over time
   - Analyze learning curves
   - Find optimal training length

4. **Advanced Challenges**
   - Can the AI beat Perfect difficulty?
   - Train with faster ball speeds
   - Multi-ball Pong?

## 🛠️ Technical Details

### Neural Network Architecture
```
Input Layer:  6 neurons (ball_x, ball_y, ball_dx, ball_dy, paddle_a_y, paddle_b_y)
Hidden Layer: 16 neurons (ReLU activation)
Output Layer: 3 neurons (Q-values for UP, DOWN, STAY)
```

### Training Parameters
- **Learning Rate**: 0.001
- **Discount Factor**: 0.95 (how much to value future rewards)
- **Batch Size**: 32 experiences
- **Memory Size**: 10,000 experiences
- **Initial Epsilon**: 1.0 (100% exploration)
- **Min Epsilon**: 0.01 (always explore a little)
- **Epsilon Decay**: 0.9995 per game

## 💡 Tips for Best Results

1. **Start with Visual Training** (100 games) to understand the learning process
2. **Then Fast Train** (1000+ games) for serious performance
3. **Save Frequently** - The model auto-saves but don't lose your progress!
4. **Be Patient** - Real learning takes 500-1000+ games
5. **Experiment** - Try different opponent difficulties

## 🎯 Success Metrics

Your AI is learning well if:
- ✅ Win rate increases over time
- ✅ Paddle tracks ball more accurately
- ✅ Fewer "own goals" from missing the ball
- ✅ Better positioning before ball arrives

Your AI needs more training if:
- ❌ Win rate stuck at 0-10%
- ❌ Still moving randomly
- ❌ Doesn't react to ball direction
- ❌ Frequently moves away from ball

## 🤝 Contributing

Ideas for enhancements:
- Add visualization of neural network activations
- Plot learning curves in real-time
- Implement other RL algorithms (A3C, PPO)
- Add human vs AI mode
- Tournament mode with multiple AIs

## 📚 Learn More

Want to understand the techniques used?
- **Reinforcement Learning**: Sutton & Barto's RL book
- **Deep Q-Learning**: DeepMind's DQN paper (2015)
- **Expert Systems**: Classic AI textbooks (Russell & Norvig)

## ⚡ Performance

Training speed (approximate):
- **Visual Mode**: ~10 games/minute
- **Headless Mode**: ~100-200 games/second

Hardware dependent, but headless mode is **~1000x faster**!

## 🎉 Have Fun!

The most rewarding part is watching your AI go from completely clueless to actually competitive. It's like watching a baby learn to walk!

Train well, and may your neural network converge quickly! 🚀🤖

---

**Made with ❤️ for AI education**
