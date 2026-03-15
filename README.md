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

Watch the neural network learn to beat the expert:

```bash
python train_ai.py
```

**Visual Training** (recommended first):
- Watch the AI learn in real-time
- See spin values when the AI discovers useful spins
- Track win rate improving over time

**Fast Training** (headless):
- Much faster (no graphics)
- Train for 1000+ games quickly
- Models auto-save and resume

### 3. Watch Progress

Early training:
```
Game 1-10:   Win Rate: ~10% (Random flailing)
Game 10-50:  Win Rate: ~30% (Learning to hit ball)
Game 50-100: Win Rate: ~40% (Discovering patterns)
Game 100+:   Win Rate: ~50%+ (Starting to exploit spin!)
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

## 🎓 Next Steps & Enhancements

Potential improvements to explore:

1. **Add More Physics**:
   - Variable ball speed on impact
   - Power shots (faster volleys)
   - Friction on paddles

2. **Enhanced Learning**:
   - Larger neural network
   - CNN for spatial understanding
   - Curiosity-driven exploration

3. **Multi-Stage Learning**:
   - Train against Easy → Medium → Hard → Perfect
   - Curriculum learning approach

4. **Analysis Tools**:
   - Plot win rate over time
   - Visualize learned strategies
   - Heatmap of preferred actions

## 🏆 Success Metrics

Your Learning AI is doing well when:
- ✅ Win rate > 45% against 90% speed expert
- ✅ Consistently creates spin (values > 10)
- ✅ Wins increase when spin/curve enabled vs disabled
- ✅ Learns to stay near center when ball is far

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
