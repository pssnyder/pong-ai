# Paddle Control & AI Tuning Guide

This guide explains how to configure and tune paddle control parameters for both the Learning AI and Expert AI to achieve desired behavior and reduce jittery movements.

## Quick Start

### Using Presets

The easiest way to configure AIs is using the built-in presets:

```python
from pong_learning_ai import LearningAI, ControlPresets
from pong_expert_ai import PhysicsExpertAI, ExpertPresets

# Learning AI with ultra-smooth movement
learning_ai = LearningAI(paddle_config=ControlPresets.ultra_smooth())

# Expert AI at medium difficulty
expert_ai = PhysicsExpertAI(control_config=ExpertPresets.medium())
```

### Available Presets

**Learning AI Presets (ControlPresets):**
- `reactive()` - Fast, responsive, may be jittery
- `balanced()` - Good mix of speed and stability (default)
- `patient()` - Strategic, smooth but slower to react
- `ultra_smooth()` - Minimal jitter, best visual quality

**Expert AI Presets (ExpertPresets):**
- `perfect()` - Nearly unbeatable
- `hard()` - Very challenging
- `medium()` - Balanced training opponent (default)
- `easy()` - Good for early training

## Configuration Location

All configurations are at the **top** of each AI file for easy tuning:

- **Learning AI:** `pong_learning_ai.py` lines 10-140
- **Expert AI:** `pong_expert_ai.py` lines 10-110
- **Training Script:** `training/train_ai.py` lines 20-60

## Detailed Parameter Guide

### Learning AI Parameters (PaddleControlConfig)

#### Decision Timing Parameters
These control how frequently the AI makes decisions - the key to reducing jitter!

```python
decision_interval = 3  # How many game frames between AI decisions
```
- **Lower (1-2):** More reactive, instinctive play, can be jittery
- **Higher (4-6):** More patient, strategic play, smoother movement
- **Recommended:** 3-4 for training, 5-6 for ultra-smooth visuals

```python
action_commitment_frames = 2  # Min frames to stick with an action
```
- **Lower (1-2):** Can change mind quickly, may flicker
- **Higher (3-5):** Commits to decisions longer, smoother
- **Recommended:** 2-3 for responsive, 4-5 for smooth

#### Paddle Physics Parameters
Control maximum speed and acceleration:

```python
max_velocity = 45.0  # Maximum paddle speed (pixels/frame)
```
- **Lower (35-40):** Slower, more deliberate movement
- **Higher (45-50):** Faster, more aggressive play
- **Recommended:** 38-42 for training

```python
acceleration = 8.0  # How quickly paddle reaches max speed
```
- **Lower (5-7):** Gradual, smooth acceleration
- **Higher (9-12):** Quick, snappy movement
- **Recommended:** 7-9

```python
deceleration_factor = 0.6  # How quickly paddle slows down (0-1)
```
- **Lower (0.4-0.5):** Gradual slowdown, drifty feel
- **Higher (0.7-0.8):** Quick stops, responsive
- **Recommended:** 0.6-0.7

#### PID Controller Parameters
**The most important settings for preventing oscillation and jitter!**

```python
kp = 0.4   # Proportional gain - strength of response to error
ki = 0.008 # Integral gain - corrects persistent offset errors  
kd = 0.3   # Derivative gain - dampens oscillations
```

**Tuning PID for smooth movement:**

1. **Too much oscillation (wobbling)?**
   - Increase `kd` (0.4-0.5) for stronger damping
   - Decrease `kp` (0.3-0.35) to reduce response strength

2. **Too sluggish (slow to reach target)?**
   - Increase `kp` (0.45-0.5) for stronger response
   - Decrease `kd` (0.2-0.25) to allow faster movement

3. **Persistent offset (not quite reaching target)?**
   - Increase `ki` (0.01-0.015) to accumulate error correction
   - Be careful: too high causes overshoot!

**Recommended PID combinations:**
- **Ultra Smooth:** `kp=0.30, ki=0.010, kd=0.50` (high damping)
- **Balanced:** `kp=0.40, ki=0.008, kd=0.30` (default)
- **Responsive:** `kp=0.50, ki=0.005, kd=0.20` (low damping)

#### Dead Zone Parameters
Prevents micro-adjustments when close to target:

```python
dead_zone_threshold = 8.0  # Stop moving if within this distance
```
- **Lower (5-7):** More precise positioning
- **Higher (9-12):** Less jitter near target
- **Recommended:** 8-10

### Expert AI Parameters (ExpertControlConfig)

#### Speed & Response
```python
paddle_speed_multiplier = 0.9  # Speed handicap (0.7-1.0)
reaction_frames = 2            # Frames of response delay
```
- Lower multiplier and higher reaction frames = easier opponent
- Use 0.85-0.90 with 2-3 frames for training
- Use 0.95-1.0 with 0-1 frames for challenge

#### Calculation Frequency
```python
calculation_interval = 4  # Recalculate target every N frames
```
- **Lower (2-3):** More responsive but may be jittery
- **Higher (5-7):** Smoother but less reactive
- **Recommended:** 4-5

#### Freeze Zone (Prevents Spin Generation)
```python
freeze_distance_before = 150  # Stop moving when ball this close
freeze_distance_after = 100   # Stay frozen until ball this far
```
- Larger freeze zones = expert generates less spin
- Smaller freeze zones = more spin, harder to exploit
- **Recommended:** 150/100 for balanced play

#### PID Parameters
Same as Learning AI - tune for smooth expert movement:
```python
kp = 0.35   # Proportional gain
ki = 0.005  # Integral gain
kd = 0.25   # Derivative gain
```

## Custom Configuration Example

Create your own custom configurations in `training/train_ai.py`:

```python
def get_training_configs():
    # CUSTOM LEARNING AI - Patient and smooth
    learning_config = PaddleControlConfig(
        decision_interval=5,        # Very patient decisions
        action_commitment_frames=4,  # Long commitment
        max_velocity=40.0,          # Moderate speed
        kp=0.32,                    # Gentle response
        ki=0.012,                   # Moderate integral
        kd=0.48,                    # Strong damping
        dead_zone_threshold=10.0    # Larger dead zone
    )
    
    # CUSTOM EXPERT AI - Easier opponent
    expert_config = ExpertControlConfig(
        paddle_speed_multiplier=0.80,  # 80% speed
        reaction_frames=3,             # Slower reactions
        calculation_interval=6,        # Less frequent updates
        kp=0.30,                       # Gentler
        kd=0.35                        # More damped
    )
    
    return learning_config, expert_config
```

## Troubleshooting

### Problem: Learning AI is too jittery

**Solution 1: Increase decision patience**
```python
decision_interval=5           # Was: 3
action_commitment_frames=4    # Was: 2
```

**Solution 2: Increase PID damping**
```python
kp=0.30    # Was: 0.40 (reduce response)
kd=0.50    # Was: 0.30 (increase damping)
```

**Solution 3: Lower max velocity**
```python
max_velocity=38.0  # Was: 45.0
```

### Problem: Learning AI overshoots targets

**Solution: Increase damping and dead zone**
```python
kd=0.45                      # Increase damping
dead_zone_threshold=12.0     # Increase dead zone
deceleration_factor=0.7      # Faster stopping
```

### Problem: Learning AI doesn't reach targets

**Solution: Increase integral gain**
```python
ki=0.012   # Was: 0.008
```
Or decrease dead zone:
```python
dead_zone_threshold=6.0  # Was: 8.0
```

### Problem: Expert AI is too slow/fast

**Solution: Adjust speed multiplier**
```python
paddle_speed_multiplier=0.85  # Adjust between 0.7-1.0
```

### Problem: Both AIs oscillate/wobble

**Solution: Universal PID fix for both AIs**
```python
kp=0.30   # Reduce proportional (was 0.35-0.40)
kd=0.50   # Increase derivative (was 0.25-0.30)
```

## Testing Your Configuration

After making changes, test with a short training run:

```bash
python .\training\train_ai.py
# Choose: 1 (Visual Training)  
# Games: 5 (just a few to observe behavior)
```

Watch the paddle movement:
- **Good:** Smooth glides, deliberate movements, minimal wobbling
- **Bad:** Rapid flickering, constant micro-adjustments, oscillation

## Recommended Workflow

1. **Start with presets:** Use `ControlPresets.ultra_smooth()` and `ExpertPresets.medium()`

2. **Tune gradually:** Change one parameter at a time

3. **Test visually:** Run 5-10 games with visible=True to observe behavior

4. **Focus on PID first:** Get oscillation under control before adjusting timing

5. **Balance speed vs smoothness:** Find the sweet spot for your use case

## Performance Impact

- **Decision intervals:** Higher = less computation, smoother
- **PID smoothing:** Minimal impact, always recommended
- **Action commitment:** No performance impact, prevents flickering

The configuration system adds negligible overhead but significantly improves visual quality and stability!
