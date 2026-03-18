# Understanding Win Rates and Expected Performance

## 🎯 Win Rate Perspective

### All win rates shown are from the **LEARNING AI's perspective**

- **0% win rate** = Learning AI loses every game (Expert wins 100%)
- **50% win rate** = Learning AI wins half the games  
- **100% win rate** = Learning AI wins every game (Expert loses 100%)

### Expected Performance Timeline

| Training Stage | Learning AI Win Rate | Expert Win Rate | What's Happening |
|---------------|---------------------|-----------------|------------------|
| **Untrained (0-50 games)** | ~0-5% | ~95-100% | AI making random moves, expert dominates |
| **Early Learning (50-150)** | ~15-25% | ~75-85% | AI learns basic positioning |
| **Phase 1 Complete (500+)** | ~40-50% | ~50-60% | AI masters straight physics |
| **Phase 2 Complete (1000+)** | ~50-60% | ~40-50% | AI exploits spin vs expert |
| **Phase 3 Complete (1500+)** | ~55-65% | ~35-45% | AI adapts to speed changes |

## ❌ Common Issues & Solutions

### Issue 1: "AI paddle doesn't move"

**Cause**: Untrained/undertrained neural network predicts STAY action constantly

**Why**: Network weights are random/near-zero → outputs ~[0, 0, 0] → argmax picks index 0 or 1 (STAY)

**Solution**: 
```bash
# Need at least 100-200 training games to see movement
python train_quick.py 200

# Check if model is learning:
python diagnose.py  # Shows action distribution
```

**Expected Action Distribution**:
- Untrained: STAY ~80-100%, UP ~0-10%, DOWN ~0-10%
- After 100 games: STAY ~40-60%, UP ~20-30%, DOWN ~20-30%
- After 500 games: STAY ~30-40%, UP ~30-35%, DOWN ~30-35%

### Issue 2: "Score shows 0-11 but only 1 point scored"

**Possible Causes**:
1. Game ended at 11 points (correct)  - one side scored 11, other scored 0
2. Display bug showing wrong scores
3. Model file corrupted with NaN values → paddle doesn't move → expert scores 11-0

**Debug Steps**:
```bash
python diagnose.py  # Check model health and test scoring
```

Look for:
- ✅ "No NaN/Inf values found" = Model OK
- ❌ "Found NaN or Inf values" = Model corrupted, retrain!
- Check "AI Action Distribution" - if 100% STAY, AI can't defend → 0-11 loss

### Issue 3: "96% win rate - whose perspective?"

**Answer**: That's the **Learning AI's** win rate!

If you see 96% win rate, it means:
- Learning AI wins 96 out of 100 games
- Expert only wins 4 out of 100 games
- **This is GOOD** for the Learning AI (far exceeds expectations!)

**However**: If this is after only 10 games, the model likely has **numerical errors** (weights exploded to infinity). Run `diagnose.py` to check.

## 🔧 Expected Training Results

### After 10 Games (Quick Test)
```
Learning AI Win Rate: 0-10%  (0-1 wins out of 10)
Average Score: 0-2 points per game
Expert wins most games 11-0 or 11-1
```

### After 100 Games
```
Learning AI Win Rate: 20-30%  (20-30 wins out of 100)
Average Score: 3-5 points per game
AI starts to track ball and block some shots
```

### After 500 Games (Phase 1 Complete)
```
Learning AI Win Rate: 40-50%  (200-250 wins out of 500)
Average Score: 5-7 points per game
AI masters basic physics, competitive games
```

## 🧪 How to Verify Your Model is Working

Run the diagnostic tool:
```bash
python diagnose.py
```

### Healthy Model Output:
```
✅ No NaN/Inf values found
📊 Action Distribution over 20 samples:
  UP   :   7 (35.0%)
  DOWN :   6 (30.0%)
  STAY :   7 (35.0%)
✅ AI shows varied actions - some learning detected!
```

### Problematic Model Output:
```
❌ Found NaN or Inf values in weights!
   Model is CORRUPTED - retrain from scratch!

OR

⚠️ AI ALWAYS chooses STAY - model hasn't learned yet!
   Need more training games (~100-200 minimum)
```

## 🎮 Testing Your Trained AI

### Visual Test (Watch AI Play)
```bash
python evaluate_ai.py
# Select option 3 (Quick test - Phase 1)
```

**What to Look For**:
- ✅ Paddle moves up and down tracking ball
- ✅ AI scores some points (even 1-2 is progress!)
- ✅ Action distribution shows varied moves
- ❌ Paddle stays frozen (untrained)
- ❌ Score 0-11 consistently (not learning)

### Benchmark Test (100 Games Headless)
```bash
python evaluate_ai.py
# Select option 2, choose model, run 100 games
```

**Expected Results**:
```
After 100 training games:
  Learning AI: 2.5 points/game, 20% win rate

After 500 training games:
  Learning AI: 6.0 points/game, 45% win rate
```

## 📊 Training Progress Indicators

### Good Progress ✅
```
Game 100: AI 5-11 Expert [LOSS] | Overall: W:25 L:75 T:0 (25.0% WR)
Game 200: AI 8-11 Expert [LOSS] | Overall: W:65 L:135 T:0 (32.5% WR)
Game 500: AI 11-9 Expert [WIN] | Overall: W:220 L:280 T:0 (44.0% WR)
```
- Win rate steadily increasing
- Scores getting closer (5-11 → 11-9)
- Some wins happening!

### Problem Signs ⚠️
```
Game 100: AI 0-11 Expert [LOSS] | Overall: W:0 L:100 T:0 (0.0% WR)
Game 200: AI 0-11 Expert [LOSS] | Overall: W:0 L:200 T:0 (0.0% WR)
Game 500: AI 0-11 Expert [LOSS] | Overall: W:0 L:500 T:0 (0.0% WR)
```
- No improvement over time
- Every game 0-11
- 0% win rate after hundreds of games
- **Solution**: Check for NaN values with `diagnose.py`, may need to retrain

## 🔍 Quick Diagnostic Checklist

1. **Check model file exists**:
   ```bash
   ls models/progressive/phase1_final.pkl
   ```

2. **Run diagnostic tool**:
   ```bash
   python diagnose.py
   ```

3. **Look for**:
   - ✅ No NaN/Inf in weights
   - ✅ Action variety (not 100% STAY)
   - ✅ Game scoring works correctly
   - ✅ Win rate matches training games

4. **If problems found**:
   ```bash
   # Delete corrupted model
   rm models/progressive/*.pkl
   
   # Retrain from scratch
   python train_quick.py 500
   ```

## 💡 Key Takeaways

1. **Win Rate = Learning AI wins / total games** (not expert wins)
2. **Expert should win ~95-100% at start** (AI is random)
3. **Goal: Get to ~40-50% after Phase 1** (500 games)
4. **0-11 scores are normal early on** (AI can't defend yet)
5. **Paddle not moving = undertrained model** (needs 100+ games)
6. **Use `diagnose.py` to check model health** before evaluation

---

**Summary**: The expert is SUPPOSED to dominate early. A 0% Learning AI win rate after 10 games is completely normal. Only after hundreds of games will the AI learn enough to compete!
