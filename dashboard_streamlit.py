"""
Streamlit Dashboard for Pong AI Training Analysis
Real-time monitoring and visualization of training progress, model insights, and physics validation

Run with: streamlit run dashboard_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Pong AI Dashboard",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        color: white;
    }
    .phase-1 { background-color: #2ecc71; }
    .phase-2 { background-color: #f39c12; }
    .phase-3 { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏓 Pong AI Training Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Controls")
refresh_button = st.sidebar.button("🔄 Refresh Data", use_container_width=True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=False)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# File paths
MODEL_PATH = 'models/pong_learning_ai.pkl'
TELEMETRY_PATH = 'telemetry.json'

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Training Overview", 
    "🧠 Model Insights", 
    "🎮 Physics Monitor",
    "📈 Performance Analysis",
    "🔍 Decision Explorer"
])

# ============================================================================
# Tab 1: Training Overview
# ============================================================================
with tab1:
    st.header("Training Progress & Statistics")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            games_played = model_data.get('games_played', 0)
            wins = model_data.get('win_count', 0)
            losses = model_data.get('loss_count', 0)
            ties = model_data.get('tie_count', 0)
            epsilon = model_data.get('epsilon', 0)
            
            win_rate = wins / games_played if games_played > 0 else 0
            
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎮 Games Played", f"{games_played:,}")
            with col2:
                st.metric("🏆 Wins", wins, delta=f"{win_rate*100:.1f}%")
            with col3:
                st.metric("❌ Losses", losses)
            with col4:
                st.metric("⚖️ Ties", ties)
            with col5:
                st.metric("🎲 Exploration", f"{epsilon*100:.1f}%")
            
            # Current phase
            current_phase_name = model_data.get('current_phase_name', 'Unknown')
            phase_num = 1 if 'Phase 1' in current_phase_name else (2 if 'Phase 2' in current_phase_name else 3)
            phase_class = f"phase-{phase_num}"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Training Phase</h3>
                <span class="phase-badge {phase_class}">{current_phase_name}</span>
                <p style="margin-top: 1rem; color: #666;">
                    Phase transitions: {model_data.get('phase_transition_games', [])}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Win rate over time
            st.subheader("📈 Win Rate Progression")
            
            # Calculate win rate windows (last 100 games moving average)
            window_size = 100
            if games_played >= window_size:
                num_windows = games_played // window_size
                window_data = []
                
                for i in range(num_windows):
                    window_end = (i + 1) * window_size
                    window_data.append({
                        'Games': window_end,
                        'Win Rate': 0  # Would need game-by-game history
                    })
                
                if window_data:
                    df_windows = pd.DataFrame(window_data)
                    fig = px.line(df_windows, x='Games', y='Win Rate', 
                                title="Win Rate Trend (100-game windows)")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Record breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Win/Loss/Tie Distribution")
                fig = go.Figure(data=[go.Pie(
                    labels=['Wins', 'Losses', 'Ties'],
                    values=[wins, losses, ties],
                    marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6']),
                    hole=0.4
                )])
                fig.update_layout(showlegend=True, height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎲 Epsilon Decay Progress")
                epsilon_history = []
                decay_rate = 0.995
                current_epsilon = 1.0
                
                for game in range(0, min(games_played, 2000), 10):
                    epsilon_history.append({
                        'Game': game,
                        'Epsilon': current_epsilon
                    })
                    current_epsilon *= decay_rate
                
                df_epsilon = pd.DataFrame(epsilon_history)
                fig = px.line(df_epsilon, x='Game', y='Epsilon',
                            title="Exploration Rate Decay")
                fig.add_hline(y=epsilon, line_dash="dash", 
                            annotation_text=f"Current: {epsilon:.3f}")
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    else:
        st.warning("⚠️ No trained model found. Start training to see statistics!")

# ============================================================================
# Tab 2: Model Insights
# ============================================================================
with tab2:
    st.header("Neural Network Analysis")
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            W1 = model_data['W1']
            W2 = model_data['W2']
            b1 = model_data['b1']
            b2 = model_data['b2']
            
            input_size = model_data.get('input_size', W1.shape[0])
            hidden_size = model_data.get('hidden_size', W1.shape[1])
            output_size = model_data.get('output_size', W2.shape[1])
            
            # Architecture
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📥 Input Features", input_size)
            with col2:
                st.metric("🧠 Hidden Neurons", hidden_size)
            with col3:
                st.metric("📤 Output Actions", output_size)
            
            st.markdown("**Input Features:** ball_x, ball_y, ball_dx, ball_dy, ball_spin, paddle_y, paddle_vel, opp_y, opp_vel")
            st.markdown("**Output Actions:** DOWN (0), STAY (1), UP (2)")
            
            # Weight analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔗 Input Layer Weights (W1)")
                st.write(f"Shape: {W1.shape}")
                
                # Weight statistics
                st.write(f"**Statistics:**")
                st.write(f"- Mean: {np.mean(W1):.4f}")
                st.write(f"- Std: {np.std(W1):.4f}")
                st.write(f"- Min: {np.min(W1):.4f}")
                st.write(f"- Max: {np.max(W1):.4f}")
                
                # Weight distribution
                fig = go.Figure(data=[go.Histogram(x=W1.flatten(), nbinsx=50)])
                fig.update_layout(title="Weight Distribution", 
                                xaxis_title="Weight Value", 
                                yaxis_title="Count",
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔗 Hidden Layer Weights (W2)")
                st.write(f"Shape: {W2.shape}")
                
                # Weight statistics
                st.write(f"**Statistics:**")
                st.write(f"- Mean: {np.mean(W2):.4f}")
                st.write(f"- Std: {np.std(W2):.4f}")
                st.write(f"- Min: {np.min(W2):.4f}")
                st.write(f"- Max: {np.max(W2):.4f}")
                
                # Weight distribution
                fig = go.Figure(data=[go.Histogram(x=W2.flatten(), nbinsx=50)])
                fig.update_layout(title="Weight Distribution",
                                xaxis_title="Weight Value",
                                yaxis_title="Count",
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (magnitude of weights from input features)
            st.subheader("📊 Feature Importance (Input Weight Magnitudes)")
            
            feature_names = ['ball_x', 'ball_y', 'ball_dx', 'ball_dy', 'ball_spin', 
                           'my_paddle_y', 'my_paddle_vel', 'opp_y', 'opp_vel']
            
            importance = np.sum(np.abs(W1), axis=1)
            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                        title="Which input features influence the network most?")
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of W1
            st.subheader("🌡️ Weight Heatmap: Input → Hidden")
            fig = go.Figure(data=go.Heatmap(
                z=W1,
                x=[f'H{i}' for i in range(hidden_size)],
                y=feature_names,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title="Input Layer Weights", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error analyzing model: {str(e)}")
    else:
        st.warning("⚠️ No model to analyze")

# ============================================================================
# Tab 3: Physics Monitor
# ============================================================================
with tab3:
    st.header("Game Physics Validation")
    
    if os.path.exists(TELEMETRY_PATH):
        try:
            with open(TELEMETRY_PATH, 'r') as f:
                telemetry = json.load(f)
            
            if telemetry and len(telemetry.get('frame', [])) > 0:
                df = pd.DataFrame(telemetry)
                
                # Physics validation metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ball_speed = np.sqrt(df['ball_dx']**2 + df['ball_dy']**2)
                    st.metric("⚡ Avg Ball Speed", f"{ball_speed.mean():.2f}")
                with col2:
                    st.metric("💫 Max Spin", f"{df['ball_spin'].abs().max():.2f}")
                with col3:
                    st.metric("🎯 Paddle A Range", f"{df['paddle_a_y'].max() - df['paddle_a_y'].min():.1f}")
                with col4:
                    st.metric("📊 Frames Logged", len(df))
                
                # Ball trajectory
                st.subheader("🏐 Ball Trajectory Visualization")
                fig = go.Figure()
                
                # Color by frame (time progression)
                fig.add_trace(go.Scatter(
                    x=df['ball_x'],
                    y=df['ball_y'],
                    mode='lines+markers',
                    marker=dict(
                        size=4,
                        color=df['frame'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Frame")
                    ),
                    line=dict(width=1, color='rgba(100,100,100,0.3)'),
                    name='Ball Path'
                ))
                
                # Add paddles
                fig.add_shape(type="rect", x0=-360, x1=-340, y0=-60, y1=60,
                            fillcolor="blue", opacity=0.3, line_width=0)
                fig.add_shape(type="rect", x0=340, x1=360, y0=-60, y1=60,
                            fillcolor="red", opacity=0.3, line_width=0)
                
                fig.update_layout(
                    title="Ball Movement Pattern",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    width=800,
                    height=500,
                    xaxis=dict(range=[-400, 400]),
                    yaxis=dict(range=[-300, 300])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Ball Speed Over Time")
                    df['ball_speed'] = np.sqrt(df['ball_dx']**2 + df['ball_dy']**2)
                    fig = px.line(df, x='frame', y='ball_speed',
                                title="Ball speed should increase on paddle hits")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🌀 Spin Evolution")
                    fig = px.line(df, x='frame', y='ball_spin',
                                title="Spin applied by paddle velocity")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Paddle tracking accuracy
                st.subheader("🎯 Paddle Tracking Accuracy")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    df['paddle_a_error'] = df['paddle_a_target'] - df['paddle_a_y']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['frame'], y=df['paddle_a_y'], name='Actual', mode='lines'))
                    fig.add_trace(go.Scatter(x=df['frame'], y=df['paddle_a_target'], name='Target', mode='lines', line=dict(dash='dash')))
                    fig.update_layout(title="Paddle A: Target vs Actual", xaxis_title="Frame", yaxis_title="Position")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Avg Tracking Error", f"{df['paddle_a_error'].abs().mean():.2f}")
                
                with col2:
                    df['paddle_b_error'] = df['paddle_b_target'] - df['paddle_b_y']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['frame'], y=df['paddle_b_y'], name='Actual', mode='lines'))
                    fig.add_trace(go.Scatter(x=df['frame'], y=df['paddle_b_target'], name='Target', mode='lines', line=dict(dash='dash')))
                    fig.update_layout(title="Paddle B: Target vs Actual", xaxis_title="Frame", yaxis_title="Position")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Avg Tracking Error", f"{df['paddle_b_error'].abs().mean():.2f}")
                
            else:
                st.info("📊 No telemetry data captured yet. Enable telemetry during visual training.")
        
        except Exception as e:
            st.error(f"❌ Error loading telemetry: {str(e)}")
    else:
        st.info("📊 No telemetry file found. Run visual training with telemetry enabled.")

# ============================================================================
# Tab 4: Performance Analysis
# ============================================================================
with tab4:
    st.header("Detailed Performance Metrics")
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            # Paddle control config
            paddle_config = model_data.get('paddle_config', {})
            
            st.subheader("🎮 Current Paddle Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Decision Timing:**")
                st.write(f"- Decision Interval: {paddle_config.get('decision_interval', 'N/A')} frames")
                st.write(f"- Action Commitment: {paddle_config.get('action_commitment_frames', 'N/A')} frames")
            
            with col2:
                st.write("**Movement Physics:**")
                st.write(f"- Max Velocity: {paddle_config.get('max_velocity', 'N/A')}")
                st.write(f"- Acceleration: {paddle_config.get('acceleration', 'N/A')}")
                st.write(f"- Deceleration: {paddle_config.get('deceleration_factor', 'N/A')}")
            
            with col3:
                st.write("**PID Controller:**")
                st.write(f"- Kp (Proportional): {paddle_config.get('kp', 'N/A')}")
                st.write(f"- Ki (Integral): {paddle_config.get('ki', 'N/A')}")
                st.write(f"- Kd (Derivative): {paddle_config.get('kd', 'N/A')}")
            
            # Learning parameters
            st.subheader("🧠 Learning Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Learning Rate", "0.001")
                st.metric("Discount Factor (γ)", "0.95")
                st.metric("Batch Size", "32")
            
            with col2:
                st.metric("Memory Size", "10,000 experiences")
                st.metric("Target Network Update", "Every 10 games")
                st.metric("Epsilon Decay", "0.995")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# Tab 5: Decision Explorer
# ============================================================================
with tab5:
    st.header("Decision-Making Analysis")
    st.write("Explore how the neural network makes decisions given specific game states")
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            W1 = model_data['W1']
            W2 = model_data['W2']
            b1 = model_data['b1']
            b2 = model_data['b2']
            
            st.subheader("🎮 Test a Game State")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Ball State:**")
                ball_x = st.slider("Ball X", -1.0, 1.0, 0.0)
                ball_y = st.slider("Ball Y", -1.0, 1.0, 0.0)
                ball_dx = st.slider("Ball DX", -1.0, 1.0, 0.5)
                ball_dy = st.slider("Ball DY", -1.0, 1.0, 0.0)
                ball_spin = st.slider("Ball Spin", -1.0, 1.0, 0.0)
            
            with col2:
                st.write("**My Paddle:**")
                my_y = st.slider("My Y Position", -1.0, 1.0, 0.0)
                my_vel = st.slider("My Velocity", -1.0, 1.0, 0.0)
            
            with col3:
                st.write("**Opponent:**")
                opp_y = st.slider("Opponent Y", -1.0, 1.0, 0.0)
                opp_vel = st.slider("Opponent Velocity", -1.0, 1.0, 0.0)
            
            # Create state vector
            state = np.array([[ball_x, ball_y, ball_dx, ball_dy, ball_spin, my_y, my_vel, opp_y, opp_vel]])
            
            # Forward pass
            z1 = state @ W1 + b1
            a1 = np.maximum(0, z1)  # ReLU
            z2 = a1 @ W2 + b2
            q_values = z2[0]
            
            # Display results
            st.subheader("🎯 Q-Values (Action Quality Estimates)")
            
            actions = ['DOWN', 'STAY', 'UP']
            best_action = np.argmax(q_values)
            
            col1, col2, col3 = st.columns(3)
            
            for i, (action, q_val) in enumerate(zip(actions, q_values)):
                with [col1, col2, col3][i]:
                    is_best = i == best_action
                    delta = "🏆 SELECTED" if is_best else ""
                    st.metric(
                        f"Action: {action}", 
                        f"{q_val:.4f}",
                        delta=delta
                    )
            
            # Q-value visualization
            fig = go.Figure(data=[
                go.Bar(x=actions, y=q_values, 
                      marker_color=['green' if i == best_action else 'lightgray' for i in range(3)])
            ])
            fig.update_layout(title="Q-Value Comparison", yaxis_title="Q-Value")
            st.plotly_chart(fig, use_container_width=True)
            
            # Hidden layer activation
            st.subheader("🧠 Hidden Layer Activations")
            activations = a1[0]
            fig = go.Figure(data=[go.Bar(x=list(range(len(activations))), y=activations)])
            fig.update_layout(title=f"Hidden Neuron Activity ({len(activations)} neurons)", 
                            xaxis_title="Neuron Index",
                            yaxis_title="Activation")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ No model loaded")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Quick Guide")
st.sidebar.markdown("""
**Training Overview:** Monitor win/loss statistics and curriculum phases

**Model Insights:** Analyze neural network weights and feature importance

**Physics Monitor:** Validate game physics and ball trajectories

**Performance:** Review paddle configs and learning parameters

**Decision Explorer:** Test how the network responds to different game states
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Enable telemetry during visual training to populate the Physics Monitor tab")
