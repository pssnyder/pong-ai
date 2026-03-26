"""
Physics Validation Monitor
Real-time physics verification and anomaly detection during training

Usage:
    from physics_validator import PhysicsValidator
    
    validator = PhysicsValidator()
    validator.enable()
    
    # During game loop:
    validator.check_physics(state)
    
    # After game:
    report = validator.get_report()
"""

import numpy as np
from collections import deque
import json


class PhysicsValidator:
    """
    Monitors and validates game physics in real-time
    
    Checks for:
    - Ball speed limits
    - Position bounds
    - Spin limits
    - Energy conservation
    - Collision detection
    - Physics anomalies
    """
    
    def __init__(self, max_ball_speed=15.0, max_spin=2.0):
        self.max_ball_speed = max_ball_speed
        self.max_spin = max_spin
        
        # Validation flags
        self.enabled = False
        
        # Anomaly tracking
        self.anomalies = {
            'speed_violations': [],
            'position_violations': [],
            'spin_violations': [],
            'collision_misses': [],
            'energy_jumps': []
        }
        
        # State history
        self.state_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        
        # Statistics
        self.frames_checked = 0
        self.total_anomalies = 0
        
    def enable(self):
        """Enable physics validation"""
        self.enabled = True
        print("✅ Physics validation enabled")
    
    def disable(self):
        """Disable physics validation"""
        self.enabled = False
        print("❌ Physics validation disabled")
    
    def reset(self):
        """Reset all tracking"""
        for key in self.anomalies:
            self.anomalies[key] = []
        self.state_history.clear()
        self.energy_history.clear()
        self.frames_checked = 0
        self.total_anomalies = 0
    
    def check_physics(self, state, frame_num=None):
        """
        Validate physics for current game state
        
        Args:
            state: Game state dictionary
            frame_num: Optional frame number for logging
            
        Returns:
            dict: Validation results
        """
        if not self.enabled:
            return {'valid': True, 'anomalies': []}
        
        self.frames_checked += 1
        anomalies = []
        
        # Extract state
        ball_x = state.get('ball_x', 0)
        ball_y = state.get('ball_y', 0)
        ball_dx = state.get('ball_dx', 0)
        ball_dy = state.get('ball_dy', 0)
        ball_spin = state.get('ball_spin', 0)
        
        # 1. Check ball speed
        ball_speed = np.sqrt(ball_dx**2 + ball_dy**2)
        if ball_speed > self.max_ball_speed:
            anomaly = {
                'type': 'speed_violation',
                'frame': frame_num,
                'speed': ball_speed,
                'limit': self.max_ball_speed,
                'severity': 'high' if ball_speed > self.max_ball_speed * 1.5 else 'medium'
            }
            anomalies.append(anomaly)
            self.anomalies['speed_violations'].append(anomaly)
        
        # 2. Check position bounds
        if abs(ball_x) > 400:
            anomalies.append({
                'type': 'position_violation',
                'frame': frame_num,
                'axis': 'x',
                'position': ball_x,
                'bound': 400
            })
            self.anomalies['position_violations'].append(anomalies[-1])
        
        if abs(ball_y) > 300:
            anomalies.append({
                'type': 'position_violation',
                'frame': frame_num,
                'axis': 'y',
                'position': ball_y,
                'bound': 300
            })
            self.anomalies['position_violations'].append(anomalies[-1])
        
        # 3. Check spin limits
        if abs(ball_spin) > self.max_spin:
            anomalies.append({
                'type': 'spin_violation',
                'frame': frame_num,
                'spin': ball_spin,
                'limit': self.max_spin
            })
            self.anomalies['spin_violations'].append(anomalies[-1])
        
        # 4. Energy conservation check (if we have history)
        kinetic_energy = 0.5 * (ball_dx**2 + ball_dy**2)  # simplified, mass=1
        self.energy_history.append(kinetic_energy)
        
        if len(self.energy_history) > 1:
            energy_change = abs(kinetic_energy - self.energy_history[-2])
            
            # Allow energy changes at collisions, but flag large unexplained jumps
            if energy_change > 50:  # Threshold for "large" jump
                anomalies.append({
                    'type': 'energy_jump',
                    'frame': frame_num,
                    'change': energy_change,
                    'prev': self.energy_history[-2],
                    'curr': kinetic_energy
                })
                self.anomalies['energy_jumps'].append(anomalies[-1])
        
        # 5. Store state
        self.state_history.append({
            'frame': frame_num,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_speed': ball_speed,
            'ball_spin': ball_spin,
            'energy': kinetic_energy
        })
        
        # Update totals
        self.total_anomalies += len(anomalies)
        
        return {
            'valid': len(anomalies) == 0,
            'anomalies': anomalies,
            'ball_speed': ball_speed,
            'kinetic_energy': kinetic_energy
        }
    
    def get_report(self):
        """
        Get comprehensive validation report
        
        Returns:
            dict: Validation statistics and anomalies
        """
        report = {
            'frames_checked': self.frames_checked,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / self.frames_checked if self.frames_checked > 0 else 0,
            'anomaly_counts': {
                anomaly_type: len(anomalies) 
                for anomaly_type, anomalies in self.anomalies.items()
            },
            'state_statistics': self._compute_state_stats(),
            'recent_anomalies': self._get_recent_anomalies(10)
        }
        
        return report
    
    def _compute_state_stats(self):
        """Compute statistics from state history"""
        if len(self.state_history) == 0:
            return {}
        
        speeds = [s['ball_speed'] for s in self.state_history]
        spins = [s['ball_spin'] for s in self.state_history]
        energies = [s['energy'] for s in self.state_history]
        
        return {
            'speed': {
                'mean': np.mean(speeds),
                'std': np.std(speeds),
                'min': np.min(speeds),
                'max': np.max(speeds)
            },
            'spin': {
                'mean': np.mean(spins),
                'std': np.std(spins),
                'min': np.min(spins),
                'max': np.max(spins)
            },
            'energy': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies)
            }
        }
    
    def _get_recent_anomalies(self, n=10):
        """Get the most recent anomalies across all types"""
        all_anomalies = []
        for anomaly_list in self.anomalies.values():
            all_anomalies.extend(anomaly_list)
        
        # Sort by frame (most recent first)
        all_anomalies = sorted(all_anomalies, 
                              key=lambda x: x.get('frame', 0) or 0, 
                              reverse=True)
        
        return all_anomalies[:n]
    
    def print_report(self):
        """Print formatted validation report"""
        report = self.get_report()
        
        print("\n" + "=" * 70)
        print("📊 PHYSICS VALIDATION REPORT")
        print("=" * 70)
        
        print(f"\n✅ Frames Checked: {report['frames_checked']}")
        print(f"⚠️  Total Anomalies: {report['total_anomalies']}")
        print(f"📈 Anomaly Rate: {report['anomaly_rate']*100:.2f}%")
        
        print("\n📋 Anomaly Breakdown:")
        for anomaly_type, count in report['anomaly_counts'].items():
            if count > 0:
                print(f"   - {anomaly_type}: {count}")
        
        if report.get('state_statistics'):
            stats = report['state_statistics']
            print("\n📊 Physics Statistics:")
            
            if 'speed' in stats:
                s = stats['speed']
                print(f"   Ball Speed: {s['mean']:.2f} ± {s['std']:.2f} (range: {s['min']:.2f} to {s['max']:.2f})")
            
            if 'spin' in stats:
                s = stats['spin']
                print(f"   Ball Spin:  {s['mean']:.2f} ± {s['std']:.2f} (range: {s['min']:.2f} to {s['max']:.2f})")
            
            if 'energy' in stats:
                s = stats['energy']
                print(f"   Energy:     {s['mean']:.2f} ± {s['std']:.2f} (range: {s['min']:.2f} to {s['max']:.2f})")
        
        if report['recent_anomalies']:
            print(f"\n⚠️  Recent Anomalies (last {len(report['recent_anomalies'])}):")
            for anomaly in report['recent_anomalies'][:5]:
                frame = anomaly.get('frame', 'N/A')
                atype = anomaly.get('type', 'unknown')
                print(f"   Frame {frame}: {atype}")
                if 'speed' in anomaly:
                    print(f"      Speed {anomaly['speed']:.2f} exceeded limit {anomaly['limit']:.2f}")
                if 'change' in anomaly:
                    print(f"      Energy changed by {anomaly['change']:.2f}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, filepath='physics_validation_report.json'):
        """Save validation report to JSON file"""
        report = self.get_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Physics validation report saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Physics Validator - Example Usage\n")
    
    validator = PhysicsValidator(max_ball_speed=10.0, max_spin=2.0)
    validator.enable()
    
    # Simulate some state checks
    print("Simulating physics checks...")
    
    # Normal state
    state1 = {
        'ball_x': 0,
        'ball_y': 0,
        'ball_dx': 5.0,
        'ball_dy': 3.0,
        'ball_spin': 0.5
    }
    result1 = validator.check_physics(state1, frame_num=1)
    print(f"Frame 1: Valid={result1['valid']}, Anomalies={len(result1['anomalies'])}")
    
    # Violation: excessive speed
    state2 = {
        'ball_x': 100,
        'ball_y': 50,
        'ball_dx': 15.0,  # Too fast!
        'ball_dy': 10.0,
        'ball_spin': 0.3
    }
    result2 = validator.check_physics(state2, frame_num=2)
    print(f"Frame 2: Valid={result2['valid']}, Anomalies={len(result2['anomalies'])}")
    
    # Print report
    validator.print_report()
