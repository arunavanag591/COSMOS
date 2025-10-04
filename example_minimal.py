"""
Minimal COSMOS Example
======================

This example demonstrates how to use COSMOS for real-time odor simulation
with just a few lines of code.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simple import - no need to handle data paths or imports
import cosmos

def main():
    """
    Minimal example showing real-time odor simulation using COSMOS.
    
    This replaces the complex setup from the original example with a single line.
    """
    
    # Create predictor with one line - handles all data loading internally
    model = cosmos.predictor('desert-hws')
    
    # Simulate a trajectory
    dt = 0.005
    tsim = np.arange(0, 10, dt)
    x_pos = np.sin(tsim * 2 * np.pi * 0.5) + 2
    y_pos = np.sin(tsim * 2 * np.pi * 4)
    
    # Run the odor simulator
    odors = []
    for i in range(len(tsim)):
        current_odor = model.step_update(x_pos[i], y_pos[i], dt)
        odors.append(current_odor)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(tsim, x_pos)
    plt.title('X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    
    plt.subplot(2, 2, 2)
    plt.plot(tsim, y_pos)
    plt.title('Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    
    plt.subplot(2, 2, 3)
    plt.plot(x_pos, y_pos)
    plt.title('Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    
    plt.subplot(2, 2, 4)
    plt.plot(tsim, odors)
    plt.title('Odor Concentration')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Simulated {len(odors)} odor samples")
    print(f"Mean concentration: {np.mean(odors):.3f}")
    print(f"Max concentration: {np.max(odors):.3f}")
    print(f"Min concentration: {np.min(odors):.3f}")


def list_models_example():
    """Show available models."""
    print("Available COSMOS models:")
    models = cosmos.list_available_models()
    for model in models:
        print(f"  - {model['name']}: {model['description']}")


if __name__ == "__main__":
    # Show available models
    list_models_example()
    print()
    
    # Run the main example
    main()