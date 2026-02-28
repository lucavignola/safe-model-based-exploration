#!/usr/bin/env python3
import sys
print("Testing SBSRL...")

try:
    from experiments.pendulum_gp_full_exp.experiment import run_sbsrl
    print("✓ Successfully imported SBSRL")
    
    # Try creating minimal experiment
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU
    
    print("✓ Starting minimal SBSRL test...")
    result = run_sbsrl(episode_length=1, num_particles=1, num_samples=2, num_elites=1)
    print("✓ SBSRL test completed successfully!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ SBSRL test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)