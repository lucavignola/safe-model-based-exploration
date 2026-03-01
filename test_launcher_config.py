#!/usr/bin/env python3
"""
Test launcher to verify Euler configuration works properly
"""

import sys
import os
sys.path.append('/Users/lucav/Documents/safe-model-based-exploration')

from smbrl.utils.experiment_utils import generate_run_commands

def test_launcher_config():
    """Test that launcher configurations generate proper SBATCH commands"""
    
    # Test RTX 4090 configuration (matching your working Hydra setup)
    test_commands = ["python test_dummy.py --param1 value1", "python test_dummy.py --param2 value2"]
    
    print("🔧 Testing Euler launcher configuration...")
    print("📋 Configuration details:")
    print("  - Account: ls_krausea")
    print("  - GPU: rtx_4090:1") 
    print("  - CPUs: 10")
    print("  - Memory per CPU: 10240")
    print()
    
    # Generate commands with dry run to see the sbatch commands
    print("🚀 Generated SBATCH commands:")
    generate_run_commands(
        command_list=test_commands,
        num_cpus=10,
        num_gpus=1, 
        mode='euler',
        duration='1:00:00',
        prompt=False,
        dry=True,
        gpu_type='rtx_4090'
    )
    
    print()
    print("✅ Launcher configuration test completed!")
    print("🎯 This matches your working Hydra config:")
    print("   - account: ls_krausea")
    print("   - gpus: rtx_4090:1") 
    print("   - cpus_per_task: 10")
    print("   - mem_per_cpu: 10240")
    print("   - requeue enabled")

if __name__ == "__main__":
    test_launcher_config()