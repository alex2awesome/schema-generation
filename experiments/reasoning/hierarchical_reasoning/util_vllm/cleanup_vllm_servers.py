#!/usr/bin/env python3
"""
Utility script to clean up orphaned VLLM servers.

This script helps clean up VLLM servers that may have been left running
after keyboard interrupts or other unexpected exits.

Usage:
    python cleanup_vllm_servers.py
    python cleanup_vllm_servers.py --port 8000
    python cleanup_vllm_servers.py --kill-all
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def find_vllm_processes():
    """Find all VLLM server processes."""
    try:
        # Find processes with 'vllm.entrypoints.openai.api_server' in command line
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
            return pids
        return []
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to ps if pgrep not available
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            pids = []
            for line in result.stdout.split('\n'):
                if 'vllm.entrypoints.openai.api_server' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            pass
            return pids
        except:
            return []


def kill_process_group(pid):
    """Kill a process and its group."""
    try:
        # Try to kill the process group first
        if hasattr(os, 'killpg'):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(2)
                # Check if still alive and force kill
                try:
                    os.kill(pid, 0)  # Check if process exists
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        else:
            # Fallback for systems without killpg
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True
    except ProcessLookupError:
        return False
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False


def cleanup_pid_files():
    """Clean up any leftover PID files."""
    here = Path(__file__).parent
    pid_files = list(here.glob("vllm_server_*.pid"))
    
    for pid_file in pid_files:
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running
            try:
                os.kill(pid, 0)
                print(f"📋 Found PID file {pid_file.name} with running process {pid}")
                return pid
            except ProcessLookupError:
                # Process not running, remove stale PID file
                pid_file.unlink()
                print(f"🗑️  Removed stale PID file {pid_file.name}")
                
        except (ValueError, OSError):
            # Invalid PID file, remove it
            try:
                pid_file.unlink()
                print(f"🗑️  Removed invalid PID file {pid_file.name}")
            except:
                pass
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned VLLM servers")
    parser.add_argument(
        "--port",
        type=int,
        help="Specific port to clean up (will look for PID file)"
    )
    parser.add_argument(
        "--kill-all",
        action="store_true",
        help="Kill all VLLM server processes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force kill without confirmation"
    )
    
    args = parser.parse_args()
    
    print("🔍 Checking for VLLM servers...")
    
    # Clean up PID files and get any running processes from them
    pid_from_file = cleanup_pid_files()
    
    # Find all VLLM processes
    vllm_pids = find_vllm_processes()
    
    if args.port:
        print(f"🎯 Looking for VLLM server on port {args.port}...")
        here = Path(__file__).parent
        pid_file = here / f"vllm_server_{args.port}.pid"
        
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    target_pid = int(f.read().strip())
                
                if target_pid in vllm_pids:
                    if not args.force:
                        response = input(f"Kill VLLM server on port {args.port} (PID {target_pid})? [y/N]: ")
                        if response.lower() != 'y':
                            print("❌ Cancelled")
                            return
                    
                    print(f"🔪 Killing VLLM server PID {target_pid}...")
                    if kill_process_group(target_pid):
                        print("✅ Server killed successfully")
                        pid_file.unlink()
                    else:
                        print("❌ Failed to kill server")
                else:
                    print(f"❌ No running VLLM server found for port {args.port}")
                    pid_file.unlink()
                    
            except (ValueError, OSError) as e:
                print(f"❌ Error reading PID file: {e}")
        else:
            print(f"❌ No PID file found for port {args.port}")
    
    elif args.kill_all:
        if not vllm_pids:
            print("✅ No VLLM servers found")
            return
            
        print(f"🎯 Found {len(vllm_pids)} VLLM server process(es): {vllm_pids}")
        
        if not args.force:
            response = input(f"Kill all {len(vllm_pids)} VLLM server(s)? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                return
        
        killed_count = 0
        for pid in vllm_pids:
            print(f"🔪 Killing VLLM server PID {pid}...")
            if kill_process_group(pid):
                killed_count += 1
                print(f"✅ Killed PID {pid}")
            else:
                print(f"❌ Failed to kill PID {pid}")
        
        print(f"🎯 Killed {killed_count}/{len(vllm_pids)} VLLM servers")
        
        # Clean up any remaining PID files
        cleanup_pid_files()
    
    else:
        if vllm_pids:
            print(f"📋 Found {len(vllm_pids)} VLLM server process(es): {vllm_pids}")
            print("🔧 Use --kill-all to kill all servers or --port <port> to kill a specific one")
        else:
            print("✅ No VLLM servers found")
    
    # Show GPU usage
    try:
        print("\n📊 Current GPU usage:")
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Could not get GPU usage (nvidia-smi not available)")


if __name__ == "__main__":
    main() 