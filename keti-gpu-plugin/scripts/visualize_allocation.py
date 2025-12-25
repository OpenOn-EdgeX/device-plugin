#!/usr/bin/env python3
"""
KETI Allocator 실시간 시각화

EdgeNode의 KETI Allocator에서 상태를 가져와 시각화합니다.
"""

import subprocess
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def get_allocator_state():
    """kubectl exec로 Allocator 상태 조회"""
    # First get pod name
    get_pod_cmd = [
        "kubectl", "get", "pods", "-n", "edge-system",
        "-l", "app=keti-allocator",
        "-o", "jsonpath={.items[0].metadata.name}"
    ]
    pod_result = subprocess.run(get_pod_cmd, capture_output=True, text=True)
    if pod_result.returncode != 0:
        return None
    pod_name = pod_result.stdout.strip()

    cmd = [
        "kubectl", "exec", "-n", "edge-system", pod_name, "--",
        "python3", "-c",
        "import urllib.request; resp = urllib.request.urlopen('http://127.0.0.1:7070/state'); print(resp.read().decode())"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None

def get_allocation_log():
    """kubectl exec로 Allocation 로그 조회"""
    # First get pod name
    get_pod_cmd = [
        "kubectl", "get", "pods", "-n", "edge-system",
        "-l", "app=keti-allocator",
        "-o", "jsonpath={.items[0].metadata.name}"
    ]
    pod_result = subprocess.run(get_pod_cmd, capture_output=True, text=True)
    if pod_result.returncode != 0:
        return None
    pod_name = pod_result.stdout.strip()

    cmd = [
        "kubectl", "exec", "-n", "edge-system", pod_name, "--",
        "python3", "-c",
        "import urllib.request; resp = urllib.request.urlopen('http://127.0.0.1:7070/allocation_log'); print(resp.read().decode())"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None

def visualize_state(state, output_path="/root/workspace/Resource_Scheduler/allocation_live.png"):
    """현재 할당 상태 시각화"""
    gpus = state.get("gpus", {})
    tenants = state.get("tenants", {})
    ts = state.get("ts", time.time())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'KETI GPU Resource Allocation - Live View\n{datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")}',
                 fontsize=14, fontweight='bold')

    # Left: GPU SM usage bar chart
    ax1 = axes[0]
    for gpu_id_str, gpu_info in gpus.items():
        gpu_id = int(gpu_id_str)
        total = gpu_info.get("total_sms", 84)
        used = gpu_info.get("used_sms", 0)
        free = total - used

        # Stack bar for each tenant
        bottom = 0
        colors = plt.cm.Set2.colors
        i = 0
        for tid, tinfo in tenants.items():
            if tinfo.get("gpu_id") == gpu_id:
                sm_count = tinfo.get("sm_count", 0)
                label = tid.split("/")[-1] if "/" in tid else tid
                ax1.barh(f"GPU {gpu_id}", sm_count, left=bottom,
                        color=colors[i % len(colors)],
                        label=f"{label} ({sm_count} SMs)", edgecolor='black', linewidth=1)
                # Add text
                if sm_count > 5:
                    ax1.text(bottom + sm_count/2, gpu_id, f"{label}\n{sm_count}",
                            ha='center', va='center', fontsize=9, fontweight='bold')
                bottom += sm_count
                i += 1

        # Free space
        if free > 0:
            ax1.barh(f"GPU {gpu_id}", free, left=bottom,
                    color='#90EE90', label=f"Free ({free} SMs)",
                    edgecolor='black', linewidth=1, alpha=0.5)
            ax1.text(bottom + free/2, gpu_id, f"Free\n{free}",
                    ha='center', va='center', fontsize=9, color='gray')

    ax1.set_xlabel("SM Count", fontsize=12)
    ax1.set_xlim(0, 90)
    ax1.set_title("GPU SM Allocation by Tenant", fontsize=12, fontweight='bold')
    ax1.axvline(x=84, color='red', linestyle='--', linewidth=2, label='Total (84 SMs)')
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: Pie chart
    ax2 = axes[1]
    if tenants:
        labels = []
        sizes = []
        colors_pie = plt.cm.Set2.colors

        for tid, tinfo in tenants.items():
            sm_count = tinfo.get("sm_count", 0)
            sm_pct = tinfo.get("sm_pct", 0)
            if sm_count:
                label = tid.split("/")[-1] if "/" in tid else tid
                labels.append(f"{label}\n{sm_count} SMs ({sm_pct:.0f}%)")
                sizes.append(sm_count)

        # Add free SMs
        total_sms = 84
        total_used = sum(sizes)
        free_sms = total_sms - total_used
        if free_sms > 0:
            labels.append(f"Free\n{free_sms} SMs ({free_sms*100/total_sms:.0f}%)")
            sizes.append(free_sms)
            colors_list = list(colors_pie[:len(sizes)-1]) + ['#90EE90']
        else:
            colors_list = list(colors_pie[:len(sizes)])

        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors_list,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes),
            shadow=True
        )
        ax2.set_title("SM Distribution", fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No tenants allocated", ha='center', va='center', fontsize=14)
        ax2.set_title("SM Distribution", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def visualize_timeline(log_data, output_path="/root/workspace/Resource_Scheduler/allocation_timeline_live.png"):
    """할당 타임라인 시각화"""
    log = log_data.get("log", [])
    if not log:
        print("No allocation log data")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Process log into timeline
    tenant_data = {}
    all_times = []
    base_time = log[0]["ts"] if log else time.time()

    for entry in log:
        t = entry["ts"] - base_time
        all_times.append(t)
        tenant = entry["tenant"].split("/")[-1] if "/" in entry["tenant"] else entry["tenant"]
        event = entry["event"]
        sm_count = entry.get("sm_count", 0)

        if tenant not in tenant_data:
            tenant_data[tenant] = {"times": [], "sms": [], "events": []}

        tenant_data[tenant]["times"].append(t)
        if event == "allocate":
            tenant_data[tenant]["sms"].append(sm_count)
        elif event == "release":
            tenant_data[tenant]["sms"].append(0)
        tenant_data[tenant]["events"].append(event)

    # Plot
    colors = plt.cm.Set2.colors
    for i, (tenant, data) in enumerate(tenant_data.items()):
        times = data["times"]
        sms = data["sms"]
        color = colors[i % len(colors)]

        # Extend to current time with step plot
        extended_times = times + [all_times[-1] + 1]
        extended_sms = sms + [sms[-1] if sms else 0]

        ax.fill_between(extended_times, 0, extended_sms, step='post',
                       alpha=0.6, color=color, label=tenant)
        ax.step(extended_times, extended_sms, where='post',
               color=color, linewidth=2)

        # Mark events
        for t, sm, ev in zip(times, sms, data["events"]):
            marker = '^' if ev == 'allocate' else 'v'
            ax.scatter(t, sm, marker=marker, s=100, color=color, edgecolors='black', zorder=5)

    ax.axhline(y=84, color='red', linestyle='--', linewidth=2, label='Total SMs (84)')
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("SM Count", fontsize=12)
    ax.set_title("SM Allocation Timeline - Real Events", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("KETI Allocator Live Visualization")
    print("=" * 60)

    # Get current state
    print("\n[1] Fetching allocation state from EdgeNode...")
    state = get_allocator_state()
    if state:
        print(f"    GPUs: {list(state.get('gpus', {}).keys())}")
        print(f"    Tenants: {list(state.get('tenants', {}).keys())}")

        # Visualize state
        print("\n[2] Generating state visualization...")
        visualize_state(state)
    else:
        print("    Failed to get state")

    # Get allocation log
    print("\n[3] Fetching allocation log...")
    log_data = get_allocation_log()
    if log_data and log_data.get("log"):
        print(f"    Log entries: {len(log_data['log'])}")

        # Visualize timeline
        print("\n[4] Generating timeline visualization...")
        visualize_timeline(log_data)
    else:
        print("    No log data or failed to fetch")

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - /root/workspace/Resource_Scheduler/allocation_live.png")
    print("  - /root/workspace/Resource_Scheduler/allocation_timeline_live.png")

if __name__ == "__main__":
    main()
