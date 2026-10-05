import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path.cwd()
RUN_ROOT = ROOT / "outputs/replan1-ablation/scene-0-9_policy-0"
SIM_PY = ROOT / "cache/venvs/robocasa/bin/python"

def stop_owned_process(process):
    """Only stop the child process started by this script."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

def occupied_port():
    result = subprocess.run(
        ["ss", "-H", "-ltn", "sport = :8000"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())

def interrupted(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, interrupted)

if RUN_ROOT.exists():
    raise SystemExit(f"目标已存在，停止以避免混合实验：{RUN_ROOT}")
if occupied_port():
    raise SystemExit("8000 已被占用；请在旧服务终端按 Ctrl+C。")

RUN_ROOT.mkdir(parents=True)
source_hashes = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in (
        "src/robocasa_eval/main.py",
        "src/robocasa_eval/serve_policy.py",
        "scripts/run_policy.sh",
    )
}
git_head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

sim_env = os.environ.copy()
sim_env.update({
    "CUDA_VISIBLE_DEVICES": "",
    "MUJOCO_GL": "osmesa",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "LD_LIBRARY_PATH": (
        str(ROOT / "cache/rendering/lib")
        + (":" + os.environ["LD_LIBRARY_PATH"]
           if os.environ.get("LD_LIBRARY_PATH") else "")
    ),
    "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
    "TMPDIR": str(ROOT / "cache/tmp"),
    "PYTHONPATH": str(ROOT / "src"),
})

summary = []
try:
    for seed in range(10):
        if occupied_port():
            raise RuntimeError("8000 已被占用；停止，不干扰现有服务。")

        directory = RUN_ROOT / f"seed-{seed}"
        directory.mkdir()
        metadata = {
            "task": "OpenDrawer",
            "split": "pretrain",
            "scene_seed": seed,
            "policy_seed": 0,
            "policy_rng_reset": "fresh_server_before_episode",
            "num_trials": 1,
            "replan_steps": 1,
            "resize_size": 224,
            "obj_registries": ["lightwheel"],
            "generative_textures": "100p",
            "gpu_index": 0,
            "git_head": git_head,
            "source_sha256": source_hashes,
        }
        (directory / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        server = None
        evaluator = None
        print(f"\n=== scene seed {seed}/9；policy seed 0 ===", flush=True)
        try:
            with (directory / "server.log").open("w") as server_log:
                # This script checks GPU 0 and port availability.
                # Its service defaults to --policy-seed=0.
                server = subprocess.Popen(
                    ["bash", "scripts/run_policy.sh", "0"],
                    cwd=ROOT, stdout=server_log,
                    stderr=subprocess.STDOUT,
                )

                deadline = time.monotonic() + 240
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"服务提前退出，请查看 {directory / 'server.log'}"
                        )
                    log = (directory / "server.log").read_text(
                        errors="replace"
                    )
                    if "server listening on 127.0.0.1:8000" in log:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("服务启动超过 240 秒，停止。")
                    time.sleep(1)

                command = [
                    str(SIM_PY), "-m", "robocasa_eval.main",
                    "--smoke-lightwheel-task", "OpenDrawer",
                    "--num-trials", "1",
                    "--seed", str(seed),
                    "--replan-steps", "1",
                    "--resize-size", "224",
                    "--trace-actions",
                    "--log-dir", str(directory),
                    "--host", "127.0.0.1",
                    "--port", "8000",
                ]
                with (directory / "evaluation.log").open("w") as eval_log:
                    evaluator = subprocess.Popen(
                        command, cwd=ROOT, env=sim_env,
                        stdout=eval_log, stderr=subprocess.STDOUT,
                    )
                    code = evaluator.wait(timeout=1800)
                    if code != 0:
                        raise RuntimeError(
                            f"评测退出码 {code}，请查看 "
                            f"{directory / 'evaluation.log'}"
                        )

                paths = list(directory.rglob("rollout_0_trace.jsonl"))
                if len(paths) != 1:
                    raise RuntimeError("没有取得唯一轨迹文件。")
                rows = [
                    json.loads(line)
                    for line in paths[0].read_text().splitlines()
                    if line.strip()
                ]
                if not rows:
                    raise RuntimeError("轨迹为空。")
                if any(
                    not row["post"].get("target_drawer")
                    for row in rows
                ):
                    raise RuntimeError("轨迹缺少目标抽屉信息。")

                opening = [
                    min(row["post"]["target_drawer"]["door_state"].values())
                    for row in rows
                ]
                result = {
                    "scene_seed": seed,
                    "policy_seed": 0,
                    "success": any(row["success"] for row in rows),
                    "steps": len(rows),
                    "peak_opening": max(opening),
                    "final_opening": opening[-1],
                    "prompt": rows[0]["prompt"],
                    "fixture_name": (
                        rows[0]["pre"]["target_drawer"]["fixture_name"]
                    ),
                    "trace": str(paths[0].relative_to(ROOT)),
                }
                summary.append(result)
                (RUN_ROOT / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            stop_owned_process(evaluator)
            stop_owned_process(server)

            # Allow our service's GPU allocation to be released.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                used = subprocess.check_output(
                    ["nvidia-smi", "--id=0",
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                if used == "0":
                    break
                time.sleep(1)

    successes = sum(item["success"] for item in summary)
    print(f"\n完成：{successes}/{len(summary)} 成功。", flush=True)
    print(f"汇总：{RUN_ROOT / 'summary.json'}", flush=True)
except KeyboardInterrupt:
    print("\n已中止；已完成的结果保留，自己的子进程已清理。")
    raise SystemExit(130)
