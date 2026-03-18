# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# References
# - https://www.kaggle.com/code/huikang/arc-agi-2-code-approach
# - https://www.kaggle.com/code/huikang/r1-distill-qwen-tir
# ```
# uv run python3 kaggle.py
# ```

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Configuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:53:22.703099Z","iopub.execute_input":"2025-12-29T08:53:22.703191Z","iopub.status.idle":"2025-12-29T08:53:22.705880Z","shell.execute_reply.started":"2025-12-29T08:53:22.703180Z","shell.execute_reply":"2025-12-29T08:53:22.705539Z"}}
# Please check
# - Accelerator
# - Internet
# - These five variables
# - Check notebook name
# - On the save and run page - Advanced settings - Run with GPU for this session
serve_vllm_on_kaggle = True
run_all_questions_on_kaggle = False  # ignored for submissions

# additional settings
save_communication_enabled = True
maybe_collaborate_enabled = True
replication_count_for_commit_runs = 50
model_path = "/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/14"

assert serve_vllm_on_kaggle is True

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:53:22.706627Z","iopub.execute_input":"2025-12-29T08:53:22.706743Z","iopub.status.idle":"2025-12-29T08:53:22.717113Z","shell.execute_reply.started":"2025-12-29T08:53:22.706733Z","shell.execute_reply":"2025-12-29T08:53:22.716712Z"}}
import os
import time

start_time = time.time()
total_available_time = (9 * 60 + 58) * 60
final_cutoff_time = start_time + total_available_time


def is_on_kaggle_commit() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_interactive() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_submission() -> bool:
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


def is_on_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


REMOTE_VLLM_URL = "NOT_AVAILABLE"
if is_on_kaggle() and serve_vllm_on_kaggle:
    # Do not use remote LLM, do not attempt to read secrets
    pass
else:
    # Does not work without Internet or on submission
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    REMOTE_VLLM_URL = secrets.get_secret("REMOTE_VLLM_URL")

if is_on_kaggle():
    # set to False on Kaggle so it does not cause issues with creating many files
    save_communication_enabled = False

if is_on_kaggle_submission():
    total_available_time = (4 * 60 + 58) * 60  # 5 hours

# Some debugger warnings on Kaggle
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["NO_COLOR"] = "1"

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:53:22.717538Z","iopub.execute_input":"2025-12-29T08:53:22.717650Z","iopub.status.idle":"2025-12-29T08:53:22.726871Z","shell.execute_reply.started":"2025-12-29T08:53:22.717640Z","shell.execute_reply":"2025-12-29T08:53:22.726483Z"}}
# print settings
print(f"{is_on_kaggle()=}")
print(f"{is_on_kaggle_interactive()=}")
print(f"{is_on_kaggle_commit()=}")
print(f"{serve_vllm_on_kaggle=}")
print(f"{run_all_questions_on_kaggle=}")
print(f"{REMOTE_VLLM_URL[::-1][:13][::-1]=}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Setup

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:53:22.727306Z","iopub.execute_input":"2025-12-29T08:53:22.727415Z","iopub.status.idle":"2025-12-29T08:53:56.292445Z","shell.execute_reply.started":"2025-12-29T08:53:22.727405Z","shell.execute_reply":"2025-12-29T08:53:56.291952Z"}}
import subprocess

if is_on_kaggle():
    subprocess.run(
        [
            "pip",
            "uninstall",
            "--yes",
            "tensorflow",
            "matplotlib",
            "keras",
            "scikit-learn",
        ]
    )

# %% [code] {"_kg_hide-output":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:53:56.292993Z","iopub.execute_input":"2025-12-29T08:53:56.293130Z","iopub.status.idle":"2025-12-29T08:55:09.839897Z","shell.execute_reply.started":"2025-12-29T08:53:56.293116Z","shell.execute_reply":"2025-12-29T08:55:09.839470Z"}}
import os
import subprocess


def cache_model(
    path, exts=(".bin", ".pt", ".safetensors"), num_workers=None, chunk_mb=256
):
    """
    Pre-read model weight files into the OS page cache to speed up later loads.

    Args:
        path        : Directory containing model files, or a single file path.
        exts        : File extensions treated as model weight files.
        num_workers : Number of threads (default = min(CPU cores, 8)).
        chunk_mb    : Size of each read chunk in MB.
    Returns:
        Total bytes read (int).
    """
    import os
    import multiprocessing
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath):
        """Sequentially read an entire file in chunks."""
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    # Collect files to read
    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        raise ValueError(f"No model files found under: {path}")

    # Decide number of worker threads
    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")

    t0 = time.time()
    total_bytes = 0

    # Read files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f"[cache_model] total read ≈ {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")

    return total_bytes


if is_on_kaggle():
    cache_model(
        model_path,
        num_workers=16,
        chunk_mb=1024,
    )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-12-29T08:55:09.840750Z","iopub.execute_input":"2025-12-29T08:55:09.840894Z","iopub.status.idle":"2025-12-29T08:55:26.265314Z","shell.execute_reply.started":"2025-12-29T08:55:09.840882Z","shell.execute_reply":"2025-12-29T08:55:26.264840Z"}}
import numpy as np
import torch

cutoff_times: list[float] = [
    float(x)
    for x in np.linspace(
        final_cutoff_time, start_time + total_available_time / 3, 50 + 1
    )
]  # will be repartitioned at solve
cutoff_times.pop()


def reallocate_time(cutoff_times: list[float]) -> None:
    """Reallocate cutoff_times in-place with equal intervals, except last interval could be larger."""
    n = len(cutoff_times)
    if n <= 1:
        return

    # n = 1 -> no change
    # n = 2 -> 1 + log10(2) / 2 = 1.15 times the other intervals
    # n = 10 -> 1 + log10(10) / 2 = 1.5 times the other intervals
    # n = 50 -> 1 + log10(50) / 2 = 2.35 times the other intervals
    # interval = (cutoff_times[0] - time.time()) / (n + np.log10(n))
    interval = 750
    cutoff_times[-1] = min(cutoff_times[-1], time.time() + interval)


from datetime import datetime

# Create timestamped run directory with subdirectories
os.makedirs("runs", exist_ok=True)
RUN_DIR = f"runs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
SOLUTIONS_DIR = f"{RUN_DIR}/solutions"
FINDINGS_DIR = f"{RUN_DIR}/findings"
COMMUNICATIONS_DIR = f"{RUN_DIR}/communications"

os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(FINDINGS_DIR, exist_ok=True)
os.makedirs(COMMUNICATIONS_DIR, exist_ok=True)

# Initialize stats CSV file
import pandas as pd

STATS_CSV_PATH = "stats.csv"  # ok to not gitignore
STATS_COLUMNS = [
    "question_id",
    "final_answer",
    "time_taken",
    "time_available",
    "active_solvers",
    "answers",
    "proposals",
    "answer_history",
    "proposal_history",
    "submission_history",
    "total_tokens",
    "backtrack_counts",
    "main_tokens",
    "total_main_tokens",
    "support_tokens",
    "total_support_tokens",
    "tool_use_counts",
    "num_findings",
    "num_acceptances",
    "num_rejections",
]

if __name__ == "__main__":
    pd.DataFrame(columns=pd.Index(STATS_COLUMNS)).to_csv(STATS_CSV_PATH, index=False)


def save_stats(
    question_id: str,
    final_answer: int,
    time_taken: float,
    time_available: float,
    active_solvers: set[int],
    answers: dict[int, int],
    proposals: dict[int, int],
    answer_history: dict[int, list[int | None]],
    proposal_history: dict[int, list[int | None]],
    submission_history: dict[int, list[int]],
    backtrack_counts: dict[int, int],
    main_tokens: dict[int, int],
    support_tokens: dict[int, int],
    tool_use_counts: dict[int, int],
    num_findings: int,
    num_acceptances: int,
    num_rejections: int,
) -> None:
    """Append stats for a question to the CSV file."""
    row_data = {
        "question_id": question_id,
        "final_answer": final_answer,
        "time_taken": round(time_taken, 1),
        "time_available": round(time_available, 1),
        "active_solvers": str(sorted(active_solvers)),
        "answers": str(answers),
        "proposals": str(proposals),
        "answer_history": str(dict(answer_history)),
        "proposal_history": str(dict(proposal_history)),
        "submission_history": str(dict(submission_history)),
        "total_tokens": sum(main_tokens.values()) + sum(support_tokens.values()),
        "backtrack_counts": str(dict(backtrack_counts)),
        "main_tokens": str(dict(main_tokens)),
        "total_main_tokens": sum(main_tokens.values()),
        "support_tokens": str(dict(support_tokens)),
        "total_support_tokens": sum(support_tokens.values()),
        "tool_use_counts": str(dict(tool_use_counts)),
        "num_findings": num_findings,
        "num_acceptances": num_acceptances,
        "num_rejections": num_rejections,
    }
    assert list(row_data.keys()) == STATS_COLUMNS, (
        f"Column mismatch: {list(row_data.keys())} != {STATS_COLUMNS}"
    )
    row = pd.DataFrame([row_data])
    row.to_csv(STATS_CSV_PATH, mode="a", header=False, index=False)


if is_on_kaggle():
    if serve_vllm_on_kaggle:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1
    else:
        # Check internet access is available when using remote inference
        import urllib.request
        from urllib.error import URLError

        try:
            urllib.request.urlopen("https://modal.com", timeout=5)
            print("Internet access confirmed")
        except (URLError, TimeoutError) as e:
            raise RuntimeError(
                "Internet access required when serve_vllm_on_kaggle=False"
            ) from e

        # Check that you are not wasting Kaggle GPUs
        assert not torch.cuda.is_available()
        assert torch.cuda.device_count() == 0

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Serve vLLM

# %% [code] {"execution":{"iopub.status.busy":"2025-12-29T08:55:49.519192Z","iopub.execute_input":"2025-12-29T08:55:49.519437Z","iopub.status.idle":"2025-12-29T08:55:49.529406Z","shell.execute_reply.started":"2025-12-29T08:55:49.519422Z","shell.execute_reply":"2025-12-29T08:55:49.528995Z"},"jupyter":{"outputs_hidden":false}}
if is_on_kaggle():
    subprocess.run(["ls", "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"])

# %% [code] {"execution":{"iopub.status.busy":"2025-12-29T08:55:49.672010Z","iopub.execute_input":"2025-12-29T08:55:49.672206Z","iopub.status.idle":"2025-12-29T08:55:49.675012Z","shell.execute_reply.started":"2025-12-29T08:55:49.672193Z","shell.execute_reply":"2025-12-29T08:55:49.674598Z"},"jupyter":{"outputs_hidden":false}}
with open("a-vllm.log", "w") as f:
    f.write("")

# %% [code] {"execution":{"iopub.status.busy":"2025-12-29T08:55:50.842639Z","iopub.execute_input":"2025-12-29T08:55:50.842856Z","iopub.status.idle":"2025-12-29T08:55:50.850331Z","shell.execute_reply.started":"2025-12-29T08:55:50.842843Z","shell.execute_reply":"2025-12-29T08:55:50.849897Z"},"jupyter":{"outputs_hidden":false}}
import subprocess

num_generations = 8
max_model_len = 131072


def start_vllm_server() -> subprocess.Popen[bytes]:
    """Start vLLM server in the background"""
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "3"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#troubleshooting
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
        "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
    )

    command: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        f"{num_generations + 4}",  # this eats into available KV cache
        "--gpu-memory-utilization",
        "0.96",  # any higher may not have enough for graph capture
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--async-scheduling",  # https://docs.vllm.ai/en/latest/cli/run-batch/?h=async+scheduling#-scheduler-cls
        "--max-num-batched-tokens",  # affects KV cache available
        "2048",  # https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill
        "--max-model-len",
        f"{max_model_len}",
    ]

    # Start the process in the background
    with open("/kaggle/working/a-vllm.log", "w") as logfile:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )

    print("Logs: /kaggle/working/a-vllm.log")
    return process


# Start the server
if is_on_kaggle() and serve_vllm_on_kaggle:
    vllm_process: subprocess.Popen[bytes] = start_vllm_server()

# %% [code] {"execution":{"iopub.status.busy":"2025-12-29T08:55:51.046803Z","iopub.execute_input":"2025-12-29T08:55:51.047007Z","iopub.status.idle":"2025-12-29T08:55:56.972784Z","shell.execute_reply.started":"2025-12-29T08:55:51.046995Z","shell.execute_reply":"2025-12-29T08:55:56.972265Z"},"jupyter":{"outputs_hidden":false}}
import os

from openai import OpenAI, Stream
from openai.types import Completion

# Point the client to vLLM server (local on Kaggle, Modal otherwise)
if is_on_kaggle() and serve_vllm_on_kaggle:
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
else:
    os.environ["OPENAI_API_BASE"] = REMOTE_VLLM_URL
    if is_on_kaggle():
        # openai_harmony uses TIKTOKEN_ENCODINGS_BASE to read pre-downloaded files
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
            "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
        )
os.environ["OPENAI_API_KEY"] = "sk-local"  # any non-empty string

client: OpenAI = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"]
)

# %% [code] {"jupyter":{"outputs_hidden":false}}
import time


def await_client(printing: bool = False) -> None:
    for _ in range(15 * 60):
        time.sleep(1)
        try:
            model_list = client.models.list()
            if printing:
                print(model_list)
        except NameError:
            raise  # maybe you did not run the cell initializing client
        except Exception:
            continue
        break
    else:
        raise


if is_on_kaggle():
    # the server should now start in 10 minutes
    await_client()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Code execution

# %% [code] {"jupyter":{"outputs_hidden":false}}
class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel.
    Extracted from gpt_oss.tools.python_docker.docker_tool.
    Thread-safe: creates its own ZMQ context for use within a single thread.
    """

    def __init__(self, timeout: float = 8.0) -> None:
        import zmq
        from jupyter_client.blocking.client import BlockingKernelClient
        from jupyter_client.manager import KernelManager

        self._default_timeout = timeout
        # Create a dedicated ZMQ context for this session (thread-safe)
        self._zmq_context = zmq.Context()
        self._km = KernelManager(context=self._zmq_context)
        # Disable IPython history to avoid SQLite "database is locked" errors
        # when multiple kernels run concurrently
        self._km.start_kernel(extra_arguments=["--HistoryManager.enabled=False"])
        self._client: BlockingKernelClient = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=None)
        # Disable colors and use plain tracebacks to avoid IPython cascade errors
        # (IPython 7.x has a bug where get_records() returns None and find_recursion crashes)
        # Plain mode shows line numbers while avoiding the VerboseTB code path
        self._client.execute("%colors NoColor", store_history=False)
        self._client.execute("%xmode Plain", store_history=False)
        # Limit traceback depth to avoid huge output from deep recursion
        self._client.execute("import sys; sys.tracebacklimit = 10", store_history=False)
        # Track msg_id of a timed-out execution that may still be running
        self._pending_msg_id: str | None = None
        # Track if kernel may be stuck in uninterruptible C code
        self._kernel_may_be_stuck: bool = False

    def _drain_pending_output(self) -> str:
        """Drain output from a previous timed-out execution. Interrupts if still running.

        If the kernel doesn't respond to interrupt (e.g., stuck in C code like BLAS),
        restarts the kernel to ensure the next execution can proceed.
        """
        # Check if kernel is stuck from a previous immediate interrupt that didn't respond
        # This flag is ONLY set when we sent an interrupt and waited 2s without getting
        # a KeyboardInterrupt or idle status - meaning the kernel is truly stuck in C code
        if self._kernel_may_be_stuck:
            self._kernel_may_be_stuck = False
            # Kernel already didn't respond to interrupt - restart it
            self._restart_kernel()
            return "[Previous execution killed - kernel restarted due to unresponsive C code]\n"

        if self._pending_msg_id is None:
            return ""

        msg_id = self._pending_msg_id
        self._pending_msg_id = None
        client = self._client

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        execution_finished = False

        # Drain any available output without blocking long
        while True:
            try:
                msg = client.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                break

            parent_header = msg["parent_header"]
            if (
                not parent_header
                or "msg_id" not in parent_header
                or parent_header["msg_id"] != msg_id
            ):
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                text = content["text"]
                if content["name"] == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content["traceback"]
                stderr_parts.append("\n".join(traceback_data))
            elif msg_type in {"execute_result", "display_data"}:
                data = content["data"]
                if "text/plain" in data:
                    text = data["text/plain"]
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content["execution_state"] == "idle":
                execution_finished = True
                break

        # If still running, try to interrupt it
        if not execution_finished:
            self._km.interrupt_kernel()
            # Wait for interrupt to take effect (with timeout)
            interrupt_timeout = 2.0  # seconds to wait for interrupt response
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=interrupt_timeout)
                except queue.Empty:
                    # Kernel didn't respond to interrupt - it's stuck in C code
                    # Restart kernel to ensure next execution can proceed
                    self._restart_kernel()
                    return "[Previous execution killed - kernel restarted due to unresponsive C code]\n"
                parent_header = msg["parent_header"]
                if (
                    not parent_header
                    or "msg_id" not in parent_header
                    or parent_header["msg_id"] != msg_id
                ):
                    continue
                msg_type = msg["msg_type"]
                content = msg["content"]
                if msg_type == "stream":
                    text = content["text"]
                    if content["name"] == "stdout":
                        stdout_parts.append(text)
                    else:
                        stderr_parts.append(text)
                elif msg_type == "error":
                    traceback_data = content["traceback"]
                    stderr_parts.append("\n".join(traceback_data))
                elif msg_type == "status" and content["execution_state"] == "idle":
                    break

        # Drain shell channel
        while True:
            try:
                reply = client.get_shell_msg(timeout=0.1)
                parent_header = reply["parent_header"]
                if (
                    parent_header
                    and "msg_id" in parent_header
                    and parent_header["msg_id"] == msg_id
                ):
                    break
            except queue.Empty:
                break

        # Combine output
        output = "".join(stdout_parts)
        if stderr_parts:
            output = (
                f"{output.rstrip()}\n{''.join(stderr_parts)}"
                if output
                else "".join(stderr_parts)
            )

        if output.strip():
            end_marker = (
                "[End previous output]"
                if execution_finished
                else "[End previous output - interrupted]"
            )
            return f"[Previous execution output]\n{output.rstrip()}\n{end_marker}\n"
        return ""

    def _restart_kernel(self) -> None:
        """Restart the kernel, preserving the session but losing state."""
        import contextlib

        from jupyter_client.blocking.client import BlockingKernelClient

        print("[LocalJupyterSession] Restarting kernel due to unresponsive state")

        # Clear stuck flag
        self._kernel_may_be_stuck = False
        self._pending_msg_id = None

        # Stop current client channels
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        # Restart the kernel (kills current process, starts new one)
        self._km.restart_kernel(now=True)

        # Create new client and reconnect
        self._client: BlockingKernelClient = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=None)
        # Re-apply color/traceback settings
        self._client.execute("%colors NoColor", store_history=False)
        self._client.execute("%xmode Plain", store_history=False)
        self._client.execute("import sys; sys.tracebacklimit = 10", store_history=False)

    def execute(self, code: str, continue_executing_on_timeout: bool = False) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""
        # Drain any pending output from previous timed-out execution
        pending_output = self._drain_pending_output()

        client = self._client
        effective_timeout = self._default_timeout
        msg_id = client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty:
                if continue_executing_on_timeout:
                    # Deferred interruption: let kernel continue, interrupt on next execute()
                    self._pending_msg_id = msg_id
                    error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
                else:
                    # Immediate interruption - send interrupt and wait for response
                    self._km.interrupt_kernel()
                    # Wait up to 2 seconds for interrupt to take effect
                    interrupt_responded = False
                    interrupt_timeout = 2.0
                    while True:
                        try:
                            int_msg = client.get_iopub_msg(timeout=interrupt_timeout)
                        except queue.Empty:
                            # No response - kernel stuck in C code
                            break
                        parent_header = int_msg["parent_header"]
                        if (
                            not parent_header
                            or "msg_id" not in parent_header
                            or parent_header["msg_id"] != msg_id
                        ):
                            continue
                        int_msg_type = int_msg["msg_type"]
                        int_content = int_msg["content"]
                        if int_msg_type == "error":
                            # Got KeyboardInterrupt - kernel responded
                            traceback_data = int_content["traceback"]
                            stderr_parts.append("\n".join(traceback_data))
                            interrupt_responded = True
                        elif (
                            int_msg_type == "status"
                            and int_content["execution_state"] == "idle"
                        ):
                            interrupt_responded = True
                            break
                    if not interrupt_responded:
                        # Kernel didn't respond to interrupt - stuck in C code
                        self._kernel_may_be_stuck = True
                    error_msg = "[TIMEOUT] Execution interrupted."
                # Return partial output with timeout message
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                result = f"{partial_output.rstrip()}\n{error_msg}".lstrip()
                return f"{pending_output}{result}" if pending_output else result

            parent_header = msg["parent_header"]
            if (
                not parent_header
                or "msg_id" not in parent_header
                or parent_header["msg_id"] != msg_id
            ):
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                text = content["text"]
                if content["name"] == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content["traceback"]
                stderr_parts.append("\n".join(traceback_data))
            elif msg_type in {"execute_result", "display_data"}:
                data = content["data"]
                if "text/plain" in data:
                    text = data["text/plain"]
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content["execution_state"] == "idle":
                break

        # Drain the shell channel to capture final execution status
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty:
                if continue_executing_on_timeout:
                    # Shell channel timeout - use deferred interruption
                    self._pending_msg_id = msg_id
                    error_msg = "[TIMEOUT] Execution still running. Will drain remaining output on next call."
                else:
                    # Immediate interruption - kernel may be stuck in C code
                    self._km.interrupt_kernel()
                    self._kernel_may_be_stuck = (
                        True  # Will check and restart on next execute()
                    )
                    error_msg = "[TIMEOUT] Execution interrupted."
                partial_output = "".join(stdout_parts)
                if stderr_parts:
                    partial_output = (
                        f"{partial_output.rstrip()}\n{''.join(stderr_parts)}"
                        if partial_output
                        else "".join(stderr_parts)
                    )
                result = f"{partial_output.rstrip()}\n{error_msg}".lstrip()
                return f"{pending_output}{result}" if pending_output else result

            reply_parent = reply["parent_header"]
            if (
                reply_parent
                and "msg_id" in reply_parent
                and reply_parent["msg_id"] == msg_id
            ):
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = "[WARN] No output available. Use print() to output anything to stdout to receive the output"

        return f"{pending_output}{stdout}" if pending_output else stdout

    def close(self) -> None:
        import contextlib

        # Stop client channels first (closes ZMQ sockets)
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        # Shutdown kernel process
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)

        # Cleanup kernel manager resources
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

        # Destroy ZMQ context - use destroy() instead of term() for immediate cleanup
        with contextlib.suppress(Exception):
            self._zmq_context.destroy(linger=0)

    def __del__(self) -> None:
        # Guard against Python shutdown (when sys.meta_path is None)
        import sys

        if sys.meta_path is not None:
            self.close()


def execute_python_code(session: LocalJupyterSession, script: str) -> str:
    """Execute Python code in a stateful Jupyter session."""
    return session.execute(script)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Token processing

# %% [code] {"execution":{"iopub.status.busy":"2025-11-24T08:13:52.195997Z","iopub.execute_input":"2025-11-24T08:13:52.196102Z","iopub.status.idle":"2025-11-24T08:14:00.409293Z","shell.execute_reply.started":"2025-11-24T08:13:52.196092Z","shell.execute_reply":"2025-11-24T08:14:00.408863Z"},"jupyter":{"outputs_hidden":false}}
# Initialize openai-harmony encoding for GPT-OSS models
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
stop_token_ids: list[int] = list(harmony_encoding.stop_tokens_for_assistant_actions())

# Token IDs for <|return|> and <|end|> - used to fix up token stream when continuing conversation
# Per OpenAI Harmony docs: replace trailing <|return|> with <|end|> before appending next turn
# https://cookbook.openai.com/articles/openai-harmony
RETURN_TOKEN_ID = 200002
END_TOKEN_ID = 200007

# Python tool configuration for gpt-oss (extracted from gpt_oss.tools.python_docker.docker_tool)
# Using dangerously_use_local_jupyter backend - stateful execution via Jupyter kernel
import queue

from openai_harmony import Author, TextContent, ToolNamespaceConfig

# Stateful Python tool instruction (matches how the model was trained)
PYTHON_TOOL_INSTRUCTION = """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning.
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution. Internet access for this session is disabled.
""".strip()


python_tool_config = ToolNamespaceConfig(
    name="python", description=PYTHON_TOOL_INSTRUCTION, tools=[]
)


def make_python_tool_response(output: str, channel: str | None = None) -> Message:
    """Create a tool response message for the Python tool."""
    content = TextContent(text=output)
    author = Author(role=Role.TOOL, name="python")
    message = Message(author=author, content=[content]).with_recipient("assistant")
    if channel:
        message = message.with_channel(channel)
    return message


def build_prompt_token_ids(
    system_content: str,
    user_content: str,
    reasoning_effort: ReasoningEffort,
    enable_python_tool: bool = False,
) -> list[int]:
    """Convert system and user content to token IDs using harmony format."""
    system_content_obj = SystemContent.new().with_reasoning_effort(reasoning_effort)
    if enable_python_tool:
        # Enable Python tool using with_tools() for stateless mode
        system_content_obj = system_content_obj.with_tools(python_tool_config)
    system_message = Message.from_role_and_content(Role.SYSTEM, system_content_obj)
    developer_message = Message.from_role_and_content(
        Role.DEVELOPER, DeveloperContent.new().with_instructions(system_content)
    )
    user_message = Message.from_role_and_content(Role.USER, user_content)
    convo = Conversation.from_messages(
        [system_message, developer_message, user_message]
    )
    return list(
        harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    )


import random
import string


def save_communication(
    content: str, prefix: str = "", enabled: bool = save_communication_enabled
) -> str:
    """Save content to communications/ with a random 6 alphanumeric hash."""
    if not enabled:
        return ""
    hash_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    filename = f"{COMMUNICATIONS_DIR}/{prefix}{hash_str}.txt"
    with open(filename, "w") as f:
        f.write(content)
    return filename


def append_user_turn_token_ids(
    all_token_ids: list[int], user_content: str
) -> list[int]:
    """Append a new user turn to the token IDs."""
    new_user_message = Message.from_role_and_content(Role.USER, user_content)
    user_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([new_user_message]), Role.ASSISTANT
        )
    )
    return all_token_ids + user_tokens


import time


def append_tool_response_token_ids(
    all_token_ids: list[int], tool_response: Message
) -> list[int]:
    """Append a tool response to the token IDs."""
    tool_tokens = list(
        harmony_encoding.render_conversation_for_completion(
            Conversation.from_messages([tool_response]), Role.ASSISTANT
        )
    )
    return all_token_ids + tool_tokens

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T11:23:07.544491Z","iopub.execute_input":"2025-11-25T11:23:07.544919Z","iopub.status.idle":"2025-11-25T11:23:07.550311Z","shell.execute_reply.started":"2025-11-25T11:23:07.5449Z","shell.execute_reply":"2025-11-25T11:23:07.549876Z"}}
import os
import time

import requests
from cachetools import TTLCache, cached


@cached(cache=TTLCache(maxsize=51, ttl=5))
def get_gpu_kv_cache_usage(question_id: str | None = None) -> float:
    # Parse vLLM /metrics endpoint using configured base URL
    # https://docs.vllm.ai/en/latest/design/metrics/
    try:
        base_url = os.environ["OPENAI_API_BASE"]
        # Remove /v1 suffix to get metrics endpoint
        metrics_url = base_url.replace("/v1", "/metrics")
        resp = requests.get(metrics_url, timeout=5)
        for line in resp.text.split("\n"):
            # vllm:kv_cache_usage_perc is the metric for KV cache usage
            if line.startswith("vllm:kv_cache_usage_perc"):
                value = float(line.split()[-1])
                return value * 100  # convert to percentage
    except (requests.RequestException, ValueError, IndexError):
        pass
    return -1


question_id_to_latest_termination_time: dict[str, float] = {}


def maybe_terminate_solver_for_gpu_usage(
    question_id: str,
    gpu_usage_thresholds: tuple[float, ...] = (
        1.00,  # 0
        1.00,  # 1
        1.00,  # 2
        0.90,  # 3
        0.80,  # 4
        0.80,  # 5
        0.80,  # 6
        0.80,  # 7
        0.80,  # 8
        0.70,  # 9
        0.60,  # 10
    ),
) -> None:
    """Terminate excess solvers based on GPU KV cache usage.

    Args:
        question_id: The question being solved.
        gpu_usage_thresholds: Descending list of thresholds mapping to allowed solvers.
            E.g. [0.95, 0.85, 0.75, 0.65, 0.60, 0.55, 0.50, 0.45] means:
            > 0.95 -> 0 solver
            > 0.85 -> 1 solver
            > ...
            > 0.45 -> 7 solvers
    """
    allowed_active_solvers = (
        num_generations * 2
    )  # which is the number of concurrent queries

    gpu_kv_cache_usage = get_gpu_kv_cache_usage(question_id) / 100
    for allowed_active_solvers, threshold in enumerate(gpu_usage_thresholds):
        if gpu_kv_cache_usage > threshold:
            break

    active_solvers = question_id_to_active_solver_indexes[question_id]
    if len(active_solvers) <= allowed_active_solvers:
        return

    if question_id_to_latest_termination_time[question_id] + 25 > time.time():
        print(f"Skip terminating a solver at {100 * gpu_kv_cache_usage:.1f}")
        return

    print(f"Terminating a solver at {100 * gpu_kv_cache_usage:.1f}")
    solver_index_to_answer = question_id_to_solver_index_to_answer[question_id]
    print(f"{solver_index_to_answer=}")
    solver_index_to_proposal = question_id_to_solver_index_to_proposal[question_id]
    print(f"{solver_index_to_proposal=}")
    active_solver_indexes = question_id_to_active_solver_indexes[question_id]
    print(f"{active_solver_indexes=}")

    solver_index_to_tool_use_count = question_id_to_solver_index_to_tool_use_count[
        question_id
    ]

    # Sort by (has_attempted_proposal, has_answer_or_proposal, has_answer, tool_use_count)
    # Lower values = higher priority for termination
    sorted_solvers = sorted(
        active_solvers,
        key=lambda solver_index: (
            solver_index in question_id_to_proposed_solver_indexes[question_id],
            solver_index in solver_index_to_answer
            or solver_index in solver_index_to_proposal,
            solver_index in solver_index_to_answer,
            solver_index_to_tool_use_count[solver_index],
        ),
    )

    for active_solver_index in sorted_solvers:
        active_solvers.discard(active_solver_index)
        print(
            f"Terminated Solver {active_solver_index}, Solvers {active_solvers} remaining"
        )
        vote_answer(question_id)
        # only terminate one at a time
        question_id_to_latest_termination_time[question_id] = time.time()
        return

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:22:55.289753Z","iopub.execute_input":"2025-11-24T08:22:55.289878Z","iopub.status.idle":"2025-11-24T08:23:00.176618Z","shell.execute_reply.started":"2025-11-24T08:22:55.289861Z","shell.execute_reply":"2025-11-24T08:23:00.176183Z"}}
if is_on_kaggle_interactive():
    test_prompt_ids = build_prompt_token_ids(
        system_content="Reply your answer in \\boxed{}",
        user_content="How many r are there in strawberry?",
        reasoning_effort=ReasoningEffort.HIGH,
    )
    resp: Completion = client.completions.create(
        model="vllm-model",
        prompt=test_prompt_ids,
        max_tokens=1024,
        temperature=1.0,
        extra_body=dict(
            min_p=0.02, stop_token_ids=stop_token_ids, return_token_ids=True
        ),
    )
    save_communication(harmony_encoding.decode(test_prompt_ids), prefix="test-")

    print("Token IDs:", resp.choices[0].token_ids)  # type: ignore[attr-defined]

    print(resp.choices[0].text)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Text processing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:23:00.326586Z","iopub.execute_input":"2025-11-24T08:23:00.326712Z","iopub.status.idle":"2025-11-24T08:23:00.333663Z","shell.execute_reply.started":"2025-11-24T08:23:00.326702Z","shell.execute_reply":"2025-11-24T08:23:00.333234Z"}}
def extract_boxed_text(text: str) -> str:
    """Extract text inside \\boxed{} from LaTeX-formatted text"""
    import re

    pattern: str = r"oxed{(.*?)}"
    matches: list[str] = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def is_valid_answer_string(text: str) -> bool:
    try:
        if int(text) == float(text):
            answer = int(text)
            if 0 <= answer <= 99_999:
                if answer == 0:
                    return False
                if answer == 1:
                    return False
                if answer == 12345:
                    return False
                # now AIMO answers no longer need modulo
                return True
    except Exception:
        pass
    return False


def save_solver_trace(
    question_id: str,
    solver_index: int,
    tool_call_count: int,
    all_token_ids: list[int],
    backtrack_count: int | None = None,
) -> None:
    detokenized_text = harmony_encoding.decode(all_token_ids)
    boxed_text = extract_boxed_text(detokenized_text)
    answer_suffix = "NA"
    if is_valid_answer_string(boxed_text):
        answer_suffix = boxed_text
    total_tokens = len(all_token_ids)
    backtrack_part = (
        f"-backtrack-{backtrack_count}" if backtrack_count is not None else ""
    )
    base_path = f"{SOLUTIONS_DIR}/{question_id}/{solver_index:01d}-{total_tokens:05d}-{tool_call_count:02d}{backtrack_part}-{answer_suffix}"
    with open(f"{base_path}-tokens.txt", "w") as f:
        for token_id in all_token_ids:
            f.write(f"{token_id}\n")
    with open(f"{base_path}-text.txt", "w") as f:
        f.write(detokenized_text)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:23:00.334105Z","iopub.execute_input":"2025-11-24T08:23:00.334228Z","iopub.status.idle":"2025-11-24T08:23:00.341278Z","shell.execute_reply.started":"2025-11-24T08:23:00.334218Z","shell.execute_reply":"2025-11-24T08:23:00.340907Z"}}
from collections import Counter

completed_question_ids: set[str] = set()
question_id_to_solver_index_to_answer: dict[str, dict[int, int]] = {}
question_id_to_solver_index_to_answer_history: dict[
    str, dict[int, list[int | None]]
] = {}
question_id_to_solver_index_to_proposal_history: dict[
    str, dict[int, list[int | None]]
] = {}
question_id_to_solver_index_to_submission_history: dict[str, dict[int, list[int]]] = {}
question_id_to_solver_index_to_proposal: dict[str, dict[int, int]] = {}
question_id_to_active_solver_indexes: dict[str, set[int]] = {}
question_id_to_proposed_solver_indexes: dict[
    str, set[int]
] = {}  # has solver ever attempted an answer
question_id_to_solver_index_to_backtrack_count: dict[str, dict[int, int]] = {}
question_id_to_solver_index_to_main_token_length: dict[str, dict[int, int]] = {}
question_id_to_solver_index_to_support_token_length: dict[str, dict[int, int]] = {}
question_id_to_solver_index_to_tool_use_count: dict[str, dict[int, int]] = {}


def get_vote_string(question_id: str, reader_solver_index: int) -> str:
    solver_index_to_answer = question_id_to_solver_index_to_answer[question_id]
    print(f"{solver_index_to_answer=}")
    solver_index_to_proposal = question_id_to_solver_index_to_proposal[question_id]
    print(f"{solver_index_to_proposal=}")
    active_solver_indexes = question_id_to_active_solver_indexes[question_id]
    print(f"{active_solver_indexes=}")
    lines = ["These are the currently submitted answers:"]
    for solver_index in sorted(active_solver_indexes):
        solver_annotation = ""
        if solver_index == reader_solver_index:
            solver_annotation = " (you)"
        if (
            solver_index in solver_index_to_answer
            and solver_index in solver_index_to_proposal
        ):
            answer = solver_index_to_answer[solver_index]
            proposal = solver_index_to_proposal[solver_index]
            if answer == proposal:
                lines.append(
                    f"Solver {solver_index}{solver_annotation} is submitting {answer}"
                )
            else:
                lines.append(
                    f"Solver {solver_index}{solver_annotation} is submitting {answer}, but considering {proposal}"
                )
        elif solver_index in solver_index_to_proposal:
            proposal = solver_index_to_proposal[solver_index]
            lines.append(
                f"Solver {solver_index}{solver_annotation} is considering {proposal}"
            )
        else:
            lines.append(
                f"Solver {solver_index}{solver_annotation} has not submitted an answer."
            )
    return "\n".join(lines)


def vote_answer(question_id: str, force_answer: bool = False) -> int | None:
    """Vote for the best answer, with the side effect of adding to completed_question_ids"""
    solver_index_to_answer = question_id_to_solver_index_to_answer[question_id]
    print(f"{solver_index_to_answer=}")
    solver_index_to_proposal = question_id_to_solver_index_to_proposal[question_id]
    print(f"{solver_index_to_proposal=}")
    active_solver_indexes = question_id_to_active_solver_indexes[question_id]
    print(f"{active_solver_indexes=}")
    solver_index_to_backtrack_count = question_id_to_solver_index_to_backtrack_count[
        question_id
    ]
    print(f"{solver_index_to_backtrack_count=}")

    answer = vote_answer_inner(
        solver_index_to_answer,
        solver_index_to_proposal,
        active_solver_indexes,
        solver_index_to_backtrack_count,
        force_answer,
    )
    if answer is not None:
        # so that we do not need to remember to do this
        # when we edit logic in vote_answer_inner
        completed_question_ids.add(question_id)
    return answer


def vote_answer_inner(
    solver_index_to_answer: dict[int, int],
    solver_index_to_proposal: dict[int, int],
    active_solver_indexes: set[int],
    solver_index_to_backtrack_count: dict[int, int],
    force_answer: bool,
) -> int | None:
    """Vote for the best answer"""

    if len(active_solver_indexes) == 0:
        force_answer = True

    answers_from_active_solver_indexes = [
        solver_index_to_answer[solver_index]
        for solver_index in active_solver_indexes
        if solver_index in solver_index_to_answer  # if there is an answer
        and solver_index in solver_index_to_proposal
        and solver_index_to_answer[solver_index]  # answer is matching current proposal
        == solver_index_to_proposal[solver_index]
    ]
    proposal_from_active_solver_indexes = [
        proposal
        for solver_index, proposal in solver_index_to_proposal.items()
        if solver_index in active_solver_indexes
    ]

    # Compute backtrack count sum per proposal (from active solvers)
    proposal_to_backtrack_count: dict[int, int] = defaultdict(int)
    for solver_index in active_solver_indexes:
        if proposal := solver_index_to_proposal.get(solver_index):
            proposal_to_backtrack_count[proposal] += (
                solver_index_to_backtrack_count.get(solver_index, 0)
            )

    # tie breaking: active proposals count, answers count, backtrack count, answer value
    active_proposal_counter = Counter(proposal_from_active_solver_indexes)
    answer_counter = Counter(answers_from_active_solver_indexes)

    ranked_proposals = sorted(
        active_proposal_counter,
        key=lambda proposal: (
            active_proposal_counter[proposal],
            answer_counter[proposal],
            proposal_to_backtrack_count[proposal],
            proposal,
        ),
    )

    best_answer = ranked_proposals[-1] if ranked_proposals else 12453

    if force_answer:
        print(f"force_answer, Current GPU usage {get_gpu_kv_cache_usage()}")
        return best_answer

    def check_threshold(
        answer_threshold: tuple[int | None, int | None, int | None],
        proposal_threshold: tuple[int | None, int | None, int | None],
    ) -> bool:
        """
        Check if voting thresholds are met.
        Threshold tuple: (match, mismatch, unsubmitted), None means don't care.
        match: at least this many matches (>=)
        mismatch: at most this many mismatches (<=)
        unsubmitted: at most this many unsubmitted (<=)
        """
        answer_match, answer_mismatch, answer_unsubmitted = answer_threshold
        proposal_match, proposal_mismatch, proposal_unsubmitted = proposal_threshold

        # Compute answer stats
        answer_match_count = answers_from_active_solver_indexes.count(best_answer)
        answer_mismatch_count = (
            len(answers_from_active_solver_indexes) - answer_match_count
        )
        answer_unsubmitted_count = len(active_solver_indexes) - len(
            answers_from_active_solver_indexes
        )

        # Check answer thresholds
        if answer_match is not None:
            if answer_match_count < answer_match:
                return False
        # mismatch is worse than unsubmitted, so:
        # - mismatch_count <= mismatch_allowance
        # - mismatch_count + unsubmitted_count <= mismatch_allowance + unsubmitted_allowance
        if answer_mismatch is not None:
            if answer_mismatch_count > answer_mismatch:
                return False
            if answer_unsubmitted is not None:
                if (
                    answer_mismatch_count + answer_unsubmitted_count
                    > answer_mismatch + answer_unsubmitted
                ):
                    return False

        # Compute proposal stats
        proposal_match_count = proposal_from_active_solver_indexes.count(best_answer)
        proposal_mismatch_count = (
            len(proposal_from_active_solver_indexes) - proposal_match_count
        )
        proposal_unsubmitted_count = len(active_solver_indexes) - len(
            proposal_from_active_solver_indexes
        )

        # Check proposal thresholds
        if proposal_match is not None:
            if proposal_match_count < proposal_match:
                return False
        # mismatch is worse than unsubmitted, so:
        # - mismatch_count <= mismatch_allowance
        # - mismatch_count + unsubmitted_count <= mismatch_allowance + unsubmitted_allowance
        if proposal_mismatch is not None:
            if proposal_mismatch_count > proposal_mismatch:
                return False
            if proposal_unsubmitted is not None:
                if (
                    proposal_mismatch_count + proposal_unsubmitted_count
                    > proposal_mismatch + proposal_unsubmitted
                ):
                    return False

        def fmt(t: tuple[int | None, int | None, int | None]) -> str:
            return "/".join("?" if x is None else str(x) for x in t)

        print(
            f"Vote for {best_answer} passed with answer {fmt(answer_threshold)}, proposal {fmt(proposal_threshold)}"
        )
        return True

    # (match / mismatch / unsubmitted)
    if check_threshold((None, 0, 0), (None, None, None)):
        return best_answer

    if check_threshold((None, None, None), (None, 0, 0)):
        return best_answer

    if best_answer <= 3:
        # answer is unlikely to be a small number
        return None

    # If last question, spend all the time
    if len(cutoff_times) <= 1:
        return None

    if check_threshold((0, 0, None), (2, 0, None)):
        return best_answer

    if check_threshold((0, 0, None), (3, 0, None)):
        return best_answer

    if check_threshold((0, 1, None), (4, 1, None)):
        return best_answer
    
    if check_threshold((0, 0, None), (2, 1, 0)):
        return best_answer

    if check_threshold((0, 0, None), (1, 0, 2)):
        return best_answer

    # otherwise spend all the time if there is a dissent without corroboration
    return None

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Collaboration logic

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Shared findings storage per question (simple list)

from dataclasses import dataclass


@dataclass
class Finding:
    statement: str
    question_id: str
    author_solver_index: int
    author_solver_token_count: int
    proposal: int | None
    accepted_solver_indexes_and_reviews: dict[int, str]
    rejected_solver_indexes_and_reviews: dict[int, str]

    @property
    def reviewed_solver_indexes(self) -> set[int]:
        return set(self.accepted_solver_indexes_and_reviews.keys()) | set(
            self.rejected_solver_indexes_and_reviews.keys()
        )


question_id_to_findings: dict[str, list[Finding]] = {"": []}

question_id_to_solver_index_to_proposals_read: dict[str, dict[int, set[int]]] = {}
question_id_to_solver_index_to_findings_read: dict[str, dict[int, set[int]]] = {}


def get_finding_texts_for_solver(
    question_id: str,
    solver_index: int,
    skip_update_read_state: bool = False,
    skip_none: bool = True,
) -> list[str]:
    """Get findings from other solvers that this solver hasn't seen yet."""
    findings = question_id_to_findings[question_id]
    proposals_read = question_id_to_solver_index_to_proposals_read[question_id][
        solver_index
    ]
    findings_read = question_id_to_solver_index_to_findings_read[question_id][
        solver_index
    ]
    current_solver_proposal = question_id_to_solver_index_to_proposal[question_id].get(
        solver_index
    )
    if current_solver_proposal is None:
        return []

    solver_index_to_finding_texts: dict[int, str] = {}
    solver_seen_in_session = set()
    for finding_idx in range(len(findings) - 1, -1, -1):
        finding = findings[finding_idx]
        if finding.author_solver_index == solver_index:
            continue
        if finding.author_solver_index in solver_seen_in_session:
            continue
        # Skip if already read this finding
        if finding_idx in findings_read:
            continue
        if skip_none and finding.proposal is None:
            continue
        already_read_proposal = finding.proposal in proposals_read
        if already_read_proposal:
            finding.rejected_solver_indexes_and_reviews[solver_index] = (
                "Already read proposal"
            )
            continue
        if finding.proposal == current_solver_proposal:
            finding.rejected_solver_indexes_and_reviews[solver_index] = (
                "Same proposal as current solver"
            )
            continue
        else:
            finding.accepted_solver_indexes_and_reviews[solver_index] = (
                "Proposals differ for now"
            )
        finding_string = (
            f"Another solver has provided this solution:\n\n{finding.statement}"
        )
        solver_index_to_finding_texts[finding.author_solver_index] = finding_string
        solver_seen_in_session.add(finding.author_solver_index)
        if not skip_update_read_state:
            findings_read.add(finding_idx)
            if finding.proposal is not None:
                proposals_read.add(finding.proposal)

    finding_texts = list(solver_index_to_finding_texts.values())
    if finding_texts:
        return finding_texts
    if skip_none is True:
        return get_finding_texts_for_solver(
            question_id, solver_index, skip_update_read_state, skip_none=False
        )
    return finding_texts


def get_finding_text(
    all_token_ids: list[int],
    question_id: str,
    solver_index: int,
) -> str:
    """
    Use a separate LLM call to summarize useful findings from the current conversation.
    Takes a copy of all_token_ids (the full conversation context) and appends a summarization request.
    Returns the finding text, or an empty string if no finding was extracted.
    Terminates early if the question is solved.
    """

    finding_prompt = """
Summarize your current approach in LESS THAN 250 characters, for other solvers to replicate your solution.

Include:
- key insights
- key intermediate results
- key pitfalls

Reminder
- Your job NOW is to SUMMARIZE.
- The limit is 250 characters. Keep your response very, very concise. Only include the important information.
""".strip()

    finding_prefix = "Insights:"

    # Append summarization request to the conversation context
    all_token_ids = append_user_turn_token_ids(all_token_ids, finding_prompt)
    finding_input_tokens = all_token_ids + harmony_encoding.encode(
        "<|channel|>analysis<|message|>My job here is to summarize. I must follow the character limit.<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{finding_prefix}",
        allowed_special="all",
    )

    # Use streaming completion for summarization
    finding_stream: Stream[Completion] = client.completions.create(
        model="vllm-model",
        prompt=finding_input_tokens,
        max_tokens=1024,
        temperature=0,
        stream=True,
        extra_body=dict(stop_token_ids=stop_token_ids, return_token_ids=True),
    )
    save_communication(
        harmony_encoding.decode(finding_input_tokens), prefix="findings-"
    )
    finding_text = finding_prefix
    for chunk in finding_stream:
        question_id_to_solver_index_to_support_token_length[question_id][
            solver_index
        ] += 1
        if chunk.choices[0].text:
            finding_text += chunk.choices[0].text
        if question_id in completed_question_ids:
            break
        if chunk.choices[0].finish_reason:
            break
    finding_stream.close()

    # do not submit answers in findings
    finding_text = finding_text.replace("\\boxed{", "{answer = ")

    if not finding_text:
        print("Warning: empty finding_text")
        return ""

    return finding_text


def get_proposal(
    all_token_ids: list[int],
    question_id: str,
    solver_index: int,
) -> int | None:
    """
    Use a separate LLM call to return the current proposal.
    """

    # Note: previously approved findings are not included because they should already be in conversation history

    proposal_prompt = """
Have you computed a final answer? If you have computed a final answer, return that integer answer in \\boxed{}.
If you have not computed a final answer, return \\boxed{None} instead.
""".strip()

    # Append proposal request to the conversation context
    all_token_ids = append_user_turn_token_ids(all_token_ids, proposal_prompt)
    proposal_text_prefix = "\\boxed{"
    proposal_input_tokens = all_token_ids + harmony_encoding.encode(
        f"<|channel|>final<|message|>{proposal_text_prefix}", allowed_special="all"
    )

    # Use non-streaming completion for proposal
    proposal_resp = client.completions.create(
        model="vllm-model",
        prompt=proposal_input_tokens,
        max_tokens=10,
        temperature=0,
        extra_body=dict(stop_token_ids=stop_token_ids, return_token_ids=True),
    )
    question_id_to_solver_index_to_support_token_length[question_id][solver_index] += (
        len(getattr(proposal_resp.choices[0], "token_ids", []))
    )
    save_communication(
        harmony_encoding.decode(proposal_input_tokens), prefix="proposal-"
    )
    proposal_text = proposal_text_prefix + proposal_resp.choices[0].text
    boxed_text = extract_boxed_text(proposal_text)
    if is_valid_answer_string(boxed_text):
        return int(boxed_text)
    return None


def get_answer_confidence(
    all_token_ids: list[int],
    answer: int,
    question_id: str,
    solver_index: int,
    enabled: bool = maybe_collaborate_enabled,
) -> bool:
    """
    Use a separate LLM call to get the confidence of the currently proposed answer.
    """
    if not enabled:
        return True

    confidence_prompt = f"""
Did you confirm that {answer} is wrong?
Start your reply with exactly either `I have confirmed that {answer} is wrong` or `I have yet to confirm that {answer} is wrong`.
""".strip()

    # need to close the assistant turn
    all_token_ids = all_token_ids + harmony_encoding.encode(
        "<|end|>", allowed_special="all"
    )

    # Append confidence request to the conversation context
    all_token_ids = append_user_turn_token_ids(all_token_ids, confidence_prompt)
    assistant_prefix = "I have"
    confidence_input_tokens = all_token_ids + harmony_encoding.encode(
        f"<|channel|>final<|message|>{assistant_prefix}", allowed_special="all"
    )

    # Use non-streaming completion for confidence check
    confidence_resp = client.completions.create(
        model="vllm-model",
        prompt=confidence_input_tokens,
        max_tokens=32,
        temperature=0,
        extra_body=dict(stop_token_ids=stop_token_ids, return_token_ids=True),
    )
    question_id_to_solver_index_to_support_token_length[question_id][solver_index] += (
        len(getattr(confidence_resp.choices[0], "token_ids", []))
    )
    confidence_text = assistant_prefix + confidence_resp.choices[0].text.replace(
        "\n", " "
    )
    save_communication(
        harmony_encoding.decode(confidence_input_tokens), prefix="confidence-"
    )
    if f"I have confirmed that {answer} is wrong" in confidence_text:
        print(f"get_answer_confidence False {answer} {confidence_text}")
        return False
    print(f"get_answer_confidence True {answer} {confidence_text}")
    return True


def contains_sublist(big: list, small: list):
    return any(
        big[i : i + len(small)] == small for i in range(len(big) - len(small) + 1)
    )


# <|start|>assistant<|channel|>final<|message|>
ASSISTANT_STARTER: list[int] = [200006, 173781, 200005, 17196, 200008]


def backtrack_all_tokens(all_token_ids: list[int], backtrack_count: int) -> list[int]:
    all_token_ids = all_token_ids.copy()
    for _ in range(backtrack_count):
        tokens_popped = 0
        if all_token_ids and contains_sublist(all_token_ids[:-1], ASSISTANT_STARTER):
            tokens_popped += 1
            all_token_ids.pop()  # make sure initial all_token_ids does not end with ASSISTANT_STARTER
            while (
                len(all_token_ids) >= len(ASSISTANT_STARTER)
                and all_token_ids[-len(ASSISTANT_STARTER) :] != ASSISTANT_STARTER
            ):
                all_token_ids.pop()
                tokens_popped += 1
            all_token_ids.pop()  # drop <|message|>
        print(f"Backtracking, {tokens_popped=}")
    return all_token_ids


def delete_answer(
    all_token_ids: list[int],
    question_id: str,
    solver_index: int,
) -> None:
    # need two calls to actually delete
    current_answer = question_id_to_solver_index_to_answer[question_id].get(
        solver_index
    )
    if current_answer is not None:  # check again in case of race condition
        if (
            question_id_to_solver_index_to_answer_history[question_id][solver_index][-1]
            is None
        ):
            del question_id_to_solver_index_to_answer[question_id][solver_index]
            print(f"Solver {solver_index} deleted current answer {current_answer}")

            current_proposal = question_id_to_solver_index_to_proposal[question_id].get(
                solver_index
            )
            if (
                current_proposal is not None
                and len(question_id_to_active_solver_indexes) >= 4
            ):
                # do not delete proposal, we want corroboration to happen
                del question_id_to_solver_index_to_proposal[question_id][solver_index]
                print(
                    f"Solver {solver_index} deleted current proposal {current_proposal}"
                )
        else:
            question_id_to_solver_index_to_answer_history[question_id][
                solver_index
            ].append(None)
            print(
                f"Solver {solver_index} is planning to delete current answer {current_answer}"
            )
        maybe_terminate_solver_for_gpu_usage(question_id)
        vote_answer(question_id)
        maybe_create_finding(all_token_ids, question_id, solver_index)
    else:
        print(
            f"Solver {solver_index} is attempting to delete an answer that no longer exists"
        )


def update_proposal(
    all_token_ids: list[int],
    question_id: str,
    solver_index: int,
) -> None:
    current_answer = question_id_to_solver_index_to_answer[question_id].get(
        solver_index
    )
    if current_answer is not None:
        answer_still_confident = get_answer_confidence(
            all_token_ids.copy(), current_answer, question_id, solver_index
        )
        if not answer_still_confident:
            delete_answer(all_token_ids, question_id, solver_index)
        else:
            if (
                question_id_to_solver_index_to_answer_history[question_id][
                    solver_index
                ][-1]
                is None
            ):
                print(
                    f"Solver {solver_index} is not planning to delete current answer {current_answer}"
                )
                question_id_to_solver_index_to_answer_history[question_id][
                    solver_index
                ].pop()

    # get proposed answer from current solver
    new_proposal = get_proposal(all_token_ids.copy(), question_id, solver_index)
    if solver_index not in question_id_to_solver_index_to_proposal[question_id]:
        if new_proposal is None:
            # no current proposal, no new proposal
            pass
        else:
            # proposing first answer
            question_id_to_solver_index_to_proposal[question_id][solver_index] = (
                new_proposal
            )
            question_id_to_solver_index_to_proposal_history[question_id][
                solver_index
            ].append(new_proposal)
            print(
                f"Solver {solver_index} is proposing its first proposal {new_proposal} at token {len(all_token_ids)}"
            )
            maybe_terminate_solver_for_gpu_usage(question_id)
            vote_answer(question_id)
            maybe_create_finding(all_token_ids, question_id, solver_index)
    else:
        current_proposal = question_id_to_solver_index_to_proposal[question_id][
            solver_index
        ]

        if current_proposal != new_proposal:
            if new_proposal is None:
                if (
                    len(question_id_to_active_solver_indexes[question_id]) <= 5
                    and cutoff_times[-1] - time.time() >= 120
                ):
                    print(f"Proposal deletion blocked for Solver {solver_index}")
                else:
                    print(f"Proposal deletion allowed for Solver {solver_index}")
                    del question_id_to_solver_index_to_proposal[question_id][
                        solver_index
                    ]
            else:
                question_id_to_solver_index_to_proposal[question_id][solver_index] = (
                    new_proposal
                )
            question_id_to_solver_index_to_proposal_history[question_id][
                solver_index
            ].append(new_proposal)
            print(
                f"Solver {solver_index} is proposing proposal {new_proposal} at token {len(all_token_ids)} to overturn proposal {current_proposal}"
            )
            maybe_terminate_solver_for_gpu_usage(question_id)
            vote_answer(question_id)
            maybe_create_finding(all_token_ids, question_id, solver_index)

    for _ in range(10):
        if question_id in completed_question_ids:
            break
        time.sleep(1)
    question_id_to_proposing_answer[question_id].discard(
        solver_index
    )  # make sure there are no early returns


def maybe_update_proposal(
    token_ids: list[int],
    question_id: str,
    solver_index: int,
    enabled: bool = maybe_collaborate_enabled,
) -> None:
    """Update proposed answer if not already working on it."""
    if not enabled:
        return
    if solver_index in question_id_to_proposing_answer[question_id]:
        return
    question_id_to_proposing_answer[question_id].add(solver_index)
    threading.Thread(
        target=update_proposal,
        args=(token_ids.copy(), question_id, solver_index),
    ).start()


def create_finding(
    all_token_ids: list[int], question_id: str, solver_index: int
) -> None:
    """Create a finding from the current solver and share it."""
    all_token_ids = all_token_ids.copy()
    finding_text = get_finding_text(all_token_ids.copy(), question_id, solver_index)
    author_proposal = question_id_to_solver_index_to_proposal[question_id].get(
        solver_index
    )
    finding = Finding(
        statement=finding_text,
        question_id=question_id,
        author_solver_index=solver_index,
        author_solver_token_count=len(all_token_ids),
        proposal=author_proposal,
        accepted_solver_indexes_and_reviews={},
        rejected_solver_indexes_and_reviews={},
    )
    question_id_to_findings[question_id].append(finding)
    if len(finding_text) >= 250:
        truncated_finding_text = (
            finding_text.replace("\n", " ")[:100]
            + "..."
            + finding_text.replace("\n", " ")[::-1][:100][::-1]
        )
    else:
        truncated_finding_text = finding_text.replace("\n", " ")
    print(
        f"Solver {solver_index:01d} shared Finding {len(question_id_to_findings[question_id]) - 1}: {len(finding_text)=} {truncated_finding_text}"
    )
    for _ in range(90):  # reduce compute load
        if question_id in completed_question_ids:
            break
        time.sleep(1)
    question_id_to_creating_finding[question_id].discard(solver_index)


import threading
from collections import defaultdict

# Track ongoing finding creation per (question_id, solver_index)
question_id_to_creating_finding: defaultdict[str, set[int]] = defaultdict(set)
question_id_to_proposing_answer: defaultdict[str, set[int]] = defaultdict(set)


def maybe_create_finding(
    token_ids: list[int],
    question_id: str,
    solver_index: int,
    enabled: bool = maybe_collaborate_enabled,
) -> None:
    """Create finding if not already working on it."""
    if not enabled:
        return
    if solver_index in question_id_to_creating_finding[question_id]:
        return
    question_id_to_creating_finding[question_id].add(solver_index)
    threading.Thread(
        target=create_finding, args=(token_ids.copy(), question_id, solver_index)
    ).start()


def save_findings(question_id: str, final_answer: int, time_taken: float) -> None:
    """Save all findings for a question to findings/ directory as JSON."""
    import json

    findings = question_id_to_findings[question_id]
    data = {
        "question_id": question_id,
        "final_answer": final_answer,
        "time_taken": round(time_taken, 1),
        "findings": [
            {
                "finding_idx": finding_idx,
                "solver_index": finding.author_solver_index,
                "finding": finding.statement,
                "accepted_solver_indexes": list(
                    finding.accepted_solver_indexes_and_reviews.keys()
                ),
                "rejected_solver_indexes": list(
                    finding.rejected_solver_indexes_and_reviews.keys()
                ),
                "author_solver_token_count": finding.author_solver_token_count,
            }
            for finding_idx, finding in enumerate(findings)
        ],
    }
    with open(f"{FINDINGS_DIR}/{question_id}.json", "w") as f:
        json.dump(data, f, indent=2)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Solve question

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:53.213762Z","iopub.execute_input":"2025-11-24T08:26:53.214218Z","iopub.status.idle":"2025-11-24T08:26:53.221603Z","shell.execute_reply.started":"2025-11-24T08:26:53.214205Z","shell.execute_reply":"2025-11-24T08:26:53.221218Z"}}
def solve_question(
    question_text: str,
    question_id: str = "",
    solver_index: int = 0,
) -> str:
    await_client()
    print(f"Client connected for Solver {solver_index}")
    if question_id in completed_question_ids:
        return ""
    if time.time() >= cutoff_times[-1]:
        return ""

    # Initialize per-solver finding state
    question_id_to_solver_index_to_proposals_read[question_id][solver_index] = set()
    question_id_to_solver_index_to_findings_read[question_id][solver_index] = set()

    system_content = """\
You will solve the problem and return the final answer in \\boxed{}.
The answer is expected to be an integer between 0 and 99999, inclusive.

You may be provided solutions from other solvers.
- Reminder: if you find something wrong in your intermediate step, be sure to recompute all the variables from the intermediate step to the answer."""

    # Create a dedicated Jupyter session for this solver
    # Each solve_question call gets its own isolated session
    jupyter_session = LocalJupyterSession(timeout=10.0)
    execute_python_code(jupyter_session, "import sympy as sp")
    print(f"Solver {solver_index} created jupyter_session")

    # try finally loop is for the jupyter_session
    try:
        # Build initial prompt as token IDs with Python tool enabled
        all_token_ids: list[int] = build_prompt_token_ids(
            system_content=system_content,
            user_content=question_text,
            reasoning_effort=ReasoningEffort.HIGH,
            enable_python_tool=True,  # Enable Python tool for code execution
        )

        generated_token_count = 0
        generated_token_count_current_iteration = 0
        tool_call_count = 0

        for iteration in range(1024):
            # Solve question loop
            # until conditions to terminate is fulfilled

            # Use StreamableParser to process streaming tokens, mainly for tool use
            # reused over resample_completion
            stream_parser = StreamableParser(harmony_encoding, role=Role.ASSISTANT)

            while True:
                # Tool use loop to handle tool calls within each iteration
                # until no tool call use, or exit_solve_question is True
                exit_solve_question = False
                continue_tool_use_loop = False
                resample_completion = False  # always False for now, but leaving it here

                # safeguard against throwing error
                if len(all_token_ids) >= max_model_len - 16384:
                    print(f"Terminating Solver {solver_index} by length")
                    exit_solve_question = True
                    break

                # Use streaming with completions API
                stream: Stream[Completion] = client.completions.create(
                    model="vllm-model",
                    prompt=all_token_ids,
                    max_tokens=max_model_len - len(all_token_ids) - 8192,
                    temperature=1.0,
                    stream=True,
                    extra_body=dict(
                        min_p=0.02,
                        stop_token_ids=stop_token_ids,
                        return_token_ids=True,
                    ),
                )
                # save_communication(
                #     harmony_encoding.decode(all_token_ids), prefix="main-"
                # )

                number_of_paragraphs_after_proposal_match = -1

                for chunk in stream:
                    generated_token_count += 1
                    generated_token_count_current_iteration += 1
                    # Get token IDs from the chunk (vLLM extension)
                    chunk_token_ids = getattr(chunk.choices[0], "token_ids", None)
                    if chunk_token_ids:
                        # (we assume len(chunk_token_ids) > 1 is rare)
                        # Process tokens through harmony parser for text
                        # track generated token length per solver
                        question_id_to_solver_index_to_main_token_length[question_id][
                            solver_index
                        ] += 1
                        for token_id in chunk_token_ids:
                            # if you use stream-interval > 1, it is more involved
                            all_token_ids.append(token_id)
                            stream_parser.process(token_id)

                    # Check finish_reason to see if generation completed naturally
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason:
                        break

                    if (
                        solver_index
                        not in question_id_to_active_solver_indexes[question_id]
                    ):
                        print(
                            f"Final generation for Solver {solver_index} before finally terminating"
                        )
                        maybe_create_finding(
                            all_token_ids + [END_TOKEN_ID], question_id, solver_index
                        )
                        # stop generating if we have finalized on an answer
                        exit_solve_question = True
                        break
                    if question_id in completed_question_ids:
                        # stop generating if we have finalized on an answer
                        exit_solve_question = True
                        break
                    if time.time() >= cutoff_times[-1]:
                        # so that we do not incorrectly delete solver IDs
                        completed_question_ids.add(question_id)
                        exit_solve_question = True
                        break

                    if all_token_ids[-2:] == [17196, 200008]:  # final<|message|>
                        # So that the model just prints the answer instead of explaining
                        # should trigger after backtracking
                        answer_token_ids_to_force = [59, 172278, 90]  # \\boxed{
                        for answer_token_id_to_force in answer_token_ids_to_force:
                            all_token_ids.append(answer_token_id_to_force)
                            stream_parser.process(answer_token_id_to_force)
                        resample_completion = True

                    if resample_completion:
                        break

                    # Also get text
                    chunk_text = chunk.choices[0].text

                    if (
                        "\n\n" in chunk_text  # paragraph end
                        and stream_parser.current_recipient is None
                        and generated_token_count_current_iteration >= 1024
                        # do not immediately drop previous proposal / answer
                        and number_of_paragraphs_after_proposal_match == 3
                        # require a few more paragraphs before sending to proposal
                    ):
                        maybe_create_finding(all_token_ids, question_id, solver_index)

                        if (
                            solver_index
                            in question_id_to_proposed_solver_indexes[question_id]
                            # only recalculate after you have proposal
                            # so this path will not generate the first proposal
                            # this means the first proposal has to be boxed
                        ):
                            maybe_update_proposal(
                                all_token_ids,
                                question_id,
                                solver_index,
                            )
                            # reset markers
                            number_of_paragraphs_after_proposal_match = -1

                    if "\n\n" in chunk_text:
                        if number_of_paragraphs_after_proposal_match >= 0:
                            number_of_paragraphs_after_proposal_match += 1

                    for matching_text in [
                        "answer",
                        "final",
                    ]:
                        if matching_text in chunk_text.lower():
                            number_of_paragraphs_after_proposal_match = 0

                    if (
                        solver_index
                        in question_id_to_solver_index_to_proposal[question_id]
                    ):
                        for matching_text in [
                            "oops",
                            "mistake",
                            "missed",
                            "missing",
                            "wrong",
                            "correct",
                        ]:
                            if matching_text in chunk_text.lower():
                                number_of_paragraphs_after_proposal_match = 0

                    # termination logic based on GPU KV cache usage
                    if len(all_token_ids) > 20_000 and len(all_token_ids) % 1000 == 0:
                        # check for len(all_token_ids) > 20_000 is there so that we do not mysteriously drop questions at start
                        # len(all_token_ids) % 100 == 0 so that we do not check all at once
                        # ideally we want to directly read the kv cache usage even if inactive
                        # it is possible to have huge jump on cache usage from the long output
                        # the gpu pruning thresholds here is less strict here
                        maybe_terminate_solver_for_gpu_usage(question_id)

                    # instead of exit_solve_question = True
                    # we want some rule based checks
                    if chunk_text and "}" in chunk_text:
                        # match "}" in chunk_text so that we do not need to run harmony_encoding.decode every time
                        # \\boxed + 5 digits
                        text_suffix = harmony_encoding.decode(all_token_ids[-20:])
                        # we could assume all_token_ids is at least 20 tokens long
                        boxed_text = extract_boxed_text(text_suffix)
                        if is_valid_answer_string(boxed_text):
                            answer = int(boxed_text)

                            # update proposal
                            question_id_to_solver_index_to_proposal[question_id][
                                solver_index
                            ] = answer
                            question_id_to_solver_index_to_proposal_history[
                                question_id
                            ][solver_index].append(answer)

                            if stream_parser.current_channel != "analysis":
                                # matches 'final'
                                # but backtracking is not guaranteed to produce this
                                if (
                                    solver_index
                                    not in question_id_to_proposed_solver_indexes[
                                        question_id
                                    ]
                                ):
                                    # so that we do not jump straight to the answer
                                    print(
                                        f"Solver {solver_index} will propose before answering."
                                    )
                                else:
                                    # update answer
                                    prev_answer = question_id_to_solver_index_to_answer[
                                        question_id
                                    ].get(solver_index)
                                    if prev_answer != answer:
                                        question_id_to_solver_index_to_answer_history[
                                            question_id
                                        ][solver_index].append(answer)
                                    question_id_to_solver_index_to_answer[question_id][
                                        solver_index
                                    ] = answer

                            question_id_to_proposed_solver_indexes[question_id].add(
                                solver_index
                            )

                            vote_answer(question_id)

                            print(
                                f"Solver {solver_index} boxed {answer} in "
                                f"{stream_parser.current_channel} to {stream_parser.current_recipient}"
                            )

                            if (
                                stream_parser.current_channel == "analysis"
                                and stream_parser.current_recipient is None
                                and len(
                                    get_finding_texts_for_solver(
                                        question_id,
                                        solver_index,
                                        skip_update_read_state=True,
                                    )
                                )
                                > 0
                                # if a solver boxes before final
                                # they will be forced to look at differing solutions first
                                # which means they might not reach final
                            ):
                                print(
                                    f"Forcing final and read finding for Solver {solver_index}"
                                )
                                # <|end|><|start|>assistant<|channel|>final<|message|>\\boxed{<integer>}<|end|>
                                force_termination_token_ids = (
                                    [
                                        200007,
                                        200006,
                                        173781,
                                        200005,
                                        17196,
                                        200008,
                                        59,
                                        172278,
                                        90,
                                    ]
                                    + harmony_encoding.encode(boxed_text)
                                    + [92, 200007]
                                )
                                for (
                                    force_termination_token_id
                                ) in force_termination_token_ids:
                                    all_token_ids.append(force_termination_token_id)
                                break

                            # vote passes (only if condition is met)
                            # -> question_id in completed_question_ids
                            # -> exit_solve_question

                # so that we free the GPU resources
                # Note: do not break between client.completions.create and stream.close()
                stream.close()

                # Replace <|return|> with <|end|> to maintain properly formed messages
                # This is required by OpenAI Harmony format when continuing conversation
                if all_token_ids and all_token_ids[-1] == RETURN_TOKEN_ID:
                    all_token_ids[-1] = END_TOKEN_ID

                if exit_solve_question:
                    break

                if resample_completion:
                    # reuse stream_parser
                    continue

                # Check if the last parsed message is a tool call
                # After streaming, parser.messages contains the parsed Message objects
                parsed_messages = stream_parser.messages
                # resample_completion False, reset stream_parser
                stream_parser = StreamableParser(harmony_encoding, role=Role.ASSISTANT)

                if parsed_messages:
                    last_message = parsed_messages[-1]
                    if (
                        last_message.recipient is not None
                        and last_message.recipient.startswith("python")
                    ):
                        continue_tool_use_loop = True
                        tool_call_count += 1
                        question_id_to_solver_index_to_tool_use_count[question_id][
                            solver_index
                        ] += 1
                        # Extract Python code from the message content
                        python_code = ""
                        if last_message.content:
                            first_block = last_message.content[0]
                            if isinstance(first_block, TextContent):
                                python_code = first_block.text
                        if python_code:
                            print(
                                f"Solver {solver_index:01d} iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}",
                                flush=True,
                            )
                            # Execute the code using stateful Jupyter session
                            output = execute_python_code(jupyter_session, python_code)
                            if len(output) > 10_000:
                                output = output[:3000] + "(truncated)" + output[-3000:]

                            # Create python tool response message
                            tool_response = make_python_tool_response(
                                output, channel=last_message.channel
                            )
                            # Append tool response tokens
                            all_token_ids = append_tool_response_token_ids(
                                all_token_ids, tool_response
                            )

                if not continue_tool_use_loop:
                    break

            if exit_solve_question:
                break

            # if it keeps repeating the same proposal, backtrack
            current_proposal = question_id_to_solver_index_to_proposal[question_id].get(
                solver_index
            )
            submissions = question_id_to_solver_index_to_submission_history[
                question_id
            ][solver_index]
            if current_proposal is not None:
                submissions.append(current_proposal)
                if len(submissions) >= 2 and len(set(submissions[-2:])) == 1:
                    submissions.pop()
                    submissions.pop()
                    print(
                        f"Backtracking Solver {solver_index} for repeating proposal {current_proposal}"
                    )
                    # notebook state is still the same after backtracking, which should be ok
                    question_id_to_solver_index_to_backtrack_count[question_id][
                        solver_index
                    ] += 1
                    # Save trace before backtracking
                    if question_id:
                        save_solver_trace(
                            question_id,
                            solver_index,
                            tool_call_count,
                            all_token_ids,
                            backtrack_count=question_id_to_solver_index_to_backtrack_count[
                                question_id
                            ][solver_index],
                        )
                    all_token_ids = backtrack_all_tokens(
                        all_token_ids, backtrack_count=2
                    )
                    continue

            maybe_create_finding(all_token_ids, question_id, solver_index)
            # flush findings
            finding_texts = get_finding_texts_for_solver(question_id, solver_index)

            boxed_text = extract_boxed_text(harmony_encoding.decode(all_token_ids))
            print(
                f"Solver {solver_index:01d} iteration {iteration:01d} tool {tool_call_count:02d} token {len(all_token_ids):05d}"
            )
            if not is_valid_answer_string(boxed_text):
                print(f"Solver {solver_index} follow-up - ask boxed answer")
                user_follow_up = (
                    "Figure out the correct answer. "
                    "The answer is expected to be an integer between 0 and 99999, inclusive. "
                    "Place your final answer in \\boxed{}. "
                    "Do not give up. Do not guess the answer. Do not put a placeholder. "
                    "If you are uncertain, continue working on the problem. There is no time limit."
                )
            else:
                print(f"Solver {solver_index} follow-up - continue verifying")

                follow_up_instructions = ""
                if len(finding_texts) == 0:
                    follow_up_instructions = (
                        "Scrutinize your solution."
                        + "\nIf you spot any critical mistake in your solution, work towards figuring out the correct answer."
                        + "\nPrioritize scrutinizing your solution so you can find any mistakes as soon as possible."
                    )
                else:
                    follow_up_instructions = (
                        "\n\n".join(finding_texts)
                        + "\n\nScrutinize your solution, using other solutions as a reference."
                        + "\nIf you spot any critical mistake in your solution, work towards figuring out the correct answer."
                        + "\nPrioritize scrutinizing your solution so you can find any mistakes as soon as possible."
                    )
                # I am allowed to do \\boxed{{}} here because extract_boxed_text ignores empty boxed content
                # Maybe reinstate the comment on missing findings
                user_follow_up = f"""\
{get_vote_string(question_id, solver_index)}

{follow_up_instructions}"""

            # Append user follow-up as token IDs
            all_token_ids = append_user_turn_token_ids(all_token_ids, user_follow_up)
            generated_token_count_current_iteration = 0

        detokenized_text = harmony_encoding.decode(all_token_ids)
        boxed_text = extract_boxed_text(detokenized_text)

        if question_id not in completed_question_ids:
            # Discarded to avoid stalemate, to not block voting
            question_id_to_active_solver_indexes[question_id].discard(solver_index)
            vote_answer(question_id)

        if question_id:
            # save information into file
            if is_valid_answer_string(boxed_text):
                print(
                    f"Solver {solver_index:01d} token {len(all_token_ids):05d} submits {boxed_text}"
                )
            save_solver_trace(
                question_id,
                solver_index,
                tool_call_count,
                all_token_ids,
            )

        return boxed_text

    finally:
        # Always clean up the Jupyter session when done
        if jupyter_session is not None:
            print(
                f"Cleaning up Jupyter session for Solver {solver_index} in {question_id}"
            )
            jupyter_session.close()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:26:54.553208Z","iopub.execute_input":"2025-11-24T08:26:54.553692Z","iopub.status.idle":"2025-11-24T08:27:03.475341Z","shell.execute_reply.started":"2025-11-24T08:26:54.553671Z","shell.execute_reply":"2025-11-24T08:27:03.474837Z"}}
if is_on_kaggle_interactive():
    solve_question("What is 1+1?")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:27:06.296266Z","iopub.execute_input":"2025-11-24T08:27:06.296867Z","iopub.status.idle":"2025-11-24T08:27:06.300859Z","shell.execute_reply.started":"2025-11-24T08:27:06.29685Z","shell.execute_reply":"2025-11-24T08:27:06.300371Z"}}
def solve(question_text: str, question_id: str = "") -> int:
    print(f"processing {question_id}")
    reallocate_time(cutoff_times)
    time_available = cutoff_times[-1] - time.time()
    print(f"time_available {time_available:.1f}s")
    question_start_time = time.time()
    os.makedirs(f"{SOLUTIONS_DIR}/{question_id}", exist_ok=True)
    question_id_to_solver_index_to_answer[question_id] = {}
    question_id_to_solver_index_to_answer_history[question_id] = defaultdict(list)
    question_id_to_solver_index_to_proposal_history[question_id] = defaultdict(list)
    question_id_to_solver_index_to_submission_history[question_id] = defaultdict(list)
    question_id_to_solver_index_to_proposal[question_id] = {}
    question_id_to_active_solver_indexes[question_id] = set(range(num_generations))
    question_id_to_proposed_solver_indexes[question_id] = set()
    question_id_to_findings[question_id] = []
    question_id_to_solver_index_to_proposals_read[question_id] = {}
    question_id_to_solver_index_to_findings_read[question_id] = {}
    question_id_to_creating_finding[question_id] = set()
    question_id_to_solver_index_to_backtrack_count[question_id] = defaultdict(int)
    question_id_to_solver_index_to_main_token_length[question_id] = defaultdict(int)
    question_id_to_solver_index_to_support_token_length[question_id] = defaultdict(int)
    question_id_to_solver_index_to_tool_use_count[question_id] = defaultdict(int)
    question_id_to_latest_termination_time[question_id] = time.time()
    completed_question_ids.discard(question_id)  # just in case question_id collides

    if question_id and time.time() > cutoff_times[-1]:
        print("timeout did not solve")
        return 12314

    get_gpu_kv_cache_usage(
        question_id
    )  # run once to prevent running in the first batch of execution

    # Start solver threads
    # I suspect that init LocalJupyterSession(timeout=10.0) can stall
    for solver_index in range(num_generations):
        threading.Thread(
            target=solve_question,
            args=(question_text, question_id, solver_index),
        ).start()

    # Poll every second until completed or time runs out
    for _ in range(int(time_available) + 1):
        if question_id in completed_question_ids:
            break
        time.sleep(1)
    else:
        print("Solve timeout - continuing to submission")

    completed_question_ids.add(question_id)  # if not already set
    final_answer = vote_answer(question_id, force_answer=True)
    assert final_answer is not None
    time_taken = time.time() - question_start_time
    print(f"Submitting {final_answer} for {question_id} in {time_taken:.1f}s")
    save_findings(question_id, final_answer, time_taken)
    save_stats(
        question_id=question_id,
        final_answer=final_answer,
        time_taken=time_taken,
        time_available=time_available,
        active_solvers=question_id_to_active_solver_indexes[question_id],
        answers=question_id_to_solver_index_to_answer[question_id],
        proposals=question_id_to_solver_index_to_proposal[question_id],
        answer_history=dict(question_id_to_solver_index_to_answer_history[question_id]),
        proposal_history=dict(
            question_id_to_solver_index_to_proposal_history[question_id]
        ),
        submission_history=dict(
            question_id_to_solver_index_to_submission_history[question_id]
        ),
        backtrack_counts=dict(
            question_id_to_solver_index_to_backtrack_count[question_id]
        ),
        main_tokens=dict(question_id_to_solver_index_to_main_token_length[question_id]),
        support_tokens=dict(
            question_id_to_solver_index_to_support_token_length[question_id]
        ),
        tool_use_counts=dict(
            question_id_to_solver_index_to_tool_use_count[question_id]
        ),
        num_findings=len(question_id_to_findings[question_id]),
        num_acceptances=sum(
            len(finding.accepted_solver_indexes_and_reviews)
            for finding in question_id_to_findings[question_id]
        ),
        num_rejections=sum(
            len(finding.rejected_solver_indexes_and_reviews)
            for finding in question_id_to_findings[question_id]
        ),
    )
    return final_answer

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-24T08:27:06.481067Z","iopub.execute_input":"2025-11-24T08:27:06.481501Z","iopub.status.idle":"2025-11-24T08:27:19.404564Z","shell.execute_reply.started":"2025-11-24T08:27:06.481486Z","shell.execute_reply":"2025-11-24T08:27:19.404121Z"}}
if is_on_kaggle_interactive():
    solve("What is 1+1?")


if not is_on_kaggle() and __name__ == "__main__":
    # stack trace without inference_server is easier to debug
    print("solving")

    question_id = "dd7f5e"
    question_text = """
Let $\\mathcal{F}$ be the set of functions $\\alpha \\colon \\mathbb{Z}\\to \\mathbb{Z}$ for which there are only finitely many $n \\in \\mathbb{Z}$ such that $\\alpha(n) \\neq 0$. 

For two functions $\\alpha$ and $\\beta$ in $\\mathcal{F}$, define their product $\\alpha\\star\\beta$ to be $\\sum\\limits_{n\\in\\mathbb{Z}} \\alpha(n)\\cdot \\beta(n)$. Also, for $n\\in\\mathbb{Z}$, define a shift operator $S_n \\colon \\mathcal{F}\\to \\mathcal{F}$ by $S_n(\\alpha)(t)=\\alpha(t+n)$ for all $t \\in \\mathbb{Z}$.

A function $\\alpha \\in \\mathcal{F}$ is called \\emph{shifty} if 
\\begin{itemize}
    \\item $\\alpha(m)=0$ for all integers $m<0$ and $m>8$ and
    \\item There exists $\\beta \\in \\mathcal{F}$ and integers $k \\neq l$ such that for all $n \\in \\mathbb{Z}$
    \\begin{equation*}
        S_n(\\alpha)\\star\\beta =
        \\begin{cases}
            1 & n \\in \\{k,l\\} \\\\
            0 & n \\not \\in \\{k,l\\}
        \\end{cases}
        \\; .
    \\end{equation*}
\\end{itemize}
How many shifty functions are there in $\\mathcal{F}$?
""".strip()

    #     question_id = "92ba6a"
    #     question_text = """
    # Alice and Bob are each holding some integer number of sweets. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?
    # """.strip()

    #     question_id = "641659"
    #     question_text = """
    # Let $ABC$ be a triangle with $AB \\neq AC$, circumcircle $\\Omega$, and incircle $\\omega$. Let the contact points of $\\omega$ with $BC$, $CA$, and $AB$ be $D$, $E$, and $F$, respectively. Let the circumcircle of $AFE$ meet $\\Omega$ at $K$ and let the reflection of $K$ in $EF$ be $K'$. Let $N$ denote the foot of the perpendicular from $D$ to $EF$. The circle tangent to line $BN$ and passing through $B$ and $K$ intersects $BC$ again at $T \\neq B$.

    # Let sequence $(F_n)_{n \\geq 0}$ be defined by $F_0 = 0$, $F_1 = 1$ and for $n \\geq 2$, $F_n = F_{n-1} + F_{n-2}$. Call $ABC$ $n$\\emph{-tastic} if $BD = F_n$, $CD = F_{n+1}$, and $KNK'B$ is cyclic. Across all $n$-tastic triangles, let $a_n$ denote the maximum possible value of $\\frac{CT \\cdot NB}{BT \\cdot NE}$. Let $\\alpha$ denote the smallest real number such that for all sufficiently large $n$, $a_{2n} < \\alpha$. Given that $\\alpha = p + \\sqrt{q}$ for rationals $p$ and $q$, what is the remainder when $\\left\\lfloor p^{q^p} \\right\\rfloor$ is divided by $99991$?
    # """.strip()

    question_id = "86e8e5"
    question_text = """
Let $n \\geq 6$ be a positive integer. We call a positive integer $n$-Norwegian if it has three distinct positive divisors whose sum is equal to $n$. Let $f(n)$ denote the smallest $n$-Norwegian positive integer. Let $M=3^{2025!}$ and for a non-negative integer $c$ define
\\begin{equation*}
    g(c)=\\frac{1}{2025!}\\left\\lfloor \\frac{2025! f(M+c)}{M}\\right\\rfloor.
\\end{equation*}
We can write
\\begin{equation*}
    g(0)+g(4M)+g(1848374)+g(10162574)+g(265710644)+g(44636594)=\\frac{p}{q}
\\end{equation*}
where $p$ and $q$ are coprime positive integers. What is the remainder when $p+q$ is divided by $99991$?
""".strip()

    #     question_id = "pe_818"
    #     question_text = """
    # The SET® card game is played with a pack of $81$ distinct cards. Each card has four features (Shape, Color, Number, Shading). Each feature has three different variants (e.g. Color can be red, purple, green).

    # A SET consists of three different cards such that each feature is either the same on each card or different on each card.

    # For a collection $C_n$ of $n$ cards, let $S(C_n)$ denote the number of SETs in $C_n$. Then define $F(n) = \\sum\\limits_{C_n} S(C_n)^4$ where $C_n$ ranges through all collections of $n$ cards (among the $81$ cards).
    # You are given $F(3) = 1080$ and $F(6) = 159690960$.

    # Find $F(12)$.

    # Give your answer modulo $99991$.
    # """

    os.makedirs(f"{SOLUTIONS_DIR}/{question_id}", exist_ok=True)
    solve(question_text, question_id)
    exit()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Submission server

# %% [code] {"_kg_hide-output":true,"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-11-24T02:04:57.769Z"}}
import os

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_3_inference_server

if is_on_kaggle():
    pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    ).drop("answer", axis=1).to_csv("reference.csv", index=False)


if is_on_kaggle():
    if run_all_questions_on_kaggle is True:
        # otherwise we need to change cutoff_times
        replication_count_for_commit_runs = min(replication_count_for_commit_runs, 5)

    df = pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    ).drop("answer", axis=1)
    dfs = []
    for replication_idx in range(replication_count_for_commit_runs):
        df_copy = df.copy()
        df_copy["id"] = df_copy["id"] + f"_{replication_idx}"
        dfs.append(df_copy)
    pd.concat(dfs, ignore_index=True).to_csv("reference.csv", index=False)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 99999, inclusive.
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    question_id: str = id_.item(0)
    question_text: str = problem.item(0)

    if not run_all_questions_on_kaggle:  # should be ignored for submissions
        if is_on_kaggle_commit():
            if serve_vllm_on_kaggle:
                # to only run for hard problems
                if not (
                    # "Norwe" in question_text  # noqa: E713
                    # or "Alice" in question_text
                    # or "tournament" in question_text
                    # or "KNK" in question_text
                    "shifty" in question_text
                ):
                    print("on kaggle commit serving vllm, skipping question")
                    # not popping cutoff_times
                    return pl.DataFrame({"id": id_, "answer": 12315})

    if not is_on_kaggle():
        # if you want to debug a particular question locally
        # prefer to use without inference server for easier error tracing
        if not (
            "shifty" in question_text  # noqa: E713
            or "tournament" in question_text
            or "KNK" in question_text
            or "Norwe" in question_text
        ):
            print("not on kaggle, skipping question")
            # not popping cutoff_times
            return pl.DataFrame({"id": id_, "answer": 12315})

    if "Norwe" in question_text:
        question_text = """
Let $\sigma(n)$ be the sum of all the divisors of the positive integer $n$, for example:

$\sigma(10) = 1+2+5+10 = 18$.

Define $T(N)$ to be the sum of all numbers $n \le N$ such that when the fraction $\frac{\sigma(n)}{n}$ is written in its lowest form $\frac ab$, the denominator is a power of 3 i.e. $b = 3^k, k &gt; 0$.

You are given $T(100) = 270$ and $T(10^6) = 26089287$.

Find $T(10^{14})$.

Give your answer modulo $99991$.
""".strip()
    
    # Make a prediction
    prediction = solve(question_text, question_id=question_id)
    completed_question_ids.add(question_id)
    cutoff_times.pop()
    return pl.DataFrame({"id": id_, "answer": prediction})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict  # type: ignore[arg-type]
)

print("Starting submission server")
if __name__ == "__main__":
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("reference.csv",))

# %% [code] {"_kg_hide-input":false,"jupyter":{"outputs_hidden":false}}
