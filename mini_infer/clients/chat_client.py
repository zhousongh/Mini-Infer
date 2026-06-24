"""Interactive chat client for the local mini-infer HTTP server."""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat client for mini-infer")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default="mini-infer")
    parser.add_argument("--system", type=str, default="你是一个简洁、专业的中文助手。")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true", help="disable SSE streaming output")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="respect HTTP(S)_PROXY env vars even for localhost",
    )
    parser.add_argument(
        "--quick-dry-run",
        action="store_true",
        help="auto-start a temporary local dry-run server for quick testing",
    )
    parser.add_argument(
        "--quick-model-path",
        type=str,
        default="",
        help="auto-start a temporary local real-model server with the given model path",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device for temporary real-model server")
    return parser.parse_args(argv)


@dataclass(slots=True)
class _ServerHandle:
    proc: subprocess.Popen[bytes]
    log_path: str


def _build_opener(base_url: str, use_env_proxy: bool) -> urllib.request.OpenerDirector:
    if use_env_proxy:
        return urllib.request.build_opener()
    host = urllib.parse.urlparse(base_url).hostname
    if host in {"127.0.0.1", "localhost"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(_format_url_error(exc)) from exc


def _stream_chat(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[str, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    finish_reason = "stop"
    try:
        with opener.open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)
                    parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(_format_url_error(exc)) from exc
    print()
    return "".join(parts), finish_reason


def _non_stream_chat(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[str, str]:
    response = _post_json(opener, url, payload, timeout)
    choice = response["choices"][0]
    text = choice["message"]["content"]
    print(text)
    return text, choice.get("finish_reason") or "stop"


def _print_help() -> None:
    print("commands: /clear reset history, /history show history, /help show help, exit quit")


def _format_url_error(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
        return (
            "无法连接到本地 mini-infer 服务。\n"
            "请先启动服务，例如：\n"
            "  conda run -n ai-infra python serve.py --dry-run --port 8000\n"
            "或真实模型：\n"
            "  conda run -n ai-infra python serve.py --model /path/to/Qwen2.5-7B-Instruct --port 8000"
        )
    return f"request failed: {reason}"


def _print_stdin_hint() -> None:
    print("chat.py 需要交互式 stdin。")
    print("如果你在用 conda run，请改用：")
    print("  conda run --no-capture-output -n ai-infra python chat.py")
    print("或者先激活环境再运行：")
    print("  conda activate ai-infra")
    print("  python chat.py")


def _check_server(opener: urllib.request.OpenerDirector, base_url: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"server returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(_format_url_error(exc)) from exc


def _base_url_host_port(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, parsed.port
    if parsed.scheme == "https":
        return host, 443
    return host, 80


def _cleanup_process(handle: _ServerHandle | None) -> None:
    if handle is None:
        return
    proc = handle.proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3.0)
    try:
        os.unlink(handle.log_path)
    except FileNotFoundError:
        return

def _read_log_tail(log_path: str, limit: int = 3000) -> str:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    return text[-limit:].strip()


def _start_temporary_server(args: argparse.Namespace) -> _ServerHandle:
    host, port = _base_url_host_port(args.base_url)
    if host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("quick mode 仅支持本地服务地址（127.0.0.1 / localhost）。")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmd = [sys.executable, "serve.py", "--port", str(port)]
    if args.quick_dry_run:
        cmd.append("--dry-run")
        mode = "dry-run"
    else:
        cmd.extend(["--model", args.quick_model_path, "--device", args.device])
        mode = "real-model"

    log_file = tempfile.NamedTemporaryFile(
        prefix="mini_infer_quick_chat_",
        suffix=".log",
        delete=False,
    )
    log_path = log_file.name
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    proc = subprocess.Popen(  # noqa: S603 - local trusted script in current repo.
        cmd,
        cwd=repo_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log_file.close()
    handle = _ServerHandle(proc=proc, log_path=log_path)
    atexit.register(_cleanup_process, handle)
    if args.quick_dry_run:
        print(f"[info] starting temporary {mode} server on http://{host}:{port}/v1 ...")
        print("[info] dry-run uses fake token output like [1] [2]; use --real for real model output.")
    else:
        print(f"[info] starting temporary {mode} server on http://{host}:{port}/v1 ...")
        print(f"[info] model={args.quick_model_path}  device={args.device}")
    return handle


def _ensure_server(args: argparse.Namespace, opener: urllib.request.OpenerDirector) -> _ServerHandle | None:
    try:
        _check_server(opener, args.base_url, min(args.timeout, 5.0))
        return None
    except Exception:
        if not (args.quick_dry_run or args.quick_model_path):
            raise

    handle = _start_temporary_server(args)
    startup_error: Exception | None = None
    max_wait = 60.0 if not args.quick_dry_run else 10.0
    interval = 0.2
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            _check_server(opener, args.base_url, 1.0)
            print("[info] temporary server is ready.")
            return handle
        except Exception as exc:  # noqa: BLE001 - keep polling until timeout.
            startup_error = exc
            if handle.proc.poll() is not None:
                break
            time.sleep(interval)

    log_tail = _read_log_tail(handle.log_path)
    _cleanup_process(handle)
    assert startup_error is not None
    if log_tail:
        raise RuntimeError(
            "temporary server failed to start.\n"
            f"{startup_error}\n\n"
            "--- server log tail ---\n"
            f"{log_tail}"
        ) from startup_error
    raise RuntimeError(f"temporary server failed to start: {startup_error}") from startup_error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not sys.stdin.isatty():
        _print_stdin_hint()
        return 2
    opener = _build_opener(args.base_url, args.use_env_proxy)
    chat_url = args.base_url.rstrip("/") + "/chat/completions"
    messages: list[dict[str, str]] = []
    server_proc: _ServerHandle | None = None
    if args.system:
        messages.append({"role": "system", "content": args.system})

    try:
        server_proc = _ensure_server(args, opener)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise startup failure.
        print(f"[error] {exc}")
        return 2

    print(f"mini-infer chat -> {args.base_url}  model={args.model}")
    _print_help()

    try:
        while True:
            try:
                user_text = input("\n你: ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("\n[exit]")
                return 130

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                return 0
            if user_text == "/help":
                _print_help()
                continue
            if user_text == "/clear":
                messages = [{"role": "system", "content": args.system}] if args.system else []
                print("[history cleared]")
                continue
            if user_text == "/history":
                for message in messages:
                    print(f"{message['role']}: {message['content']}")
                continue

            messages.append({"role": "user", "content": user_text})
            payload = {
                "model": args.model,
                "messages": messages,
                "stream": not args.no_stream,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }

            print("模型: ", end="", flush=True)
            try:
                if args.no_stream:
                    answer, _ = _non_stream_chat(opener, chat_url, payload, args.timeout)
                else:
                    answer, _ = _stream_chat(opener, chat_url, payload, args.timeout)
            except KeyboardInterrupt:
                print("\n[interrupted]")
                messages.pop()
                continue
            except Exception as exc:  # noqa: BLE001 - CLI should show concise failure.
                print(f"\n[error] {exc}")
                messages.pop()
                continue

            messages.append({"role": "assistant", "content": answer})
    finally:
        _cleanup_process(server_proc)
