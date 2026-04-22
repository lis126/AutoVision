#!/usr/bin/env python3
"""
Open-Ad-Batch: configurable third-party API batch poster generator.

The script keeps creative inference out of the calling agent context by using a
small text-model API call per round, then sends the final prompt to the image
model API. It assumes OpenAI-compatible endpoints by default and allows explicit
endpoint overrides for third-party platforms.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ASPECT_RATIO = ""
DEFAULT_SIZE = ""
DEFAULT_TOTAL = 1
DEFAULT_COOLDOWN_SECONDS = 0
DEFAULT_TEMPLATE = "hanfu-character-sheet"
DEFAULT_IMAGE_BACKEND = "codex"
DEFAULT_IMAGE_MODEL = "gpt-image-2-all"
DEFAULT_IMAGE_API_MODE = "chat"
DEFAULT_TEXT_MODEL = "gpt-5.4"
DEFAULT_WORKSPACE_ROOT = Path.home() / "AutoVision" / "open-ad-batch-runs"
SKILL_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCES: dict[str, str] = {}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            ENV_SOURCES[key] = str(path)


def load_env_defaults() -> None:
    load_env_file(SKILL_ROOT / ".env")
    load_env_file(Path.cwd() / ".env")


def env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def env_source(*names: str) -> str:
    for name in names:
        if os.environ.get(name):
            return ENV_SOURCES.get(name, "environment")
    return "default" if names else "-"


def shutil_which(command: str) -> str | None:
    return shutil.which(command)


def prompt_if_missing(value: str | None, label: str, interactive: bool, secret: bool = False) -> str:
    if value:
        return value
    if not interactive:
        raise SystemExit(f"Missing {label}. Set it with an environment variable or CLI option.")
    if secret:
        entered = getpass.getpass(f"{label}: ").strip()
    else:
        entered = input(f"{label}: ").strip()
    if not entered:
        raise SystemExit(f"{label} is required.")
    return entered


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    return cleaned or "product"


def default_workspace(product: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_WORKSPACE_ROOT / f"{slugify(product or 'task')}-{stamp}"


def normalize_workspace(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return value.expanduser()
    return Path(str(value)).expanduser()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def template_values(args: argparse.Namespace, round_num: int, used_styles: list[str]) -> dict[str, Any]:
    avoid_styles = used_styles[-8:]
    return {
        "template": args.template,
        "product": args.product,
        "round": round_num,
        "total": args.total,
        "aspect_ratio": args.aspect_ratio or "由推理模型根据目标图片模型和版式自行选择",
        "size": args.size or "由推理模型根据目标图片模型自行选择",
        "language": args.language,
        "requirements": args.requirements or "无",
        "role_setting": args.role_setting or "无",
        "avoid_styles_text": "、".join(avoid_styles) if avoid_styles else "无",
        "avoid_styles_json": json.dumps(avoid_styles, ensure_ascii=False),
        "date_utc": now_iso(),
    }


def render_template_value(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(SafeFormatDict(values))
    if isinstance(value, list):
        return [render_template_value(item, values) for item in value]
    if isinstance(value, dict):
        return {key: render_template_value(item, values) for key, item in value.items()}
    return value


def candidate_template_paths(name_or_path: str, workspace: Path) -> list[Path]:
    raw = Path(name_or_path).expanduser()
    candidates = [raw]
    if raw.suffix:
        candidates.extend([
            workspace / "templates" / raw.name,
            SKILL_ROOT / "assets" / "templates" / raw.name,
        ])
    else:
        for suffix in (".json", ".txt"):
            candidates.extend([
                workspace / "templates" / f"{name_or_path}{suffix}",
                SKILL_ROOT / "assets" / "templates" / f"{name_or_path}{suffix}",
            ])
    return candidates


def resolve_template_path(name_or_path: str, workspace: Path) -> Path:
    for path in candidate_template_paths(name_or_path, workspace):
        if path.exists() and path.is_file():
            return path
    searched = "\n".join(f"- {path}" for path in candidate_template_paths(name_or_path, workspace))
    raise SystemExit(f"Template not found: {name_or_path}\nSearched:\n{searched}")


def load_template(args: argparse.Namespace) -> dict[str, Any]:
    if args.workspace is None:
        args.workspace = DEFAULT_WORKSPACE_ROOT
    path = resolve_template_path(args.template, args.workspace)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            template = json.load(f)
    else:
        template = {
            "name": path.stem,
            "system": "Return strict JSON only. Do not include markdown or commentary.",
            "user": path.read_text(encoding="utf-8"),
            "response_format": {"type": "json_object"},
        }
    template["_path"] = str(path)
    return template


def load_role_setting(args: argparse.Namespace) -> str:
    parts = []
    if args.role_setting:
        parts.append(args.role_setting.strip())
    if args.role_setting_file:
        path = Path(args.role_setting_file).expanduser()
        if not path.exists():
            raise SystemExit(f"Role setting file not found: {path}")
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in parts if part)


def load_final_prompt(args: argparse.Namespace) -> str:
    parts = []
    if args.final_prompt:
        parts.append(args.final_prompt.strip())
    if args.final_prompt_file:
        path = Path(args.final_prompt_file).expanduser()
        if not path.exists():
            raise SystemExit(f"Final prompt file not found: {path}")
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in parts if part)


def build_text_messages(template: dict[str, Any], values: dict[str, Any]) -> list[dict[str, str]]:
    rendered = render_template_value(template, values)
    role_title = rendered.get("role_title", "模板提示词生成器")
    template_name = rendered.get("template_name") or rendered.get("name") or values.get("template", "")
    system = rendered.get("system") or (
        f"你是【{role_title}】。你的任务是根据模板名、输入信息和最终提示词格式，"
        "输出可直接传给图片模型的最终提示词。只返回严格 JSON，不要解释，不要输出 Markdown。"
    )
    if "reasoning_model_input" in rendered or "final_prompt_format" in rendered:
        user = {
            "template_name": template_name,
            "role_title": f"【{role_title}】",
            "reasoning_model_input": rendered.get("reasoning_model_input", {}),
            "final_prompt_format": rendered.get("final_prompt_format", ""),
            "response_contract": rendered.get("response_contract", {
                "style": "本轮风格名称，用于历史去重",
                "final_prompt": "完整图片生成提示词"
            }),
        }
    else:
        user = rendered.get("user", "")
    if not isinstance(user, str):
        user = json.dumps(user, ensure_ascii=False)
    messages = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(user)})
    return messages


def list_templates(workspace: Path) -> None:
    workspace = workspace or DEFAULT_WORKSPACE_ROOT
    roots = [SKILL_ROOT / "assets" / "templates", workspace / "templates"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".json", ".txt"}:
                continue
            description = ""
            if path.suffix.lower() == ".json":
                try:
                    with path.open("r", encoding="utf-8") as f:
                        description = str(json.load(f).get("description", ""))
                except Exception:
                    description = ""
            found.append((path.stem, str(path), description))
    if not found:
        print("No templates found.")
        return
    for name, path, description in found:
        suffix = f" - {description}" if description else ""
        print(f"{name}: {path}{suffix}")


def env_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def configure(args: argparse.Namespace) -> None:
    env_path = Path(args.config_path).expanduser() if args.config_path else SKILL_ROOT / ".env"
    print(f"Writing local config to: {env_path}")
    print("This file should stay local and must not be committed.")

    base_url = input("API base URL (for example https://your-platform.example/v1): ").strip()
    if not base_url:
        raise SystemExit("API base URL is required.")
    api_key = getpass.getpass("API key: ").strip()
    if not api_key:
        raise SystemExit("API key is required.")
    image_model = input(f"Image model name [{DEFAULT_IMAGE_MODEL}]: ").strip() or DEFAULT_IMAGE_MODEL
    text_model = input(f"Text/reasoning model [{DEFAULT_TEXT_MODEL}]: ").strip() or DEFAULT_TEXT_MODEL
    response_format = input("Image response_format [b64_json, or none if unsupported]: ").strip() or "b64_json"

    lines = [
        "# Local Open Ad Batch configuration. Do not commit this file.",
        f"AD_API_BASE_URL={env_quote(base_url)}",
        f"AD_API_KEY={env_quote(api_key)}",
        f"AD_IMAGE_MODEL={env_quote(image_model)}",
        f"AD_IMAGE_API_MODE={env_quote(DEFAULT_IMAGE_API_MODE)}",
        f"AD_TEXT_MODEL={env_quote(text_model)}",
        f"AD_IMAGE_RESPONSE_FORMAT={env_quote(response_format)}",
    ]
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Config saved.")


def mask_secret(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def doctor(args: argparse.Namespace) -> None:
    display_workspace = args.workspace or DEFAULT_WORKSPACE_ROOT
    rows = [
        ("skill_root", str(SKILL_ROOT), "detected"),
        ("workspace", str(display_workspace), "argument/default"),
        ("template", args.template, env_source("AD_TEMPLATE", "AD_TEMPLATE_NAME")),
        ("api_base_url", args.image_base_url or args.text_base_url or "missing", env_source("AD_IMAGE_BASE_URL", "AD_TEXT_BASE_URL", "AD_API_BASE_URL")),
        ("api_key", mask_secret(args.image_api_key or args.text_api_key), env_source("AD_IMAGE_API_KEY", "AD_TEXT_API_KEY", "AD_API_KEY", "OPENAI_API_KEY")),
        ("image_backend", args.image_backend, env_source("AD_IMAGE_BACKEND")),
        ("text_model", args.text_model or "missing", env_source("AD_TEXT_MODEL")),
        ("image_model", args.image_model or "missing", env_source("AD_IMAGE_MODEL")),
        ("image_api_mode", args.image_api_mode, env_source("AD_IMAGE_API_MODE")),
        ("image_endpoint", args.image_endpoint or ("OpenAI-compatible /chat/completions" if args.image_api_mode == "chat" else "OpenAI-compatible /images/generations"), env_source("AD_IMAGE_ENDPOINT")),
        ("text_endpoint", args.text_endpoint or "OpenAI-compatible /chat/completions", env_source("AD_TEXT_ENDPOINT")),
        ("image_response_format", args.response_format or "not sent", env_source("AD_IMAGE_RESPONSE_FORMAT")),
        ("aspect_ratio", args.aspect_ratio or "not sent; reasoning model chooses", env_source("AD_ASPECT_RATIO")),
        ("size", args.size or "not sent", env_source("AD_IMAGE_SIZE")),
    ]
    print("Open Ad Batch effective configuration:")
    for key, value, source in rows:
        print(f"- {key}: {value} ({source})")
    problems = []
    if args.image_backend == "api" and not (args.image_base_url or args.image_endpoint):
        problems.append("AD_IMAGE_BASE_URL/AD_API_BASE_URL or AD_IMAGE_ENDPOINT is required.")
    if args.image_backend == "api" and not args.image_api_key:
        problems.append("AD_IMAGE_API_KEY/AD_API_KEY is required.")
    if args.image_backend == "codex" and not shutil_which(args.codex_command):
        problems.append(f"Codex CLI command not found: {args.codex_command}")
    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("\nConfig looks runnable.")


def log(workspace: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "batch_run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def join_endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


def http_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {text[:500]}") from exc


def find_latest_codex_image(since_ts: float | None = None) -> Path | None:
    roots = [
        Path.home() / ".codex" / "generated_images",
        Path.cwd() / "generated_images",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp"):
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if since_ts is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= since_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_codex_cli(args: argparse.Namespace, prompt: str, round_dir: Path) -> tuple[bytes, str, dict[str, Any]]:
    prompt_file = round_dir / "_codex_prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    started = time.time()
    cmd = [
        args.codex_command,
        "exec",
        "--full-auto",
        "--skip-git-repo-check",
        "--cd",
        str(round_dir),
    ]
    with prompt_file.open("r", encoding="utf-8") as stdin:
        result = subprocess.run(
            cmd,
            stdin=stdin,
            text=True,
            capture_output=True,
            timeout=args.codex_timeout,
        )
    (round_dir / "codex_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (round_dir / "codex_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI failed with exit code {result.returncode}. See codex_stderr.txt.")
    image_path = find_latest_codex_image(started)
    if not image_path:
        raise RuntimeError("Codex CLI completed, but no generated image was found in ~/.codex/generated_images.")
    return image_path.read_bytes(), image_path.suffix or ".png", {
        "backend": "codex",
        "source_image_path": str(image_path),
        "stdout_path": str(round_dir / "codex_stdout.txt"),
        "stderr_path": str(round_dir / "codex_stderr.txt"),
    }


def download_url(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "open-ad-batch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def decode_image_reference(ref: str, timeout: int) -> tuple[bytes, str]:
    ref = ref.strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        parsed = urllib.parse.urlparse(ref)
        return download_url(ref, timeout), Path(parsed.path).suffix or ".png"
    if ref.lower().startswith("data:") and "," in ref:
        header, b64 = ref.split(",", 1)
        mime = header.split(";", 1)[0].replace("data:", "")
        return base64.b64decode(b64), mimetypes.guess_extension(mime) or ".png"
    return base64.b64decode(ref), ".png"


def extract_image_reference(text: str) -> str | None:
    markdown = re.search(r"!\[[^\]]*\]\(([^)]+)\)", text)
    if markdown:
        return markdown.group(1).strip()
    data_url = re.search(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", text)
    if data_url:
        return data_url.group(0).replace("\n", "").replace(" ", "")
    url = re.search(r"https?://[^\s)\"']+", text)
    if url:
        return url.group(0)
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def infer_brief(
    args: argparse.Namespace,
    round_num: int,
    used_styles: list[str],
    text_api_key: str | None,
    template: dict[str, Any],
) -> dict[str, str]:
    if args.final_prompt:
        values = template_values(args, round_num, used_styles)
        prompt = str(render_template_value(args.final_prompt, values)).strip()
        return {
            "style": f"precomputed-final-prompt-{round_num}",
            "headline": str(template.get("role_title", "")),
            "prompt": prompt,
        }

    if args.no_infer or not text_api_key or not args.text_model:
        style = f"fresh commercial direction {round_num}"
        return {
            "style": style,
            "headline": "",
            "prompt": build_fallback_prompt(args, style, used_styles),
        }

    endpoint = args.text_endpoint or join_endpoint(args.text_base_url, "/chat/completions")
    values = template_values(args, round_num, used_styles)
    messages = build_text_messages(template, values)
    payload = {
        "model": args.text_model,
        "messages": messages,
        "temperature": args.text_temperature,
    }
    response_format = template.get("response_format", {"type": "json_object"})
    if response_format:
        payload["response_format"] = response_format
    data = http_json(endpoint, payload, text_api_key, args.timeout)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        brief = extract_json_object(content)
    except Exception:
        style = f"fallback direction {round_num}"
        return {
            "style": style,
            "headline": "",
            "prompt": build_fallback_prompt(args, style, used_styles),
        }
    prompt = str(brief.get("final_prompt") or brief.get("prompt") or "").strip()
    if not prompt:
        prompt = build_fallback_prompt(args, str(brief.get("style") or ""), used_styles)
    return {
        "style": str(brief.get("style") or f"round-{round_num}").strip(),
        "headline": str(brief.get("headline") or brief.get("role_title") or "").strip(),
        "prompt": prompt,
    }


def build_fallback_prompt(args: argparse.Namespace, style: str, used_styles: list[str]) -> str:
    avoid = "、".join(used_styles[-8:]) if used_styles else "无"
    parts = [
        f"模板：{args.template}",
        f"输出标识：{args.product}",
        f"画幅：{args.aspect_ratio or '由推理模型根据目标图片模型和版式自行选择'}",
        f"本轮方向：{style or '按模板要求生成'}",
        f"历史需避开：{avoid}",
    ]
    if args.role_setting:
        parts.append(f"角色设定：{args.role_setting}")
    if args.requirements:
        parts.append(f"额外要求：{args.requirements}")
    parts.append("请严格遵循所选模板的最终提示词格式，生成可直接传给图片模型的中文提示词。")
    return "\n".join(parts)


def parse_image_response(data: dict[str, Any], timeout: int) -> tuple[bytes, str]:
    item: Any = None
    if isinstance(data.get("choices"), list) and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content", "")
        if isinstance(content, str):
            ref = extract_image_reference(content)
            if ref:
                return decode_image_reference(ref, timeout)
    if isinstance(data.get("data"), list) and data["data"]:
        item = data["data"][0]
    elif isinstance(data.get("images"), list) and data["images"]:
        item = data["images"][0]
    elif data.get("image"):
        item = data["image"]
    else:
        item = data

    if isinstance(item, str):
        ref = extract_image_reference(item) or item
        return decode_image_reference(ref, timeout)

    if not isinstance(item, dict):
        raise RuntimeError(f"Unsupported image response shape: {json.dumps(data)[:500]}")

    b64 = item.get("b64_json") or item.get("base64") or item.get("image_base64")
    if b64:
        if "," in b64 and b64.lower().startswith("data:"):
            header, b64 = b64.split(",", 1)
            mime = header.split(";", 1)[0].replace("data:", "")
            return base64.b64decode(b64), mimetypes.guess_extension(mime) or ".png"
        return base64.b64decode(b64), ".png"

    url = item.get("url") or item.get("image_url")
    if url:
        return decode_image_reference(url, timeout)

    raise RuntimeError(f"No image URL/base64 found in response: {json.dumps(data)[:500]}")


def generate_image(
    args: argparse.Namespace,
    image_api_key: str,
    prompt: str,
) -> tuple[bytes, str, dict[str, Any]]:
    if args.image_api_mode == "chat":
        endpoint = args.image_endpoint or join_endpoint(args.image_base_url, "/chat/completions")
        payload = {
            "model": args.image_model,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        endpoint = args.image_endpoint or join_endpoint(args.image_base_url, "/images/generations")
        payload = {
            "model": args.image_model,
            "prompt": prompt,
            "n": 1,
        }
        if args.size:
            payload["size"] = args.size
        if args.response_format and args.response_format.lower() != "none":
            payload["response_format"] = args.response_format
        if args.aspect_ratio:
            payload["aspect_ratio"] = args.aspect_ratio
    if args.extra_image_params:
        payload.update(json.loads(args.extra_image_params))
    data = http_json(endpoint, payload, image_api_key, args.timeout)
    image_bytes, ext = parse_image_response(data, args.timeout)
    return image_bytes, ext, data


def generate_image_with_backend(
    args: argparse.Namespace,
    image_api_key: str | None,
    prompt: str,
    round_dir: Path,
) -> tuple[bytes, str, dict[str, Any]]:
    if args.image_backend == "codex":
        return run_codex_cli(args, prompt, round_dir)
    if not image_api_key:
        raise RuntimeError("Image API key is required when --image-backend api.")
    return generate_image(args, image_api_key, prompt)


def load_context(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    default = {
        "product": args.product,
        "total": args.total,
        "done": 0,
        "round": 0,
        "state": "IDLE",
        "history": [],
        "created_at": now_iso(),
    }
    ctx = read_json(path, default)
    ctx["product"] = args.product
    ctx["total"] = args.total
    return ctx


def status(args: argparse.Namespace) -> None:
    if args.workspace is None:
        args.workspace = DEFAULT_WORKSPACE_ROOT
    ctx_path = args.workspace / "context.json"
    if not ctx_path.exists():
        print(f"No context found at {ctx_path}")
        return
    ctx = read_json(ctx_path, {})
    print(
        f"{ctx.get('product', '-')}: {ctx.get('done', 0)}/{ctx.get('total', 0)} "
        f"state={ctx.get('state', '-')} round={ctx.get('round', 0)}"
    )


def run(args: argparse.Namespace) -> None:
    image_api_key = args.image_api_key
    if args.image_backend == "api":
        image_api_key = prompt_if_missing(args.image_api_key, "Image API key", args.interactive, secret=True)
    text_api_key = args.text_api_key or image_api_key
    args.product = prompt_if_missing(args.product, "Product name", args.interactive)
    if args.workspace is None:
        args.workspace = default_workspace(args.product)
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.role_setting = load_role_setting(args)
    args.final_prompt = load_final_prompt(args)
    template = load_template(args)

    ctx_path = args.workspace / "context.json"
    if args.reset and ctx_path.exists():
        ctx_path.unlink()
    ctx = load_context(ctx_path, args)
    product_dir = args.workspace / "images" / f"{slugify(args.product)}_batch"
    product_dir.mkdir(parents=True, exist_ok=True)

    start = max(int(ctx.get("round", 0)) + 1, 1)
    used_styles = [
        item.get("style", "")
        for item in ctx.get("history", [])
        if item.get("status") == "ok" and item.get("style")
    ]

    log(args.workspace, f"Product: {args.product}")
    log(args.workspace, f"Workspace: {args.workspace}")
    log(args.workspace, f"Image model: {args.image_model}")
    log(args.workspace, f"Template: {template.get('name', args.template)} ({template.get('_path')})")
    log(args.workspace, f"Rounds: {start}-{args.total}, cooldown={args.cooldown}s")

    for round_num in range(start, args.total + 1):
        round_dir = product_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        ctx["state"] = "RUNNING"
        ctx["round"] = round_num
        ctx["round_workspace"] = str(round_dir)
        write_json(ctx_path, ctx)
        log(args.workspace, f"Round {round_num}/{args.total}: workspace {round_dir}")
        log(args.workspace, f"Round {round_num}/{args.total}: inferring brief")

        try:
            brief = infer_brief(args, round_num, used_styles, text_api_key, template)
            prompt_path = round_dir / "final_prompt.txt"
            prompt_path.write_text(brief["prompt"], encoding="utf-8")

            log(args.workspace, f"Round {round_num}/{args.total}: generating image")
            image_bytes, ext, raw = generate_image_with_backend(args, image_api_key, brief["prompt"], round_dir)
            image_path = round_dir / f"image{ext}"
            image_path.write_bytes(image_bytes)

            meta = {
                "round": round_num,
                "round_workspace": str(round_dir),
                "status": "ok",
                "style": brief["style"],
                "headline": brief["headline"],
                "prompt_path": str(prompt_path),
                "image_path": str(image_path),
                "image_model": args.image_model,
                "image_backend": args.image_backend,
                "text_model": args.text_model if not args.no_infer else None,
                "template": template.get("name", args.template),
                "template_path": template.get("_path"),
                "time": now_iso(),
            }
            if args.save_raw_response:
                raw_path = round_dir / "raw_response.json"
                write_json(raw_path, raw)
                meta["raw_response_path"] = str(raw_path)
            manifest_path = round_dir / "manifest.json"
            write_json(manifest_path, meta)
            meta["manifest_path"] = str(manifest_path)

            ctx["done"] = max(int(ctx.get("done", 0)), round_num)
            ctx.setdefault("history", []).append(meta)
            used_styles.append(brief["style"])
            log(args.workspace, f"Saved: {image_path}")
        except Exception as exc:
            error_meta = {
                "round": round_num,
                "round_workspace": str(round_dir),
                "status": "fail",
                "error": str(exc),
                "time": now_iso(),
            }
            write_json(round_dir / "manifest.json", error_meta)
            ctx.setdefault("history", []).append({
                "round": round_num,
                "round_workspace": str(round_dir),
                "status": "fail",
                "error": str(exc),
                "time": now_iso(),
            })
            write_json(ctx_path, ctx)
            log(args.workspace, f"Round {round_num} failed: {exc}")
            if not args.continue_on_error:
                raise SystemExit(1) from exc

        write_json(ctx_path, ctx)
        if round_num < args.total and args.cooldown > 0:
            ctx["state"] = "COOLING"
            write_json(ctx_path, ctx)
            log(args.workspace, f"Cooldown: {args.cooldown}s")
            time.sleep(args.cooldown)

    ctx["state"] = "DONE"
    write_json(ctx_path, ctx)
    log(args.workspace, f"Completed: {ctx.get('done', 0)}/{args.total}")
    log(args.workspace, f"Images: {product_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate Chinese ad posters with a third-party API.")
    parser.add_argument("command", nargs="?", choices=["run", "status", "templates", "configure", "doctor"], default="run")
    parser.add_argument("--config-path", default=env_first("AD_CONFIG_PATH"))
    parser.add_argument("--product", default=env_first("AD_PRODUCT"))
    parser.add_argument("--total", type=int, default=int(env_first("AD_TOTAL", default=str(DEFAULT_TOTAL))))
    parser.add_argument(
        "--workspace",
        type=Path,
        default=normalize_workspace(env_first("AD_WORKSPACE")),
        help="Run workspace. Defaults to ~/AutoVision/open-ad-batch-runs/<product>-<timestamp> so skill installs stay clean.",
    )
    parser.add_argument("--reset", action="store_true", help="Delete this workspace context.json before running, avoiding shell-level Remove-Item/rm commands.")
    parser.add_argument("--cooldown", type=int, default=int(env_first("AD_COOLDOWN_SECONDS", default=str(DEFAULT_COOLDOWN_SECONDS))))
    parser.add_argument("--aspect-ratio", default=env_first("AD_ASPECT_RATIO", default=DEFAULT_ASPECT_RATIO), help="Optional ratio hint for templates and APIs. Empty means the reasoning model should choose.")
    parser.add_argument("--size", default=env_first("AD_IMAGE_SIZE", default=DEFAULT_SIZE), help="Optional image API size. Empty means do not send a size field.")
    parser.add_argument("--language", default=env_first("AD_LANGUAGE", default="zh-CN"))
    parser.add_argument("--requirements", default=env_first("AD_REQUIREMENTS", default=""))
    parser.add_argument("--template", default=env_first("AD_TEMPLATE", "AD_TEMPLATE_NAME", default=DEFAULT_TEMPLATE))
    parser.add_argument("--role-setting", default=env_first("AD_ROLE_SETTING", default=""))
    parser.add_argument("--role-setting-file", default=env_first("AD_ROLE_SETTING_FILE"))
    parser.add_argument("--final-prompt", default=env_first("AD_FINAL_PROMPT", default=""))
    parser.add_argument("--final-prompt-file", default=env_first("AD_FINAL_PROMPT_FILE"))

    parser.add_argument("--image-backend", choices=["codex", "api"], default=env_first("AD_IMAGE_BACKEND", default=DEFAULT_IMAGE_BACKEND), help="Default codex uses Codex CLI; api uses the configured HTTP image API.")
    parser.add_argument("--codex-command", default=env_first("AD_CODEX_COMMAND", default="codex"))
    parser.add_argument("--codex-timeout", type=int, default=int(env_first("AD_CODEX_TIMEOUT_SECONDS", default="900")))

    parser.add_argument("--image-base-url", default=env_first("AD_IMAGE_BASE_URL", "AD_API_BASE_URL"))
    parser.add_argument("--image-endpoint", default=env_first("AD_IMAGE_ENDPOINT"))
    parser.add_argument("--image-api-key", default=env_first("AD_IMAGE_API_KEY", "AD_API_KEY", "OPENAI_API_KEY"))
    parser.add_argument("--image-model", default=env_first("AD_IMAGE_MODEL", default=DEFAULT_IMAGE_MODEL))
    parser.add_argument("--image-api-mode", choices=["chat", "images"], default=env_first("AD_IMAGE_API_MODE", default=DEFAULT_IMAGE_API_MODE), help="Use chat for gpt-image-2-all prompt adherence; use images for /images/generations compatibility.")
    parser.add_argument("--response-format", default=env_first("AD_IMAGE_RESPONSE_FORMAT", default="b64_json"))
    parser.add_argument("--extra-image-params", default=env_first("AD_EXTRA_IMAGE_PARAMS"))

    parser.add_argument("--text-base-url", default=env_first("AD_TEXT_BASE_URL", "AD_API_BASE_URL"))
    parser.add_argument("--text-endpoint", default=env_first("AD_TEXT_ENDPOINT"))
    parser.add_argument("--text-api-key", default=env_first("AD_TEXT_API_KEY", "AD_API_KEY", "OPENAI_API_KEY"))
    parser.add_argument("--text-model", default=env_first("AD_TEXT_MODEL", default=DEFAULT_TEXT_MODEL))
    parser.add_argument("--text-temperature", type=float, default=float(env_first("AD_TEXT_TEMPERATURE", default="0.8")))
    parser.add_argument("--no-infer", action="store_true", default=env_first("AD_NO_INFER", default="").lower() in {"1", "true", "yes"})

    parser.add_argument("--timeout", type=int, default=int(env_first("AD_TIMEOUT_SECONDS", default="180")))
    parser.add_argument("--interactive", action="store_true", help="Prompt for missing product/key values instead of failing fast.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-raw-response", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command in {"status", "templates", "configure", "doctor"}:
        return
    missing = []
    if args.image_backend == "api" and not (args.image_endpoint or args.image_base_url):
        missing.append("AD_IMAGE_BASE_URL/AD_API_BASE_URL or --image-endpoint")
    has_final_prompt = bool(args.final_prompt or args.final_prompt_file)
    if not has_final_prompt and not args.no_infer and args.text_model and not (args.text_endpoint or args.text_base_url):
        missing.append("AD_TEXT_BASE_URL/AD_API_BASE_URL or --text-endpoint")
    if missing:
        raise SystemExit(
            "Missing required configuration: "
            + ", ".join(missing)
            + "\nRun `python scripts/run_batch.py configure` to create a local .env, or set the variables manually."
        )


def main() -> None:
    load_env_defaults()
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.command == "status":
        status(args)
    elif args.command == "templates":
        list_templates(args.workspace)
    elif args.command == "configure":
        configure(args)
    elif args.command == "doctor":
        doctor(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
