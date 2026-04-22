---
name: open_ad_batch
description: Generate images by selecting a named prompt template and role setting, using this skill's third-party API script. Use when the user wants OpenClaw to turn a template name plus character/product/ad setting into final image prompts and batch-generate images without OpenClaw image_generate.
---

# Open Ad Batch

Use this skill to run a template-driven image generation batch through `scripts/run_batch.py`. Do not use OpenClaw `image_generate` for this workflow unless the user explicitly redirects you; image generation is handled by this skill's third-party HTTP API caller.

## Core Flow

The intended prompt flow is:

```text
template name + role setting
  -> isolated prompt builder returns final_prompt
  -> script sends final_prompt to the third-party image model
```

In OpenClaw, prefer `sessions_spawn` for the prompt-builder step when the tool is available. It creates an isolated background agent run and keeps prompt construction out of the main session context. The script's own text API call remains the fallback when `sessions_spawn` is unavailable or the user wants the script to be self-contained.

Official OpenClaw `sessions_spawn` fields to use for a one-shot native sub-agent:

```json
{
  "task": "根据模板名 hanfu-character-sheet 和以下角色设定，严格输出 JSON：{\"final_prompt\":\"...\"}。不得解释。",
  "runtime": "subagent",
  "mode": "run",
  "runTimeoutSeconds": 120
}
```

Notes:

- `runtime` defaults to OpenClaw's native sub-agent runtime; set `runtime: "acp"` only for Codex/Claude Code/Gemini-style ACP harness sessions.
- Use `runTimeoutSeconds`, not `timeoutSeconds`.
- Use `mode: "run"` for one-shot prompt generation. `mode: "session"` requires `thread: true` and is for persistent bound sessions.
- Optional fields include `model`, `thinking`, `label`, `agentId`, `cleanup`, and `sandbox`.

After the sub-agent returns `final_prompt`, write it to a file and call:

```bash
python scripts/run_batch.py ^
  --template hanfu-character-sheet ^
  --product "角色名或项目名" ^
  --final-prompt-file ".\final-prompt.txt" ^
  --total 1
```

## Runtime Inputs

Infer these from the user request where possible:

- Template: default `hanfu-character-sheet`.
- Role setting: pass with `--role-setting` or `--role-setting-file`.
- Product/output label: pass with `--product`; for character sheets, use the character or project name.
- Total rounds: default `1`.
- Workspace: default `<current workspace>/open-ad-batch-output`.
- Cooldown: default `0`.
- Aspect ratio: treat `--aspect-ratio` as a preference/API hint. The template may instruct the reasoning model to choose the best vertical ratio for the target model and layout.

Required third-party image API config:

```bash
AD_IMAGE_BASE_URL=https://your-platform.example/v1
AD_IMAGE_API_KEY=...
AD_IMAGE_MODEL=your-image-model
```

Recommended text/reasoning API config:

```bash
AD_TEXT_BASE_URL=https://your-platform.example/v1
AD_TEXT_API_KEY=...
AD_TEXT_MODEL=gpt-5.4
```

`AD_API_BASE_URL` and `AD_API_KEY` can be used when text and image calls share one platform.

For first-time local deployment, run:

```bash
python scripts/run_batch.py configure
```

This prompts for API base URL, API key, image model, and text/reasoning model, then writes a local `.env`. The default text/reasoning model is `gpt-5.4`. Do not commit `.env`.

## Templates

Built-in templates live in `assets/templates/`.

Current built-in template:

- `hanfu-character-sheet`: 汉服角色设定集海报。输入角色设定，输出一个可直接传给图片模型的最终提示词。

List templates:

```bash
python scripts/run_batch.py templates
```

Run with inline role setting:

```bash
python scripts/run_batch.py ^
  --template hanfu-character-sheet ^
  --product "角色名或项目名" ^
  --role-setting "姓名、身份、朝代气质、服饰偏好、场景要求..." ^
  --total 1 ^
  --cooldown 0
```

Run with a role-setting file:

```bash
python scripts/run_batch.py ^
  --template hanfu-character-sheet ^
  --product "角色名或项目名" ^
  --role-setting-file ".\role-setting.txt"
```

Run with a precomputed final prompt from `sessions_spawn`:

```bash
python scripts/run_batch.py ^
  --template hanfu-character-sheet ^
  --product "角色名或项目名" ^
  --final-prompt-file ".\final-prompt.txt"
```

Template variables available to JSON templates:

```text
{product}, {role_setting}, {round}, {total}, {aspect_ratio}, {size},
{language}, {requirements}, {avoid_styles_text}, {avoid_styles_json}, {date_utc}
```

Each template should include:

- `role_title`: the bracketed role identity sent to the reasoning model.
- `reasoning_model_input`: the task facts sent to the reasoning model.
- `final_prompt_format`: the exact final prompt structure to fill.
- `response_contract`: require `final_prompt` as the primary output field.

## Output

The script writes `context.json`, logs, and one isolated workspace folder per round:

```text
<workspace>/
├── context.json
├── logs/
│   └── batch_run.log
└── images/
    └── <product>_batch/
        └── round_001/
            ├── final_prompt.txt
            ├── image.png
            └── manifest.json
```

Re-running the same command resumes from `context.json`.

## API Compatibility

By default:

- Text: `POST {base_url}/chat/completions`
- Image: `POST {base_url}/images/generations`

Use `AD_TEXT_ENDPOINT` or `AD_IMAGE_ENDPOINT` for custom third-party paths. Set `AD_IMAGE_RESPONSE_FORMAT=none` if the image API rejects OpenAI-style `response_format`.

## GitHub Deployment Check

Before publishing this skill repository, verify:

```bash
python -m py_compile scripts/run_batch.py
python scripts/run_batch.py templates
```

After another OpenClaw installs the repo, the expected first command is:

```bash
python scripts/run_batch.py configure
```
