# Windows 中文编码问题解决方案

## 问题现象

在 Windows 上使用 Python `subprocess.run()` 或 `os.system()` 调用 Codex CLI 并传入中文 prompt 时：

```
error: unexpected argument '...乱码...' found
```

或中文提示词被截断/解析错误。

## 根本原因

- Windows 命令行默认使用 **GBK** (代码页 936)
- Python `subprocess` 在传递字符串参数时，会按系统默认编码转换
- Codex CLI 接收到的参数是 GBK 编码的中文，但解析为 UTF-8，导致乱码

## 已验证失败的方案

| 方案 | 结果 |
|------|------|
| `subprocess.run(["codex", "exec", prompt], encoding='utf-8')` | ❌ 失败 |
| `subprocess.run(..., stdin=subprocess.PIPE)` + `communicate(prompt.encode('utf-8'))` | ❌ 失败 |
| 临时 `.cmd` 文件（UTF-8 无 BOM） | ❌ 失败 |
| PowerShell `Start-Process` | ❌ 失败（中文解析错误） |
| `os.system(f'codex exec "{prompt}"')` | ❌ 失败 |

## 唯一成功方案：type 管道

```batch
@echo off
chcp 65001 >nul
type "prompt.txt" | codex exec --full-auto --skip-git-repo-check --cd "{WORKSPACE}"
```

### 为什么有效？

1. **`type` 命令** 直接读取 UTF-8 编码的 `.txt` 文件，不经过 Python subprocess 的参数转换
2. **管道 `|`** 将文件内容作为 stdin 传递给 Codex CLI，stdin 在 Windows 上默认继承 UTF-8（当代码页设为 65001 时）
3. **Codex CLI** 支持从 stdin 读取 prompt（`Reading prompt from stdin...`），完美接收完整中文

### 代码页设置

- `chcp 65001` 将当前命令行会话切换为 UTF-8
- 仅影响当前 `.cmd` 执行会话，不影响系统全局设置
- 配合 `.cmd` 文件的 **UTF-8 BOM** 编码写入，确保 `cmd.exe` 正确识别文件中的中文路径

## 实现要点

### 1. 提示词写入文件

```python
with open("prompt.txt", 'w', encoding='utf-8') as f:
    f.write(prompt_text)
```

### 2. .cmd 文件使用 UTF-8 BOM

```python
with open("run.cmd", 'w', encoding='utf-8-sig') as f:  # utf-8-sig = with BOM
    f.write('@echo off\n')
    f.write('chcp 65001 >nul\n')
    f.write('type "prompt.txt" | codex exec --full-auto\n')
```

### 3. 通过 os.system 执行

```python
import os
exit_code = os.system('cmd /c "run.cmd"')
```

## 限制

- 此方案仅适用于 Codex CLI 支持 stdin 读取的场景
- 需要 Codex CLI 版本 ≥ 0.120.0（支持 `--cd` 和 `--full-auto`）
- 图片输出目录固定为 `~/.codex/generated_images`，需扫描复制

## 替代方案（非 Windows）

在 macOS/Linux 上，Python `subprocess` 默认使用 UTF-8，可直接传参：

```python
subprocess.run(["codex", "exec", "--full-auto", prompt_text])
```

无需管道方案。

## 参考

- Python issue: [subprocess on Windows with non-ASCII arguments](https://bugs.python.org/issue1759845)
- Microsoft Docs: [Code pages](https://docs.microsoft.com/en-us/windows/console/code-pages)
