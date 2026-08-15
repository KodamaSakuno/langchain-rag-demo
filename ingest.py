import json
import re
from pathlib import Path

from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS, CHUNKS_PATH, DOCS_DIR

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
IMPORT_RE = re.compile(r"^import\s+.*?from\s+['\"].*?['\"];?\s*$")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")

ALLOWED_SUFFIX = (".md", ".mdx", ".txt")


def clean_mdx(text: str) -> str:
    text = FRONTMATTER_RE.sub("", text)
    lines = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code:
            lines.append(line)
            continue
        s = line.strip()
        if IMPORT_RE.match(s):
            continue
        # 去掉独占一行的 JSX 包裹标签（如 <Tabs>、</TabItem>）
        if s.startswith("<") and s.endswith(">") and len(s) < 60 and "`" not in s:
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_blocks(text: str):
    blocks = []
    header_stack = []  # [(level, title)]
    buf = []
    in_code = False
    cur_path = ""

    def flush_para():
        nonlocal buf
        t = "\n".join(buf).strip()
        if t:
            blocks.append({"kind": "para", "text": t, "header_path": cur_path})
        buf = []

    for line in text.splitlines():
        is_fence = line.strip().startswith("```")
        if in_code:
            buf.append(line)
            if is_fence:
                blocks.append({"kind": "code", "text": "\n".join(buf), "header_path": cur_path})
                buf, in_code = [], False
            continue
        if is_fence:
            flush_para()
            buf, in_code = [line], True
            continue
        m = HEADER_RE.match(line)
        if m:
            flush_para()
            level, title = len(m.group(1)), m.group(2).strip()
            header_stack = [(lv, t) for lv, t in header_stack if lv < level]
            header_stack.append((level, title))
            cur_path = " > ".join(t for _, t in header_stack)
            continue
        if not line.strip():
            flush_para()
            continue
        buf.append(line)

    flush_para()
    if buf:  # 未闭合的代码块兜底：整块保留
        blocks.append({"kind": "code", "text": "\n".join(buf), "header_path": cur_path})
    return blocks


def pack_chunks(blocks, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS):
    chunks, cur, cur_len = [], [], 0
    for b in blocks:
        t = b["text"]
        if cur and cur_len + len(t) > max_chars:
            chunks.append(cur)
            overlap, olen = [], 0
            for prev in reversed(cur):
                if prev["kind"] == "code" or olen + len(prev["text"]) > overlap_chars:
                    break
                overlap.insert(0, prev)
                olen += len(prev["text"])
            cur, cur_len = overlap, olen
        cur.append(b)
        cur_len += len(t)
    if cur:
        chunks.append(cur)
    return chunks


def blocks_to_chunk(group, source):
    text = "\n\n".join(b["text"] for b in group)
    header_path = next((b["header_path"] for b in reversed(group) if b["header_path"]), "")
    has_code = any(b["kind"] == "code" for b in group)
    return {"text": text, "source": source, "header_path": header_path, "has_code": has_code}


def ingest_all(docs_dir=DOCS_DIR, out_path=CHUNKS_PATH):
    docs_dir = Path(docs_dir)
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in ALLOWED_SUFFIX)
    if not files:
        raise SystemExit(f"{docs_dir} 下没有找到 .md/.mdx/.txt 文档，请先把 LangChain 文档放进来（见 README 快速开始）")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in files:
            rel = str(p.relative_to(docs_dir))
            text = clean_mdx(p.read_text(encoding="utf-8", errors="ignore"))
            for i, group in enumerate(pack_chunks(parse_blocks(text))):
                chunk = blocks_to_chunk(group, rel)
                chunk["chunk_id"] = f"{rel}::{i}"
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
    print(f"共处理 {len(files)} 个文件，生成 {n} 个块 → {out_path}")


if __name__ == "__main__":
    ingest_all()
