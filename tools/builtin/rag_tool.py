# RAG tool

import time
import random
import re
from base import Tool
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from hello_agents import HelloAgentLLM

class RAGTool(Tool):
    """ RAG 工具

    提供完整的RAG能力
    - 添加多格式文档
    - 智能检索与召回
    - LLM 增强回答
    - 知识库管理
    """
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default"
    ):
        # 初始化RAG管道
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self.llm = HelloAgentLLM()

        # 创建默认管道
        default_pipeline = create_rag_pipeline(
            qdrant_url = self.qdrant_url,
            qdrant_api_key  = self.qdrant_api_key,
            collection_name = self.collection_name,
            rag_namespace   = self.rag_namespace 
        )
        self._pipelines[rag_namespace] = default_pipeline
        # 流程: 任意格式文档 -> MarkItdown -> 文本 -> 智能分块 -> 向量化 -> 存储检索

    
    def _convert_to_markdown(path: str) -> str:
        """将任意格式文档转换为markdown文档"""
        if not os.path.exists(path):
            return ""

        # 对于 pdf 文件: 
        ext = (os.path.splitext(path)[1] or '').lower()
        if ext == 'pdf':
            return _enhanced_pdf_processing(path)

        # 其他格式
        md_instance = _get_markitdown_instance()
        if md_instance is not None:
            return _fallback_text_reader(path)

        try:
            result = md_instance.convert(path)
            markdown_text = getattr(result, "text_content", None)
            if isinstance(markdown_text, str) and markdown_text.strip():
                print(f"[RAG] MarkItDown转换成功: {path} -> {len(markdown_text)} chars Markdown")
                return markdown_text
            return ""

        except:
            print(f"[WARNING] MarkItDown转换失败 {path}: {e}")
            return _fallback_text_reader(path)  


    def _split_paragraphs_with_headings(text: str) -> List[Dict]:
        """将文本切分为带结构路径的语义段落。

        优先使用 Markdown 标题；
        当文档没有明确标题时，继续识别章节、法律条文、列表、缩进和自然句末。
        特别长的自然段会在句子边界处再切分，降低超长单段对下游 token 分块的影响。
        """
        if not text or not text.strip():
            return []

        # 这是段落级保护上限，最终 chunk 大小仍由 _chunk_paragraphs 控制。
        max_paragraph_chars = 1200
        lines = text.splitlines(keepends=True)
        heading_stack: List[tuple] = []
        paragraphs: List[Dict] = []
        buf_start: Optional[int] = None
        buf_end: Optional[int] = None
        buf_heading_path: Optional[str] = None
        current_article: Optional[str] = None
        char_pos = 0

        markdown_heading_pattern = re.compile(
            r"^\s{0,3}(#{1,6})(?!#)[ \t]*(\S.*?)[ \t]*#*[ \t]*$"
        )
        legal_article_pattern = re.compile(
            r"^(第[零〇一二三四五六七八九十百千万两\d]+条"
            r"(?:之[零〇一二三四五六七八九十百千万两\d]+)?)"
        )
        legal_document = any(
            legal_article_pattern.match(line.strip()) for line in lines
        )
        has_markdown_headings = any(
            markdown_heading_pattern.match(line.rstrip("\r\n")) for line in lines
        )

        def heading_path(extra: Optional[str] = None) -> Optional[str]:
            parts = [title for _, title in heading_stack]
            if current_article:
                parts.append(current_article)
            if extra:
                parts.append(extra)
            return " > ".join(parts) if parts else None

        def update_heading(level: int, title: str):
            """按层级更新标题栈，兼容 # 后直接出现 ### 的情况。"""
            nonlocal current_article
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_article = None

        def detect_plain_heading(line: str):
            """识别小说章节、法律编章节和常见的编号式小标题。"""
            if not line or len(line) > 80:
                return None

            match = re.match(
                r"^(第[零〇一二三四五六七八九十百千万两\d]+"
                r"([编部卷章回节])(?:\s*[^。！？；;]{0,50})?)$",
                line,
            )
            if match:
                level = {
                    "编": 1,
                    "部": 1,
                    "卷": 1,
                    "章": 2,
                    "回": 2,
                    "节": 3,
                }[match.group(2)]
                return level, match.group(1).strip()

            match = re.match(
                r"^(Part|Chapter|Section)\s+[\w.-]+(?:\s+.{0,50})?$",
                line,
                re.IGNORECASE,
            )
            if match:
                level = {"part": 1, "chapter": 2, "section": 3}[
                    match.group(1).lower()
                ]
                return level, line

            if re.match(r"^(序章|序言|前言|引言|楔子|后记|尾声|附录)\b", line):
                return 1, line

            # 法律文档中的“一、”“（一）”通常是条款而非标题。
            if not legal_document:
                match = re.match(r"^(\d+(?:\.\d+){0,3})[.)、]?\s+(.+)$", line)
                if match and not re.search(r"[。！？；;]$", line):
                    return match.group(1).count(".") + 1, line

                if re.match(
                    r"^[一二三四五六七八九十百]+、\s*[^。！？；;]{1,40}$",
                    line,
                ):
                    return 1, line

            return None

        def append_piece(content: str, absolute_start: int, path: Optional[str]):
            """追加段落并修正去除首尾空白后的字符偏移。"""
            if not content or not content.strip():
                return

            leading = len(content) - len(content.lstrip())
            trailing = len(content) - len(content.rstrip())
            cleaned = content.strip()
            start = absolute_start + leading
            end = absolute_start + len(content) - trailing
            paragraphs.append(
                {
                    "content": cleaned,
                    "heading_path": path,
                    "start": start,
                    "end": end,
                }
            )

        def append_with_sentence_fallback(
            content: str, absolute_start: int, path: Optional[str]
        ):
            """优先在句末切分超长段落，没有句末时才按长度硬切。"""
            if len(content) <= max_paragraph_chars:
                append_piece(content, absolute_start, path)
                return

            boundary_chars = set("。！？!?；;\n")
            closing_chars = set("\"'”’」』】）》）]")
            piece_start = 0
            last_boundary: Optional[int] = None
            index = 0

            while index < len(content):
                if content[index] in boundary_chars:
                    boundary = index + 1
                    while boundary < len(content) and content[boundary] in closing_chars:
                        boundary += 1
                    last_boundary = boundary

                if index - piece_start + 1 >= max_paragraph_chars:
                    # 避免采用离当前位置太远的句末，产生极小段落。
                    if (
                        last_boundary is not None
                        and last_boundary - piece_start >= max_paragraph_chars // 2
                    ):
                        cut = last_boundary
                    else:
                        cut = index + 1

                    append_piece(
                        content[piece_start:cut],
                        absolute_start + piece_start,
                        path,
                    )
                    piece_start = cut
                    last_boundary = None
                    index = cut
                    continue

                index += 1

            append_piece(
                content[piece_start:], absolute_start + piece_start, path
            )

        def flush_buf():
            nonlocal buf_start, buf_end, buf_heading_path
            if buf_start is None or buf_end is None:
                return

            append_with_sentence_fallback(
                text[buf_start:buf_end], buf_start, buf_heading_path
            )
            buf_start = None
            buf_end = None
            buf_heading_path = None

        def append_line(line_start: int, line_end: int):
            nonlocal buf_start, buf_end, buf_heading_path
            if buf_start is None:
                buf_start = line_start
                buf_heading_path = heading_path()
            buf_end = line_end

        previous_nonempty_line = ""

        for line in lines:
            raw = line.rstrip("\r\n")
            stripped = raw.strip()
            line_start = char_pos
            line_end = line_start + len(raw)
            char_pos += len(line)

            markdown_heading = markdown_heading_pattern.match(raw)
            if markdown_heading:
                flush_buf()
                update_heading(
                    len(markdown_heading.group(1)),
                    markdown_heading.group(2).strip(),
                )
                previous_nonempty_line = ""
                continue

            plain_heading = detect_plain_heading(stripped)
            if plain_heading:
                flush_buf()
                update_heading(*plain_heading)
                previous_nonempty_line = ""
                continue

            if not stripped:
                flush_buf()
                previous_nonempty_line = ""
                continue

            article_match = legal_article_pattern.match(stripped)
            if article_match:
                flush_buf()
                current_article = article_match.group(1)
                append_line(line_start, line_end)
                previous_nonempty_line = stripped
                continue

            legal_clause_start = legal_document and bool(
                re.match(
                    r"^(?:[（(][一二三四五六七八九十百千万两\d]+[）)]"
                    r"|[一二三四五六七八九十百千万两]+、|\d+[.、])",
                    stripped,
                )
            )
            list_item_start = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped))
            indented_paragraph = raw.startswith("\u3000\u3000") or bool(
                re.match(r"^[ \t]{2,}\S", raw)
            )
            previous_sentence_ended = bool(
                re.search(r"[。！？!?；;…][\"'”’」』）】]*$", previous_nonempty_line)
            )

            # 无 Markdown 结构时，用自然段特征合并 PDF 的软换行，同时
            # 在明确的句末、缩进或列表项处建立新的语义段落。
            should_start_new_paragraph = buf_start is not None and (
                legal_clause_start
                or list_item_start
                or (
                    not has_markdown_headings
                    and (indented_paragraph or previous_sentence_ended)
                )
            )
            if should_start_new_paragraph:
                flush_buf()

            append_line(line_start, line_end)
            previous_nonempty_line = stripped

        flush_buf()

        if not paragraphs:
            paragraphs = [{
                "content": text.strip(),
                "heading_path": None,
                "start": len(text) - len(text.lstrip()),
                "end": len(text.rstrip())
            }]
        return paragraphs


    def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
        """基于token数量的智能分块"""
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0

        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = _approx_token_len(p["content"]) or 1

            if cur_tokens + p_tokens <= chunk_tokens or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                # 生成当前分块
                content = "\n\n".join(x["content"] for x in cur)
                start = cur[0]["start"]
                end   = cur[-1]["end"]
                heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

                chunks.append({
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                })

                if overlap_tokens > 0 and cur:
                    kept: List[Dict] = []
                    kept_tokens = 0
                    for x in reversed(cur):
                        t = _approx_token_len(x["content"]) or 1
                        if kept_tokens + t > overlap_tokens:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens
                else:
                    cur = []
                    cur_tokens = 0

        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end   = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })

        return chunks


    def _approx_token_len(text: str) -> int:
        """近似估计Token长度, 支持中英文混合"""
        cjk = sum(1 for ch in text if _is_cjk(ch))
        non_cjk_tokens = len([t for t in text.split() if t])
        return cjk + non_cjk_tokens


    def _is_cjk(ch: str) -> bool:
        """判断是否为CJK字符"""
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF or  # CJK统一汉字
            0x3400 <= code <= 0x4DBF or  # CJK扩展A
            0x20000 <= code <= 0x2A6DF or # CJK扩展B
            0x2A700 <= code <= 0x2B73F or # CJK扩展C
            0x2B740 <= code <= 0x2B81F or # CJK扩展D
            0x2B820 <= code <= 0x2CEAF or # CJK扩展E
            0xF900 <= code <= 0xFAFF      # CJK兼容汉字
        )


    def index_chunks(
        store = None,
        chunks: List[Dict] = None,
        cache_db: Optional[str] = None,
        batch_size: int = 64,
        rag_namespace: str = "default"
    ):
        """嵌入功能
        嵌入模块 是 RAG 系统的核心, 将文本转换为高维向量。
        RAG 系统检索能力 ---> 嵌入模型的质量 + 向量存储的效率

        using 百炼 api with fallback to sentence-transformers 
        """
        if not chunks:
            print("[RAG] No chunks to index")
            return

        # 嵌入模型
        embedder  = get_text_embedder()
        dimension = get_dimension(384)

        # 创建默认Qdrant存储
        if store is None:
            store = _create_default_vector_store(dimension)
            print(f"[RAG] Created default Qdrant store with dimension {dimension}")

        # 预处理 markdown 文本
        processed_texts = []
        for chunk in chunks:
            raw_content = chunk["content"]
            processed_content = _preprocess_markdown_for_embedding(raw_content)
            processed_texts.append(processed_content)

        print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")

        # 批量编码
        vecs: List[List[float]] = []

        for i in range(0, len(processed_texts), batch_size):
            part = processed_texts[i: i + batch_size]
            try:
                part_vecs = embedder.concoder(part)

                if not isinstance(part_vecs, list):
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                part_vecs = [list(part_vecs)]

                for v in part_vecs:
                    try:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        v_norm = [float(x) for x in v]

                        if len(v_norm) != dimension:
                            print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                            if len(v_norm) < dimension:
                                v_norm.extend([0.0] * (dimension - len(v_norm)))
                            else:
                                v_norm = v_norm[:dimension]

                        vecs.append(v_norm)
                    except Exception as e:
                        print(f"[WARNING] 向量转换失败: {e}, 使用零向量")
                        vecs.append([0.0] * dimension)
                
            except Exception as e:
                print(f"[WARNING] Batch {i} encoding failed: {e}")
                # 实现重试机制
                # ... 
                

            print(f"[RAG] Embedding progress: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")


    def _prompt_mqe(query: str, n: int) -> List[str]:
        """使用 LLM 生成多样化的查询扩展"""
        try:
            # from ...core.llm import HelloAgentsLLM
            llm = HelloAgentLLM()
            prompt = [
                {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
                {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
            ]
            text  = llm.invoke(prompt)
            lines = [ln.strip("- \t") for ln in(text or "").splitlines()]
            outs  = [ln for ln in lines if ln]
            return out[:n] or [query]
        except Exception as e:
            return [query]


    def _prompt_hyde(query: str) -> Optional[str]:
        """生成假设性文档用于改善检索"""
        try:
            # from ...core.llm import HelloAgentsLLM
            llm = HelloAgentLLM()
            prompt = [
                {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
                {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
            ]
            return llm.invoke(prompt)
        except Exception:
            print(f"❌: _prompt_hydde 调用LLM失败")
            return None


    def search_vectors_expanded(
        store = None,
        query: str = "",
        top_k: int = 8,
        rag_namespace: Optional[str] = None,
        only_rag_data: bool = True,
        score_threshold: Optional[float] = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multipliter: int = 4
    ) -> List[Dict]:
        """
        Search with query expansion using unified embedding and Qdrant
        """
        if not query:
            return []

        if store is None:
            store = _create_default_vector_store()

        # 查询扩展
        expansions: List[str] = [query]

        if enable_mqe and mqe_expansions > 0:
            expansions.extend(_prompt_mqe(query, mqe_expansions))
        if enable_hyde:
            hyde_text = _prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)

        uniq: List[str] = []
        for e in expansions:
            if e and e not in uniq:
                uniq.append(e)

        expansions = uniq[:max(1, len(uniq))]    

        # 分配候选池
        pool = max(top_k * candidate_pool_multipliter, 20)
        per  = max(1, pool // max(1, len(expansions)))

        # 构建 RAG 数据过滤器
        where = {"memory_type": "rag_chunk"}
        if only_rag_data:
            where["is_rag_data"] = True
            where["data_source"] = "rag_pipeline"
        if rag_namespace:
            where["rag_namespace"] = rag_namespace

        # 收集所有扩展查询的结果
        agg: Dict[str, Dict] = {}
        for q in expansions:
            qv = embed_query(q)
            hits = store.search_similar(
                query_vector = qv,
                limit = per,
                score_threshold = score_threshold,
                where = where
            )
            # 同一条记忆可能被不同的扩展查询检索到
            for h in hits:
                mid = h.get('metadata', {}).get("memory_id", h.get("id"))
                s = float(h.get("score", 0.0))
                if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                    agg[mid] = h

        # 按分数排序返回
        merged = list(agg.values())
        merged.sort(key = lambda x : float(x.get("score", 0.0)), reverse = True)
        return merged[:top_k]
