# 实现 MemoryTool, 作为记忆系统统一接口
# 按自顶向下实现，“统一入口，分发处理”

# - MemoryTool: 专注于用户的接口和参数处理
# - MemoryManager: 负责核心的记忆管理

from typing import Dict, List
from tools.base import Tool
import datetime

def execute(self, action: str, **kwargs) -> str:
    """执行记忆操作
    
    支持操作: 
    - add: 四种类型: working / episodic / semantic / perceptual
    - search: 
    - summary:
    - stats: 获取统计信息
    - update:
    - remove:
    - forget: 
    - consolidate: 整合记忆
    - clear_all: 清空记忆
    """

    # 分发处理
    if action == "add":
        return self._add_memory(**kwargs)
    elif action == "search":
        return self._search_memory(**kwargs)
    elif action == "summary":
        return self._get_summary(**kwargs)
    elif action == "stats":
        return self._get_stats(**kwargs)
    # ..... 


# op1: add_memory
# 1. 会话 id管理, 每个记忆都会会话归属
# 2. 自动推断数据文件类型
# 3. 上下文信息自动补充
def _add_memory(
    self, 
    content: str = "",
    memory_type: str = "working",
    importance: float = 0.5,
    file_path:  str = None,
    modality:   str = None, # 信息模态
    **metadata
) -> str:
    """添加记忆"""
    try:
        # 确保会话 id 存在
        if self.current_session_id is None:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 感知记忆文件
        if memory_type == "perceptual" and file_path:
            inferred = modality or self._infer_modality(file_path)
            metadata.setdefault("modality", inferred)
            metadata.setdefault("raw_data", file_path)

        # 添加会话信息到元数据
        metadata.update({
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat()
        })

        memory_id = self.memory_manager.add_memory(
            content = content,
            memory_type = memory_type,
            importance  = importance,
            metadata = metadata,
            auto_classfiy = False,
        )

        return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

    except Exception as e:
        return f"❌ 添加记忆失败: {str(e)}"


# op2: search_memory
def _search_memory(
    self,
    query: str,
    limit: int = 5,
    memory_types: List[str] = None,
    memory_type: str = None,
    min_importance: float = 0.1
) -> str:
    """搜索记忆"""
    try:
        if memory_type and not memory_types:
            memory_types = [memory_type]

        results = self.memory_manager.retrieve_memories(
            query = query,
            limit = limit,
            memory_types = memory_types,
            min_importance = min_importance
        )

        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆"

        # 格式化结果
        formatted_results = []
        formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

        for i, memory in enumerate(results, 1):
            memory_type_label = {
                "working": "工作记忆",
                "episodic": "情景记忆", 
                "semantic": "语义记忆",
                "perceptual": "感知记忆"
            }.get(memory.memory_type, memory.memory_type)

            content_preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
            formatted_results.append(
                f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.importance:.2f})"
            )

        return "\n".join(formatted_results)
        
    except Exception as e:
        pass


# op3: forget_memory
# 1. -- 删除不重要的记忆
# 2. -- 删除过时的记忆
# 3. -- 当接近上限时，删除最不重要的记忆 
def _forget_memory(
    self,
    strategy: str = "importance_based",
    threshold: float = 0.1,
    max_age_days: int = 30
) -> str:
    """遗忘记忆"""
    try:
        count = self.memory_manager.forget_memories(
            strategy  = strategy,
            threshold = threshold,
            max_age_days = max_age_days
        )
        return f"🧹 已遗忘 {count} 条记忆 (策略: {strategy})"
    except Exception as e:
        return f"❌ 遗忘记忆失败: {e}"


# op4: consolidate_memory
# 整合记忆 --> 将重要的短期记忆变为长期
def _consolidate_memory(
    self,
    from_type: str = "working",
    to_type:   str = "episodic",
    importance_threshold: float = 0.7
) -> str:
    try:
        count = self.memory_manager.consolidate_memories(
            from_type = from_type,
            to_type = to_type,
            importance_threshold = importance_threshold
        )
        return f"🔄 已整合 {count} 条记忆为长期记忆 ({from_type} → {to_type}，阈值={importance_threshold})"

    except Exception as e:
        return f"❌ 错误: 整合记忆失败: {e}"


class MemoryTool(Tool):
    """记忆工具 - 为Agent提供记忆功能"""
    def __init__(self, user_id: str = "default_user", memory_config: MemoryConfig = None, memory_types: List[str] = None):
        super().__init__(
            name = "memory",
            description = "记忆工具 -- 存储 / 检索对话历史和经验"
        )
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types  = memory_types  or ["working", "episodic", "semantic"] # 多模态功能手动开启
        self.memory_manager = MemoryManager(
            config = self.memory_config,
            user_id = user_id,
            enable_working  = "working" in self.memory_types,
            enable_episodic = "episodic" in self.memory_types,
            enable_semantic = "semantic" in self.memory_types,
            enable_perceptual = "perceptual" in self.memory_types
        )


class MemoryManager:
    """记忆管理器 - 统一的记忆接口"""
    def __init__(
        self,
        config:  Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working:  bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False
    ):
        self.config  = config or MemoryConfig()
        self.user_id = user_id
        self.store = MemoryStore(self.config)
        self.retriever = MemoryRetriever(self.store, self.config)

        self.memory_types = {}

        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config, self.store)   # self.memory_types = {"working": <WorkingMemory对象>}
        if enable_episodic:
            self.memory_types['episodic'] = EpisodicMemory(self.config, self.store)
        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config, self.store)
        if enable_perceptual:
            self.memory_types['perceptual'] = PerceptualMemory(self.config, self.store)

        
# 四种记忆类型的实现
class WorkingMemory:
    """工作记忆实现

    特点：
    - 容量有限 (默认50条)+ TTL自动清理
    - 纯内存存储，访问速度极快
    - 混合检索: TF-IDF向量化 + 关键词匹配
    """
    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl or 60
        self.memories = []


    def add(self, memory_item: MemoryItem) -> str:
        self._expire_old_memories() # 先清理过期记忆

        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory()

        self.memories.append(memory_item)
        return memory_item.id  # 返回会话 id
    

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索: TF-IDF向量化 + 关键词匹配"""
        self._expire_old_memories()
        
        # 尝试TF-IDF向量检索
        vector_scores = self._try_tfidf_search(query)
        
        # 计算综合分数
        scored_memories = []
        for memory in self.memories:
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)
            
            # 混合评分
            base_relevance = vector_score * 0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score
            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)
            
            final_score = base_relevance * time_decay * importance_weight
            if final_score > 0:
                scored_memories.append((final_score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]


class EpisodicMemory:
    """情景记忆实现

    特点：
    - SQLite + Qdrant混合存储架构
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """
    def __init__(self, config: MemoryConfig):
        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)
        self.embedder = create_embedding_model_with_fallback()
        self.sessions = {} # 会话索引


    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 创建情景对象
        episode = Episode(
            episode_id = memory_item.id,
            session_id = memory_item.metadata.get("session_id", "default"),
            timestamp = memory_item.timestamp,
            content = memory_item.content,
            context = memory_item.medata
        )

        # 更新对话索引
        session_id = episode.session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 持久化存储
        self._persist_episode(episode)
        return memory_item.id


    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索： 结构化过滤 + 语义向量检索"""
        # 1. 结构化过滤
        candidate_ids = self._structed_filter(**kwargs)

        # 2. 向量化检索
        hits = self._vector_search(query, limit * 5, kwargs.get("user_id"))

        # 3. 综合评分
        results = []
        for hit in hits:
            if self._should_include(hit, candidate_ids, **kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item(hit)
                results.append((score, memory_item))

        results.sort(key = lambda x : x[0], reverse = True)
        return [item for _ , item in results[:limit]]


    def _calculate_episode_score(self, hit) -> float:
        """情景记忆评分算法"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"]["timestamp"])
        importance = hit["metadata"].get("importance", 0.5)
        
        # 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × 重要性权重
        base_relevance = vec_score * 0.8 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)
        
        return base_relevance * importance_weight


class SemanticMemory(BaseMemory):
    """语义记忆实现
    - 最复杂的部分，负责存储抽象的概念、规则和知识

    特点：
    - 使用 HuggingFace 中文预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量(Qdrant) + 图(Neo4j) + 语义推理
    
    功能:
    - 能够进行快速的检索 + 复杂的关系推理
    """
    def __init__(self, config: MemoryConfig, storage_backend = None):
        super().__init__(config, storage_backend)

        self.embedding_model = get_text_embedder()
        self.vector_store = QdrantConnectionManager.get_instance(**qdrant_config)
        self.graph_store  = Neo4jGraphStore(**neo4j_config)
        # 实体和关系缓存
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.nlp = self._init_nlp()


    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        # 1. 生成嵌入
        embedding = self.embedding_model.encode(memory_item.content)

        # 2. 提取实体和关系
        entities  = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content)

        # 3. 存储到数据库
        for entity in entities:
            self._add_entity_to_graph(entity, memory_item)

        for relation in relations:
            self._add_relation_to_graph(relation, memory_item)

        metadata = {
            "memory_id": memory_item.id,
            "entities": [e.entity_id for e in entities],
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
        self.vector_store.add_vectors(
            vectors  = [embedding.tolist()],
            metadata = [metadata],
            ids = [memory_item.id]
        )
        

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆"""
        # 1. 向量检索
        vector_results = self._vector_search(query, limit * 2, user_id)
        
        # 2. 图检索
        graph_results = self._graph_search(query, limit * 2, user_id)
        
        # 3. 混合排序
        combined_results = self._combine_and_rank_results(
            vector_results, graph_results, query, limit
        )
        
        return combined_results[:limit]


    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        """混合排序结果"""
        combined = {}
        
        # 合并向量和图检索结果
        for result in vector_results:
            combined[result["memory_id"]] = {
                **result,
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0
            }
        
        for result in graph_results:
            memory_id = result["memory_id"]
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            else:
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0)
                }
        
        # 计算混合分数
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)
            
            # 基础相似度得分
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            
            # 重要性权重 [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)
            
            # 最终得分：相似度 * 重要性权重
            combined_score = base_relevance * importance_weight
            result["combined_score"] = combined_score
        
        # 排序并返回
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        return sorted_results[:limit]


class PerceptualMemory(BaseMemory):
    """感知记忆实现

    特点:
    - 多模态数据分析
    - 跨模态相似性搜索
    - 感知信息的语义理解
    """
    def __init__(self, config: MemoryConfig, storage_backend = None):
        super().__init__(config, storage_backend)

        # 多模态编码
        self.text_embedder = get_text_embedder()
        self._clip_embedder = self._init_clip_model() # 图像理解
        self._clap_embedder = self._init_clap_model() # 视频理解

        # 按模态分离，进行向量存储
        self.vector_stores = {
            "text": QdrantConnectionManager.get_instance(
                collection_name = "perceptual_text",
                vector_size = self.vector_dim
            ),
            "image": QdrantConnectionManager.get_instance(
                collection_name = "perceptual_image",
                vector_size = self._image_dim
            ),
            "audio": QdrantConnectionManager.get_instance(
                collection_name = "perceptual_audio",
                vector_size = self._audio_dim
            ) 
        }


    def retrieve(self, query: str, limit: int = 5, **kwargs) -> str:
        """ 检索感知记忆 (同模态 / 跨模态)"""
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality  = kwargs.get("query_modality", target_modality or "text")

        # 同模态向量检索
        try:
            query_vector = self._encode_data(query, query_modality)
            store = self._get_vector_store_for_modality(target_modality or query_modality)

            where = {"memory_type": "perceptual"}
            if user_id:
                where["user_id"] = user_id
            if target_modality:
                where["target_modality"] = target_modality

            hits = store.search_similar(
                query_vector = query_vector,
                limit = max(limit * 5, 20),
                where = where
            )

        except Exception as e:
            hits = []

        # 融合排序
        results = []
        for hit in hits:
            vector_score  = float(hit.get("score", 0.0))
            recency_score = self._calculate_recency_score(hit["metadata"]["timestamp"])
            importance = hit["metadata"].get("importance", 0.5)

            base_relevance = vector_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (importance * 0.4) 
            combined_score = base_relevance * importance_weight

            results.append((combined_score, self._create_memory_item(hit)))

        results.sort(key = lambda x : x[0], reverse = True)
        return [item for _, item in results[:limit]]


    def _calculate_recency_score(self, timestamp: str) -> float:
        """计算时间近因性得分"""
        try:
            memory_time  = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600

            decay_factor  = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24) 

            return max(0.1, recency_score)
        except Exception as e:
            return 0.5 # 默认分数


    