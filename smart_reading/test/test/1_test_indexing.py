from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

cfg = QaConfig()
pipeline = IndexingPipeline(cfg)

# 从文件构建
pdf_file_path = "/\data\sample_document .pdf"
file_hash = pipeline.build_from_file(pdf_file_path, "sk-e6a85c002d804560a7ef3d332d0075e9")
print(f"索引构建完成，file_hash: {file_hash}")