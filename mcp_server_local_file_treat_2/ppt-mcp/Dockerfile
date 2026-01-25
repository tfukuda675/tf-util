FROM python:3.11-slim

# uv (uvx) を入れる
RUN pip install --no-cache-dir uv

# PowerPoint MCP server をインストール
# (uvx で実行する想定)
RUN uv pip install --system office-powerpoint-mcp-server

WORKDIR /work
CMD ["uvx", "--from", "office-powerpoint-mcp-server", "ppt_mcp_server"]
