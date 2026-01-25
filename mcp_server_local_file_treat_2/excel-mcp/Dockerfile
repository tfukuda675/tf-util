FROM python:3.11-slim

RUN pip install --no-cache-dir uv
RUN uv pip install --system excel-mcp-server

WORKDIR /work
CMD ["uvx", "excel-mcp-server", "stdio"]
