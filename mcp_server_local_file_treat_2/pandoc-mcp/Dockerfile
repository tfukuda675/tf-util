FROM python:3.11-slim

# OS deps: pandoc + TeX (PDF生成に必要) + フォント類
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-recommended \
    texlive-latex-extra \
    lmodern \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# uv (uvx) と mcp-pandoc
RUN pip install --no-cache-dir uv
RUN uv pip install --system mcp-pandoc

WORKDIR /work

# MCP は stdio で動かす
CMD ["uvx", "mcp-pandoc"]
