FROM yuzutech/kroki:latest

# 日本語フォントをインストール
USER root
RUN apt-get update && \
    apt-get install -y fonts-noto-cjk fonts-noto-cjk-extra && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv

# フォント設定ファイルを作成
RUN mkdir -p /usr/share/fonts/truetype/noto

# フォントキャッシュを更新
RUN fc-cache -f -v

USER kroki