FROM ubuntu:24.04

# 必要なツールのインストール
RUN apt-get update && apt-get install -y \
    curl tar sudo git ca-certificates libicu70 \
 && apt-get clean

# ワーキングディレクトリ
WORKDIR /runner

# GitHub Actions Runnerのダウンロード（後で entrypoint で実行）
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
