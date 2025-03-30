FROM ubuntu:24.04

 # 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    curl tar sudo git ca-certificates libicu70 \
    podman uidmap slirp4netns fuse-overlayfs jq \
&&  apt-get clean


# Podman rootless 設定（非rootユーザでPodman使えるようにする）
# github runner用ユーザー作成
RUN useradd -m runner
USER runner
WORKDIR /home/runner

# Podman configディレクトリを作成
RUN mkdir -p /home/runner/.config/containers

# containers.conf をコピー
COPY containers.conf /home/runner/.config/containers/containers.conf


# actions-runnerセットアップ
RUN mkdir actions-runner
WORKDIR /home/runner/actions-runner
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
