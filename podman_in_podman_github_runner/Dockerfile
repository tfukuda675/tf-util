FROM ubuntu:24.04

 # 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    curl tar sudo git ca-certificates libicu74 \
    podman uidmap slirp4netns fuse-overlayfs jq \
&&  apt-get clean


# github runner用ユーザー作成
RUN useradd -m runner


# sudo状態で処理
COPY launch_runner.sh /home/runner/launch_runner.sh
RUN chmod +x /home/runner/launch_runner.sh && chown runner:runner /home/runner/launch_runner.sh


# Podman rootless 設定（非rootユーザでPodman使えるようにする）
USER runner
WORKDIR /home/runner

# Podman configディレクトリを作成
RUN mkdir -p /home/runner/.config/containers

# containers.conf をコピー
COPY containers.conf /home/runner/.config/containers/containers.conf


# actions-runnerセットアップ
RUN mkdir actions-runner
WORKDIR /home/runner/actions-runner

ENTRYPOINT ["/home/runner/launch_runner.sh"]
