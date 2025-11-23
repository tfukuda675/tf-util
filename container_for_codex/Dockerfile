# Node.jsのベースイメージを使用する
FROM node:latest

# Codex CLIをグローバルにインストールする
#RUN npm install -g @codex/cli
RUN npm install @codex/cli

# コンテナ起動時にCodex CLIを実行するように設定する (例: 任意のコマンドを実行)
CMD ["codex", "init"]
