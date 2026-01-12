FROM node:22-alpine

WORKDIR /app

# package.jsonをコピー
COPY package*.json ./

# 依存関係をインストール
RUN npm install --production

# サーバーコードをコピー
COPY server.js ./

# ドキュメントディレクトリ
RUN mkdir -p /documents

# 非rootユーザーで実行
USER node

CMD ["node", "server.js"]
