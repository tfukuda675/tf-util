FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージと依存関係をインストール
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git python3 python3-pip nodejs npm \
      ruby ruby-dev build-essential openjdk-17-jre-headless \
      fonts-noto-cjk fonts-noto-cjk-extra \
      libxml2-dev libxslt1-dev \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Rubyのgemsをインストール（Asciidoctor関連）
RUN gem install asciidoctor asciidoctor-pdf asciidoctor-diagram asciidoctor-kroki \
    rouge coderay

# 数式処理のためのMathJaxをセットアップ
RUN npm install -g mathjax-node-cli

# 作業ディレクトリを設定
WORKDIR /workspace

# 日本語フォントの設定
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

CMD ["bash"]