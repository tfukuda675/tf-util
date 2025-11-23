# AsciiDoc PDF/HTML 生成用Makefile

.PHONY: build clean up down rebuild test help

# デフォルトターゲット
all: build

# Docker環境のビルドと起動
up:
	@echo "Docker環境を起動中..."
	docker compose up -d
	@echo "Krokiサーバーの起動を待機中..."
	@sleep 10
	@echo "環境が準備されました"

# Docker環境の停止
down:
	@echo "Docker環境を停止中..."
	docker compose down

# Docker環境の再構築
rebuild:
	@echo "Docker環境を再構築中..."
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@sleep 10

# PDF/HTML生成
build: up
	@echo "PDF/HTMLを生成中..."
	docker compose exec asciidoctor chmod +x /workspace/build.sh
	docker compose exec asciidoctor /workspace/build.sh

# 出力ディレクトリのクリーンアップ
clean:
	@echo "出力ディレクトリをクリーンアップ中..."
	rm -rf output/
	@echo "クリーンアップが完了しました"

# テスト実行（生成されたファイルの確認）
test: build
	@echo "生成されたファイルを確認中..."
	@if [ -f output/sample.pdf ]; then \
		echo "✓ PDF生成成功: output/sample.pdf"; \
	else \
		echo "✗ PDF生成失敗"; \
	fi
	@if [ -f output/sample.html ]; then \
		echo "✓ HTML生成成功: output/sample.html"; \
	else \
		echo "✗ HTML生成失敗"; \
	fi

# ヘルプ
help:
	@echo "利用可能なコマンド:"
	@echo "  make up       - Docker環境を起動"
	@echo "  make down     - Docker環境を停止"
	@echo "  make build    - PDF/HTMLを生成"
	@echo "  make clean    - 出力ディレクトリをクリーンアップ"
	@echo "  make rebuild  - Docker環境を再構築"
	@echo "  make test     - 生成されたファイルをテスト"
	@echo "  make help     - このヘルプを表示"
