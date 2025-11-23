# AsciiDoc to PDF/HTML Converter

AsciiDocファイルからPDFとHTMLを生成するDockerベースのシステムです。Krokiを使用したMermaid図やPlantUML図の生成、日本語フォント対応、LaTeX数式処理をサポートしています。

## 機能

- AsciiDocからPDF/HTML生成
- 日本語フォント対応
- Krokiを使用した図表生成（Mermaid, PlantUML等）
- LaTeX数式処理
- コンテナベースの環境（環境依存の最小化）

## 必要な環境

- Docker
- Docker Compose
- Make（オプション）

## 使用方法

### 1. クイックスタート

```bash
# Docker環境を起動してPDF/HTML生成
make build

# または手動で実行
docker-compose up -d
docker-compose exec asciidoctor chmod +x /workspace/build.sh
docker-compose exec asciidoctor /workspace/build.sh
```

### 2. 利用可能なコマンド

```bash
make up       # Docker環境を起動
make down     # Docker環境を停止
make build    # PDF/HTMLを生成
make clean    # 出力ディレクトリをクリーンアップ
make rebuild  # Docker環境を再構築
make test     # 生成されたファイルをテスト
make help     # ヘルプを表示
```

### 3. 出力ファイル

生成されたファイルは `output/` ディレクトリに保存されます：
- `output/sample.pdf` - PDF版
- `output/sample.html` - HTML版

## ファイル構成

```
asciidoc_to_html_pad/
├── Dockerfile                    # Asciidoctor環境の定義
├── docker-compose.yml           # DockerとKroki環境の構成
├── asciidoctor-pdf-theme.yml    # PDF用日本語テーマ
├── build.sh                     # ビルドスクリプト
├── Makefile                     # 簡単なコマンド実行用
├── README.md                    # このファイル
├── docs/
│   └── sample.adoc              # サンプルAsciiDocファイル
└── output/                      # 生成されるPDF/HTMLの出力先
    ├── sample.pdf
    └── sample.html
```

## カスタマイズ

### 新しいAsciiDocファイルの追加

1. `docs/` ディレクトリに `.adoc` ファイルを配置
2. `build.sh` を編集して新しいファイルを処理対象に追加

### PDFテーマの変更

`asciidoctor-pdf-theme.yml` を編集してフォント、色、レイアウトをカスタマイズできます。

### Kroki設定の変更

`docker-compose.yml` の `kroki` サービス設定を変更して、セキュリティやリクエストサイズの制限を調整できます。

## トラブルシューティング

### Krokiサーバーに接続できない場合

```bash
# Krokiコンテナが起動しているか確認
docker-compose ps

# Krokiサーバーのヘルスチェック
curl http://localhost:8000/health
```

### 日本語フォントが表示されない場合

1. コンテナ内のフォントパスを確認：
   ```bash
   docker-compose exec asciidoctor ls /usr/share/fonts/opentype/noto/
   ```

2. `asciidoctor-pdf-theme.yml` のフォントパスが正しいか確認

### PDF生成でエラーが発生する場合

```bash
# 詳細なログを確認
docker-compose exec asciidoctor asciidoctor-pdf --trace docs/sample.adoc
```