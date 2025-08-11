# AsciiDoc Simple

AsciiDocを使用してPDFやHTMLを生成するためのDockerベースのプロジェクトです。日本語フォント対応とLaTeX数式表記に対応しています。

## 機能

- AsciiDoc → PDF変換
- AsciiDoc → HTML変換  
- 日本語フォント対応（Noto CJKフォント使用）
- LaTeX数式表記対応
- シンタックスハイライト対応

## ファイル構成

```
.
├── Dockerfile           # asciidoctorコンテナの設定
├── docker-compose.yml   # Docker Compose設定
├── japanese-theme.yml   # 日本語PDF用テーマ設定
├── sample.adoc          # サンプルAsciiDocファイル
└── README.md           # このファイル
```

## セットアップ

### 1. コンテナのビルドと起動

```bash
# コンテナをビルド
docker-compose build

# コンテナを起動
docker-compose up -d
```

### 2. コンテナに入る

```bash
docker-compose exec asciidoctor bash
```

## 使用方法

### PDF生成

```bash
# 基本的なPDF生成
docker-compose exec asciidoctor asciidoctor-pdf sample.adoc

# 数式対応のPDF生成
docker-compose exec asciidoctor asciidoctor-pdf -r asciidoctor-mathematical sample.adoc
```

### HTML生成

```bash
# 基本的なHTML生成
docker-compose exec asciidoctor asciidoctor sample.adoc

# 数式対応のHTML生成
docker-compose exec asciidoctor asciidoctor -r asciidoctor-mathematical sample.adoc
```

### 両方を同時生成

```bash
docker-compose exec asciidoctor bash -c "asciidoctor -r asciidoctor-mathematical sample.adoc && asciidoctor-pdf -r asciidoctor-mathematical sample.adoc"
```

## サンプルファイルの内容

`sample.adoc`には以下の要素が含まれています：

- 基本的なテキストフォーマット（太字、イタリック）
- リスト（順序あり・なし）
- コードブロック（シンタックスハイライト付き）
- 表
- LaTeX数式記法
- 注釈・警告ブロック
- 引用ブロック

## LaTeX数式の記述方法

```asciidoc
[latexmath]
++++
x = \frac{-b \pm \sqrt{ b^2-4ac }}{2a}
++++
```

## カスタマイズ

### 日本語フォントテーマの変更

`japanese-theme.yml`を編集することで、フォントサイズやスタイルをカスタマイズできます。

### 新しいAsciiDocファイルの作成

1. `.adoc`拡張子でファイルを作成
2. ドキュメントヘッダーに以下を追加：

```asciidoc
= ドキュメントタイトル
作成者 <email@example.com>
v1.0, 2025-08-11
:doctype: book
:toc: left
:sectnums:
:source-highlighter: highlight.js
:pdf-theme: japanese-theme.yml
:stem: latexmath
```

## トラブルシューティング

### 日本語が表示されない場合

- コンテナを再ビルドしてください
- フォントパスが正しいか`japanese-theme.yml`を確認してください

### 数式が表示されない場合

- `-r asciidoctor-mathematical`オプションを付けて実行してください
- `:stem: latexmath`属性がファイルに含まれているか確認してください

## 開発環境での使用

コンテナ起動後、ファイルを編集すると自動的に反映されます。ホストマシンのファイルとコンテナ内のファイルは同期されています。

## コンテナの停止・削除

```bash
# コンテナを停止
docker-compose down

# コンテナとイメージを削除
docker-compose down --rmi all
```