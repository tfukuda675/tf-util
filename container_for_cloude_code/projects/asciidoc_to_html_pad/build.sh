#!/bin/bash

# AsciiDoc から PDF/HTML 生成スクリプト
set -e

# 色付きメッセージ用の関数
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 出力ディレクトリを作成
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

print_info "AsciidoctorとKrokiを使用してPDF/HTML生成を開始します..."

# sample.adoc ファイルの存在確認
if [ ! -f "docs/sample.adoc" ]; then
    print_error "docs/sample.adoc が見つかりません"
    exit 1
fi

# Krokiサーバーの起動待機
print_info "Krokiサーバーの起動を待機中..."
timeout=60
count=0
while ! curl -s http://kroki:8000/health > /dev/null 2>&1; do
    if [ $count -ge $timeout ]; then
        print_error "Krokiサーバーが起動しませんでした (${timeout}秒でタイムアウト)"
        exit 1
    fi
    sleep 1
    count=$((count + 1))
done
print_success "Krokiサーバーが起動しました"

# PDF生成
print_info "PDFを生成中..."
asciidoctor-pdf \
    --attribute kroki-server-url=http://kroki:8000 \
    --attribute pdf-themesdir=. \
    --attribute pdf-theme=asciidoctor-pdf-theme \
    --attribute allow-uri-read \
    --out-dir $OUTPUT_DIR \
    docs/sample.adoc

if [ $? -eq 0 ]; then
    print_success "PDF生成が完了しました: $OUTPUT_DIR/sample.pdf"
else
    print_error "PDF生成でエラーが発生しました"
    exit 1
fi

# HTML生成
print_info "HTMLを生成中..."
asciidoctor \
    --attribute kroki-server-url=http://kroki:8000 \
    --attribute allow-uri-read \
    --attribute source-highlighter=rouge \
    --out-dir $OUTPUT_DIR \
    docs/sample.adoc

if [ $? -eq 0 ]; then
    print_success "HTML生成が完了しました: $OUTPUT_DIR/sample.html"
else
    print_error "HTML生成でエラーが発生しました"
    exit 1
fi

print_success "すべての変換が正常に完了しました！"
print_info "生成されたファイル:"
ls -la $OUTPUT_DIR/