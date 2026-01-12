import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import pdfParse from 'pdf-parse';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCUMENTS_DIR = '/documents';

// MCPサーバーの作成
const server = new Server(
  {
    name: 'local-file-treat',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// ツール一覧を提供
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'list_files',
        description: 'ドキュメントディレクトリ内のファイル一覧を取得します。ファイル名、サイズ、更新日時などの情報を返します。',
        inputSchema: {
          type: 'object',
          properties: {},
          required: [],
        },
      },
      {
        name: 'read_file',
        description: 'ドキュメントディレクトリ内の指定されたファイルの内容を読み取ります。テキストファイルとPDFファイルに対応しています。',
        inputSchema: {
          type: 'object',
          properties: {
            filename: {
              type: 'string',
              description: '読み取るファイルの名前',
            },
          },
          required: ['filename'],
        },
      },
    ],
  };
});

// ツールの実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (name === 'list_files') {
      const files = await fs.readdir(DOCUMENTS_DIR);
      const fileDetails = await Promise.all(
        files.map(async (file) => {
          const filePath = path.join(DOCUMENTS_DIR, file);
          const stats = await fs.stat(filePath);
          return {
            name: file,
            size: stats.size,
            isDirectory: stats.isDirectory(),
            modified: stats.mtime.toISOString(),
          };
        })
      );

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(fileDetails, null, 2),
          },
        ],
      };
    }

    if (name === 'read_file') {
      const { filename } = args;
      if (!filename) {
        throw new Error('filename is required');
      }

      const filePath = path.join(DOCUMENTS_DIR, filename);

      // セキュリティ: ディレクトリトラバーサル対策
      const realPath = await fs.realpath(filePath);
      if (!realPath.startsWith(DOCUMENTS_DIR)) {
        throw new Error('Access denied: path traversal detected');
      }

      // ファイルの拡張子を確認
      const ext = path.extname(filename).toLowerCase();

      if (ext === '.pdf') {
        // PDFファイルの場合
        const dataBuffer = await fs.readFile(filePath);
        const pdfData = await pdfParse(dataBuffer);
        
        return {
          content: [
            {
              type: 'text',
              text: `PDFファイル: ${filename}\n\nページ数: ${pdfData.numpages}\n\n内容:\n${pdfData.text}`,
            },
          ],
        };
      } else {
        // テキストファイルの場合
        const content = await fs.readFile(filePath, 'utf-8');
        
        return {
          content: [
            {
              type: 'text',
              text: content,
            },
          ],
        };
      }
    }

    throw new Error(`Unknown tool: ${name}`);
  } catch (error) {
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// サーバーの起動
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('MCP Local File Treat Server running on stdio');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
