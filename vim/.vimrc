" ===============<<  Vim Path Setting   >>=============== 
set runtimepath=$HOME/.vim/vim,$VIMRUNTIME

" ===============<<    Tab Space        >>=============== 
" タブの画面上での幅
set tabstop=4

" タブをスペースに展開しない (expandtab:展開する)
set expandtab

" tabstopを変えずに空白を含めることにより、見た目のtabstopを変える
set softtabstop=4

" 改行時に前の行のインデントを継続する
set autoindent

" 自動的に構文を確認し、インデント量を調整する (noautoindent:インデントしない)
set smartindent

" インデント量を設定する (初期値は"8")
set shiftwidth=4

" クリップボードを共有
set clipboard+=unnamed

" viminfoファイルを作成しない
:set viminfo=

" ===============<<    plogin setting   >>=============== 
" シンタックス
syntax on

" ===============<<    Tab 保管         >>=============== 
" TABによる補完でリスト表示
set wildmode=list:longest


" ===============<<    Encode           >>=============== 
" エンコード
set encoding=utf-8
set fileencoding=utf-8
scriptencoding utf-8


" ===============<<    Color            >>=============== 
set background=dark
" colorscheme iceberg
colorscheme PaperColor
"let g:molokai_original = 1


" ===============<<    文字列検索       >>=============== 
" インクリメンタルサーチ. １文字入力毎に検索を行う
"set incsearch " うるさいので使わない

" 検索パターンに大文字小文字を区別しない
set ignorecase

" 検索パターンに大文字を含んでいたら大文字小文字を区別する
set smartcase

" 検索結果をハイライト
set hlsearch

" ===============<<    command line     >>=============== 
" ステータスラインを常に表示
set laststatus=2

" 現在のモードを表示
set showmode

" 打ったコマンドをステータスラインの下に表示
set showcmd

" ステータスラインの右側にカーソルの位置を表示する
set ruler



" ===============<<    change keymap    >>=============== 
"マップコマンドとモードの対応。詳細は以後に。
"     コマンド                    モード
":map   :noremap  :unmap     ノーマル、ビジュアル、選択、オペレータ待機
":nmap  :nnoremap :nunmap    ノーマル
":vmap  :vnoremap :vunmap    ビジュアル、選択
":smap  :snoremap :sunmap    選択
":xmap  :xnoremap :xunmap    ビジュアル
":omap  :onoremap :ounmap    オペレータ待機
":map!  :noremap! :unmap!    挿入、コマンドライン
":imap  :inoremap :iunmap    挿入
":lmap  :lnoremap :lunmap    挿入、コマンドライン、Lang-Arg
":cmap  :cnoremap :cunmap    コマンドライン
":tmap  :tnoremap :tunmap    端末ジョブ
"
"{訳注: Lang-Argについては language-mapping を参照}
"
"上記のマップコマンドの対応表:
"                                                        map-table
"       モード  | Norm | Ins | Cmd | Vis | Sel | Opr | Term | Lang |
"コマンド       +------+-----+-----+-----+-----+-----+------+------+
"[nore]map      | yes  |  -  |  -  | yes | yes | yes |  -   |  -   |
"n[nore]map     | yes  |  -  |  -  |  -  |  -  |  -  |  -   |  -   |
"[nore]map!     |  -   | yes | yes |  -  |  -  |  -  |  -   |  -   |
"i[nore]map     |  -   | yes |  -  |  -  |  -  |  -  |  -   |  -   |
"c[nore]map     |  -   |  -  | yes |  -  |  -  |  -  |  -   |  -   |
"v[nore]map     |  -   |  -  |  -  | yes | yes |  -  |  -   |  -   |
"x[nore]map     |  -   |  -  |  -  | yes |  -  |  -  |  -   |  -   |
"s[nore]map     |  -   |  -  |  -  |  -  | yes |  -  |  -   |  -   |
"o[nore]map     |  -   |  -  |  -  |  -  |  -  | yes |  -   |  -   |
"t[nore]map     |  -   |  -  |  -  |  -  |  -  |  -  | yes  |  -   |
"l[nore]map     |  -   | yes | yes |  -  |  -  |  -  |  -   | yes  |


" Macの時だけ Command <-> Control
if has('mac')
    nnoremap <D-d> <C-d>
    nnoremap <D-u> <C-u>
    nnoremap <C-v> <Nop>
endif


" ===============<<    plogin setting   >>=============== 
" indent hilight
let g:indent_guides_enable_on_vim_startup = 1

autocmd BufNewFile *.py 0r $HOME/.vim/template/python.txt



" ===============<<    Misc             >>=============== 
set noundofile
