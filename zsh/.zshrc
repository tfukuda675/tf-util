# Created by newuser for 5.9
#
#    _________________________
#___/ Lang Settings           \_________________________
#
export lang=jp.utf-8
export lc_all=en_us.utf-8

umask 027

#    _________________________
#___/  Usefull                \_________________________
#

# 補完機能を有効にする
autoload -Uz compinit && compinit

# 補完スクリプトのpath設定

# Set on .zshrc_`uname` file

# OS共通のものがautoloadにはいる
fpath=($ZDOTDIR/autoload/*(N-/) $fpath)

# 補完スクリプトのload
# -U は、呼び出し側のシェルで alias 設定を設定していたとしても、
# 中の関数側ではその影響を受けなくなるというオプション。
# ls は alias を設定してることが多いと思うけど、この場合はそのままの ls が実行される。
# -z は関数を zsh 形式で読み込むというオプション。
#
# color一覧
autoload -Uz colorlist

# 補完候補を一覧を表示
setopt auto_list

# 補完メニューをカーソルで選択可能にする
zstyle ':completion:*:default' menu select=1

# 色設定読を有効にする
autoload -Uz colors
colors

# ディレクトリ名でcd
setopt auto_cd

# cd後に実行
function chpwd() { /bin/ls -FhG; pwd; }

# コマンドのスペルミスを指摘
setopt correct

# 補完候補表示時にビープ音を鳴らさない
setopt nolistbeep

# 候補が多い場合は詰めて表示
setopt list_packed

# Ctrl+sのロック, Ctrl+qのロック解除を無効にする
setopt no_flow_control


#    _________________________
#___/  History                \_________________________
#

# ディレクトリ履歴
DIRSTACKSIZE=30
setopt auto_pushd

# コマンド履歴ファイル
HISTFILE=$ZDOTDIR/.zsh_history

# コマンド履歴数
HISTSIZE=10000
SAVEHIST=20000

# 直前のコマンドの重複を削除
setopt hist_ignore_dups

# 同じコマンドをヒストリに残さない
setopt hist_ignore_all_dups

# 同時に起動したzshの間でヒストリを共有
setopt share_history

# 上下キーで入力補完
autoload history-search-end
zle -N history-beginning-search-backward-end history-search-end
zle -N history-beginning-search-forward-end history-search-end

# コマンド履歴検索をVim風に
bindkey "^P" history-beginning-search-backward
bindkey "^N" history-beginning-search-forward

#    _________________________
#___/  VSC                    \_________________________
#
autoload -Uz vcs_info
zstyle ':vcs_info:*' enable git 
zstyle ':vcs_info:git:*' check-for-changes true
zstyle ':vcs_info:git:*' stagedstr "%F{yellow}!"
zstyle ':vcs_info:git:*' unstagedstr "%F{red}+"
zstyle ':vcs_info:*' formats "%F{green}%c%u[%b]%f"
zstyle ':vcs_info:*' actionformats '[%b|%a]'
precmd () { vcs_info }
#precmd_vcs_info() { vcs_info }
#precmd_functions+=( precmd_vcs_info )

#    _________________________
#___/  Prompt                 \_________________________
#

# Set on .zshrc_`uname` file

#    _________________________
#___/  Apps                   \_________________________
#

# Set on .zshrc_`uname` file

#    _________________________
#___/  Alias                  \_________________________
#

# ls Color Settings
export CLICOLOR=true
export LSCOLORS=gxfxxxxxcxxxxxxxxxgxgx
export LS_COLORS='di=1;34:ln=1;35:ex=32'
zstyle ':completion:*' list-colors 'di=1;34' 'ln=1;35' 'ex=32'

# ls alias
## 表示設定
# -a  : .付きも表示
# -a  : .付きも表示、「.」と「..」以外の隠しファイル含め全表示
# -l  : 詳細表示
# -F  : 名前の末尾に種類を意味する記号を表示
# -G  : 種類に応じて色を変更して表示
# -h  : [K]/[M]/[G]付きで表示
## 順序変更
# -t  : 日時を最近のものから並び替えて表示
# -R  : サブディレクトリの内容も表示
# -S  : サイズの大きいものから並び替えて表示
# -r  : 名前を降順に表示
alias ls="ls -GhF"
alias ll="ls -l"
alias lm="ls -Al"
alias lt="ls -Altr"
alias lS="ls -AlSr"


alias u="cd ../"
alias uu="cd ../../"
alias uuu="cd ../../../"
alias grep='grep --color=auto'
alias du="du -h"


#    _________________________
#___/  OS Dependent           \_________________________
#

# What OS are we running?
if [[ $(uname) == "Darwin" ]]; then
    [ -f $ZDOTDIR/.zshrc_`uname` ] && . $ZDOTDIR/.zshrc_`uname`
elif [[ $(uname) == "Linux" ]]; then
    [ -f $ZDOTDIR/.zshrc_`uname` ] && . $ZDOTDIR/.zshrc_`uname`
fi


