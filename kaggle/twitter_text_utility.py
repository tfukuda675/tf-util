import os
import re

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
tqdm.pandas()


from xml.sax.saxutils import unescape
import emoji


def tweet_clean_text(sentence):
    if sentence is None:
        return ''
    
    # 特殊文字デコード
    sentence = unescape(sentence)

    # ユーザー名削除
    sentence = re.sub(r'@[0-9a-zA-Z_:]*', "", sentence)

    # ハッシュタグ削除　非貪欲マッチへ
    sentence = re.sub('#.*?\s', " ", sentence)

    # URL削除
    sentence = re.sub(r'(https?)(:\/\/[-_.!~*\'()a-zA-Z0-9;\/?:\@&=+\$,%#]+)', "", sentence)

    # RT削除
    sentence = re.sub(r'(RT ).*$', "", sentence)

    # 「w」はすべて半角1文字に
    sentence = re.sub(r'ｗ+', "w", sentence)
    sentence = re.sub(r'w+', "w", sentence)

    # 「・」を3点リーダにまとめる
    sentence = re.sub(r'・・+', "…", sentence)
    sentence = re.sub(r'･･+', "…", sentence)
    sentence = re.sub(r'\.\.+', "…", sentence)

    # 「！」はすべて半角単!に
    sentence = re.sub(r'！+', "!", sentence)
    sentence = re.sub(r'!+', "!", sentence)

    # 「？」はすべて半角単?に
    sentence = re.sub(r'？+', "?", sentence)
    sentence = re.sub(r'\?+', "?", sentence)

    # 連続する「っ」「ッ」をつめる
    sentence = re.sub(r'っ+', "っ", sentence)
    sentence = re.sub(r'ッ+', "ッ", sentence)

    # 連続する「あ行」をつめる（3文字以上）
    sentence = re.sub(r'あああ+', "あああ", sentence)
    sentence = re.sub(r'いいい+', "いいい", sentence)
    sentence = re.sub(r'ううう+', "ううう", sentence)
    sentence = re.sub(r'えええ+', "えええ", sentence)
    sentence = re.sub(r'おおお+', "おおお", sentence)
    sentence = re.sub(r'ぁぁぁ+', "ぁぁぁ", sentence)
    sentence = re.sub(r'ぃぃぃ+', "ぃぃぃ", sentence)
    sentence = re.sub(r'ぅぅぅ+', "ぅぅぅ", sentence)
    sentence = re.sub(r'ぇぇぇ+', "ぇぇぇ", sentence)
    sentence = re.sub(r'ぉぉぉ+', "ぉぉぉ", sentence)
    sentence = re.sub(r'アアア+', "アアア", sentence)
    sentence = re.sub(r'イイイ+', "イイイ", sentence)
    sentence = re.sub(r'ウウウ+', "ウウウ", sentence)
    sentence = re.sub(r'エエエ+', "エエエ", sentence)
    sentence = re.sub(r'オオオ+', "オオオ", sentence)
    sentence = re.sub(r'ァァァ+', "ァァァ", sentence)
    sentence = re.sub(r'ィィィ+', "ィィィ", sentence)
    sentence = re.sub(r'ゥゥゥ+', "ゥゥゥ", sentence)
    sentence = re.sub(r'ェェェ+', "ェェェ", sentence)
    sentence = re.sub(r'ォォォ+', "ォォォ", sentence)

    # 連続する「笑」をつめる
    sentence = re.sub(r'笑+', "笑", sentence)

    # 連続する「b」をつめる
    sentence = re.sub(r'ｂ+', "b", sentence)
    sentence = re.sub(r'b+', "b", sentence)

    # 連続する「＞」「＜」をつめる
    sentence = re.sub(r'＞+', "＞", sentence)
    sentence = re.sub(r'＜+', "＜", sentence)
    sentence = re.sub(r'>+', ">", sentence)
    sentence = re.sub(r'<+', "<", sentence)

    # 連続する「ー」「～」をつめる
    sentence = re.sub(r'ー+', "ー", sentence)
    sentence = re.sub(r'～+', "～", sentence)
    sentence = re.sub(r'~+', "～", sentence)
    sentence = re.sub(r'〜+', "～", sentence)

    # 連続する3点リーダをつめる
    sentence = re.sub(r'…+', "…", sentence)

    # 連続する読点をつめる
    sentence = re.sub(r'、+', "、", sentence)

    # 連続する「！？」「？！」をつめる
    sentence = re.sub(r'(！？)+', "!?", sentence)
    sentence = re.sub(r'(？！)+', "?!", sentence)

    # 連続する絵文字をつめる
    sentence = tweet_prepare_emoji(sentence)

    # 「（」「）」はすべて半角に
    sentence = sentence.replace("（", "(")
    sentence = sentence.replace("）", ")")

    # スペースすべて削除
    #sentence = sentence.replace(" ", "")
    #sentence = sentence.replace("　", "")

    # 改行を半角スペースに変換
    sentence = sentence.replace("\r\n", " ")
    sentence = sentence.replace("\r", " ")
    sentence = sentence.replace("\n", " ")

    # 句点をすべて半角スペースに変換
    #sentence = sentence.replace("。", " ")

    # 「！」「？」に半角スペースをつける
    sentence = sentence.replace("！", "！ ")
    sentence = sentence.replace("？", "？ ")

    # 「！w」から半角スペースを消す
    sentence = sentence.replace("！ w", "！w ")

    # 「！？」「？！」から半角スペースを消す
    sentence = sentence.replace("！ ？ ", "！？ ")
    sentence = sentence.replace("？ ！ ", "？！ ")

    # 「)」前から半角スペースを消す
    sentence = sentence.replace(" )", ")")

    # 「」」前から半角スペースを消す
    sentence = sentence.replace(" 」", "」")

    # 「w」の前から半角スペースを消す
    sentence = sentence.replace(" w", "w")

    # 全角空白を半角Spaceへ
    sentence = sentence.replace("ㅤ", " ")
    
    # 連続するスペースをつめる
    sentence = re.sub(r'\s+', " ", sentence)
    sentence = re.sub(r'\s+', " ", sentence)

    # 余白削除
    sentence = sentence.strip()

    # ペアになってないカッコを修正
    sentence = tweet_prepare_pair(sentence, '(', ')')
    sentence = tweet_prepare_pair(sentence, '[', ']')
    sentence = tweet_prepare_pair(sentence, '「', '」')
    sentence = tweet_prepare_pair(sentence, '『', '』')
    sentence = tweet_prepare_pair(sentence, '【', '】')
    sentence = tweet_prepare_pair(sentence, '［', '］')
    
    # レビュー**件を削除
    sentence = re.sub(r'レビュー\d+件','',sentence)

    # AAっぽいツイートは除外
    #if sentence.find('＿＿') >= 0 or sentence.find('人人人') >= 0 \
    #    or sentence.find('|') >= 0 or sentence.find('｜') >= 0 or sentence.find('／') >= 0 or sentence.find('＼') >= 0 \
    #    or sentence.find('┏') >= 0 or sentence.find('┓') >= 0 or sentence.find('┗') >= 0 or sentence.find('┛') >= 0 \
    #    or sentence.find('┳') >= 0 or sentence.find('┻') >= 0 or sentence.find('━') >= 0 or sentence.find('┃') >= 0:
    #    return ''

    return sentence

def tweet_prepare_emoji(text):
    prepared_text = ""
    prev_emoji = ''
    for char in list(text):
        if char in emoji.UNICODE_EMOJI:
            if prev_emoji == char:
                continue
            prev_emoji = char
        prepared_text += char
    return prepared_text

def tweet_prepare_pair(text, start_char, end_char):
    start_pos = 0
    end_pos = 0
    hit_flag = False
    prepared_text = ""
    for char in list(text):
        if char == start_char:
            if hit_flag:
                start_pos += 1
                prepared_text += text[start_pos:end_pos]
                start_pos = end_pos
            else:
                prepared_text += text[start_pos:end_pos]
                start_pos = end_pos
            hit_flag = True
        elif char == end_char:
            if hit_flag:
                prepared_text += text[start_pos:end_pos]
                start_pos = end_pos
            else:
                prepared_text += text[start_pos:end_pos]
                start_pos = end_pos + 1
            hit_flag = False
        end_pos += 1
    prepared_text += text[start_pos:end_pos]
    if hit_flag:
        prepared_text += end_char
    return prepared_text


def tweet_transformer_data(texts,tokenizer,device):

    sen_max_length = max([len(ids) for ids in tokenizer(texts)["input_ids"]])
       
    encoding = tokenizer(
        texts,
        padding="max_length", 
        truncation=True,
        add_special_tokens = True,
        return_tensors='pt',
        max_length=sen_max_length
    )
    input_ids = encoding['input_ids'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    return input_ids, token_type_ids, attention_mask, sen_max_length

