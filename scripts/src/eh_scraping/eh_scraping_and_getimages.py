#! /Users/tfuku/Tools/miniforge3/envs/py311/bin/python3

import os
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

import click

#     _________________
#____/  Logger設定      \____________________
#
def setup_logger(name, logfile='logger_log.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    fh = FileHandler(logfile)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


    return logger

logger = setup_logger(__name__)


#     _________________
#____/  Class          \____________________
#
class TGTPAGE:
    def __init__(self, driver=None, imgobjs=list(), dir=str()):
        self.driver     =   driver
        self.imgobjs    =   imgobjs
        self.dir        =   dir
        self.imgtops    =   list()

    def find_image_topurls(self):
        driver = self.driver

        # 特定のIDのdivを取得
        div = driver.find_element(By.ID, 'gdt')

        # その下にある全ての a タグを取得
        a_tags = div.find_elements(By.TAG_NAME, 'a')

        # hrefをすべて取得して表示
        for a in a_tags:
            href    =   a.get_attribute('href')
            self.imgtops.append(href)
            imgobj  =   IMGPAGE(driver=self.driver, url=href, dir=self.dir)
            self.imgobjs.append(imgobj)

        # URL 確認
        logger.debug(self.imgtops)

    def find_image_urls(self):
        for obj in self.imgobjs:
            obj.find_image_url()
            obj.get_image()


class IMGPAGE:
    def __init__(self, driver=None, url=str(), dir=str()): 
        self.driver     =   driver
        self.topurl     =   url
        self.dir        =   dir
        self.imgurl     =   str()

    def find_image_url(self):
        self.driver.get(self.topurl)

        # ページが完全に読み込まれるまで待機（必要に応じて調整）
        self.driver.implicitly_wait(10)

        # 特定のIDのdivを取得
        div     =   self.driver.find_element(By.ID, 'i3')

        # その下にある全ての a タグを取得
        img     =   div.find_element(By.ID, 'img')

        # その下のimgのsrcを取得
        self.imgurl =   img.get_attribute('src')

        # URL 確認
        logger.debug(self.topurl)
        logger.debug(self.imgurl)


    def get_image(self):
        ## urlの一番うしろを取得し保存するファイル名で利用
        basename    =   os.path.basename(self.topurl)
        ## 後ろ側を4桁でゼロ埋めする。
        basename_id =   basename.split("-")[0]
        basename_no =   basename.split("-")[1].zfill(4)
        basename    =   basename_id + "-" + basename_no

        response    =   requests.get(self.imgurl)
        img_ext     =   os.path.splitext(self.imgurl)[1]
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        save_path   =   f"./{self.dir}/{basename}{img_ext}"

        if os.path.exists(save_path):
            logger.info(f"ファイル有り {save_path}")
        else:
            logger.info(f"ダウンロード {self.imgurl} ファイルを {save_path} で保存")
            with open(save_path, "wb") as f:
                f.write(response.content)


#     _________________
#____/  Click設定       \____________________
#

@click.command()
@click.option('--tgts', '-t', required=True,  multiple=True)
@click.option('--dir', '-d', required=True)
@click.option('--page', '-p', required=False, default=0)
@click.option('--stop', '-s', required=False, default=100)
def run(tgts, dir, page, stop):
    for t in tgts:
        logger.info(f"Target URL : {t}")

        # Chromeドライバーのインストール
        driver_path = ChromeDriverManager().install()
        logger.info(f"Chrome Driver Path : {driver_path}")

        # ヘッドレスモード（ブラウザを表示しない）
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')

        # Chromeドライバーの起動
        driver = webdriver.Chrome(service=Service(driver_path), options=options)

        # 最後のページかどうか確認しながら、ページ送り
        n_page  =   page
        end_flg =   False
        while n_page <= stop:
            logger.debug(f'{n_page}')
            url = f"{t}?p={n_page}"
            logger.info(f'次のURLを確認中 : {url}')

            # サイトにアクセス
            driver.get(url)

            # ページが完全に読み込まれるまで待機（必要に応じて調整）
            driver.implicitly_wait(10)

            # 最終ページか確認
            try:
                element = driver.find_element(By.CLASS_NAME, 'ptdd')
                logger.debug(f'<もしくは>が見つかりました: {element.text}')
                elements = driver.find_elements(By.CSS_SELECTOR, '.ptdd')
                for elem in elements:
                    logger.debug(f'<か>どちらかを確認: {elem.text}')
                    if elem.text == ">":
                        logger.debug('最終ページです')
                        end_flg = True
                        break

            except NoSuchElementException:
                logger.debug('>ボタンが見つかりませんでした')
                logger.debug('最終ページではありません')


            # ページごとに処理
            pages = TGTPAGE(driver=driver,imgobjs=list(),dir=dir)
            logger.debug(f"ページごとのオブジェクト : {pages}")
            pages.find_image_topurls()
            pages.find_image_urls()

            del pages


            # 最終ページの場合はwhileを抜ける
            if end_flg:
                break

            n_page += 1


def main():
    run()

if __name__ == '__main__':
    main()
