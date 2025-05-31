#! /users/tfuku/tools/miniforge3/envs/py311/bin/python3

import multiprocessing as multi
from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from multiprocessing import Pool

import click
import polars as pl
import yaml


#     ____________________
# ____/ [*] logger設定      \____________________
#
def setup_logger(name, logfile="logger_log.log"):
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # create file handler with a info log level
    fh = FileHandler(logfile)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # create console handler with a info log level
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch_formatter = Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%y-%m-%d %h:%m:%s"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


logger = setup_logger(__name__)


#     ____________________
# ____/ [*] click設定       \____________________
# Clockについてはこちら
# https://click.palletsprojects.com/en/stable/parameters/
@click.command()
@click.option("--config", "-c", required=True, type=click.File(mode="r"))
def run(config):
    cfgs = yaml.safe_load(config)
    p = Pool(multi.cpu_count())
    results = p.map(polars_text_process, cfgs)
    p.close()

    logger.info(results)


#     ____________________
# ____/ [*] functions      \____________________
#


def polars_text_process(cfg):
    df = pl.read_csv(
        cfg["file"], separator="\n", has_header=False, new_columns=["texts"]
    )
    df = df.lazy()
    logger.info(df)
    logger.info(cfg["regexp"])

    for key, reg in cfg["regexp"].items():
        logger.info(reg)
        df = df.with_columns(pl.col("texts").str.extract(reg, 1).alias(key))
    df = df.collect()
    logger.info(df)
    return df


def main():
    run()


if __name__ == "__main__":
    main()
