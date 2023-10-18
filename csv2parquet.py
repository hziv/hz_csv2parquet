#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv

"""
CSV to pandas parquet converter.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, error, DEBUG
from multiprocessing.pool import ThreadPool
from os import cpu_count
from os.path import isdir, isfile, join, split, splitext
from typing import Union, List

from glob2 import glob
from numpy import ndarray
from pandas import read_csv, Series

"""
=========
CONSTANTS
=========
"""

DESCRIPTION = \
    "CSV to pandas parquet converter.\n\n" \
    "Mainly used to accelerate file reading and reduce file sizes."

"""
================
GLOBAL VARIABLES
================
"""


"""
=========
FUNCTIONS
=========
"""


def multiprocessing_wrapper(func, inputs: Union[list, ndarray, Series], enable_multiprocessing: bool = False):
    assert isinstance(inputs, (list, ndarray, Series))
    assert isinstance(enable_multiprocessing, bool)
    if func is not None:
        if isinstance(func, list):  # list of functions
            for f in func:
                multiprocessing_wrapper(f, inputs, enable_multiprocessing)  # recursively call for each function
        if enable_multiprocessing:
            cpu_cnt = cpu_count()
            debug(f"{cpu_cnt} processors found")
            process_pool = ThreadPool(cpu_cnt)
            process_pool.map(func, sorted(inputs))
        else:
            for inp in sorted(inputs):
                func(inp)
        debug(f"multiprocessing_command finished running {len(inputs)}")
    else:
        error("func argument to multiprocessing_command is None")


def csv_file_to_parquet_file(src: str, dest: str = '') -> str:
    assert isinstance(src, str)
    assert isfile(src), f"expecting {src} to point to a file"
    assert isinstance(dest, str)
    is_dir_dest = isdir(dest)
    if dest == '' or is_dir_dest:  # use the same filename
        dest_filename = f"{splitext(split(src)[1])[0]}.parquet"
        if is_dir_dest:
            dest = join(dest, dest_filename)
        elif dest == '':
            dest = join(split(src)[0], dest_filename)
    debug(f"reading {split(src)[1]} file")
    df = read_csv(src, index_col=False, skipinitialspace=True, low_memory=True)
    debug(f"{split(src)[1]} read with size {df.shape} and columns:\n{list(df.columns)}")
    df.to_parquet(dest, index=False)
    debug(f"{df.shape} stored to {split(dest)[1]}")
    return dest


def csv_directory_to_parquet_directory(src_dir: str) -> List[str]:
    assert isinstance(src_dir, str)
    assert isdir(src_dir), f"expecting {src_dir} to point to directory"
    source_files = glob(f"{src_dir}/**/*.csv", case_sensitive=False)
    debug(f"found {len(source_files)} CSV files in {src_dir}")
    if len(source_files) > 0:
        multiprocessing_wrapper(csv_file_to_parquet_file, source_files, enable_multiprocessing=True)
    error(f"no CSV files found in {src_dir}")
    return ['']


"""
====================
CSV TO PARQUET CLASS
====================
"""


class CsvToParquet:
    """
    Relabel summary_stats.csv.
    """

    _src_path = ''
    _dest_path = ''

    def __init__(self, src_path: str, dest_path: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_path, str)
        self._dest_path = dest_path

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def run(self):
        """
        Main program.
        """
        if isdir(self._src_path):
            return csv_directory_to_parquet_directory(self._src_path)
        if isfile(self._src_path):
            return csv_file_to_parquet_file(self._src_path, self._dest_path)
        error(f"invalid source path {self._src_path}")


"""
========================
ARGUMENT SANITY CHECKING
========================
"""


class ArgumentsAndConfigProcessing:
    """
    Argument parsing and default value population (from config).
    """

    _src_path = ''
    _dest_dir = ''

    def __init__(self, src_path: str, dest_path: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_path, str)
        self._dest_dir = dest_path

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def run(self):
        """
        Main program.
        """
        collator = CsvToParquet(src_path=self._src_path,
                                dest_path=self._dest_dir)
        collator.run()


"""
======================
COMMAND LINE INTERFACE
======================
"""


def main():
    """ Argument Parser and Main Class instantiation. """

    # ---------------------------------
    # Parse arguments
    # ---------------------------------

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    no_extension_default_name = parser.prog.rsplit('.', 1)[0]
    parser.add_argument('src', nargs=1, type=str, help='source path')
    parser.add_argument('dest', nargs='?', type=str, default='', help='destination path')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('-d', '--debug', help='sets verbosity to display debug level messages',
                        action="store_true")

    args = parser.parse_args()

    # ---------------------------------
    # Preparing LogFile formats
    # ---------------------------------

    assert isinstance(args.src, list)
    assert len(args.src) == 1
    assert isinstance(args.src[0], str)
    assert isinstance(args.dest, str)

    log_filename = f'{no_extension_default_name}.log'
    try:
        basicConfig(filename=log_filename, filemode='a', datefmt='%Y/%m/%d %I:%M:%S %p', level=DEBUG,
                    format='%(asctime)s, %(threadName)-8s, %(name)-15s %(levelname)-8s - %(message)s')
    except PermissionError as err:
        raise PermissionError(f'Error opening log file {log_filename}. File might already be opened by another '
                              f'application. Error: {err}\n')

    console = StreamHandler()
    if args.debug:
        console.setLevel(DEBUG)
    formatter = Formatter('%(threadName)-8s, %(name)-15s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    getLogger('').addHandler(console)

    getLogger('main')
    info(f"Successfully opened log file named: {log_filename}")
    debug(f"Program run with the following arguments: {str(args)}")

    # ---------------------------------
    # Debug mode
    # ---------------------------------

    assert isinstance(args.debug, bool)

    # ---------------------------------
    # Instantiation
    # ---------------------------------

    arg_processing = ArgumentsAndConfigProcessing(src_path=args.src[0],
                                                  dest_path=args.dest)
    arg_processing.run()
    debug('Program execution completed. Starting clean-up.')


if __name__ == "__main__":
    main()
