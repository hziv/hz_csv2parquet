#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv

"""
Add country column.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, warning, error, DEBUG
from os.path import split, splitext

from glob2 import glob
from pandas import DataFrame, Series, read_csv, read_parquet
from geolocator import osm  # pip install geocoder

from csv2parquet import multiprocessing_wrapper

"""
=========
CONSTANTS
=========
"""

DESCRIPTION = ("Add country column.\n\n"
               "Reads Parquet or CSV, uses Latitude and Longitude columns to populate\n"
               "Country column and stores to the same format.")

DEFAULT_FILENAME_SUFFIX = "_with_country"

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


def read_file(path: str) -> DataFrame:
    assert isinstance(path, str)
    file_type = splitext(path)[1].lower()
    if file_type.endswith('csv'):
        df = read_csv(path, index_col=False, skipinitialspace=True, low_memory=True)
    elif file_type.endswith('parquet'):
        df = read_parquet(path)
    else:
        msg = f"unsupported file {split(path)[1]}"
        error(msg)
        raise FileNotFoundError(msg)
    debug(f"file {split(path)[1]} read with size {df.shape} and columns: {list(df.columns)}")
    return df


def write_file(df: DataFrame, path: str, file_type: str):
    assert isinstance(df, DataFrame)
    assert isinstance(path, str)
    assert isinstance(file_type, str)
    file_type = file_type.lower()
    if file_type.endswith('csv'):
        df.to_csv(path, index=False)
    elif file_type.endswith('parquet'):
        df.to_parquet(path, index=False)
    else:
        msg = f"unsupported file_type: {file_type}"
        error(msg)
        raise ValueError(msg)
    debug(f"written {df.shape[0]} rows into {split(path)[1]}")


def add_suffix_to_filename(path: str, suffix: str) -> str:
    assert isinstance(path, str)
    assert isinstance(suffix, str)
    path_without_extension, extension = splitext(path)
    return f"{path_without_extension}{suffix}.{extension}"


"""
====================
CSV TO PARQUET CLASS
====================
"""


class AddCountry:
    """
    Relabel summary_stats.csv.
    """

    _src_path = ''
    _dest_suffix = ''

    def __init__(self, src_path: str, dest_suffix: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_suffix, str)
        self._dest_suffix = dest_suffix

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def add_country_to_file(self, src_path: str):
        def get_country(coordinates: Series) -> str:
            location = None
            try:
                location = osm((coordinates["Latitude"], coordinates["Longitude"]), method='reverse')
            except ValueError as e:
                warning(f"unknown location for {coordinates} - {e}")
            if location:
                return location.json['country']
            else:
                return "No country"

        file_type = splitext(src_path)[1].lower()
        df = read_file(src_path)
        df["Country"] = df[["Latitude", "Longitude"]].apply(get_country, axis=1)
        write_file(df, add_suffix_to_filename(src_path, self._dest_suffix), file_type)

    def run(self):
        """
        Main program.
        """
        files = glob(self._src_path)
        len_files = len(files)
        debug(f"found {len_files} files")

        multiprocessing_wrapper(self.add_country_to_file, files, enable_multiprocessing=len_files > 1)
        debug(f"finished processing {len_files} files")


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
    _dest_suffix = ''

    def __init__(self, src_path: str, dest_suffix: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_suffix, str)
        self._dest_suffix = dest_suffix

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def run(self):
        """
        Main program.
        """
        add_country = AddCountry(src_path=self._src_path,
                                 dest_suffix=self._dest_suffix)
        add_country.run()


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
    parser.add_argument('src', nargs=1, type=str, help='source path, supports wildcards')
    parser.add_argument('dest_suffix', nargs='?', type=str, default=DEFAULT_FILENAME_SUFFIX,
                        help=f'destination filename suffix, {DEFAULT_FILENAME_SUFFIX} by default (unspecified)')

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
    assert isinstance(args.dest_suffix, str)

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
                                                  dest_suffix=args.dest_suffix)
    arg_processing.run()
    debug('Program execution completed. Starting clean-up.')


if __name__ == "__main__":
    main()
