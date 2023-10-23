#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv

"""
Add country column.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, error, DEBUG
from os.path import split, splitext
from math import pi
from typing import Union, List

from glob2 import glob
from pandas import DataFrame, read_csv, read_parquet
from pyproj import Proj

from csv2parquet import multiprocessing_wrapper

"""
=========
CONSTANTS
=========
"""

DESCRIPTION = ("Aggregate geographically.\n\n"
               "Reads Parquet or CSV, uses Latitude and Longitude to aggregate\n"
               "JDSs values in selectable mode [mean, median, max].")

DEFAULT_FILENAME_SUFFIX = "_geo_aggregated"

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


def convert_meter_to_angle(meters: Union[int, float] = 10) -> float:
    assert isinstance(meters, (int, float))
    earth_radius = 6371  # km average
    return meters / (earth_radius * 1000) * 180 / pi


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
        df.to_csv(path)  # , index=False)
    elif file_type.endswith('parquet'):
        df.to_parquet(path)  # , index=False)
    else:
        msg = f"unsupported file_type: {file_type}"
        error(msg)
        raise ValueError(msg)
    debug(f"written {df.shape[0]} rows into {split(path)[1]}")


def add_suffix_to_filename(path: str, suffix: str) -> str:
    assert isinstance(path, str)
    assert isinstance(suffix, str)
    path_without_extension, extension = splitext(path)
    return f"{path_without_extension}{suffix}{extension}"


"""
====================
CSV TO PARQUET CLASS
====================
"""


class GeoAggregator:
    """
    Relabel summary_stats.csv.
    """

    _src_path = ''
    _dest_suffix = ''
    _df = DataFrame()
    _aggregation_function = ''
    _size = 0

    def __init__(self, src_path: str, dest_suffix: str, mode: str, size: float) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_suffix, str)
        self._dest_suffix = dest_suffix
        assert isinstance(mode, str)
        self._aggregation_function = mode.lower()
        assert isinstance(size, (int, float))
        self._size = size

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def read(self, file: str):
        self._df = read_file(file)
        self._df = self._df[["Latitude", "Longitude", "Data"]]  # only interested in these columns
        self._df["Data"] /= 10  # JDS = Data / 10

    def reduce_resolution(self, cols: Union[None, str, List[str]] = None, by: Union[None, int, float] = None):
        if cols is None:  # default
            cols = ["Latitude", "Longitude"]
        if isinstance(cols, str):
            cols = [cols]
        if by is None:  # default
            by = self._size
        for col in cols:
            self._df[col] = (self._df[col] / by).round(0).astype(int)

    def geo_aggregate(self, file: str):
        self.read(file)
        projection = Proj(proj='utm', zone=15, ellps='WGS84', preserve_units=False)
        # Apply projection Latitude & Longitude angles to meters.
        self._df['Easting'], self._df['Northing'] = projection(self._df['Longitude'].to_numpy(),
                                                               self._df['Latitude'].to_numpy())
        self.reduce_resolution(cols=['Easting', 'Northing'], by=self._size)
        grouped = self._df.groupby(['Easting', 'Northing'])
        aggregated = grouped["Data"].agg(self._aggregation_function).reset_index()
        # Convert the 'Easting' and 'Northing' back to longitude and latitude
        aggregated['Longitude'], aggregated['Latitude'] = projection(aggregated['Easting'].to_numpy() * self._size,
                                                                     aggregated['Northing'].to_numpy() * self._size,
                                                                     inverse=True)
        write_file(aggregated.drop(columns=['Easting', 'Northing']),
                   path=add_suffix_to_filename(file, self._dest_suffix),
                   file_type=splitext(file)[1].lower())

    def run(self):
        """
        Main program.
        """
        files = glob(self._src_path)
        len_files = len(files)
        debug(f"found {len_files} files")

        multiprocessing_wrapper(self.geo_aggregate, files, enable_multiprocessing=len_files > 1)
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
    _mode = ''
    _size = 0

    def __init__(self, src_path: str, dest_suffix: str, mode: str, size: float) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        self._src_path = src_path
        assert isinstance(dest_suffix, str)
        self._dest_suffix = dest_suffix
        assert isinstance(mode, str)
        self._mode = mode
        assert isinstance(size, (int, float))
        self._size = size

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def run(self):
        """
        Main program.
        """
        aggregator = GeoAggregator(src_path=self._src_path,
                                   dest_suffix=self._dest_suffix,
                                   mode=self._mode,
                                   size=self._size)
        aggregator.run()


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
    parser.add_argument('-m', dest='mode', nargs=1, type=str, default=['median'],
                        choices=['mean', 'median', 'max'], help='aggregation mode, \'median\' by default')
    parser.add_argument('-s', dest='size', nargs=1, type=float, default=[10],
                        help='aggregation size in meters, default (unspecified) 10 meters.')

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
    assert isinstance(args.mode, list)
    assert len(args.mode) == 1
    assert isinstance(args.mode[0], str)
    assert args.mode[0].lower() in ['mean', 'median', 'max']
    assert isinstance(args.size, list)
    assert len(args.size) == 1
    assert isinstance(args.size[0], (int, float))
    assert args.size[0] >= 0  # must be positive

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
                                                  dest_suffix=args.dest_suffix,
                                                  mode=args.mode[0],
                                                  size=args.size[0])
    arg_processing.run()
    debug('Program execution completed. Starting clean-up.')


if __name__ == "__main__":
    main()
