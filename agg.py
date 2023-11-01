#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv

"""
Add country column.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, error, DEBUG, ERROR
from os.path import split, splitext
from typing import Union, List

from glob2 import glob
from numpy import pi, arange, digitize
from pandas import DataFrame, read_csv, read_parquet

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

EARTH_RADIUS_IN_METERS = 6371000

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


def read_file(path: str, cols: List[str]) -> DataFrame:
    assert isinstance(path, str)
    assert isinstance(cols, list)
    assert all(isinstance(col, str) for col in cols)
    file_type = splitext(path)[1].lower()
    if file_type.endswith('csv'):
        df = read_csv(path, index_col=False, usecols=cols, skipinitialspace=True, low_memory=True)
    elif file_type.endswith('parquet'):
        df = read_parquet(path, columns=cols)  # engine='fastparquet'
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
    debug(f"{split(path)[1]} written {df.shape[0]} rows into {split(path)[1]}")


def add_suffix_to_filename(path: str, suffix: str) -> str:
    assert isinstance(path, str)
    assert isinstance(suffix, str)
    path_without_extension, extension = splitext(path)
    return f"{path_without_extension}{suffix}{extension}"


def convert_meters_to_latitude_angles(meters: Union[int, float] = 10) -> float:
    assert isinstance(meters, (int, float))
    # Calculate the circumference of the Earth at the equator
    circumference_equator = 2 * pi * EARTH_RADIUS_IN_METERS
    # Calculate the equivalent latitude grid size in degrees for the desired grid size
    lat_angle = (meters / circumference_equator) * 360
    debug(f"converted {meters} [meters] to {lat_angle} [latitude°]")
    return lat_angle


"""
=====================
GEO AGGREGATION CLASS
=====================
"""


class GeoAggregator:
    """
    Relabel summary_stats.csv.
    """

    _src_path = ''
    _dest_suffix = ''
    _df = DataFrame()
    _aggregation_function = ''
    _size_of_grid_in_angles = 0
    _latitude_bins = None

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
        self._size_of_grid_in_angles = convert_meters_to_latitude_angles(size)
        self._latitude_bins = arange(-90, 90, self._size_of_grid_in_angles, dtype=float)
        self._longitude_bins = arange(-180, 180, self._size_of_grid_in_angles, dtype=float)  # ignoring latitude

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    @staticmethod
    def read(file: str) -> DataFrame:
        filename = split(file)[1]
        debug(f"reading {filename}")
        df = read_file(file, ["Latitude", "Longitude", "Data"])
        debug(f"{filename} read with size {df.shape}")
        df["Data"] /= 10  # JDS = Data / 10
        return df

    def iterate(self, df: DataFrame) -> DataFrame:
        df['latitude_bin_id'] = digitize(df["Latitude"], self._latitude_bins) - 1  # bin indices from zero
        df['longitude_bin_id'] = digitize(df["Longitude"], self._longitude_bins) - 1  # bin indices from zero
        aggregated = (df.groupby(by=['latitude_bin_id', 'longitude_bin_id'])["Data"].
                      apply(self._aggregation_function).to_frame().reset_index())
        aggregated["Latitude"] = (self._latitude_bins[aggregated['latitude_bin_id'].values] +
                                  self._size_of_grid_in_angles / 2)
        aggregated["Longitude"] = (self._longitude_bins[aggregated['longitude_bin_id'].values] +
                                   self._size_of_grid_in_angles / 2)
        aggregated.drop(columns=['latitude_bin_id', 'longitude_bin_id'], inplace=True)
        debug(f"aggregated {self._aggregation_function} size {aggregated.shape}")
        return aggregated

    def geo_aggregate(self, file: str):
        df = self.read(file)
        aggregated = self.iterate(df)
        write_file(aggregated, path=add_suffix_to_filename(file, self._dest_suffix), file_type=splitext(file)[1].lower())

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
    assert isinstance(args.debug, bool)

    log_filename = f'{no_extension_default_name}.log'
    try:
        basicConfig(filename=log_filename, filemode='a', datefmt='%Y/%m/%d %I:%M:%S %p', level=DEBUG,
                    format='%(asctime)s, %(threadName)-8s, %(name)-15s %(levelname)-8s - %(message)s')
    except PermissionError as err:
        raise PermissionError(f'Error opening log file {log_filename}. File might already be opened by another '
                              f'application. Error: {err}\n')

    console = StreamHandler()
    console.setLevel(ERROR)
    if args.debug:
        console.setLevel(DEBUG)
    formatter = Formatter('%(threadName)-8s, %(name)-15s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    getLogger('').addHandler(console)

    getLogger('main')
    info(f"Successfully opened log file named: {log_filename}")
    debug(f"Program run with the following arguments: {str(args)}")

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
