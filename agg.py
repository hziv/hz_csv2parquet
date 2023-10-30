#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv

"""
Add country column.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, error, DEBUG
from os.path import split, splitext
from typing import Union, List

from glob2 import glob
from numpy import pi, cos, arange, count_nonzero
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


def convert_meter_to_angle(meters: Union[int, float] = 10) -> float:
    assert isinstance(meters, (int, float))
    earth_radius = 6371  # km average
    return meters / (earth_radius * 1000) * 180 / pi


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


def convert_meters_to_longitude_angle(meters: Union[int, float], latitude_angle_for_compensation: float) -> float:
    assert isinstance(meters, (int, float))
    assert isinstance(latitude_angle_for_compensation, float) and (-360 <= latitude_angle_for_compensation <= 360)
    # Convert latitude degrees to meters (approximately)
    lat_meters = (111132.92 - 559.82 * cos(2 * latitude_angle_for_compensation) +
                  1.175 * cos(4 * latitude_angle_for_compensation))
    # Convert meters to angles
    grid_size_deg = meters / lat_meters
    return grid_size_deg


"""
==================
PROGRESS BAR CLASS
==================
"""


class ProgressBar:
    """ Parse arguments. """

    # class globals
    _title_width = 20
    _width = 32
    _bar_prefix = ' |'
    _bar_suffix = '| '
    _empty_fill = ' '
    _fill = '#'
    progress_before_next = 0
    debug = False
    verbose = False
    quiet = False

    _progress = 0  # between 0 and _width -- used as filled portion of progress bar
    _increment = 0  # between 0 and (_max - _min) -- used for X/Y indication right of progress bar

    def __init__(self, text, maximum=10, minimum=0, verbosemode=''):
        """ Initialising parsing arguments.
        :param text: title of progress bar, displayed left of the progress bar
        :type text: str
        :param maximum: maximal value presented by 100% of progress bar
        :type maximum: int
        :param minimum: minimal value, zero by default
        :type minimum: int
        :param verbosemode: 'debug', 'verbose' or 'quiet'
        :type verbosemode: str
        """

        self.log = getLogger(self.__class__.__name__)
        assert isinstance(text, str)
        assert isinstance(maximum, int)
        assert isinstance(minimum, int)
        assert maximum > minimum
        self._text = text
        self._min = minimum
        self._max = maximum
        self._progress = 0
        self._increment = 0
        # LOGGING PARAMETERS
        assert isinstance(verbosemode, str)
        assert verbosemode in ['', 'debug', 'verbose', 'quiet']
        if verbosemode == 'debug':
            self.debug = True
        elif verbosemode == 'verbose':
            self.verbose = True
        elif verbosemode == 'quiet':
            self.quiet = True
        debug('{} started'.format(self._text))
        self.update()

    def __del__(self):
        """ Destructor. """

        # destructor content here if required
        debug('{} destructor completed.'.format(str(self.__class__.__name__)))

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        """
        Set length of the progress bar in characters.
        :param value: number of characters
        :type value: int
        """
        assert isinstance(value, int)
        assert 0 < value < 80
        self._width = value

    @property
    def title_width(self):
        return self._title_width

    @title_width.setter
    def title_width(self, value):
        """
        Set padding width for text before the progress bar.
        :param value: padding width in number of characters
        :type value: int
        """
        assert isinstance(value, int)
        assert 0 < value < 80
        self._title_width = value

    def next(self, n=1):
        """ Increment progress bar state.
        :param n: increment progress bar by n
        :type n: int
        """

        assert isinstance(n, int)
        assert n >= 0
        if n > 0:
            self._progress += 1 / (n * (self._max - self._min) / self._width)
            if self._progress > self._width:
                self._progress = self._width
            self._increment += n
            if float(self._progress) >= self.progress_before_next + 1 / self._width:
                self.progress_before_next = self._progress
                self.update()

    def update(self, end_char='\r'):
        """ Update progress bar on console.
        :param end_char: character used to command cursor to get back to beginning of line without carriage return.
        :type end_char: str
        """

        assert isinstance(end_char, str)
        diff = self._max - self._min
        bar = self._fill * int(self._progress)
        empty = self._empty_fill * (self._width - int(self._progress))
        if not self.debug and not self.verbose and not self.quiet:
            print("{:<{}.{}s}{}{}{}{}{}/{}".format(self._text, self._title_width, self._title_width,
                                                   self._bar_prefix, bar, empty, self._bar_suffix,
                                                   str(self._increment), str(diff)), end=end_char)

    def finish(self):
        """ Clean up and release handles. """

        self._progress = self._width
        self._increment = self._max - self._min
        if self._increment < 0:
            self._increment = 0
        self.update('\n')
        debug('{} finished'.format(self._text))


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

    @staticmethod
    def read(file: str) -> DataFrame:
        filename = split(file)[1]
        debug(f"reading {filename}")
        df = read_file(file, ["Latitude", "Longitude", "Data"])
        debug(f"{filename} read with size {df.shape}")
        df["Data"] /= 10  # JDS = Data / 10
        return df

    def reduce_resolution(self, df: DataFrame, by_meters: Union[None, int, float] = None) -> DataFrame:
        assert isinstance(df, DataFrame)
        debug(f"high resolution data size (with NaNs) {df.shape}")
        df.dropna(subset=["Latitude", "Longitude"], inplace=True)  # clean NaNs
        debug(f"high resolution data size (without NaNs) {df.shape}")
        if by_meters is None:  # default
            by_meters = self._size
        assert by_meters > 0
        if df.shape[0] > 0:  # not empty
            lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
            lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
            latitude_grid_size = convert_meters_to_latitude_angles(by_meters)
            lat_range = arange(start=lat_min, stop=lat_max, step=latitude_grid_size)
            i = 0
            for lat_grid_cell in lat_range:
                debug(f"reducing resolution for latitude [{i}/{len(lat_range)}]")
                longitude_grid_size = convert_meters_to_longitude_angle(by_meters, lat_grid_cell)  # Lat. compensated
                lat_idxs = ((lat_grid_cell <= df["Latitude"]) &
                            (df["Latitude"] < lat_grid_cell + latitude_grid_size))
                if count_nonzero(lat_idxs) > 0:
                    df.loc[lat_idxs, "Latitude"] = lat_grid_cell + latitude_grid_size / 2  # centre
                    lon_range = arange(start=lon_min, stop=lon_max, step=longitude_grid_size)
                    progress_bar = ProgressBar('longitude', len(lon_range))
                    for lon_grid_cell in lon_range:
                        lon_idxs = ((lon_grid_cell <= df["Longitude"]) &
                                    (df["Longitude"] < lon_grid_cell + longitude_grid_size))
                        if count_nonzero(lon_idxs) > 0:
                            df.loc[lon_idxs, "Longitude"] = lon_grid_cell + longitude_grid_size / 2  # centre
                        progress_bar.next()
                    progress_bar.finish()
                i += 1
        debug(f"resolution reduced by {by_meters} [meters], now size {df.shape}")
        return df

    def aggregate(self, df: DataFrame) -> DataFrame:
        assert isinstance(df, DataFrame)
        aggregated = df.groupby(["Latitude", "Longitude"])["Data"].agg(self._aggregation_function)
        debug(f"aggregated size {aggregated.shape}")
        return aggregated.to_frame().reset_index()

    def geo_aggregate(self, file: str):
        df = self.read(file)
        df = self.reduce_resolution(df, by_meters=self._size)
        df = self.aggregate(df)
        write_file(df, path=add_suffix_to_filename(file, self._dest_suffix), file_type=splitext(file)[1].lower())

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
