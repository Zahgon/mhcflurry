'''
Download MHCflurry released datasets and trained models.

Examples

Fetch the default downloads:
    $ mhcflurry-downloads fetch

Fetch a specific download:
    $ mhcflurry-downloads fetch models_class1_pan

Get the path to a download:
    $ mhcflurry-downloads path models_class1_pan

Get the URL of a download:
    $ mhcflurry-downloads url models_class1_pan

Summarize available and fetched downloads:
    $ mhcflurry-downloads info
'''
import sys
import argparse
import logging
import os
from pipes import quote
import errno
import tarfile
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from tqdm import tqdm

import posixpath
import pandas

try:
    from urllib.request import urlretrieve
    from urllib.parse import urlsplit
except ImportError:
    from urllib import urlretrieve
    from urlparse import urlsplit

from .downloads import (
    get_current_release,
    get_current_release_downloads,
    get_downloads_dir,
    get_path,
    ENVIRONMENT_VARIABLES)

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="Output less")

parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    default=False,
    help="Output more")

subparsers = parser.add_subparsers(dest="subparser_name")

parser_fetch = subparsers.add_parser('fetch')
parser_fetch.add_argument(
    'download_name',
    metavar="DOWNLOAD",
    nargs="*",
    help="Items to download")
parser_fetch.add_argument(
    "--keep",
    action="store_true",
    default=False,
    help="Don't delete archives after they are extracted")
parser_fetch.add_argument(
    "--release",
    default=get_current_release(),
    help="Release to download. Default: %(default)s")
parser_fetch.add_argument(
    "--already-downloaded-dir",
    metavar="DIR",
    help="Don't download files, get them from DIR")

parser_info = subparsers.add_parser('info')

parser_path = subparsers.add_parser('path')
parser_path.add_argument(
    "download_name",
    nargs="?",
    default='')

parser_url = subparsers.add_parser('url')
parser_url.add_argument(
    "download_name",
    nargs="?",
    default='')


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level="INFO")
    if args.verbose:
        logging.basicConfig(level="DEBUG")

    command_functions = {
        "fetch": fetch_subcommand,
        "info": info_subcommand,
        "path": path_subcommand,
        "url": url_subcommand,
        None: lambda args: parser.print_help(),
    }
    command_functions[args.subparser_name](args)


def mkdir_p(path):
    """
    Make directories as needed, similar to mkdir -p in a shell.

    From:
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    pass


def yes_no(boolean):
    pass


# For progress bar on download. See https://pypi.python.org/pypi/tqdm
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        pass


def fetch_subcommand(args):
    pass


def info_subcommand(args):
    pass


def path_subcommand(args):
    """
    Print the local path to a download
    """
    pass


def url_subcommand(args):
    """
    Print the URL(s) for a download
    """
    pass
