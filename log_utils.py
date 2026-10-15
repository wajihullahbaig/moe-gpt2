import logging
import os
import sys

from tqdm import tqdm

log = logging.getLogger('moe')


class _TqdmHandler(logging.StreamHandler):
    """Console handler that prints through tqdm so progress bars stay intact."""

    def emit(self, record):
        """Write the formatted record via tqdm."""
        tqdm.write(self.format(record))


def setup_logger(run_dir):
    """Log to the console and append to run_dir/train.log with one shared format."""
    os.makedirs(run_dir, exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter('%(asctime)s │ %(message)s', datefmt='%H:%M:%S')
    for handler in (_TqdmHandler(), logging.FileHandler(os.path.join(run_dir, 'train.log'), encoding='utf-8')):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def banner(title, width=96):
    """Log a section heading."""
    log.info('━' * width)
    log.info(f'  {title}')
    log.info('━' * width)


class Table:
    """Stream rows as an aligned text table; the header is printed before the first row.

    Columns are (title, width) pairs; a negative width left-aligns the column.
    """

    def __init__(self, columns):
        self.titles, self.widths = zip(*columns)
        self.started = False

    def _line(self, cells):
        """Join cells into one aligned line."""
        return ' │ '.join(f'{cell:{"<" if w < 0 else ">"}{abs(w)}}' for cell, w in zip(cells, self.widths))

    def row(self, *cells):
        """Log one row, printing the header first if needed."""
        if not self.started:
            log.info(self._line(self.titles))
            log.info('─┼─'.join('─' * abs(w) for w in self.widths))
            self.started = True
        log.info(self._line(cells))
