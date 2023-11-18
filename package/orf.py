"""Read the ORF file."""
from __future__ import annotations

from typing import NamedTuple
from collections.abc import Callable
import functools
import os
import pathlib
import re

import numpy

#: Regular expression to parse a line of the ORF file
ENTRY: Callable[[str], re.Match | None] = re.compile(
    r'(?P<year>\d{4})\/(?P<month>\d{2})\/(?P<day>\d{2})\s+'
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+(?P<cycle>\d+)\s+'
    r'(?P<pass>\d+)\s+\d+\s+(?P<lon>-?\d+\.\d+)\s+'
    r'(?P<lat>-?\d+\.\d+)').search


class Entry(NamedTuple):
    """Store a single entry of the ORF file.

    Args:
        date: Date of the event
        cycle_number: Cycle number
        pass_number: Pass number
        latitude: Latitude of the event
        longitude: Longitude of the event
    """
    date: numpy.datetime64
    cycle_number: int
    pass_number: int
    latitude: float
    longitude: float

    @classmethod
    def from_line(cls, line) -> Entry | None:
        """Create an entry from a line of the ORF file.

        Args:
            line: Line of the ORF file

        Returns:
            Entry or None if the line is not valid
        """
        match: re.Match[str] | None = ENTRY(line)
        if match is None:
            return None
        return cls(
            date=numpy.datetime64(
                f"{match['year']}-{match['month']}-{match['day']}T"
                f"{match['time']}", 'ms'),
            cycle_number=int(match['cycle']),
            pass_number=int(match['pass']),
            latitude=float(match['lat']),
            longitude=float(match['lon']),
        )


@functools.lru_cache(maxsize=1)
def load(filename: os.PathLike) -> dict[int, numpy.datetime64]:
    """Load an ORF file and return for each cycle numbers the first measurement
    of the pass at the equator.

    Args:
        filename: Path to the ORF file

    Returns:
        Dictionary of cycle numbers and dates.
    """
    filename = pathlib.Path(filename)
    with filename.open(encoding='UTF-8') as stream:
        entries: dict[int, numpy.datetime64] = {}
        previous_cycle: int = -1
        for line in stream:
            entry: Entry | None = Entry.from_line(line)
            if entry is None or entry.cycle_number == 0:
                continue
            # Ignore entries for information stored at the poles.
            if entry.latitude == 0:
                continue
            if previous_cycle != entry.cycle_number:
                entries[entry.cycle_number] = entry.date
            previous_cycle = entry.cycle_number
    return entries
