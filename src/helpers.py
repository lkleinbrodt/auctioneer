from config import *

import datetime

def format_elapsed_time(elapsed_time: datetime.timedelta):
    """
    Returns a human-readable string representation of the elapsed time.

    Args:
        elapsed_time (datetime.timedelta): The elapsed time to format.

    Returns:
        str: A string representation of the elapsed time in logical units (days, hours, minutes, seconds, milliseconds).

    Example:
        >>> elapsed_time = datetime.timedelta(days=2, hours=5, minutes=30, seconds=45)
        >>> format_elapsed_time(elapsed_time)
        '2 days, 5 hours, 30 minutes'
    """
    assert isinstance(elapsed_time, datetime.timedelta)
    time_units = []
    
    units = [
        ('day', 24 * 60 * 60),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1),
        ('millisecond', 0.001)
    ]
    
    counter = 0
    for unit, seconds in units:
        if counter == 3:
            break
        value = elapsed_time.total_seconds() // seconds
        if value > 0:
            time_units.append(f"{int(value)} {unit}{'s' if value > 1 else ''}")
            elapsed_time -= datetime.timedelta(seconds=value * seconds)
            counter += 1
            
    return ', '.join(time_units)