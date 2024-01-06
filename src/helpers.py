from config import *

import shutil
from s3 import S3Client
import os
import zipfile
import pytz
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


def download_champions(s3_directory, local_path = ROOT_DIR/'data/models/champions/', granularity='FIFTEEN_MINUTE'):
    
    s3 = S3Client()
    pacific_tz = pytz.timezone('US/Pacific')
    
    timestamp = datetime.datetime.now(pacific_tz).strftime("%Y%m%d")
    shutil.make_archive(
        base_name = str(ROOT_DIR/f'data/models/champions_{timestamp}'),
        format = 'zip',
        base_dir = local_path
    )
    
    if s3_directory[-1] != '/':
        s3_directory += '/'
    
    best_trials = s3.load_json(s3_directory+'best_trials.json')
    s3.download_file(s3_directory+'best_trials.json', local_path/'best_trials.json')
    
    for product_id, best_trial_info in best_trials.items():
        best_trial = best_trial_info['best_number']
        print(s3_directory+f"{product_id}_{granularity}/{best_trial}.zip")
        s3.download_file(
            s3_directory+f"{product_id}_{granularity}/{best_trial}.zip",
            local_path/f"{product_id}_{granularity}.zip"
        )
        os.makedirs(local_path/f"{product_id}_{granularity}", exist_ok=True)
        with zipfile.ZipFile(ROOT_DIR/'data/tmp.zip', 'r') as zip_ref:
            zip_ref.extractall(local_path/f"{product_id}_{granularity}")
        
        # Delete the file
        os.remove(local_path/f"{product_id}_{granularity}.zip")