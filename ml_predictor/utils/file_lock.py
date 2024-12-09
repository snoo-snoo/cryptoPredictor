import os
import time
from contextlib import contextmanager
import logging
import msvcrt  # Windows-specific file locking
import tempfile

logger = logging.getLogger(__name__)

@contextmanager
def file_lock(file_path, timeout=10):
    """Context manager for file locking with timeout (Windows compatible)"""
    lock_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(file_path)}.lock")
    start_time = time.time()
    
    while True:
        try:
            # Try to create lock file
            if not os.path.exists(lock_path):
                with open(lock_path, 'w') as lock_file:
                    # Try to acquire lock using Windows file locking
                    handle = msvcrt.get_osfhandle(lock_file.fileno())
                    msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
                    
            # If we got here, we have the lock
            try:
                yield
                break
            finally:
                try:
                    # Release lock and remove lock file
                    if os.path.exists(lock_path):
                        with open(lock_path, 'r+') as lock_file:
                            handle = msvcrt.get_osfhandle(lock_file.fileno())
                            msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
                        os.remove(lock_path)
                except Exception as e:
                    logger.warning(f"Error releasing lock: {e}")
                    
        except (IOError, OSError) as e:
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for lock on {file_path}")
                raise TimeoutError(f"Could not acquire lock for {file_path}")
            time.sleep(0.1)  # Wait before retry