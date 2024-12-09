from functools import lru_cache
import weakref
import gc

class CacheManager:
    def __init__(self, max_size=128):
        self.cache = {}
        self.max_size = max_size
        self._cleanup_threshold = max_size * 0.8

    @lru_cache(maxsize=128)
    def get_cached_data(self, key):
        return self.cache.get(key)

    def set_cached_data(self, key, value):
        if len(self.cache) >= self._cleanup_threshold:
            self.cleanup()
        self.cache[key] = weakref.proxy(value)

    def cleanup(self):
        gc.collect()
        # Remove expired weak references
        self.cache = {k: v for k, v in self.cache.items() 
                     if weakref.getweakrefcount(v) > 0} 