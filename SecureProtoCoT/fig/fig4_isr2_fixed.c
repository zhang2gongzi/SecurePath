#include <string.h>

void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    memmove(dst, src, n);
}
