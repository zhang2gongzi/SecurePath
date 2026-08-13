#include <string.h>
#include <stddef.h>

void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    if (n == 0) return;
    memcpy(dst, src, n);
}
