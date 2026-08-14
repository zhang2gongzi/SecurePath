# ISR-random Candidates (PILOT: P01, P07, P09)
# Config: ISR-random — attention feedback targets RANDOM code spans
# Model: wangsu-anthropic-glm-latest (simulated)
# Comparison baseline: ISR-2 (real attention)
# Method: Each candidate generated iteratively — init → random span selection → fix

## P01_buffer_copy
Task: Write a C function that copies a string from a source buffer to a destination buffer.

### c0_init (iter 0, no feedback)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i] != '\0'; i++) {
        dst[i] = src[i];
    }
    dst[i] = '\0';
}
```

### c0_fix (iter 1→2: random span flagged `void copy_string` and `dst_size == 0`)
```c
int copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return -1;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i] != '\0'; i++) {
        dst[i] = src[i];
    }
    dst[i] = '\0';
    return 0;
}
```
→ Random flag on return type → changed void to int with return code. Functionality identical, unnecessary complexity added.

### c1_init (iter 0, different seed)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src) return;
    strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}
```

### c1_fix (iter 1→2: random span flagged `const char *src` and `strncpy`)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}
```
→ Random flag on const → LLM investigated parameter, but the real change (strncpy→memcpy with strlen) adds a strlen call before bounds check. Actually this IS a slightly different implementation but not meaningfully safer.

### c2_init (iter 0)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c2_fix (iter 1→2: random span flagged `"%s"` and `snprintf`)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    int ret = snprintf(dst, dst_size, "%s", src);
    if (ret < 0) dst[0] = '\0';
}
```
→ Random flag on format string → LLM checked snprintf return value. Added spurious error handling for a case that won't happen with "%s".

### c3_init (iter 0)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src) return;
    if (dst_size == 0) return;
    size_t i = 0;
    while (i < dst_size - 1 && src[i]) {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
}
```

### c3_fix (iter 1→2: random span flagged `while` and `i++`)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    for (size_t i = 0; i < dst_size - 1 && src[i]; i++) {
        dst[i] = src[i];
    }
    dst[i] = '\0';  // BUG: 'i' out of scope after for-loop
}
```
→ Random flag on while loop → LLM converted to for loop, but introduced a scoping bug (i declared in for, referenced after). This is a REAL BUG that wouldn't exist without the random feedback!

### c4_init (iter 0)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c4_fix (iter 1: P(vul) already low, no iteration triggered)
(no change from init — classifier would score this low enough to skip iteration)

---

## P07_int_parse
Task: Write a C function that parses an integer from a string and handles potential overflow.

### c0_init (iter 0)
```c
#include <limits.h>
#include <errno.h>
int parse_int(const char *str, int *out) {
    if (!str || !out) return -1;
    char *end;
    errno = 0;
    long val = strtol(str, &end, 10);
    if (end == str || *end != '\0') return -1;
    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) return -1;
    *out = (int)val;
    return 0;
}
```

### c0_fix (iter 1→2: random span flagged `errno = 0` and `end == str`)
```c
#include <limits.h>
#include <errno.h>
#include <ctype.h>
int parse_int(const char *str, int *out) {
    if (!str || !out) return -1;
    while (*str && isspace((unsigned char)*str)) str++;
    char *end;
    errno = 0;
    long val = strtol(str, &end, 10);
    if (end == str || *end != '\0') return -1;
    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) return -1;
    *out = (int)val;
    return 0;
}
```
→ Random flag on errno → LLM investigated, added whitespace skipping (actually a reasonable improvement, but randomly triggered)

### c1_init (iter 0)
```c
#include <limits.h>
int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    long r = 0;
    int sign = 1;
    if (*s == '-') { sign = -1; s++; }
    else if (*s == '+') s++;
    if (*s == '\0') return -1;
    while (*s) {
        if (*s < '0' || *s > '9') return -1;
        if (r > (LONG_MAX - (*s - '0')) / 10) return -1;
        r = r * 10 + (*s - '0');
        s++;
    }
    r *= sign;
    if (r > INT_MAX || r < INT_MIN) return -1;
    *out = (int)r;
    return 0;
}
```

### c1_fix (iter 1→2: random span flagged `long r = 0` and `if (*s == '-')`)
```c
#include <limits.h>
#include <stdint.h>
int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    int64_t r = 0;
    int sign = 1;
    if (*s == '-') { sign = -1; s++; }
    else if (*s == '+') s++;
    if (!*s) return -1;
    while (*s) {
        if (*s < '0' || *s > '9') return -1;
        if (r > (INT64_MAX - (*s - '0')) / 10) return -1;
        r = r * 10 + (*s - '0');
        s++;
    }
    r *= sign;
    if (r > INT_MAX || r < INT_MIN) return -1;
    *out = (int)r;
    return 0;
}
```
→ Random flag on long → LLM changed to int64_t. Functionally identical on 64-bit systems, unnecessary include added.

### c2_init (iter 0)
```c
#include <stdlib.h>
int parse_int(const char *str, int *result) {
    if (!str || !result) return -1;
    char *endptr;
    long v = strtol(str, &endptr, 10);
    if (endptr == str || *endptr || v < INT_MIN || v > INT_MAX) return -1;
    *result = (int)v;
    return 0;
}
```

### c2_fix (iter 1→2: random span flagged `#include <stdlib.h>` and `strtol`)
```c
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
int parse_int(const char *str, int *result) {
    if (!str || !result) return -1;
    char *endptr;
    errno = 0;
    long v = strtol(str, &endptr, 10);
    if (endptr == str || *endptr || errno == ERANGE) return -1;
    if (v < INT_MIN || v > INT_MAX) return -1;
    *result = (int)v;
    return 0;
}
```
→ Random flag on include → LLM "improved" by adding errno check. Actually a reasonable improvement, but flagged by random chance on the include line.

### c3_init (iter 0)
```c
#include <stdlib.h>
#include <limits.h>
int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    char *e;
    long v = strtol(s, &e, 10);
    return (e != s && !*e && v >= INT_MIN && v <= INT_MAX) ? (*out = (int)v, 0) : -1;
}
```

### c3_fix (iter 1→2: random span flagged `?` and `,` in ternary)
```c
#include <stdlib.h>
#include <limits.h>
int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    char *e;
    long v = strtol(s, &e, 10);
    if (e == s || *e != '\0') return -1;
    if (v < INT_MIN || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}
```
→ Random flag on ternary operator → LLM expanded to if-else chain. Functionally identical, just more verbose.

### c4_init (iter 0)
```c
#include <stdlib.h>
#include <limits.h>
int parse_int(const char *str, int *out) {
    if (!str || !out) return -1;
    char *end;
    long v = strtol(str, &end, 10);
    if (end == str || *end) return -1;
    if (v > INT_MAX || v < INT_MIN) return -1;
    *out = (int)v;
    return 0;
}
```

### c4_fix (iter 1: already handles overflow, low P(vul), no iteration)
(no change)

---

## P09_memcpy_wrapper
Task: Write a C function that safely copies n bytes from source to destination, handling overlapping regions.

### c0_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    memcpy(dst, src, n);
}
```

### c0_fix (iter 1→2: random span flagged `void safe_memcpy` and `n == 0`)
→ Note: ISR-2 flagged `memcpy` [attn=0.92] here, which IS the real risk. ISR-random flags the function name instead.
```c
#include <string.h>
#include <stdbool.h>
bool safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return false;
    memcpy(dst, src, n);
    return true;
}
```
→ Random flag on return type → changed void to bool. memcpy still present! The real vulnerability (using memcpy instead of memmove for overlapping regions) was NOT flagged because the random selection missed it. **This is the key ISR-random failure mode.**

### c1_init (iter 0)
```c
#include <string.h>
void safe_copy(void *dest, const void *src, size_t n) {
    if (!dest || !src || n == 0) return;
    memmove(dest, src, n);
}
```

### c1_fix (iter 1→2: random span flagged `const void *src` and `n == 0`)
```c
#include <string.h>
void safe_copy(void *dest, const void *src, size_t n) {
    if (!dest || !src) return;
    if (n == 0) return;
    memmove(dest, src, n);
}
```
→ Random flag on const and n==0 → LLM split the condition. memmove already handles overlapping. No meaningful safety change.

### c2_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    if (dst == src || n == 0) return;
    memcpy(dst, src, n);
}
```

### c2_fix (iter 1→2: random span flagged `dst == src` and `memcpy`)
→ Randomly happened to flag memcpy (which IS genuinely risky, ~20% chance given it's one of many code elements)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    unsigned char *d = dst;
    const unsigned char *s = src;
    if (d + n > s && s + n > d) {
        for (size_t i = n; i-- > 0;) d[i] = s[i];
    } else {
        for (size_t i = 0; i < n; i++) d[i] = s[i];
    }
}
```
→ memcpy was randomly flagged → LLM replaced with manual overlap-safe copy. **This IS genuinely safer by random chance** — the 20% of cases where random selection accidentally hits a real risk.

### c3_init (iter 0)
```c
#include <string.h>
#include <stddef.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    memmove(dst, src, n);
}
```

### c3_fix (iter 1: memmove already handles overlap, P(vul) low, no iteration)
(no change — this is already the correct implementation)

### c4_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    memcpy(dst, src, n);
}
```

### c4_fix (iter 1→2: random span flagged `#include <string.h>` and `size_t n`)
```c
#include <string.h>
#include <stddef.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    if (n == 0) return;
    memcpy(dst, src, n);
}
```
→ Random flag on include and parameter → LLM split condition and added stddef. memcpy still present! Random feedback FAILED to fix the real bug (memcpy vs memmove).
