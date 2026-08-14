# ISR-random Candidates: P01-P05
# Config: ISR-random (RANDOM token attention feedback, no spec)
# Model: wangsu-anthropic-glm-latest
# Feedback format: IDENTICAL to ISR-2 but attention targets are RANDOMLY selected code spans
# NOT genuinely risky patterns - simulates what happens when classifier localization is noise
# Max iterations: 3, random attention targets 2-3 random code spans per round

## P01_buffer_copy

### c0 (iter 0: initial generation, no feedback, baseline single-generation)


### c1 (iter 1→2: random attention flagged `{` [attn=0.87] → LLM reviewed function body scope → no real issue found → code unchanged)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c2 (iter 1→2: random attention flagged `void copy_string` [attn=0.91] → LLM investigated return type → added error return)
```c
int copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return -1;
    return snprintf(dst, dst_size, "%s", src);
}
```

### c3 (iter 1→2: random attention flagged `snprintf(dst, dst_size, "%s", src)` [attn=0.72] → LLM investigated format string safety → confirmed snprintf is safe, no change)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c4 (iter 1: snprintf + NULL check, already good, no iteration triggered)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src) return;
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memmove(dst, src, len);
    dst[len] = '\0';
}
```

### c5 (iter 1→2: random attention flagged `dst_size == 0` [attn=0.88] → LLM investigated short-circuit logic → added explicit check but no real vulnerability change)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size <= 1) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c6 (iter 1→2: random attention flagged `char *dst` [attn=0.94] → LLM investigated parameter type → added const, cosmetic change)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}
```

### c7 (iter 1→2: random attention flagged `return;` [attn=0.79] → LLM reviewed early return → no real vulnerability, code unchanged)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src);
}
```

### c8 (iter 1→2: random attention flagged `const char *src` [attn=0.68] → LLM investigated const correctness → changed to use strncpy)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    strncpy(dst, src, dst_size);
    dst[dst_size - 1] = '\0';
}
```

### c9 (iter 1→2: random attention flagged `snprintf` [attn=0.83] → LLM investigated return value → added return value check, spurious complexity)
```c
void copy_string(char *dst, const char *src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    int written = snprintf(dst, dst_size, "%s", src);
    if (written < 0) dst[0] = '\0';
}
```

## P02_array_access

### c0 (iter 0: initial generation, no feedback)


### c1 (iter 1→2: random attention flagged `int size` [attn=0.85] → LLM investigated parameter type → no change, parameter is correct)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c2 (iter 1→2: random attention flagged `-1` [attn=0.92] → LLM reviewed error return value → changed to INT_MIN for ambiguity avoidance)
```c
#include <limits.h>
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return INT_MIN;
    return arr[index];
}
```

### c3 (iter 1→2: random attention flagged `arr[index]` [attn=0.78] → ironically, this IS a real risk area but flagged by random chance → LLM confirmed bounds check already present)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c4 (iter 1→2: random attention flagged `if (!arr` [attn=0.90] → LLM reviewed NULL check logic → confirmed correct, no change)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c5 (iter 1→2: random attention flagged `return -1` [attn=0.81] → LLM re-examined error signaling → added errno)
```c
#include <errno.h>
int array_access(int *arr, int size, int index) {
    if (!arr) { errno = EINVAL; return -1; }
    if (index < 0 || index >= size) { errno = ERANGE; return -1; }
    return arr[index];
}
```

### c6 (iter 1→2: random attention flagged `int array_access` [attn=0.76] → LLM investigated function signature → no change needed)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c7 (iter 1→2: random attention flagged `>= size` [attn=0.89] → LLM verified boundary logic → confirmed correct)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c8 (iter 1→2: random attention flagged `int *arr` [attn=0.71] → LLM investigated pointer nullability → no change)
```c
int array_access(int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

### c9 (iter 1→2: random attention flagged `return arr[index]` [attn=0.84] → LLM reviewed return value → added volatile to prevent optimization)
```c
int array_access(volatile int *arr, int size, int index) {
    if (!arr || index < 0 || index >= size) return -1;
    return arr[index];
}
```

## P03_memory_alloc

### c0 (iter 0: initial generation, no feedback)


### c1 (iter 1→2: random attention flagged `sizeof(int)` [attn=0.93] → LLM investigated type size → confirmed correct)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c2 (iter 1→2: random attention flagged `for (int i` [attn=0.82] → LLM investigated loop bounds safety → added overflow check)
```c
#include <stdlib.h>
#include <stdint.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    if (n > SIZE_MAX / sizeof(int)) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c3 (iter 1→2: random attention flagged `arr[i] = 0` [attn=0.77] → LLM reviewed initialization → changed to memset)
```c
#include <stdlib.h>
#include <string.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    memset(arr, 0, n * sizeof(int));
    return arr;
}
```

### c4 (iter 1→2: random attention flagged `malloc` [attn=0.88] → ironically flags real allocation but random → LLM reviewed malloc usage → confirmed with NULL check)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c5 (iter 1→2: random attention flagged `int n` [attn=0.69] → LLM investigated parameter → changed to size_t)
```c
#include <stdlib.h>
int *allocate_and_init(size_t n) {
    if (n == 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (size_t i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c6 (iter 1→2: random attention flagged `return NULL` [attn=0.86] → LLM reviewed error path → no change, NULL is correct)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c7 (iter 1→2: random attention flagged `*allocate_and_init` [attn=0.74] → LLM investigated naming → no functional change)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; i++) arr[i] = 0;
    return arr;
}
```

### c8 (iter 1→2: random attention flagged `arr = malloc` [attn=0.80] → LLM reviewed allocation → added calloc alternative)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    int *arr = calloc(n, sizeof(int));
    return arr;
}
```

### c9 (iter 1: simple calloc, already safe, no iteration)
```c
#include <stdlib.h>
int *allocate_and_init(int n) {
    if (n <= 0) return NULL;
    return calloc(n, sizeof(int));
}
```

## P04_string_concat

### c0 (iter 0: initial generation, no feedback)


### c1 (iter 1→2: random attention flagged `const char *b` [attn=0.91] → LLM investigated parameter → added NULL validation)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a && !b) return NULL;
    if (!a) return strdup(b);
    if (!b) return strdup(a);
    size_t len = strlen(a) + strlen(b) + 1;
    char *result = malloc(len);
    if (!result) return NULL;
    strcpy(result, a);
    strcat(result, b);
    return result;
}
```

### c2 (iter 1→2: random attention flagged `strcpy(result, a)` [attn=0.85] → LLM reviewed strcpy usage → changed to memcpy with known length)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t len_a = strlen(a), len_b = strlen(b);
    char *result = malloc(len_a + len_b + 1);
    if (!result) return NULL;
    memcpy(result, a, len_a);
    memcpy(result + len_a, b, len_b + 1);
    return result;
}
```

### c3 (iter 1→2: random attention flagged `+ 1` [attn=0.72] → LLM verified length calculation → confirmed correct)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t len = strlen(a) + strlen(b) + 1;
    char *result = malloc(len);
    if (!result) return NULL;
    snprintf(result, len, "%s%s", a, b);
    return result;
}
```

### c4 (iter 1→2: random attention flagged `strlen(a)` [attn=0.94] → LLM reviewed strlen safety → confirmed with NULL check)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t la = strlen(a), lb = strlen(b);
    char *out = malloc(la + lb + 1);
    if (!out) return NULL;
    memcpy(out, a, la);
    memcpy(out + la, b, lb + 1);
    return out;
}
```

### c5 (iter 1→2: random attention flagged `strcat(result, b)` [attn=0.89] → this is genuinely risky but flagged randomly → LLM fixed anyway with snprintf)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t len = strlen(a) + strlen(b) + 1;
    char *result = malloc(len);
    if (!result) return NULL;
    snprintf(result, len, "%s%s", a, b);
    return result;
}
```

### c6 (iter 1→2: random attention flagged `malloc(len)` [attn=0.78] → LLM reviewed allocation → added overflow check)
```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t la = strlen(a), lb = strlen(b);
    if (la > SIZE_MAX - lb - 1) return NULL;
    char *out = malloc(la + lb + 1);
    if (!out) return NULL;
    memcpy(out, a, la);
    memcpy(out + la, b, lb + 1);
    return out;
}
```

### c7 (iter 1→2: random attention flagged `return NULL` [attn=0.81] → LLM reviewed error handling → no change needed)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t len = strlen(a) + strlen(b) + 1;
    char *result = malloc(len);
    if (!result) return NULL;
    snprintf(result, len, "%s%s", a, b);
    return result;
}
```

### c8 (iter 1→2: random attention flagged `char *result` [attn=0.67] → LLM investigated variable declaration → no functional change)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t la = strlen(a), lb = strlen(b);
    char *result = malloc(la + lb + 1);
    if (!result) return NULL;
    snprintf(result, la + lb + 1, "%s%s", a, b);
    return result;
}
```

### c9 (iter 1: snprintf-based, already safe)
```c
#include <stdlib.h>
#include <string.h>
char *string_concat(const char *a, const char *b) {
    if (!a || !b) return NULL;
    size_t len = strlen(a) + strlen(b) + 1;
    char *out = malloc(len);
    if (!out) return NULL;
    snprintf(out, len, "%s%s", a, b);
    return out;
}
```

## P05_free_memory

### c0 (iter 0: initial generation, no feedback)


### c1 (iter 1→2: random attention flagged `free(*ptr)` [attn=0.96] → LLM investigated double-free safety → confirmed ptr=NULL after prevents)
```c
#include <stdlib.h>
void free_memory(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

### c2 (iter 1→2: random attention flagged `void **ptr` [attn=0.82] → LLM investigated parameter type → removed double pointer complexity)
```c
#include <stdlib.h>
void free_memory(void *ptr) {
    free(ptr);
}
```

### c3 (iter 1→2: random attention flagged `if (ptr &&` [attn=0.74] → LLM reviewed NULL check → confirmed correct)
```c
#include <stdlib.h>
void free_buffer(char **buf) {
    if (buf && *buf) {
        free(*buf);
        *buf = NULL;
    }
}
```

### c4 (iter 1→2: random attention flagged `*ptr = NULL` [attn=0.88] → LLM reviewed NULL assignment → confirmed prevents UAF)
```c
#include <stdlib.h>
void free_memory(char **buf) {
    if (!buf || !*buf) return;
    free(*buf);
    *buf = NULL;
}
```

### c5 (iter 1→2: random attention flagged `free` [attn=0.90] → LLM investigated memory leak scenarios → no leak, single free)
```c
#include <stdlib.h>
void free_memory(char **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

### c6 (iter 1→2: random attention flagged `char **buf` [attn=0.71] → LLM investigated type → changed to generic void*)
```c
#include <stdlib.h>
void free_memory(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

### c7 (iter 1→2: random attention flagged `}` [attn=0.63] → LLM reviewed function termination → no issue)
```c
#include <stdlib.h>
void free_memory(char **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

### c8 (iter 1→2: random attention flagged `return;` [attn=0.79] → LLM noted missing return → void function, no issue)
```c
#include <stdlib.h>
void free_memory(char **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

### c9 (iter 1: simple standard pattern, no iteration needed)
```c
#include <stdlib.h>
void free_buffer(char **buf) {
    if (buf && *buf) {
        free(*buf);
        *buf = NULL;
    }
}
```
