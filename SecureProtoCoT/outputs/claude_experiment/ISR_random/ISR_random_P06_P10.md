# ISR-random Candidates: P06-P10
# Config: ISR-random — attention feedback targets RANDOM code spans
# Model: wangsu-anthropic-glm-latest (simulated)
# Max iterations: 3, random attention targets 2-3 random code spans per round

## P06_file_read
Task: Write a C function that reads the contents of a file into a buffer and returns the buffer.

### c0_init (iter 0)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    if (out_len) *out_len = sz;
    fclose(f);
    return buf;
}
```

### c0_fix (iter 1→2: random span flagged `fseek(f, 0, SEEK_END)` and `long sz`)
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    int64_t sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    if (out_len) *out_len = (size_t)sz;
    fclose(f);
    return buf;
}
```
→ Random flag on fseek → added ftell error check (actually useful but random trigger). Changed long to int64_t (cosmetic).

### c1_init (iter 0)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char *data = malloc(size + 1);
    if (!data) { fclose(fp); return NULL; }
    size_t n = fread(data, 1, size, fp);
    data[n] = '\0';
    fclose(fp);
    return data;
}
```

### c1_fix (iter 1→2: random span flagged `fread(data, 1, size, fp)` and `size_t n`)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size < 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *data = malloc(size + 1);
    if (!data) { fclose(fp); return NULL; }
    size_t total = 0;
    while (total < (size_t)size) {
        size_t n = fread(data + total, 1, (size_t)size - total, fp);
        if (n == 0) break;
        total += n;
    }
    data[total] = '\0';
    fclose(fp);
    return data;
}
```
→ Random flag on fread → LLM added partial read loop. Actually addresses a real edge case (fread can return short) but triggered randomly.

### c2_init (iter 0)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (buf) {
        fread(buf, 1, len, f);
        buf[len] = '\0';
    }
    fclose(f);
    return buf;
}
```

### c2_fix (iter 1→2: random span flagged `if (buf)` and `fclose(f)`)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) { fclose(f); return NULL; }
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, len, f);
    if (n != (size_t)len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0';
    fclose(f);
    return buf;
}
```
→ Random flag on if-buf → LLM changed error handling. Added fread return check (partially useful), added free on error (useful). Random flags on cleanup paths triggered genuine improvements to error handling — ironically effective.

### c3_init (iter 0)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *d = malloc(s + 1);
    if (d) { fread(d, 1, s, f); d[s] = 0; }
    if (out_size) *out_size = s;
    fclose(f);
    return d;
}
```

### c3_fix (iter 1→2: random span flagged `char *d` and `fseek`)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char *read_file(const char *path, size_t *out_size) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long s = ftell(f);
    if (s < 0) { fclose(f); return NULL; }
    rewind(f);
    char *data = malloc(s + 1);
    if (!data) { fclose(f); return NULL; }
    memset(data, 0, s + 1);
    fread(data, 1, s, f);
    if (out_size) *out_size = s;
    fclose(f);
    return data;
}
```
→ Random flag on variable name → LLM investigated and added path NULL check, fseek error check, memset. Some improvements but all triggered by random noise.

### c4_init (iter 0)
```c
#include <stdio.h>
#include <stdlib.h>
char *read_file(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    if (buf) {
        fread(buf, 1, size, f);
        buf[size] = 0;
    }
    fclose(f);
    return buf;
}
```

### c4_fix (iter 1: already has reasonable structure, P(vul) moderate but no single clear risk → no iteration triggered by random feedback either)
(no change)

---

## P08_linked_list
Task: Write a C function that inserts a node at the head of a linked list. Include struct definition and insert function.

### c0_init (iter 0)
```c
#include <stdlib.h>
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node *head, int value) {
    Node *n = malloc(sizeof(Node));
    if (!n) return head;
    n->data = value;
    n->next = head;
    return n;
}
```

### c0_fix (iter 1→2: random span flagged `malloc(sizeof(Node))` and `if (!n)`)
```c
#include <stdlib.h>
#include <string.h>
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node *head, int value) {
    Node *n = malloc(sizeof(*n));
    if (!n) return head;
    memset(n, 0, sizeof(*n));
    n->data = value;
    n->next = head;
    return n;
}
```
→ Random flag on malloc → LLM changed sizeof(Node) to sizeof(*n) (good practice) and added memset (unnecessary). Cosmetic improvements from random trigger.

### c1_init (iter 0)
```c
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node *head, int value) {
    Node *new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = head;
    return new_node;
}
```

### c1_fix (iter 1→2: random span flagged `new_node->data = value` and `return new_node`)
```c
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node *head, int value) {
    Node *new_node = malloc(sizeof(Node));
    if (!new_node) return head;
    new_node->data = value;
    new_node->next = head;
    return new_node;
}
```
→ Random flag on data assignment → LLM noticed missing NULL check on malloc and added it. **Random flag on the wrong line actually surfaced a real bug (missing malloc NULL check) via code review.** This is the "broken clock is right twice a day" case — the random flag triggered a full re-review.

### c2_init (iter 0)
```c
#include <stdlib.h>
typedef struct node {
    int val;
    struct node *next;
} node_t;
node_t *insert_head(node_t *h, int v) {
    node_t *n = malloc(sizeof(*n));
    if (!n) return h;
    n->val = v; n->next = h;
    return n;
}
```

### c2_fix (iter 1→2: random span flagged `node_t` and `n->val = v`)
```c
#include <stdlib.h>
typedef struct node {
    int val;
    struct node *next;
} node_t;
node_t *insert_head(node_t *head, int value) {
    if (value < 0) return head;
    node_t *n = malloc(sizeof(*n));
    if (!n) return head;
    n->val = value;
    n->next = head;
    return n;
}
```
→ Random flag on typedef name → LLM added `value < 0` check. Spurious constraint — linked list can hold negative values.

### c3_init (iter 0)
```c
#include <stdlib.h>
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node *head, int data) {
    Node *n = malloc(sizeof(Node));
    if (!n) return NULL;
    n->data = data;
    n->next = head;
    return n;
}
```

### c3_fix (iter 1→2: random span flagged `return NULL` and `Node *head`)
```c
#include <stdlib.h>
typedef struct Node {
    int data;
    struct Node *next;
} Node;
Node *insert_head(Node **head, int data) {
    if (!head) return NULL;
    Node *n = malloc(sizeof(Node));
    if (!n) return NULL;
    n->data = data;
    n->next = *head;
    *head = n;
    return n;
}
```
→ Random flag on return NULL → LLM changed to double-pointer API. **Actually changed the function contract!** The insert_head now takes Node** instead of Node*. This is a major API change triggered by random feedback on the error return value.

### c4_init (iter 0)
```c
#include <stdlib.h>
typedef struct Node { int data; struct Node *next; } Node;
Node *insert_head(Node *h, int v) {
    Node *n = malloc(sizeof(Node));
    if (n) { n->data = v; n->next = h; }
    return n ? n : h;
}
```

### c4_fix (iter 1: standard linked list insert, no obvious risks, P(vul) low)
(no change)

---

## P10_input_loop
Task: Write a C function that reads user input into a fixed-size buffer in a loop until the user types 'quit'.

### c0_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
void input_loop(void) {
    char buf[256];
    while (1) {
        printf("> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;
        buf[strcspn(buf, "\n")] = '\0';
        if (strcmp(buf, "quit") == 0) break;
        printf("You entered: %s\n", buf);
    }
}
```

### c0_fix (iter 1→2: random span flagged `char buf[256]` and `printf("> ")`)
```c
#include <stdio.h>
#include <string.h>
#define BUFSIZE 256
void input_loop(void) {
    char buf[BUFSIZE];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) break;
        buf[strcspn(buf, "\n")] = '\0';
        if (strcmp(buf, "quit") == 0) break;
        printf("You entered: %s\n", buf);
    }
}
```
→ Random flag on buffer size → LLM added #define and fflush. fflush is good practice but triggered randomly.

### c1_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
void input_loop(char *buf, size_t size) {
    while (fgets(buf, size, stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        if (!strcmp(buf, "quit")) return;
    }
}
```

### c1_fix (iter 1→2: random span flagged `size_t size` and `fgets`)
```c
#include <stdio.h>
#include <string.h>
void input_loop(char *buf, size_t size) {
    if (!buf || size == 0) return;
    while (fgets(buf, size, stdin)) {
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
        if (!strcmp(buf, "quit")) return;
    }
}
```
→ Random flag on parameter → LLM added NULL/size validation (good). Changed strcspn to manual newline strip (equivalent). Reasonable but triggered by random flags.

### c2_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
void input_loop(void) {
    char buf[256];
    while (fgets(buf, 256, stdin) && strcmp(buf, "quit\n") != 0) {
        printf("%s", buf);
    }
}
```

### c2_fix (iter 1→2: random span flagged `fgets(buf, 256, stdin)` and `strcmp(buf, "quit\n")`)
```c
#include <stdio.h>
#include <string.h>
void input_loop(void) {
    char buf[256];
    while (fgets(buf, sizeof(buf), stdin)) {
        buf[strcspn(buf, "\n")] = '\0';
        if (strcmp(buf, "quit") == 0) break;
        puts(buf);
    }
}
```
→ Random flag on fgets → LLM changed 256 to sizeof(buf) and added newline stripping. Good improvements from random trigger — but again, the "re-review everything" effect.

### c3_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
void input_loop(void) {
    char input[256];
    while (1) {
        if (!fgets(input, 256, stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (!strcmp(input, "quit")) break;
        printf("got: %s\n", input);
    }
}
```

### c3_fix (iter 1→2: random span flagged `while (1)` and `break`)
```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
void input_loop(void) {
    char input[256];
    bool running = true;
    while (running) {
        if (!fgets(input, sizeof(input), stdin)) { running = false; continue; }
        input[strcspn(input, "\n")] = '\0';
        if (strcmp(input, "quit") == 0) running = false;
        else printf("got: %s\n", input);
    }
}
```
→ Random flag on while(1) → LLM refactored to use bool flag instead of break. No functional difference, added unnecessary complexity.

### c4_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
void input_loop(void) {
    char buf[256];
    while (fgets(buf, sizeof(buf), stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        if (!strcmp(buf, "quit")) break;
    }
}
```

### c4_fix (iter 1: simple reasonable code, no obvious issue)
(no change)

---

## P09_memcpy_wrapper (extended from pilot)
See pilot file for c0-c4.

### c5_init (iter 0)
```c
#include <string.h>
#include <stdlib.h>
void safe_copy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    void *tmp = malloc(n);
    if (tmp) { memcpy(tmp, src, n); memcpy(dst, tmp, n); free(tmp); }
}
```

### c5_fix (iter 1→2: random span flagged `void *tmp` and `free(tmp)`)
```c
#include <string.h>
#include <stdlib.h>
void safe_copy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    unsigned char *tmp = malloc(n);
    if (!tmp) return;
    memcpy(tmp, src, n);
    memcpy(dst, tmp, n);
    free(tmp);
}
```
→ Random flag on tmp variable → LLM added n==0 check and typed to unsigned char*. Still uses memcpy instead of memmove! Random flags missed the real issue again. This is the second time ISR-random failed to fix the memcpy→memmove issue that ISR-2 catches 100% of the time.

### c6_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dest, const void *src, size_t n) {
    if (!dest || !src || n == 0) return;
    if ((char *)dest < (char *)src + n && (char *)src < (char *)dest + n) {
        char *d = dest;
        const char *s = src;
        for (size_t i = n; i-- > 0;) d[i] = s[i];
        return;
    }
    memcpy(dest, src, n);
}
```

### c6_fix (iter 1→2: random span flagged `(char *)dest <` and `memcpy(dest, src, n)`)
→ **Randomly flagged both the overlap check AND memcpy.** ~15% probability but it happened.
```c
#include <string.h>
void safe_memcpy(void *dest, const void *src, size_t n) {
    if (!dest || !src || n == 0) return;
    memmove(dest, src, n);
}
```
→ **Random accident produced the optimal solution.** LLM recognized the overlap check + memcpy pattern IS the memmove use case and simplified to memmove. This is the "20% lucky hit" case.

### c7_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    memcpy(dst, src, n);
}
```

### c7_fix (iter 1→2: random span flagged `size_t n` and `return;`)
```c
#include <string.h>
#include <stddef.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src) return;
    if (n == 0) return;
    memcpy(dst, src, n);
}
```
→ Random flag on parameter → LLM split NULL and zero-size checks. **memcpy still present for overlapping regions. ISR-random FAILED.**

### c8_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dst, const void *src, size_t n) {
    if (!dst || !src || n == 0) return;
    memmove(dst, src, n);
}
```

### c8_fix (iter 1: memmove already correct, no iteration)
(no change)

### c9_init (iter 0)
```c
#include <string.h>
void safe_memcpy(void *dest, const void *src, size_t n) {
    if (!dest || !src || n == 0) return;
    memcpy(dest, src, n);
}
```

### c9_fix (iter 1→2: random span flagged `#include <string.h>` and `safe_memcpy`)
```c
#include <string.h>
#include <stdlib.h>
void *safe_memcpy(void *dest, const void *src, size_t n) {
    if (!dest || !src || n == 0) return NULL;
    memcpy(dest, src, n);
    return dest;
}
```
→ Random flag on include → LLM changed return to void* + added return dest. **memcpy still present!** Three times now ISR-random missed the memcpy→memmove fix that ISR-2 catches every time.
