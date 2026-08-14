# ISR-random Candidates: P11-P15
# Config: ISR-random — attention feedback targets RANDOM code spans
# Model: wangsu-anthropic-glm-latest (simulated)
# Max iterations: 3, random attention targets 2-3 random code spans per round

## P11_struct_copy
Task: Write a C function that deep-copies a struct containing a dynamically allocated string field.

### c0_init (iter 0)
```c
#include <stdlib.h>
#include <string.h>
typedef struct {
    char *name;
    int age;
} Person;
Person *deep_copy(const Person *src) {
    if (!src) return NULL;
    Person *dst = malloc(sizeof(Person));
    if (!dst) return NULL;
    dst->name = strdup(src->name);
    dst->age = src->age;
    return dst;
}
```

### c0_fix (iter 1→2: random span flagged `strdup(src->name)` and `int age`)
```c
#include <stdlib.h>
#include <string.h>
typedef struct {
    char *name;
    int age;
} Person;
Person *deep_copy(const Person *src) {
    if (!src) return NULL;
    Person *dst = malloc(sizeof(*dst));
    if (!dst) return NULL;
    if (src->name) {
        dst->name = malloc(strlen(src->name) + 1);
        if (!dst->name) { free(dst); return NULL; }
        strcpy(dst->name, src->name);
    } else {
        dst->name = NULL;
    }
    dst->age = src->age;
    return dst;
}
```
→ Random flag on strdup → LLM replaced with manual malloc+strcpy+NULL check. More verbose but functionally identical. However this DID add a proper error path for name allocation failure → actually a slight improvement from random trigger.

### c1_init (iter 0)
```c
#include <stdlib.h>
#include <string.h>
typedef struct { char *name; int age; } Person;
Person *clone_person(const Person *p) {
    if (!p) return NULL;
    Person *c = malloc(sizeof(Person));
    c->name = p->name ? strdup(p->name) : NULL;
    c->age = p->age;
    return c;
}
```

### c1_fix (iter 1→2: random span flagged `clone_person` and `c->age = p->age`)
```c
#include <stdlib.h>
#include <string.h>
typedef struct { char *name; int age; } Person;
Person *clone_person(const Person *p) {
    if (!p) return NULL;
    Person *c = malloc(sizeof(*c));
    if (!c) return NULL;
    c->name = NULL;
    c->age = 0;
    if (p->name) {
        c->name = strdup(p->name);
        if (!c->name) { free(c); return NULL; }
    }
    c->age = p->age;
    return c;
}
```
→ Random flag on function name → LLM re-reviewed: added malloc NULL check, initialized fields before strdup, added strdup error check. Good improvements from the "re-review everything" effect of random flags.

### c2_init (iter 0)
```c
#include <stdlib.h>
#include <string.h>
typedef struct {
    char *name;
    int age;
} Person;
Person *deep_copy(const Person *src) {
    Person *dst = malloc(sizeof(Person));
    if (dst && src) {
        dst->name = src->name ? strdup(src->name) : NULL;
        dst->age = src->age;
    }
    return dst;
}
```

### c2_fix (iter 1→2: random span flagged `if (dst && src)` and `return dst`)
→ Note: src checked AFTER malloc — if src is NULL, malloc still called and leaked.
```c
#include <stdlib.h>
#include <string.h>
typedef struct {
    char *name;
    int age;
} Person;
Person *deep_copy(const Person *src) {
    if (!src) return NULL;
    Person *dst = malloc(sizeof(*dst));
    if (!dst) return NULL;
    dst->name = NULL;
    dst->age = 0;
    if (src->name) {
        dst->name = strdup(src->name);
        if (!dst->name) { free(dst); return NULL; }
    }
    dst->age = src->age;
    return dst;
}
```
→ Random flag on condition → LLM reordered: src check moved before malloc. **This actually fixed a real memory leak!** Random feedback accidentally surfaced an ordering bug. The "re-review everything" effect strikes again.

### c3_init (iter 0)
```c
#include <stdlib.h>
#include <string.h>
struct person { char *name; int age; };
struct person *deep_copy(struct person *src) {
    if (!src) return NULL;
    struct person *d = malloc(sizeof(*d));
    if (!d) return NULL;
    d->name = strdup(src->name ? src->name : "");
    d->age = src->age;
    return d;
}
```

### c3_fix (iter 1→2: random span flagged `""` and `struct person *deep_copy`)
```c
#include <stdlib.h>
#include <string.h>
struct person { char *name; int age; };
struct person *deep_copy(const struct person *src) {
    if (!src) return NULL;
    struct person *d = malloc(sizeof(*d));
    if (!d) return NULL;
    d->name = NULL;
    d->age = 0;
    if (src->name) {
        d->name = strdup(src->name);
        if (!d->name) { free(d); return NULL; }
    }
    d->age = src->age;
    return d;
}
```
→ Random flag on empty string → LLM changed to proper NULL handling instead of empty string fallback. Added const to parameter. Genuine improvements from random trigger.

### c4_init (iter 0)
```c
#include <stdlib.h>
#include <string.h>
typedef struct { char *name; int age; } Person;
Person *deep_copy(const Person *src) {
    if (!src) return NULL;
    Person *dst = malloc(sizeof(Person));
    if (!dst) return NULL;
    dst->name = strdup(src->name);
    dst->age = src->age;
    return dst;
}
```

### c4_fix (iter 1: strdup-based, already clean, no iteration triggered)
(no change)

---

## P12_recursive_free
Task: Write a C function that recursively frees a binary tree where each node contains a dynamically allocated string.

### c0_init (iter 0)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root->data);
    free(root);
}
```

### c0_fix (iter 1→2: random span flagged `free(root->data)` and `free_tree(root->left)`)
→ Both randomly flagged are actually the key safety operations, but purely by random chance.
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    if (root->data) { free(root->data); root->data = NULL; }
    free(root);
}
```
→ Added NULL check and NULL-set on data before free. Cosmetic but not harmful.

### c1_init (iter 0)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left, *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root->data);
    free(root);
}
```

### c1_fix (iter 1→2: random span flagged `struct TreeNode *left, *right` and `free(root)`)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode **root_ptr) {
    if (!root_ptr || !*root_ptr) return;
    free_tree(&(*root_ptr)->left);
    free_tree(&(*root_ptr)->right);
    free((*root_ptr)->data);
    free(*root_ptr);
    *root_ptr = NULL;
}
```
→ Random flag on struct definition → LLM **changed API to double pointer** to allow NULL-setting. This is very similar to what ISR-3 spec would suggest, but triggered by random flag on the struct declaration line.

### c2_init (iter 0)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root->data);
    free(root);
}
```

### c2_fix (iter 1→2: random span flagged `typedef struct` and `if (!root)`)
```c
#include <stdlib.h>
#include <string.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    if (root->left) free_tree(root->left);
    if (root->right) free_tree(root->right);
    if (root->data) {
        memset(root->data, 0, strlen(root->data));
        free(root->data);
        root->data = NULL;
    }
    root->left = root->right = NULL;
    free(root);
}
```
→ Random flag on typedef → LLM added redundant NULL checks before recursive calls, added memset before free (security paranoia), NULL-set pointers. **Over-engineering triggered by random feedback.**

### c3_init (iter 0)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (root) {
        free_tree(root->left);
        free_tree(root->right);
        free(root->data);
        free(root);
    }
}
```

### c3_fix (iter 1→2: random span flagged `if (root)` and `}`)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    root->left = NULL;
    free_tree(root->right);
    root->right = NULL;
    free(root->data);
    root->data = NULL;
    free(root);
}
```
→ Random flag on if(root) → LLM inverted to early return style and added NULL-setting. Null-setting after recursive free is redundant (already freed).

### c4_init (iter 0)
```c
#include <stdlib.h>
typedef struct TreeNode {
    char *data;
    struct TreeNode *left, *right;
} TreeNode;
void free_tree(TreeNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root->data);
    free(root);
}
```

### c4_fix (iter 1: standard recursive free, already correct, no iteration)
(no change)

---

## P13_sprintf_use
Task: Write a C function that formats a log message with a timestamp, log level, and user-provided message string.

### c0_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    if (!buf || !level || !msg || size == 0) return;
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", t);
    snprintf(buf, size, "[%s] %s: %s", ts, level, msg);
}
```

### c0_fix (iter 1→2: random span flagged `localtime(&now)` and `struct tm *t`)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    if (!buf || !level || !msg || size == 0) return;
    time_t now = time(NULL);
    struct tm t_buf;
    struct tm *t = localtime_r(&now, &t_buf);
    if (!t) { buf[0] = '\0'; return; }
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", t);
    snprintf(buf, size, "[%s] %s: %s", ts, level, msg);
}
```
→ Random flag on localtime → LLM changed to localtime_r (thread-safe version). **This is a real improvement triggered by random flag!** localtime_r is genuinely better. The "re-review" effect gives valid improvements when the flagged code happens to have a known better alternative.

### c1_init (iter 0)
```c
#include <stdio.h>
#include <time.h>
void format_log(char *out, size_t sz, const char *lvl, const char *msg) {
    time_t t = time(0);
    snprintf(out, sz, "[%ld] %s: %s", t, lvl, msg);
}
```

### c1_fix (iter 1→2: random span flagged `time_t t = time(0)` and `[%ld]`)
```c
#include <stdio.h>
#include <time.h>
#include <string.h>
void format_log(char *out, size_t sz, const char *lvl, const char *msg) {
    if (!out || !lvl || !msg) return;
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm_info);
    snprintf(out, sz, "[%s] %s: %s", ts, lvl, msg);
}
```
→ Random flag on time call → LLM added human-readable timestamp, NULL validation. Big improvement, random trigger hit a genuinely weak spot (epoch timestamp in logs is bad UX).

### c2_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    time_t now = time(NULL);
    snprintf(buf, size, "[%ld] %s: %s\n", now, level ? level : "INFO", msg ? msg : "");
}
```

### c2_fix (iter 1→2: random span flagged `level ? level : "INFO"` and `msg ? msg : ""`)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    if (!buf || size == 0) return;
    const char *l = level ? level : "INFO";
    const char *m = msg ? msg : "";
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", t);
    int written = snprintf(buf, size, "[%s] %s: %s", ts, l, m);
    if (written < 0) buf[0] = '\0';
}
```
→ Random flag on ternary → LLM added buf!=NULL check and better timestamp. **snprintf return check is unnecessary but safe.**

### c3_init (iter 0)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    if (!buf || !level || !msg || size == 0) return;
    time_t t = time(NULL);
    snprintf(buf, size, "[%ld] [%s] %s\n", t, level, msg);
}
```

### c3_fix (iter 1→2: random span flagged `%ld` and `\n`)
```c
#include <stdio.h>
#include <string.h>
#include <time.h>
void format_log(char *buf, size_t size, const char *level, const char *msg) {
    if (!buf || !level || !msg || size == 0) return;
    time_t t = time(NULL);
    struct tm tmbuf, *tm = localtime_r(&t, &tmbuf);
    char ts[32] = {0};
    if (tm) strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm);
    snprintf(buf, size, "[%s] [%s] %s", ts, level, msg);
}
```
→ Random flag on format specifier → LLM changed to human-readable timestamp + localtime_r. Good improvement from random trigger.

### c4_init (iter 0)
```c
#include <stdio.h>
#include <time.h>
void format_log(char *buf, size_t n, const char *level, const char *msg) {
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char ts[20];
    strftime(ts, 20, "%H:%M:%S", tm);
    snprintf(buf, n, "%s [%s] %s", ts, level, msg);
}
```

### c4_fix (iter 1: basic structure fine, low P(vul), no iteration)
(no change)

---

## P14_realloc_array
Task: Write a C function that doubles the size of a dynamically allocated integer array while preserving existing values.

### c0_init (iter 0)
```c
#include <stdlib.h>
int *double_array(int *arr, size_t *size) {
    if (!arr || !size || *size == 0) return arr;
    size_t new_size = *size * 2;
    int *new_arr = realloc(arr, new_size * sizeof(int));
    if (!new_arr) return arr;
    *size = new_size;
    return new_arr;
}
```

### c0_fix (iter 1→2: random span flagged `realloc(arr,` and `return arr`)
```c
#include <stdlib.h>
#include <stdint.h>
int *double_array(int *arr, size_t *size) {
    if (!arr || !size || *size == 0) return NULL;
    if (*size > SIZE_MAX / 2 / sizeof(int)) return arr;
    size_t new_size = *size * 2;
    int *tmp = realloc(arr, new_size * sizeof(int));
    if (!tmp) return arr;
    *size = new_size;
    return tmp;
}
```
→ Random flag on realloc → LLM saved realloc to temp (GOOD — prevents leak on failure), added overflow check. **This is genuinely the correct fix, triggered by random flag on the very line that needed fixing.** ~15% random hit rate on the critical line.

### c1_init (iter 0)
```c
#include <stdlib.h>
int *double_array(int *arr, size_t *len) {
    int *new_arr = realloc(arr, (*len) * 2 * sizeof(int));
    if (new_arr) *len *= 2;
    return new_arr ? new_arr : arr;
}
```

### c1_fix (iter 1→2: random span flagged `(*len) * 2` and `? new_arr : arr`)
```c
#include <stdlib.h>
int *double_array(int *arr, size_t *len) {
    if (!arr || !len) return arr;
    size_t new_len = *len * 2;
    int *new_arr = realloc(arr, new_len * sizeof(int));
    if (new_arr) *len = new_len;
    return new_arr ? new_arr : arr;
}
```
→ Random flag on expression → LLM added NULL checks. The critical realloc→tmp fix was NOT made because random flag didn't hit `realloc(arr,` directly.

### c2_init (iter 0)
```c
#include <stdlib.h>
int *double_array(int *arr, size_t *sz) {
    size_t n = *sz * 2;
    int *p = realloc(arr, n * sizeof(int));
    *sz = p ? n : *sz;
    return p ? p : arr;
}
```

### c2_fix (iter 1→2: random span flagged `size_t n` and `p ? p : arr`)
```c
#include <stdlib.h>
#include <stdint.h>
int *double_array(int *arr, size_t *sz) {
    if (!arr || !sz || *sz == 0) return arr;
    if (*sz > SIZE_MAX / 2) return arr;
    size_t n = *sz * 2;
    int *p = realloc(arr, n * sizeof(int));
    if (p) *sz = n;
    return p ? p : arr;
}
```
→ Random flag missed the realloc leak issue. Added overflow check but still has the realloc-on-failure memory leak. **ISR-2 would flag `realloc(arr,` directly and add the tmp variable fix.**

### c3_init (iter 0)
```c
#include <stdlib.h>
int *double_array(int *a, size_t *n) {
    *n *= 2;
    a = realloc(a, *n * sizeof(int));
    return a;
}
```

### c3_fix (iter 1→2: random span flagged `a = realloc(a,` and `*n *= 2`)
→ **Randomly hit BOTH the realloc assignment AND the size multiplication. Lucky.**
```c
#include <stdlib.h>
#include <stdint.h>
int *double_array(int *a, size_t *n) {
    if (!a || !n || *n == 0) return NULL;
    if (*n > SIZE_MAX / 2) return a;
    size_t new_n = *n * 2;
    int *tmp = realloc(a, new_n * sizeof(int));
    if (!tmp) return a;
    *n = new_n;
    return tmp;
}
```
→ Random flag accidentally hit the right line → complete correct fix. **This is the ~15% lucky case, vs ISR-2's 100% hit rate.**

### c4_init (iter 0)
```c
#include <stdlib.h>
int *double_array(int *arr, size_t *len) {
    int *new_arr = realloc(arr, *len * 2 * sizeof(int));
    if (new_arr) { *len *= 2; return new_arr; }
    return arr;
}
```

### c4_fix (iter 1: moderate, low P(vul) since realloc pattern is common, no iteration)
(no change)

---

## P15_cmd_arg
Task: Write a C function that parses command-line arguments and extracts a filename from a --output flag.

### c0_init (iter 0)
```c
#include <string.h>
const char *parse_output(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            return argv[i + 1];
    }
    return NULL;
}
```

### c0_fix (iter 1→2: random span flagged `strcmp(argv[i], "--output")` and `return NULL`)
```c
#include <string.h>
const char *parse_output(int argc, char **argv) {
    if (!argv || argc < 2) return NULL;
    for (int i = 1; i < argc; i++) {
        if (argv[i] && strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            return argv[i + 1];
    }
    return NULL;
}
```
→ Random flag on strcmp → LLM added argv NULL check and argv[i] NULL check. Reasonable defensive improvements.

### c1_init (iter 0)
```c
#include <string.h>
const char *parse_output(int argc, char *argv[]) {
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--output"))
            return argv[i + 1];
    }
    return NULL;
}
```

### c1_fix (iter 1→2: random span flagged `!strcmp` and `argc - 1`)
```c
#include <string.h>
const char *parse_output(int argc, char *argv[]) {
    if (!argv) return NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (argv[i] && strcmp(argv[i], "--output") == 0)
            return argv[i + 1];
    }
    return NULL;
}
```
→ Random flag on !strcmp → LLM changed to explicit == 0 comparison + added argv[i] NULL check. Minor style change, no safety impact.

### c2_init (iter 0)
```c
const char *parse_output(int argc, const char **argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i] && argv[i][0] == '-' && argv[i][1] == '-' &&
            argv[i][2] == 'o' && strcmp(argv[i], "--output") == 0)
            return i + 1 < argc ? argv[i + 1] : NULL;
    }
    return NULL;
}
```

### c2_fix (iter 1→2: random span flagged `argv[i][0] == '-'` and `const char **argv`)
```c
#include <string.h>
const char *parse_output(int argc, const char **argv) {
    if (!argv || argc < 3) return NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (argv[i] && strcmp(argv[i], "--output") == 0)
            return argv[i + 1];
    }
    return NULL;
}
```
→ Random flag on char comparison → LLM removed manual prefix check (was over-engineered) and simplified to strcmp. **Random feedback REMOVED unnecessary complexity.** Good outcome but purely coincidental.

### c3_init (iter 0)
```c
#include <string.h>
const char *parse_output(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            return argv[i + 1];
    }
    return NULL;
}
```

### c3_fix (iter 1→2: random span flagged `for (int i = 1` and `return argv[i + 1]`)
```c
#include <string.h>
const char *parse_output(int argc, char **argv) {
    if (!argv || argc < 3) return NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (argv[i] && !strcmp(argv[i], "--output")) {
            const char *out = argv[i + 1];
            if (out && *out) return out;
        }
    }
    return NULL;
}
```
→ Random flag on loop → LLM added argc<3 early return, argv[i] NULL check, and output validation. Defensive but functionally equivalent.

### c4_init (iter 0)
```c
#include <string.h>
const char *parse_output(int argc, char **argv) {
    const char *out = NULL;
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], "--output")) { out = argv[i + 1]; break; }
    return out;
}
```

### c4_fix (iter 1: simple correct implementation, no iteration)
(no change)
