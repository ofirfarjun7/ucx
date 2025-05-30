/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "string.h"
#include "math.h"
#include "sys.h"
#include <ucs/config/parser.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/math.h>
#include <ucs/debug/log.h>

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <libgen.h>


const char *ucs_memunits_suffixes[] = {"", "K", "M", "G", "T", "P", "E", NULL};


void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max)
{
    char *p, *end;
    const char *pf, *pp;
    size_t length;
    time_t t;

    p = buf;
    end = buf + max - 1;
    *end = 0;
    pf = tmpl;
    while (*pf != 0 && p < end) {
        pp = strchr(pf, '%');
        if (pp == NULL) {
            strncpy(p, pf, end - p);
            p = end;
            break;
        }

        length = ucs_min(pp - pf, end - p);
        strncpy(p, pf, length);
        p += length;
        /* default length of the modifier (e.g. %p) */
        length = 2;

        switch (*(pp + 1)) {
        case 'p':
            snprintf(p, end - p, "%d", getpid());
            break;
        case 'h':
            snprintf(p, end - p, "%s", ucs_get_host_name());
            break;
        case 'c':
            snprintf(p, end - p, "%02d", ucs_get_first_cpu());
            break;
        case 't':
            t = time(NULL);
            strftime(p, end - p, "%Y-%m-%d-%H-%M-%S", localtime(&t));
            break;
        case 'u':
            snprintf(p, end - p, "%s", ucs_basename(ucs_get_user_name()));
            break;
        case 'e':
            snprintf(p, end - p, "%s", ucs_basename(ucs_get_exe()));
            break;
        case 'i':
            snprintf(p, end - p, "%u", geteuid());
            break;
        default:
            *(p++) = *pp;
            length = 1;
            break;
        }

        pf = pp + length;
        p += strlen(p);
    }
    *p = 0;
}

void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    memset(buf, 0, size);
    va_start(ap, fmt);
    vsnprintf(buf, size, fmt, ap);
    va_end(ap);
}

void ucs_strncpy_zero(char *dest, const char *src, size_t max)
{
    if (max) {
        strncpy(dest, src, max - 1);
        dest[max - 1] = '\0';
    }
}

uint64_t ucs_string_to_id(const char* str)
{
    uint64_t id = 0;
    strncpy((char*)&id, str, sizeof(id) - 1); /* Last character will be \0 */
    return id;
}

size_t ucs_string_quantity_prefix_value(char prefix)
{
    switch (prefix) {
    case 'B':
        return 1;
    case 'K':
        return UCS_KBYTE;
    case 'M':
        return UCS_MBYTE;
    case 'G':
        return UCS_GBYTE;
    case 'T':
        return UCS_TBYTE;
    default:
        return 0;
    }
}

char *ucs_memunits_to_str(size_t value, char *buf, size_t max)
{
    const char **suffix;

    if (value == UCS_MEMUNITS_INF) {
        ucs_strncpy_safe(buf, UCS_NUMERIC_INF_STR, max);
    } else if (value == UCS_MEMUNITS_AUTO) {
        ucs_strncpy_safe(buf, UCS_VALUE_AUTO_STR, max);
    } else {
        suffix = &ucs_memunits_suffixes[0];
        while ((value >= 1024) && ((value % 1024) == 0) && *(suffix + 1)) {
            value /= 1024;
            ++suffix;
        }
        ucs_snprintf_safe(buf, max, "%zu%s", value, *suffix);
    }
    return buf;
}

const char *ucs_memunits_range_str(size_t range_start, size_t range_end,
                                   char *buf, size_t max)
{
    char buf_start[64], buf_end[64];

    if (range_start == range_end) {
        snprintf(buf, max, "%s",
                 ucs_memunits_to_str(range_start, buf_start,
                                     sizeof(buf_start)));
    } else {
        snprintf(buf, max, "%s..%s",
                 ucs_memunits_to_str(range_start, buf_start, sizeof(buf_start)),
                 ucs_memunits_to_str(range_end, buf_end, sizeof(buf_end)));
    }

    return buf;
}

ucs_status_t ucs_str_to_memunits(const char *buf, void *dest)
{
    char units[3];
    int num_fields;
    size_t value;
    size_t bytes;

    /* Special value: infinity */
    if (!strcasecmp(buf, UCS_NUMERIC_INF_STR)) {
        *(size_t*)dest = UCS_MEMUNITS_INF;
        return UCS_OK;
    }

    /* Special value: auto */
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *(size_t*)dest = UCS_MEMUNITS_AUTO;
        return UCS_OK;
    }

    memset(units, 0, sizeof(units));
    num_fields = sscanf(buf, "%ld%c%c", &value, &units[0], &units[1]);
    if (num_fields == 1) {
        bytes = 1;
    } else if ((num_fields == 2) || (num_fields == 3)) {
        bytes = ucs_string_quantity_prefix_value(toupper(units[0]));
        if (!bytes || ((num_fields == 3) && tolower(units[1]) != 'b')) {
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    *(size_t*)dest = value * bytes;
    return UCS_OK;
}

char *ucs_dirname(char *path, int num_layers)
{
    while (num_layers-- > 0) {
        path = dirname(path);
        if (path == NULL) {
            return NULL;
        }
    }
    return path;
}

void ucs_vsnprintf_safe(char *buf, size_t size, const char *fmt, va_list ap)
{
    if (size == 0) {
        return;
    }

    vsnprintf(buf, size, fmt, ap);
    buf[size - 1] = '\0';
}

void ucs_snprintf_safe(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    ucs_vsnprintf_safe(buf, size, fmt, ap);
    va_end(ap);
}

char* ucs_strncpy_safe(char *dst, const char *src, size_t len)
{
    size_t length;

    if (!len) {
        return dst;
    }

    /* copy string into dst including null terminator */
    length = ucs_min(len, strnlen(src, len) + 1);

    memcpy(dst, src, length);
    dst[length - 1] = '\0';
    return dst;
}

char *ucs_strtrim(char *str)
{
    char *start, *end;

    /* point 'p' at first non-space character */
    start = str;
    while (isspace(*start)) {
        ++start;
    }

    if (*start) {
        /* write '\0' after the last non-space character */
        end = start + strlen(start) - 1;
        while (isspace(*end)) {
            --end;
        }
        *(end + 1) = '\0';
    }

    return start;
}

const char * ucs_str_dump_hex(const void* data, size_t length, char *buf,
                              size_t max, size_t per_line)
{
    static const char hexchars[] = "0123456789abcdef";
    char *p, *endp;
    uint8_t value;
    size_t i;

    p     = buf;
    endp  = buf + max - 2;
    i     = 0;
    while ((p < endp) && (i < length)) {
        if (i > 0) {
            if ((i % per_line) == 0) {
                *(p++) = '\n';
            } else if ((i % 4) == 0) {
                *(p++) = ':';
            }

            if (p == endp) {
                break;
            }
        }

        value = *(const uint8_t*)(UCS_PTR_BYTE_OFFSET(data, i));
        p[0]  = hexchars[value / 16];
        p[1]  = hexchars[value % 16];
        p    += 2;
        ++i;
    }
    *p = 0;
    return buf;
}

const char* ucs_flags_str(char *buf, size_t max,
                          uint64_t flags, const char **str_table)
{
    size_t i, len = 0;

    for (i = 0; *str_table; ++str_table, ++i) {
        if (flags & UCS_BIT(i)) { /* not using ucs_for_each_bit to silence coverity */
            snprintf(buf + len, max - len, "%s,", *str_table);
            len = strlen(buf);
        }
    }

    if (len > 0) {
        buf[len - 1] = '\0'; /* remove last ',' */
    } else {
        buf[0] = '\0';
    }

    return buf;
}

size_t ucs_string_count_char(const char *str, char c)
{
    size_t count = 0;
    const char *p;

    for (p = str; *p != '\0'; ++p) {
        if (*p == c) {
            count++;
        }
    }

    return count;
}

size_t ucs_string_common_prefix_len(const char *str1, const char *str2)
{
    const char *p1 = str1;
    const char *p2 = str2;

    /* as long as *p1==*p2, if *p1 is not '\0' then neither is *p2 */
    while ((*p1 != '\0') && (*p1 == *p2)) {
        p1++;
        p2++;
    }

    return (p1 - str1);
}

static size_t
ucs_path_common_parent_length(const char *path1, const char *path2)
{
    size_t offset, parent_length;

    offset        = 0;
    parent_length = 0;
    do {
        /* A path component ends by either a '/' or a '\0' */
        if (((path1[offset] == '/') || (path1[offset] == '\0')) &&
            ((path2[offset] == '/') || (path2[offset] == '\0'))) {
            parent_length = offset;
        }
    } while ((path1[offset] == path2[offset]) && (path1[offset++] != '\0'));

    return parent_length;
}

void ucs_path_get_common_parent(const char *path1, const char *path2,
                                char *common_path)
{
    size_t parent_length;

    parent_length = ucs_path_common_parent_length(path1, path2);
    memcpy(common_path, path1, parent_length);
    common_path[parent_length] = '\0';
}

size_t ucs_path_calc_distance(const char *path1, const char *path2)
{
    size_t common_length = ucs_path_common_parent_length(path1, path2);

    return ucs_string_count_char(path1 + common_length, '/') +
           ucs_string_count_char(path2 + common_length, '/');
}

const char* ucs_mask_str(uint64_t mask, ucs_string_buffer_t *strb)
{
    uint8_t bit;

    if (mask == 0) {
        ucs_string_buffer_appendf(strb, "<none>");
        goto out;
    }

    ucs_for_each_bit(bit, mask) {
        ucs_string_buffer_appendf(strb, "%u, ", bit);
    }

    ucs_string_buffer_rtrim(strb, ", ");

out:
    return ucs_string_buffer_cstr(strb);
}

ssize_t ucs_string_find_in_list(const char *str, const char **string_list,
                                int case_sensitive)
{
    size_t i;

    for (i = 0; string_list[i] != NULL; ++i) {
        if ((case_sensitive && (strcmp(string_list[i], str) == 0)) ||
            (!case_sensitive && (strcasecmp(string_list[i], str) == 0))) {
            return i;
        }
    }

    return -1;
}

char* ucs_string_split(char *str, const char *delim, int count, ...)
{
    char *p = str;
    size_t length;
    va_list ap;
    int i;

    va_start(ap, count);
    for (i = 0; i < count; ++i) {
        *va_arg(ap, char**) = p;
        if (p == NULL) {
            continue;
        }

        length = strcspn(p, delim);
        if (p[length] == '\0') {
            /* 'p' is last element, so point to NULL from now on */
            p = NULL;
        } else {
            /* There is another element after 'p', so point 'p' to it */
            p[length] = '\0';
            p        += length + 1;
        }
    }
    va_end(ap);

    return p;
}

ucs_status_t ucs_string_alloc_path_buffer(char **buffer_p, const char *name)
{
    char *temp_buffer = ucs_malloc(PATH_MAX, name);

    if (temp_buffer == NULL) {
        ucs_error("failed to allocate memory for %s", name);
        return UCS_ERR_NO_MEMORY;
    }

    *buffer_p = temp_buffer;
    return UCS_OK;
}

ucs_status_t ucs_string_alloc_formatted_path(char **buffer_p, const char *name,
                                             const char *fmt, ...)
{
    char *temp_buffer = NULL;
    va_list ap;
    ucs_status_t status;

    status = ucs_string_alloc_path_buffer(&temp_buffer, name);
    if (status != UCS_OK) {
        return status;
    }

    va_start(ap, fmt);
    ucs_vsnprintf_safe(temp_buffer, PATH_MAX, fmt, ap);
    va_end(ap);

    *buffer_p = temp_buffer;
    return UCS_OK;
}

ucs_status_t ucs_string_alloc_path_buffer_and_get_dirname(char **buffer_p,
                                                          const char *name,
                                                          const char *path,
                                                          const char **dir_p)
{
    ucs_status_t status;
    char *buffer;

    status = ucs_string_alloc_path_buffer(buffer_p, name);
    if (status != UCS_OK) {
        return status;
    }

    buffer = *buffer_p;
    ucs_strncpy_safe(buffer, path, PATH_MAX);
    *dir_p = dirname(buffer);
    return UCS_OK;
}
