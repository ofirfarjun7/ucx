/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_FLAGS_H
#define UCT_DEVICE_FLAGS_H

#include <ucs/sys/compiler_def.h>


/**
 * @brief Specify modifier flags for device sending functions.
 */
typedef enum {
    UCT_DEVICE_FLAG_NODELAY = UCS_BIT(0), /**< Complete before return. */
    UCT_DEVICE_FLAG_LAST    = UCT_DEVICE_FLAG_NODELAY
} uct_device_flags_t;


#endif /* UCT_DEVICE_FLAGS_H */
