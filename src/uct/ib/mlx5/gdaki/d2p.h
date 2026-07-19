/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_H_
#define UCT_D2P_H_

#include <uct/api/device/uct_device_types.h>

typedef struct {
    unsigned long long *pi;
    unsigned long long *ci;
    void               *queue_base;
    uint64_t           qp_idx;
    uint64_t           atomic_result_va;
    uint32_t           atomic_result_lkey;
    uint32_t           pad;
} uct_ib_d2p_channel_t;

typedef struct {
    uct_device_ep_t      super;
    uint8_t              log_depth;
    uint8_t              pad[2];
    uint32_t             num_channels;
    uint32_t             channel_mask;
    uct_ib_d2p_channel_t *channels;
} uct_ib_d2p_gpu_ep_t;

#endif /* UCT_D2P_H_ */
