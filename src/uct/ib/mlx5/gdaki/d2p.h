/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_H_
#define UCT_D2P_H_

#include <uct/api/device/uct_device_types.h>

#define UCT_IB_D2P_MAX_CHANNELS 4

typedef struct {
    uct_device_ep_t    super;
    unsigned long long *pi[UCT_IB_D2P_MAX_CHANNELS];
    unsigned long long *ci[UCT_IB_D2P_MAX_CHANNELS];
    void               *queue_base[UCT_IB_D2P_MAX_CHANNELS];
    uint64_t           qp_idx[UCT_IB_D2P_MAX_CHANNELS];
    uint64_t           atomic_result_va[UCT_IB_D2P_MAX_CHANNELS];
    uint32_t           atomic_result_lkey[UCT_IB_D2P_MAX_CHANNELS];
    uint8_t            log_depth;
    uint8_t            channel_mask;
    uint8_t            pad[2];
} uct_ib_d2p_gpu_ep_t;

#endif /* UCT_D2P_H_ */
