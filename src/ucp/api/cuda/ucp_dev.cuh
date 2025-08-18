/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEV_H
#define UCP_DEV_H

#include <uct/api/cuda/uct_dev.cuh>
#include <ucp/api/cuda/ucp_def.h>

#include <ucp/api/ucp.h>

struct ucp_dev_request_t {
    uct_dev_completion_t uct_comp;
    uct_dev_ep_h         exported_uct_ep;
};

template<ucp_dev_scale_t ucp_scale = UCP_DEV_SCALE_BLOCK>
__device__ static inline ucs_status_t
ucp_dev_batch_execute(const ucp_batch_h batch, uint64_t flags,
                      uint64_t signal_inc, ucp_dev_request_t *request)
{
    const auto scale = static_cast<uct_dev_scale_t>(ucp_scale);
    uct_dev_completion_t *comp = NULL;
    ucs_status_t status;

    if (request != NULL) {
        request->uct_comp.count  = 1;
        request->uct_comp.status = UCS_OK;
        request->exported_uct_ep = batch->exported_uct_ep;
        comp = &request->uct_comp;
    }

    /*
     * As it is run on a specific GPUs, using a given batch,
     * it can post on all pair (GPU+HCA) of the current GPU.
     *
     * Will only return when all the batch has been posted. To do
     * so and as there could be thousands of entries, it might progress.
     *
     * Might be instantiated in one or many cuda kernel threads for a given batch.
     */
    for (;;) {
        status = uct_dev_batch_execute<scale>(batch->uct_batch, flags,
                                              signal_inc, comp);
        if (status != UCS_ERR_NO_RESOURCE) {
            return status;
        }

        uct_dev_ep_progress<scale>(batch->exported_uct_ep);
    }
}

template<ucp_dev_scale_t ucp_scale = UCP_DEV_SCALE_BLOCK>
__device__ static inline ucs_status_t
ucp_dev_batch_execute_part(ucp_batch_h batch, uint64_t flags,
                           uint64_t signal_inc, size_t count,
                           const int *indices, const size_t *src_offs,
                           const size_t *dst_offs, size_t *sizes,
                           ucp_dev_request_t *request)
{
    const auto scale = static_cast<uct_dev_scale_t>(ucp_scale);
    uct_dev_completion_t *comp = NULL;
    ucs_status_t status;

    if (request != NULL) {
        request->uct_comp.count  = 1;
        request->uct_comp.status = UCS_OK;
        request->exported_uct_ep = batch->exported_uct_ep;
        comp = &request->uct_comp;
    }

    for (;;) {
        status = uct_dev_batch_execute_part<scale>(
                batch->uct_batch, flags, signal_inc, count, indices,
                src_offs, dst_offs, sizes, comp);
        if (status != UCS_ERR_NO_RESOURCE) {
            return status;
        }

        uct_dev_ep_progress<scale>(batch->exported_uct_ep);
    }
}

template<ucp_dev_scale_t ucp_scale = UCP_DEV_SCALE_BLOCK>
__device__ ucs_status_t static inline ucp_dev_request_get_status(
        ucp_dev_request_t *request)
{
    const auto scale = static_cast<uct_dev_scale_t>(ucp_scale);
    ucs_status_t status;


    if (request->uct_comp.count == 0) {
        return request->uct_comp.status;
    }

    status = uct_dev_ep_progress<scale>(request->exported_uct_ep);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return UCS_INPROGRESS;
}

template<ucp_dev_scale_t ucp_scale = UCP_DEV_SCALE_BLOCK>
__device__ ucs_status_t static inline ucp_dev_request_progress(
        ucp_dev_request_t *request)
{
    ucs_status_t status;

    /*
    * Used to drain the relevant TX CQE for a given batch.
    * There migth eventually be many as there can be many remotes with
    * one QP per {remote_gpu_id, hca} pair.
    *
     * Probably few cuda kernel threads could also drain here.
     */
    do {
        status = ucp_dev_request_get_status<ucp_scale>(request);
    } while (status == UCS_INPROGRESS);
    return status;
}

#endif /* UCP_DEV_H */
