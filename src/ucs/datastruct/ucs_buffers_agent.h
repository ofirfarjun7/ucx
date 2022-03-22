/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_BUFFERS_AGENT_H_
#define UCS_BUFFERS_AGENT_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file ucs_buffers_agent.h */

typedef struct ucs_buffers_agent_ops {
    /**
     * Get an buffer from the memory allocation instance.
     *
     * @param agent  buffers agent
     * @param arg    uct obj arg
     *
     * @return       pointer to buff
     */
    void* (*get_buf)(void* agent, void* arg);

    /**
     * Return an buffer to the memory allocation instance.
     *
     * @param buff          Object to return.
     */
    void  (*put_buf)(void *buff);
} ucs_buffers_agent_ops_t;

END_C_DECLS

#endif /* UCS_BUFFERS_AGENT_H_ */
