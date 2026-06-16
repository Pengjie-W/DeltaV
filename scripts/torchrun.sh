#!/usr/bin/bash
echo torchrun --nproc_per_node $PROC_PER_NODE --master_addr $MASTER_ADDR --master_port ${MASTER_PORT:-5678} --nnodes $NODE_COUNT --node_rank $NODE_RANK -- ${@:1}
torchrun --nproc_per_node $PROC_PER_NODE --master_addr $MASTER_ADDR --master_port ${MASTER_PORT:-5678} --nnodes $NODE_COUNT --node_rank $NODE_RANK -- ${@:1}
