# AI Compiler Paper


## OpenSource Project

TVM

XLA

## OSDI 20

RAMMER

## MICRO 20

Polyhedral

## PLDI 21
AKG

## LISA 21
the talk

## OSDI 22

2022 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/522950146/answer/2404943067

## MLSys22

Apollow; Mind IR

##  OSDI 23

2023 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/591516372/answer/2953251460

Welder

## OSDI 24

USHER
MonoNN
Enabling Tensor Language Model
LADDER
MScalar
Chameleon API

## NIPS 24

ASPEN

## ASPLOS 24

On the fly micro kernel

## OSDI 25
### Nanoflow
NanoFlow: Towards Optimal Large Language Model Serving Throughput

这篇文章首先抛出了一个惊人的结论：
Through a detailed analysis, we show that despite having memory-intensive components, **end-to-end LLM serving is compute bound for most common workloads and LLMs**.

Alas, most existing serving engines fall short from optimal compute utilization, because the heterogeneous operations that comprise LLM serving--compute, memory, networking--**are executed sequentially within a device**.

We propose NanoFlow, a novel serving framework that exploits intra-device parallelism, which overlaps the usage of heterogeneous resources within a single device. NanoFlow splits inputs into smaller nano-batches and duplicates operations to operate on each portion independently, enabling overlapping. NanoFlow automatically identifies the number, size, ordering, and GPU resource allocation of nano-batches to minimize the execution time, while considering the interference of concurrent operations. We evaluate NanoFlow's end-to-end serving throughput on several popular models such as LLaMA-2-70B, Mixtral 8x7B, LLaMA-3-8B, etc. With practical workloads, NanoFlow provides 1.91x throughput boost compared to state-of-the-art serving systems achieving 50% to 72% of optimal throughput across popular models.

### Mirage
Pipeline Threader
QiMeng X Compiler
Bayesian Code Diffusion
Neutrim
Principle and methodology for serial performnace optimization

