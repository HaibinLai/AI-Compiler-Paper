
LLVM, TVM
[BBuf/tvm_mlir_learn: compiler learning resources collect.](https://github.com/BBuf/tvm_mlir_learn)

## Project

### TVM

TVM最早是GW那边老师做的一个系统。他们不把这个东西成为AI编译器，认为它包括了更多的东西。这个东西的输入是python，输出是machine code。在中间有优化。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723170618532.png)

Transform your code to Relay (an IR) and try to optimize though TE 

比如
**拆分(tiling),** 比如大矩阵乘拆成小 tile（小方块）在 GPU 上做。
**向量化(vectorization),** 用 SIMD 指令
**并行化(parallelization),**  生成多线程的 parallel-for 循环
**循环展开(unrolling),** 小循环（比如 reduce over 4 个元素）常常直接完全展开
**融合(fusion)**

先 fusion 然后 Tiling → 再 Parallelization → 再 Unrolling → 再 Vectorization


[Quick Start — tvm 0.22.dev0 documentation](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html)
```txt
mod, [param_spec](https://docs.python.org/3/library/stdtypes.html#list "builtins.list") = MLPModel().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod.show()

# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor((1, 784), dtype="float32"), fc1_weight: R.Tensor((256, 784), dtype="float32"), fc1_bias: R.Tensor((256,), dtype="float32"), fc2_weight: R.Tensor((10, 256), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


### XLA

XLA (Accelerated Linear Algebra) is an open-source machine learning (ML) compiler for GPUs, CPUs, and ML accelerators.
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723172849549.png)
[TensorFlow XLA优化原理与示例 - 吴建明wujianming - 博客园](https://www.cnblogs.com/wujianming-110117/p/15333306.html)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173126047.png)



![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173150461.png)

补充，在做XlaOps时，基本上会做这三件事情：

- 算子融合 (Fusion)：把连续的逐元素运算合并为一个核函数，减少内存访问。
- 常量折叠 (Constant Folding)：计算时已知的值直接预计算。
- Layout 优化：调整张量内存布局，使后续 kernel 更高效。

| 优化                        | XLA 能不能做 | 做的层次 / 阶段                                                |
| ------------------------- | -------- | -------------------------------------------------------- |
| **拆分 (Tiling)**           | ✅ 可以     | 在后端 Lowering 和 GPU TPU 后端里做 block / warp / thread tiling |
| **向量化 (Vectorization)**   | ✅ 可以     | CPU 后端：HLO -> LLVM IR，再用 LLVM 的 Loop Vectorizer          |
| **并行化 (Parallelization)** | ✅ 可以     | GPU/TPU：kernel launch，CPU：OpenMP/线程                      |
| **循环展开 (Unrolling)**      | ✅ 可以     | HLO Lowering → LLVM → LLVM loop unroll pass              |

- 高层 → HLO IR 做 fusion、partition
- 低层 → LLVM / GPU backend 做 tiling/unroll/vectorize/parallel

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173927799.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173937743.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173944871.png)


### TensorRT

非常难装
[TensorRT SDK | NVIDIA Developer](https://developer.nvidia.com/tensorrt)

[Writing custom operators with TensorRT Python plugins - NVIDIA TensorRT Standard Python API Documentation 10.13.0](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/pluginGuide.html)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723175117813.png)


## OSDI 20

### RAMMER

上科大神作

[算子调度优化论文分享：Rammer - 知乎](https://zhuanlan.zhihu.com/p/616050345)

以往的算子优化是：算子内优化（如循环展开等等）+算子间优化（fusion）
RAMMER尝试用最小的单元：task来完成

“为了解决以上问题，Rammer通过一个称为rTask的新抽象统一了算子间和算子内的调度。rTask使调度能够打破算子边界，并允许将计算细粒度调度到设备上。RAMMER是一种统一的纯软件解决方案，它不依赖于底层硬件，因此可以被不同的加速器采用，而不是将调度分成由软件和硬件分别管理的两部分。“
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723185317787.png)
**知乎原文：**

重新定义一个问题和定位一个工作并不是在用不同的写法来写“茴”。之前我们只是在做一个广义上的kernel fusion，也没有设立起rTask和vEU的抽象。**而在弄明白本质的问题在于原本系统中两层调度的gap以后，新的抽象很快帮助我们探明了更大的优化空间**：

首先是将原本的通过cost model来选择子图进行fusion的问题，转变为了以更细粒度下的调度和资源分配问题。而得益于绝大部分情况下，神经网络计算的特征（DFG, 算子和张量）在compile time是已知的，我们因此可以将调度的开销移交给编译器，这既提升了搜索的效率也简化了系统设计。

更重要的是，**让inter operator与intra operator parallelism相互影响这个问题走进我们的视野**。举一个具体的例子，如果对于同一个算子有两种kernel实现，其中一个相较另一个消耗三倍的资源（CUDA Cores, Shared Memory, etc.），但是只取得两倍的加速，这在并行计算中是很常见的一个现象。而在此前单个算子独占整个硬件的情况下，毫无疑问我们会选择更快的实现。而我们的实验表明，在inter和intra operator两种parallelism协同调度的情况下，选择资源“性价比”最高的实现而非“最快”往往是更优的选择。这其实挑战了之前许多生成高性能算子的工作如AutoTVM[[9]](https://zhuanlan.zhihu.com/p/275837455#ref_9)等的一个基本假设，单个算子独占整个硬件表现出的计算性能，是否真的是性能调优的金标准？那么显然的，subgraph substitution (TASO[[10]](https://zhuanlan.zhihu.com/p/275837455#ref_10)) + high performance kernel (TVM)两个“optimal”相结合，并没有带来真的optimal。而我们基于新的抽象，只是浅尝一下简单的policy，就在一些场景下获得了超过现有SOTA的性能。


Rammer 的实现用五万两千行 C++ 代码，其中编译模块和[调度函数](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E8%B0%83%E5%BA%A6%E5%87%BD%E6%95%B0&zhida_source=entity)占据3k行代码，它的输入是 TensorFlow frozen graph、TorchScript 或 ONNX 格式的 DNN 模型，Rammer 首先将输入模型转换为 rOperators 的DFG，在 DFG 基础上进行常见的图优化：[常量折叠](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%B8%B8%E9%87%8F%E6%8A%98%E5%8F%A0&zhida_source=entity)、[公共子表达式消除](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%85%AC%E5%85%B1%E5%AD%90%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B6%88%E9%99%A4&zhida_source=entity)等。对于来自优化后的DFG中的每个rOperator，Rammer从不同的源（例如自动内核生成器、手动调整内核，或从其他框架中的现有算子转换而来）加载一个或多个版本的 rKernel 实现。Rammer 编译器然后将DFG划分为子图（例如基于[算法1](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E7%AE%97%E6%B3%951&zhida_source=entity)中的策略），并将每个子图编译为 rProgram。作为输出，每个 rProgram 被进一步生成为在加速器上运行的设备代码（例如GPU内核）。

Rammer 不是动态调度每个rOperator，而是将DFG编译成一个称为 rProgram（由rTask组成）的静态执行计划，并通过一个名为vDevice的软件设备抽象将其映射到硬件。


### Ansor: Generating High-Performance Tensor Programs for Deep Learning

[tvm_mlir_learn/paper_reading/Ansor Generating High-Performance Tensor Programs for Deep Learning.md at main · BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn/blob/main/paper_reading/Ansor%20Generating%20High-Performance%20Tensor%20Programs%20for%20Deep%20Learning.md)

高性能的张量化程序对于保证深度神经网络的高效执行是至关重要的。然而，在各种硬件平台上为不同的算子都获得高效的张量化程序是一件充满挑战的事。目前深度学习系统依赖硬件厂商提供的内核库或者各种搜索策略来获得高性能的张量化程序。这些方法要么需要大量的工程工作来开发特定于平台的优化代码，要么由于搜索空间受限和无效的探索策略而无法找到高性能的程序。

我们提出了Ansor，一个用于深度学习应用的张量化程序生成框架。与现有的搜索策略相比，**Ansor通过从搜索空间的分层表示中采样程序来探索更多的优化组合**。然后Ansor使用进化搜索和一个可学习的代价模型对采样程序进行微调，以确定最佳程序。Ansor可以找到现有SOTA方法的搜索空间之外的高性能程序。此外，Ansor利用scheduler同时优化深度神经网络中的多个子图。实验表明，相对于Intel CPU，ARM CPU，NVIDIA GPU，Ansor分别将神经网络的执行性能提高了3.8倍，2.6倍，和1.7倍。

**Ansor是一个自动的张量化程序生成框架。** Figure 4展示了Ansor的整体架构。Ansor的输入是一组待优化的DNN。Ansor使用Relay[42]的算符融合算法将DNN从流行的模型格式（例如ONNX，TensorFlow PB）转换为小的子图，然后Ansor为这些子图产生张量化程序。Ansor具有三个重要的组件：（1）一个程序采样器，它构建一个大的搜索空间并从中采样不同的程序。（2）微调采样程序性能的性能调试器。（3）一个任务调度器，用于分配时间资源以优化 DNN 中的多个子图。

**Program sampler.** Ansor 必须解决的一个关键挑战是为给定的计算图生成一个大的搜索空间。为了覆盖具有各种high-level结构和low-level细节的各种张量化程序，Ansor利用具有两个级别的搜索空间的分层表示：草图和注释（第 4 节）。 Ansor 将程序的high-level结构定义为草图，并将数十亿个low-level选择（例如，tile size、parallel、unroll annotations）作为注释。 这种表示允许Ansor灵活地枚举high-level结构并有效地采样low-level细节。Ansor包含一个程序采样器，它从空间中随机采样程序以提供对搜索空间的全面覆盖。

**Performance tuner.** 随机采样程序的性能不一定好。下一个挑战是对它进行微调、Ansor使用进化搜索和一个可学习的代价模型迭代式微调（第5节）。在每次迭代中，Ansor使用重新采样的新程序以及来自先前迭代的性能还不错的程序作为初始种群执行进化搜索。进化搜索通过变异和交叉对程序进行微调，执行乱序重写并解决顺序构造的局限性。 查询学习到的代价模型比实际测试快几个数量级，因此我们可以在几秒钟内评估数千个程序。

**Task scheduler.** 使用程序采样器和性能调试器允许 Ansor 为计算图找到高性能的张量化程序。 直观地说，将整个DNN视为单个计算图并为其生成完整的张量化程序可能会实现最佳性能。 然而，这是低效的，因为它必须处理搜索空间不必要的指数爆炸。 通常，编译器将DNN的大型计算图划分为几个小的子图 [11, 42]。由于 DNN 的逐层构造特性，该划分对性能的影响可以忽略不计。这带来了Ansor最终的挑战：在为多个子图生成程序时如何分配时间资源。 Ansor中的任务调度器（第 6 节）使用基于梯度下降的调度算法将资源分配给更可能提高端到端DNN性能的子图。


Ansor 的主要贡献是做到了自动寻找高效的Schedule（循环展开、合并、分块、缓存使用、改变并行度等等），**不再需要开发者在[TVM](https://zhida.zhihu.com/search?content_id=209483109&content_type=Article&match_order=1&q=TVM&zhida_source=entity)中基于Tensor Expression手写Schedule模板，大大增强了算子编译器（Tensor Compiler）的易用性并且对一些典型的算子和模型效果也很不错，算是AutoTVM的升级版（因为AutoTVM还需要手动指定需要搜索的Schedule模板**

这里可以看到，Ansor是在尝试不同的优化组合，它是个程序生成框架。然后分成了好几个section。这和之前的TVM有点类似？但是它靠划分，来加快生成速度。如果RAMMER说的是对的话，那么这样其实是不好的。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111503338.png)



## MICRO 20

### Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data

这篇文章主要还是在用多面体，然后利用了memory hierarchy。进一步利用芯片上的高速缓存，从而提升程序在CPU、GPU以及**AI芯片华为昇腾910等不同芯片上的性能**。

Polyhedral
[53年来国内唯三！清华校友获芯片顶会最佳论文提名，华为昇腾芯片性能显著提升 - 知乎](https://zhuanlan.zhihu.com/p/268210680)

在Polyhedral模型中研究如何实现一种更好的[循环分块](https://zhida.zhihu.com/search?content_id=147733024&content_type=Article&match_order=2&q=%E5%BE%AA%E7%8E%AF%E5%88%86%E5%9D%97&zhida_source=entity)和合并新组合，来优化存储进而提升程序在不同芯片上的执行效率。  
**该团队是国内少有的从事Polyhedral模型研究的团队之一。**

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190335662.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190349228.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190404733.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190416708.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190430955.png)



## PLDI 21

### AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations

把TVM的那套迁移到NPU上。

AKG

开发一个面向NPU架构的编译器仍然面临着诸多挑战，因为NPU上通常需要更复杂融合策略设计，以期充分利用NPU上的快速存储部件。编译器需要解决不同程序对并行性和局部性的需求不同的挑战，并找到最优的schedule结果。另一方面，领域特定芯片的内存层级都是多层次、多方向设计的，编译器需要解决如何在软件层面上实现自动内存管理的挑战。解决面向领域特定硬件，自动生成运算密集型算子的挑战，例如卷积，矩阵乘等。

[PLDI 2021: AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://www.pldi21.org/poster_pldi.760.html)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724110827069.png)


## LISA 21
the talk

更多的CPU Core 的用处没有那么多了。从学术上讲通信成本，锁，复杂度都很难受，商业上GPU薄纱它

[Computing Performance: On the Horizon | USENIX](https://www.usenix.org/conference/lisa21/presentation/gregg-computing)
![02c5d1bc0bc068dd957f79a8dd682658.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111035742.png)

![330b1bd1ed36bdc73cc3d05abf856d6e.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111024201.png)

## OSDI 21

### PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections

[tvm_mlir_learn/paper_reading/PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections.md at main · BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn/blob/main/paper_reading/PET%3A%20Optimizing%20Tensor%20Programs%20with%20Partially%20Equivalent%20Transformations%20and%20Automated%20Corrections.md)

标题可以翻译为：**基于部分等价变换和自动校正来优化张量化程序**。作者团队来自清华，CMU和FaceBook等。这篇论文的一作王豪杰来自清华大学。后面会介绍到这篇论文在生成突变程序集合时，要维护K个效率最高的突变程序时，使用了ASO中的代价模型和评估方式，所以作者有贾志豪大神也不奇怪。

现有的框架在图层做优化一般都是基于等价变换，也就时说变换前后的程序是完全等价的。这里等价的意思是给定相同的输入，那么变换前后的程序一定可以得到相同的输出。而这篇论文挖了一个新坑，**即做了一个新的框架PET，在优化过程中允许出现部分等价的变换**，并且设计了一个高效的搜索算法去组合完全等价以及部分等价的变换以探索更大的搜索空间。并且最终结果也比较好。

PET不关心算子的Schedule，而是从部分等价变换的新角度出发去增加并行度或者改善缓存从而达到加速效果

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724112025354.png)


## OSDI 22

- [NNFusion](https://github.com/microsoft/nnfusion): Rammer (OSDI'20), Roller (OSDI'22), Welder (OSDI'23), Cocktailer (OSDI'23), T10 (SOSP'24)

2022 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/522950146/answer/2404943067




### ROLLER: Fast and Efficient Tensor Compilation for Deep Learning

作者团队在TVM上实现并开源了roller

- 论文链接：[https://www.usenix.org/conference/osdi22/presentation/zhu](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/osdi22/presentation/zhu)
- 代码链接：[https://github.com/microsoft/nn](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nnfusion/)

无论是Ansor，AutoTVM还是PET（一部分代码生成也是基于TVM AutoTVM/Ansor的）它们都面临了同样一个问题，那就是在对算子的Schedule进行搜索时需要耗费大量的时间，在特定硬件上对一个常见的视觉模型进行自动调优和生成代码kennel需要数小时。这严重阻碍了AI编译器应用于模型部署。基于这个痛点，Roller横空出世。

现代的张量编译器虽然取得了很多的进展，但通常这些编译器都需要小时计的时间去搜索和生成高效的Kernel，这是因为现有张量编译器通常指定的搜索空间很大。为了解决编译时间长的问题，本文提出了Roller，它的核心是rTile，这是一种新的Tile抽象，它封装了和底层加速器的关键特性一致的张量shape，从而通过限制shape的选择实现高效的运行。**Roller采用了基于rTile的递归构建算法来生成目标程序（rProgram）。最终，Roller可以在几秒内就生产高效的Kernel**，性能可以媲美目前主流加速器上的其它张量编译器，并且为IPU等新的加速器生产更好的Kernel。

## SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute

针对稀疏的神经网络，生成特定的优化算子。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724112724077.png)


- 本篇文章主要探索深度神经网络的**稀疏性**，借助名为TeSA的新抽象为模型提供端到端稀疏性模式，并根据不同的稀疏性模式，产生高效的专用的operator，从而为稀疏的DNN模型提供更小的内存占用和更低的推理延迟。



### Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences

GPU调度的一篇文章

[hanosdi22.pdf](https://ipads.se.sjtu.edu.cn/_media/publications/hanosdi22.pdf)


### Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

生成LLM并行计划的一篇文章

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113043827.png)


### Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization

CMU的嘉志豪作品

试图进行联合优化

Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。

Unity 基于先前工作，定义了 DNN 并行系统中常见的六类基本形式：

数据并行（Data Parallelism）：最基本的并行形式；
模型并行（Model Parallelism）：将 DNN 模型拆分为多个子模型，分别在特定的设备训练；
空间并行（Spatial Parallelism）：对 tensor 的空间维度（如图像的高/宽）进行划分；
规约并行（Reduction Parallelism）：利用 OP 的线性。将 tensor 划分为 n 份到 n 个设备上，分别进行 OP 操作。注意每个结果 tensor 的 shape 都和最终 tensor 相同，因此只需相加进行规约；
管道并行（Pipeline Parallelism）：不同训练代数之间的并行；
指定 OP 的并行（Operator-specific Parallelsim）：对 batch 内不同的输入 sample 有不同的 weight，不需要 tensor 备份或同步；
许多并行策略是不同 cost 指标的 trade-off，因此需要不同 OP 用不同的并行方式，以达到最优性能。DNN 结构表示的传统方法是计算图，节点是 OP，边是 tensor，代数变换被表示为迭代的图替代，通过给每个节点分配并行注释来并行化模型。然而，该方法存在如下局限：

**对代数变换和并行化使用独立表示阻碍了共优化。代数变换会增删节点，但并行化需要静态计算图；**
**计算图并没有显式考虑并行带来的通信开销。这导致代数变换难以预测最终模型的性能**

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113239546.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113303099.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113319680.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113359004.png)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113415932.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113430925.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113505064.png)

[Microsoft PowerPoint - OSDI 22 - Animations Expanded.pptx](https://www.usenix.org/system/files/osdi22_slides_unger.pdf)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113540522.png)

## MLSys22


###  Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization
Apollow; Mind IR

[【MLSys 2022】论文介绍：图算融合Apollo:-基于多层规约的自动算子融合优化框架 - 知乎](https://zhuanlan.zhihu.com/p/511821386)

- **Memory stitching（内存缝合）**：在融合 phase 的 Layer II 中，把原本因为 reduction 等导致无法直接 loop fusion 的 micro-graphs，通过共享局部内存（比如 GPU 的 shared memory 或寄存器）连接起来。这允许跨 micro-graphs 的进一步融合。
- **Parallelism stitching（并行缝合）**：在融合 phase 的 Layer III 中，发掘**相互独立算子或分支之间的并行性**，把它们打包到同一个 kernel 中并行执行，最大化硬件利用率。
    

> 简单来说：
> - stitching 在这里是**扩展融合空间的一种手段**，目的是让更多算子能一起执行，不仅仅依赖传统的生产者-消费者关系，还考虑到内存层次结构和硬件并行度。
> - stitching 是对传统 loop fusion 的补充：loop fusion 注重单一路径上的数据局部性，而 stitching 注重跨路径的融合与并行。
>
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724114007756.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724114024797.png)



![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724125028517.png)


### Hydrozoa: Dynamic Hybrid-Parallel DNN Training on Serverless Containers

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724124742488.png)


##  OSDI 23

2023 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/591516372/answer/2953251460

Welder

### EINNET: Optimizing Tensor Programs with Derivation-Based Transformations

研究工作中的张量程序优化方法可以分为两类：一类是从操作符级别优化，通过搜索执行计划生成高性能内核（如TVM和AnSOR），并应用专家设计的程序变换；另一类是通过图级别变换重新组织DNN计算（TASO和PET），采用superoptimization的方法来发现图变换。现有的张量程序优化器仅使用预定义操作符变换来优化张量程序，探索了有限的性能优化机会。

在本文中，作者提出了一种探索一般张量代数变换的方法，其中节点是一般张量操作符的变换。与基于预定义操作符变换相比，一般张量代数变换构成了一个更大的优化空间。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724125840294.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724125900498.png)


### WELDER: Scheduling Deep Learning Memory Access via Tile-graph

研究员们发现当前大部分 DNN 计算的瓶颈主要在于 GPU 的访存，如这些模型对[内存带宽](https://zhida.zhihu.com/search?content_id=231010118&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E5%B8%A6%E5%AE%BD&zhida_source=entity)利用率高达96.7%，但计算核的平均利用率只有51.6%，而且随着硬件与 DNN 模型的不断发展，这两者之间的差距还会持续增大。尤其是当前的人工智能模型需要处理高保真度的数据，如更大的图像、更长的句子、更高清的图形，这些数据在计算中都占用了更多的内存带宽。同时，更高效的专有计算核（如TensorCore）也进一步加大了内存压力。

为了解决内存问题，研究员们提出了 **Welder 深度学习编译器，全面优化由通用算子组成的端到端 DNN 模型的内存访问效率。**其实，DNN 模型可以看作是由多个算子连成的一张图，整个计算过程涉及多个阶段，即数据需要流过不同的算子，在每个阶段都需要将张量切分成块，先搬运到处理器上进行计算，然后再搬运回内存，这就会造成很大的搬运开销。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724130248808.png)


### Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning

尝试在share mem上搞事情。生成更好的控制流。

基于同一套抽象和统一的中间[表示层](https://zhida.zhihu.com/search?content_id=231010118&content_type=Article&match_order=1&q=%E8%A1%A8%E7%A4%BA%E5%B1%82&zhida_source=entity)（Intermediate Representation，IR），**这四款 AI 编译器解决了当前 AI 编译器中的不同问题——并行、编译效率、内存、控制流，构成了一套完整的编译解决方案。**

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724130105889.png)


### Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724130603411.png)



## NIPS 23


### ASPEN: Breaking Operator Barriers for Efficient Parallel Execution of Deep Neural Networks
ASPEN

这篇工作提出了一个叫做 opportunistic parallelism 的概念。其基于以下的观察：大部分的 DNN 加速库在计算矩阵乘法时，会将大矩阵拆分为小矩阵（tile）来对齐寄存器大小提高计算效率。拆分之后就有更多的并行机会了，如下图（b）（d）所示，左右两侧的张量仅依赖一个父节点，不用等 operator2 计算完成 operator3 的部分就可以同时计算。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724131027580.png)


1. **移除 operator 屏障**：把 DNN 表示为跨 operator 的 **细粒度 tile 依赖图**（tile-based dataflow graph），让跨 operator 的 tile 可以并行执行。
2. **分布式调度**：为每个并行资源（线程/核）分配一个 **分布式调度引擎 (DSE)**，动态遍历和执行 tile 依赖图，遇到就绪的 tile 立刻执行，而不需要全局同步。
3. **Ready Pool 并发队列**：并行资源之间通过 Ready Pool 实现依赖信息交换，避免传统同步屏障带来的等待。

这个做法让 DNN 执行不再严格按层，而是任意就绪 tile 都可以优先执行，实现了所谓的 **opportunistic parallelism（机会并行**

这里的拆分是指拆分张量（和张量并行类似，但是文章是基于多核CPU去做的，paper中也只讨论了CPU计算，这种模式感觉和张量并行是相同的，不太确定张量并行是否有这样的工作模式）

## OSDI 24

### USHER: Holistic Interference Avoidance for Resource Optimized ML Inference
USHER

本文提出了 ML Inference Serving 系统 USHER，其核心 Insight 是使用 Holistic 的方式**最大化 GPU 资源（算力、显存）利用率，同时最小化模型的干扰**。为此，USHER 提出了三个技术：1）一个估计模型显存占用和算力需求的方法；2）一个综合优化全局GPU内存和算力的调度器（决定不同模型复制多少份，分配到哪个GPU，使用多大batch size）；3）一个将不同模型的参数相同算子合并到一起，从而降低GPU cache干扰的方法。

相比现有 Inference Serving 系统（[Shepherd](https://zhida.zhihu.com/search?content_id=245473614&content_type=Article&match_order=1&q=Shepherd&zhida_source=entity)[NSDI'23]、[GPUlet](https://zhida.zhihu.com/search?content_id=245473614&content_type=Article&match_order=1&q=GPUlet&zhida_source=entity)[ATC'22]、[AlpaServe](https://zhida.zhihu.com/search?content_id=245473614&content_type=Article&match_order=1&q=AlpaServe&zhida_source=entity)[OSDI'23]），USHER 可以提高 2.6 倍推理服务的 Goodput。

作者进一步探究了，如何进行 workload division 能够提高性能。作者测试了两个方案，给定一个 workload，其中每个模型的 batch size 为 BS，以及最少的 replication 数量为 RD。

**方案1（independent workload division）**： 每个模型都分成 2*RD 份（每份的 batch size 为 BS/2），然后每一个份都随机选一个 GPU （资源足够）执行。
**方案2（holistic workload division）**：枚举所有可能的划分方式（每个模型可能划分 [RD, 2RD] 份，batch size 可能设置为 [BS, BS/2]），随机分配到资源充足的GPU，并且最终找一个使用GPU最少的方案。 上图给出了这个两个方案的测试结果：

1. Independent workload division 可以只使用 11 个 GPU 达到之前 12 个 GPU 的吞吐
2. Holistic workload divition 可以进一步将 GPU 数量降低到 10 个

综上，作者总结了以下观察：

**观察1： 现有系统无法最大化 Cuti 与 Muti，并且 multiplexing 产生的干扰会显著降低 goodput。**
**观察2： 在空间并行的推理系统里，即使一个GPU足够满足一个Model的资源需求，将这个Model划分到多个GPU可以提高资源利用率。**
**观察3：将 Model 划分到多个 GPU 的策略应该是 Holistic 的，而不是独立考虑每个 Model。**


### MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures
MonoNN

在现代GPU架构上，传统的逐内核执行方案未能充分利用硬件资源，导致显著的非计算开销和内存访问瓶颈。本工作提出了MonoNN，这是一个创新的**机器学习编译优化框架**，它将整个神经网络编译成**单个GPU内核（kernel）**，并通过一系列优化技术，如上下文感知的指令重排、资源权衡策略和调度解耦的组调优技术，以提高硬件资源的利用率和整体推理性能。

本工作认为，解决上述问题的主要挑战如下：

1. **计算密集型算子和内存密集型算子的最优资源使用策略是冲突的**：计算密集算子要求最大化片上资源的使用；然而内存密集型算子要求并行度尽可能高（TLP），二者不可兼得；
2. **优化空间巨大**：将整个神经网络编译成单个内核后，编译器的优化空间非常庞大，导致搜索最优策略的时间难以接受。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724131926819.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724132018673.png)

Q1：MonoNN的代码生成是如何实现的，是否使用了例如TVM这样的工具？
A1：我们使用了自己搭建的代码生成器，我们会根据SLA采用一些基本的计算图优化，并且在生产中根据反馈持续优化代码。

Q2：MonoNN为什么表现比CUDA Graph更好？
A2：这个问题有个两个方面原因，首先是MonoNN会尝试优化源代码，提高内核本身的执行效率；其次，MonoNN最终编译出的单内核和正常的GPU内核一样，没有像CUDA Graph一样引入额外约束。

### Enabling Tensor Language Model to Assist in Generating High-Performance Tensor Programs for Deep Learning
Enabling Tensor Language Model

**Tensor 编译器架构**如图1所示。 Tensor编译器首先有一个计算图处理引擎将一个DL程序 (i.e., 模型) 抽象为计算图 (一种 IR) 并进行计算图优化 (如算子融合) 得到一些子图。然后使用tensor exploration框架低层化 (lowering) IR并进行优化。最后利用代码生成器 (如 [LLVM](https://zhida.zhihu.com/search?content_id=245465565&content_type=Article&match_order=1&q=LLVM&zhida_source=entity) 或 [NVCC](https://zhida.zhihu.com/search?content_id=245465565&content_type=Article&match_order=1&q=NVCC&zhida_source=entity)) 针对特定硬件平台生成后端代码。 底层化及优化是编译优化中最关键的步骤。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724132223071.png)


![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724132735053.png)



### Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation
LADDER

以 NVIDIA GPU 上 FP16 张量与 NF4 张量的矩阵乘法为例，由于硬件支持的限制，需要将 NF4 类型转换为 FP16 类型。此转换应在从 L1 到 L0 的 Transform-Load 之前完成，因此有两种安排方式：

1. **在 L2 中进行转换**
- 优点：后续从 L2 → L1 → L0 的 tTile 移动中**不占用计算单元**  
- 缺点：会在 **L2 和 L1 上占用更多内存**

**2. 在L1 上进行转换**
- 优点：**节省 L2 的内存** 并**减少 L2 的内存带宽使用**  
- 缺点：会**占用计算单元**进行类型转换  
    ——分析：
- 如果操作**受计算单元限制**，前一种选择可实现更低的延迟但占用更多内存；  
- 如果操作**受内存 I/O 限制**，后一种选择在延迟和内存占用方面都能实现更好的性能。

转成什么格式

优先选择硬件支持的**位数最接近**的 tType。因为位数更多的数值类型通常需要更多的晶体管来实现硬件指令[]，并且通常会导致性能降低。例如，在 [NVIDIA A100](https://zhida.zhihu.com/search?content_id=257233663&content_type=Article&match_order=1&q=NVIDIA+A100&zhida_source=entity) GPU 中，NF4 类型可以转换为 FP16 或 FP32 进行处理，而 LADDER 会选择 FP16 核心（312 TFLOPS）而不是 FP32（19.5 TFLOPS）。（这个选择比较简单。。）



![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724132703588.png)




### nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training

随着深度神经网络（DNN）模型规模的不断扩大，深度学习训练越来越依赖于手工设计的搜索空间，以找到高效的并行策略。然而，现有的成熟的方法在面对越来越复杂和多样化的模型架构时，常常忽略了一些潜在的优化机会，显得过于僵化，无法充分探索所有可能的并行化方案。

例如，在张量并行（Tensor Parallelism）中，现有系统一般会将一个operation的计算与数据强制划分到不同的设备上。然而，这种方式忽略了将分割后的算子放在同一设备上的可能性。在特定情况下，这种放置方式可以通过更细粒度的流水线计算减少来内存峰值占用和设备间的通信成本。

又如，在流水线并行（Pipeline Parallelism）中，现有的并行方案无法直接应用到如AlphaFold2这类3次正向转播对应1次反向转播的特殊模型。同时，现有方案还假设不同的流水线阶段只能放在不同的设备上，不允许同一设备通过分时复用来执行不同的流水线阶段。在遇到不同阶段的计算与内存需求差异较大的情况时（如大模型的嵌入表，Embedding Table），这往往会导致较低的GPU内存、计算利用率。

 设计与实现

为了解决上述问题，作者希望能提出一套抽象来更为灵活地表达更加完备的并行计划搜索空间。然而，更大的搜索空间意味着指数级增长的求解时间，这也是为何现有工作往往会手工设计合理的搜索空间。作者的解决方式是，在抽象中允许领域专家手工来提供并行策略的约束条件，从而限制搜索空间的大小。基于此思路，作者提出了nnScaler。nnScaler并非提出了新的求解算法，而是一套自动并行框架，能让领域专家灵活地设计新的求解空间，同时方便地复用现有的并行策略求解算法。

1. **操作变换**（op-trans）：将操作 op 根据变换算法 algo 分割成 n 个子op。algo可以是一般的划分（用于实现DP、TP），也可以是重计算或offloading等自定义算法。其中的op可以是一个单独的算子，也可以是一个子图。
2. **操作分配**（op-assign）：将op分配到设备d上执行。
3. **操作顺序**（op-order）：在op1和op2之间建立顺序依赖关系

通过这些原语，领域专家可以编写代码，构建任意 DNN 模型的并行化计划搜索空间。同时，模型代码与搜索空间代码可以分离开来。利用这套原语，数据流图中的每个算子和张量如何划分，以及在空间（设备间）和时间（同一设备）上如何调度，都能够得到准确的描述和解释。
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724133156533.png)



### ChameleonAPI: Automatic and Efficient Customization of Neural Networks for ML Applications

本文的工作希望能够自动化生成一个定制的模型，能够利用在特定任务下的某些输出误差不影响最终的结果，从而提升应用输出的准确率。**简而言之，就是对模型做fine tuning，训个定制化的模型**。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724133321939.png)




## ASPLOS 24

### Amanda: Unified Instrumentation Framework for Deep Neural Networks
（上交 Jingwen Leng、过敏意团队）
Amanda：深度神经网络的统一工具框架

Paper: https://dl.acm.org/doi/10.1145/3617232.3624864
Code: ~
Abstract: 深度神经网络（DNN）的成功激发了分析（如追踪（tracing））和优化（如剪枝（pruning））它们的努力。这些任务有特定的要求，在当前的执行后端（如 TensorFlow/PyTorch）中需要临时实现，这就要求开发人员管理零散的接口，并根据不同的模型调整代码。本文提出了一个名为 Amanda 的新框架，以简化这些任务的开发。本文将这些任务的实现形式化为神经网络工具（neural network instrumentation），这涉及将工具化引入 DNN 的算子级。这使本文能够将 DNN 分析和优化任务抽象为各种 DNN 模型的工具。本文用两级应用程序接口构建 Amanda，以实现统一、可扩展和高效的工具设计。用户级 API 为不同的后端提供了统一的算子粒度工具 API。同时，本文在内部设计了一套以回调为中心的应用程序接口，用于管理和优化不同后端中原始代码和工具代码的执行。通过这些设计原则，Amanda 框架可以在不同的后端（如 TensorFlow/PyTorch）和执行模式（图/eager 模式）中适应广泛的用例，如追踪、剖析（profiling）、剪枝和量化。此外，本文高效的执行管理确保了性能开销通常保持在 5%以内。


### Souffle | Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions（中科院计算所 赵家程、冯晓兵团队）
通过全局分析和张量表达优化深度学习推理

Paper: https://dl.acm.org/doi/10.1145/3617232.3624858
Code: ~
Abstract: 优化深度神经网络（DNN）的执行非常重要，但随着 DNN 复杂性的增加，优化变得越来越困难。现有的 DNN 编译器无法有效利用跨算子边界的优化机会，因此还有改进的余地。为了应对这一挑战，本文推出了一款开源编译器——Souffle，它可以跨算子边界优化 DNN 推理。Souffle 使用张量表达式创建全局张量依赖图，跟踪数据流和张量信息，并根据数据流分析和资源限制将计算图划分为子程序。在子程序中，Souffle 通过语义保留转换（semantic-preserving transformations）执行局部优化，找到优化的程序进度表，并提高指令级并行性和数据重用性。本文在英伟达 A100 GPU 上使用六个具有代表性的 DNN 模型对 Souffle 进行了评估。实验结果表明，与 TensorRT 和 Tensorflow XLA 相比，Souffle 的几何平均速度分别提高了 3.7 倍和 7.8 倍，始终优于六种最先进的 DNN 优化器。


### SoD2: Statically Optimizing Dynamic Deep Neural Network Execution（威廉与玛丽学院 Bin Ren团队，佐治亚大学 Gagan Agrawal团队）
SoD2：静态优化动态深度神经网络执行

Paper: https://dl.acm.org/doi/10.1145/3617232.3624869
Code: ~
Abstract: 尽管近年来为 DNN 开发了许多编译和运行系统，但重点主要放在静态 DNN 上。在动态 DNN 中，张量的形状和大小，甚至所使用的算子集都取决于输入和/或执行，这种情况正变得越来越普遍。本文介绍了用于优化动态 DNN 的综合框架 SoD2。本文方法的基础是对构成 DNN 的常见算子进行分类，并将这种分类用于等级和维度传播 (Rank and Dimension Propagation, RDP) 方法。该框架将算子的形状静态地确定为已知常量、符号常量或对这些常量的运算。接下来，本文利用 RDP 实现了一系列优化，如融合代码生成、执行（顺序）规划，甚至运行时内存分配计划生成。通过在 10 个新出现的动态 DNN 上对该框架进行评估，并将其与几个现有系统进行比较，本文证明了执行延迟和内存需求的减少，而 RDP 所支持的关键优化则是大部分收益的来源。本文的评估结果表明，SoD2 的运行速度比这些系统快 3.9 倍，同时节省了高达 88% 的峰值内存消耗。


### Tandem Processor: Grappling with Emerging Operators in Neural Networks


然而，人们很少关注非 GEMM 算子，它们反而被忽视了。随着深度学习的发展和进步，这些操作也变得越来越多样化，同时还出现了大量将它们与 GEMM 算子交织在一起的结构模式。然而，传统的 NPU 设计采取了相当简单的方法，要么通过一些专用块来支持这些操作，要么退回到通用处理器。这项工作旨在挑战神经加速器设计中的传统智慧，并探索一种被称为串联处理器（Tandem Processor）的片上辅助处理器的架构，以补充神经加速器中相当优化的 GEMM 单元

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724133648524.png)


### Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization

这篇文章针对动态形状神经网络，提出了一种离线生成小内核、在线高效聚合的编译方法 MikPoly，在 GPU 和 NPU 上都能取得明显性能提升，解决了动态场景下编译开销高、适应性差的问题。
这篇文章做了以下几件主要的事情（总结自文章开头和整体内容）：

✅ **研究背景与问题**  
随着动态形状（dynamic-shape）神经网络的广泛应用，传统静态形状（static-shape）张量编译器难以应对运行时形状变化带来的优化需求；而现有的动态形状编译器也往往需要预先定义形状范围，并且离线搜索成本较高，不适用于频繁变化的场景。

---

✅ **核心方法：提出 MikPoly**  
作者提出了 **MikPoly**：一种基于 **micro-kernel polymerization（微内核聚合）** 的动态形状张量编译器，通过两阶段（离线+在线）优化流程解决动态形状神经网络加速问题：

- **离线阶段**：从一个两阶段程序模板中生成针对固定形状的、高度优化的 micro-kernels（小型 tiled 内核），并基于硬件（如 GPU Tensor Core、NPU）为这些内核建立性能模型。
- **在线阶段**：在运行时根据已知形状，通过聚合离线生成的 micro-kernels，并结合轻量级代价模型，快速组合出适合当前形状的优化张量程序，实现高效执行。


## OSDI 25
### Nanoflow
NanoFlow: Towards Optimal Large Language Model Serving Throughput

这篇文章首先抛出了一个惊人的结论：
Through a detailed analysis, we show that despite having memory-intensive components, **end-to-end LLM serving is compute bound for most common workloads and LLMs**.

Alas, most existing serving engines fall short from optimal compute utilization, because the heterogeneous operations that comprise LLM serving--compute, memory, networking--**are executed sequentially within a device**.

We propose NanoFlow, a novel serving framework that exploits intra-device parallelism, which overlaps the usage of heterogeneous resources within a single device. NanoFlow splits inputs into smaller nano-batches and duplicates operations to operate on each portion independently, enabling overlapping. NanoFlow automatically identifies the number, size, ordering, and GPU resource allocation of nano-batches to minimize the execution time, while considering the interference of concurrent operations. We evaluate NanoFlow's end-to-end serving throughput on several popular models such as LLaMA-2-70B, Mixtral 8x7B, LLaMA-3-8B, etc. With practical workloads, NanoFlow provides 1.91x throughput boost compared to state-of-the-art serving systems achieving 50% to 72% of optimal throughput across popular models.

#### Introduction

LLM需要很多的资源。
LLMs are more resource-intensive than earlier DNNs due to unique characteristics.

在本文里，我们会详细研究为什么是compute bound的。Profile 理由有3：
In this paper, we show that while the self-attention opera tion is indeed memory-bound, **LLM serving is often compute bound when considered as a whole**. We provide a detailed analysis (which we also validate empirically) of the common operations used in current LLMs and find that: (1) **Batching prefill and decode requests amortizes weight loading, making general matrix multiplications (GEMMs) compute-bound.**(2) **Although batched decode request still need to load a unique KV-cache per sequence, novel optimizations such as grouped query attention (GQA) can reduce the memory loads.** (3) **As model sizes grow, the compute operations from GEMMs tend to dominate compared to self-attention.** Consequently, for many common workloads and LLMs, the total compute operations dominate network and memory I/O.

As model sizes grow, the compute operations from GEMMs tend to dominate compared to self-attention. 呃？self-attention 难道没有GEMMs吗？

However, in practice, we find that LLM serving engines are far from optimal in terms of compute utilization when measured end-to-end and compared to the hardware FLOPs. 为什么？This is because LLMs use heterogeneous operations.

So how many heterogeneous operations? Here it draws a figure:
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723154208635.png)

To address this gap, we present NanoFlow, a serving frame work that aims to maximize utilization of the workload’s resource bottleneck, when considered as a whole. **The key insight is to leverage intra-device parallelism for heteroge neous operations**. In particular, **NanoFlow splits each input batch into nano-batches and duplicates operations across nano-operations,with each nano-operation processing a single nano-batch.**

Since nano-operations operate on separate nano batches without dependencies, heterogeneous operations— **such as memory-bound and compute-bound operations—can execute simultaneously rather than sequentially**. This ap proach facilitates fine-grained pipelining across compute, memory, and network resources within a single device. As nano-operations duplicate each operation, this method in creases memory I/O for weight loading. However, when the workload is compute-bound as a whole, the additional mem ory I/O can be hidden via pipelining.

这里的计算重叠是尝试将机器的计算、内存、网络三者都压榨到极致。

### Mirage

[mirage-project/mirage: Mirage: Automatically Generating Fast GPU Kernels without Programming in Triton/CUDA](https://github.com/mirage-project/mirage)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724134855997.png)

| 瓶颈                                | Mirage 怎么解决                                                                                       | 对应哪里                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **内存墙** （device memory 访问太多）      | 自动发现 kernel fusion、新算法、block 层 fusion，让中间结果保存在 shared memory / register file，而不是反复写 device memory | µGraph 的 block / thread 层设计，fusion |
| **并行墙** （核数多，但单 kernel 并行度低）      | 自动搜索最优的 grid / block / thread 划分，比如并行哪个维度、用多少 block、大 block 还是小 block                             | µGraph Generator + Optimizer       |
| **需要新 kernel**（现有 kernel 不支持复杂融合） | 自动发现并生成全新 kernel，而不是手工写                                                                           | µGraph 生成新 kernel，验证保证正确           |

### Pipeline Threader

Tensor + SM 同时运行。

PipeThreader: Software-Defined Pipelining for Efficient DNN Execution
[OSDI 2025 论文评述 Day 3 Session 9: AI + Systems III - 知乎](https://zhuanlan.zhihu.com/p/1926740033395749827)
[PipeThreader: Software-Defined Pipelining for Efficient DNN Execution - About](https://hamerlate.github.io/publications/old/PipeThreader/)

[osdi25-cheng.pdf](https://www.usenix.org/system/files/osdi25-cheng.pdf)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724134133783.png)

[CS PhD Statements of Purpose](https://cs-sop.notion.site/CS-PhD-Statements-of-Purpose-df39955313834889b7ac5411c37b958d?p=2b8470ffc59c48bcbe09ee2a3fae01df&pm=s)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724134133785.png)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724134133786.png)


![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724134133787.png)

### QiMeng X Compiler


### Bayesian Code Diffusion



### Neutrino: Fine-grained GPU Kernel Profiling via Programmable Probing

TL;DR: 我们能否像[eBPF](https://zhida.zhihu.com/search?content_id=732884758&content_type=Answer&match_order=1&q=eBPF&zhida_source=entity)观测Linux Kernel一样观测GPU Kernel？如果能，会发现什么有意思的事情？

我们通过Assembly Level Probing实现了一个eBPF-like programming interface，允许用户在GPU Kernel内部（例如特定的指令或者调用）插入probe来获取运行时信息，以及提供了eBPF-like Map来便捷和有结构的的保存获得到的信息以供分析我们实现了一个高可用(Artifact Available, Functional, Reproducible)的在运行时插入probe的系统，核心是一个Hook Driver来在运行时捕获GPU代码和提供相应支持（比如allocate map memory）和一个Probe Engine来负责根据定义在GPU代码中以汇编代码形式插入Probe和Map我们拓展了经典的Page Reference Map/String，引入了physical time来对齐不同thread之间的访存，并且将不同thread之间并行访存的intensity表达为色深，提出了Densified Memory Access Timeline (DMAT)，希望能帮助大家Understanding GPU Kernel, Visually.

作者：积雪照明月
链接：https://www.zhihu.com/question/15731788637/answer/1918830576229684045
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

### Principle and methodology for serial performnace optimization

