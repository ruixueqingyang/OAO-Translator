/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: FuncInfo.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190506
  Description: 实现 一些通用的函数
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1. main: 主函数
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#include "FuncInfo.h"

using namespace clang;

// wfr 20191225 从 std::vector<NODE_INDEX> 中 删除所有 NODE_INDEX
void deleteNodeIndex(std::vector<NODE_INDEX> &vectorNodeIndex, NODE_INDEX indexNode){
    for (unsigned int i = 0; i < vectorNodeIndex.size(); ++i) {
        if (vectorNodeIndex[i] == indexNode) {
            vectorNodeIndex.erase(vectorNodeIndex.begin()+i);
            --i;
        }
    }
}

// wfr 20191225 删除节点指向自身的边
void deleteEdge2Self(NODE_INDEX indexSEQ, std::vector<SEQ_REGION> &vectorSEQ){
    SEQ_REGION& SEQ = vectorSEQ[indexSEQ.index];
    deleteNodeIndex(SEQ.vectorParents, indexSEQ);
    deleteNodeIndex(SEQ.vectorChildren, indexSEQ);
}
void deleteEdge2Self(NODE_INDEX indexOMP, std::vector<OMP_REGION> &vectorOMP){
    OMP_REGION& OMP = vectorOMP[indexOMP.index];
    deleteNodeIndex(OMP.vectorParents, indexOMP);
    deleteNodeIndex(OMP.vectorChildren, indexOMP);
}

// wfr 20191225 合并节点, 将 src 节点 合并到 dest 节点, 只将 src 节点空间重置 并不释放
void merge(NODE_INDEX indexDest, NODE_INDEX indexSrc, std::vector<SEQ_REGION> &vectorSEQ, std::vector<OMP_REGION> &vectorOMP){

    SEQ_PAR_NODE_BASE* pSrc = NULL;
    SEQ_PAR_NODE_BASE* pDest = NULL;
    if (indexSrc.type == NODE_TYPE::SEQUENTIAL) {
        pSrc = (SEQ_PAR_NODE_BASE*)(&vectorSEQ[indexSrc.index]);
        deleteEdge2Self(indexSrc, vectorSEQ); // 删除 指向自身的边
    } else if (indexSrc.type == NODE_TYPE::PARALLEL) {
        pSrc = (SEQ_PAR_NODE_BASE*)(&vectorOMP[indexSrc.index]);
        deleteEdge2Self(indexSrc, vectorOMP); // 删除 指向自身的边
    } else {
        std::cout << "merge 错误: indexSrc 类型是: " << indexSrc.type << std::endl;
    }

    if (indexDest.type == NODE_TYPE::SEQUENTIAL) {
        pDest = (SEQ_PAR_NODE_BASE*)(&vectorSEQ[indexDest.index]);
        deleteEdge2Self(indexDest, vectorSEQ); // 删除 指向自身的边
    } else if (indexDest.type == NODE_TYPE::PARALLEL) {
        pDest = (SEQ_PAR_NODE_BASE*)(&vectorOMP[indexDest.index]);
        deleteEdge2Self(indexDest, vectorOMP); // 删除 指向自身的边
    } else {
        std::cout << "merge 错误: indexDest 类型是: " << indexDest.type << std::endl;
    }
    
    // wfr 201912225 为了避免 Dest 节点 出现指向自身的边, 删除 Src 和 Dest 之间的边
    // 从 Src 中 删除 Dest
    deleteNodeIndex(pSrc->vectorParents, indexDest);
    deleteNodeIndex(pSrc->vectorChildren, indexDest);
    // 从 Dest 中 删除 Src
    deleteNodeIndex(pDest->vectorParents, indexSrc);
    deleteNodeIndex(pDest->vectorChildren, indexSrc);
    
    // 将 Src 节点的父节点列表 插入 Dest 节点的父节点列表, 排除重复
    // 将 Src 节点的父节点们 链接到 Dest 节点, 排除重复
    for (unsigned long i = 0; i < pSrc->vectorParents.size(); ++i) {
        NODE_INDEX indexParent = pSrc->vectorParents[i];
        SEQ_PAR_NODE_BASE* pParent = NULL;
        if (indexParent.type == NODE_TYPE::SEQUENTIAL) {
            pParent = (SEQ_PAR_NODE_BASE*)(&vectorSEQ[indexParent.index]);
        } else if (indexParent.type == NODE_TYPE::PARALLEL) {
            pParent = (SEQ_PAR_NODE_BASE*)(&vectorOMP[indexParent.index]);
        } else {
            std::cout << "merge 错误: indexParent 类型是: " << indexParent.type << std::endl;
        }

        std::vector<NODE_INDEX>::iterator iterTmpIndex;

        // 处理 pParent->vectorChildren
        iterTmpIndex = find(pParent->vectorChildren.begin(), pParent->vectorChildren.end(), indexDest);
        if (iterTmpIndex != pParent->vectorChildren.end()) { // 如果 父节点 已经链接到 Dest
            iterTmpIndex = find(pParent->vectorChildren.begin(), pParent->vectorChildren.end(), indexSrc);
            if (iterTmpIndex != pParent->vectorChildren.end()) {
                pParent->vectorChildren.erase(iterTmpIndex); // 直接删除 父节点到 Src 的链接 即可
            }
        } else {
            iterTmpIndex = find(pParent->vectorChildren.begin(), pParent->vectorChildren.end(), indexSrc);
            if (iterTmpIndex != pParent->vectorChildren.end()) {
                *iterTmpIndex = indexDest; // 父节点->Src 链接 重置为 父节点->Dest
            }
        }

        // 处理 pDest->vectorParents
        iterTmpIndex = find(pDest->vectorParents.begin(), pDest->vectorParents.end(), indexParent);
        if (iterTmpIndex == pDest->vectorParents.end()) { // 如果 父节点 没链接到 Dest
            pDest->vectorParents.push_back(indexParent); // 新建链接 父节点->Dest
        }
        // if (iterTmpIndex != pDest->vectorParents.end()) { // 如果 父节点 已经链接到 Dest
        //     iterTmpIndex = find(pDest->vectorParents.begin(), pDest->vectorParents.end(), indexSrc);
        //     if (iterTmpIndex != pDest->vectorParents.end()) {
        //         pDest->vectorParents.erase(iterTmpIndex); // 直接删除 父节点到 Src 的链接 即可
        //     }
        // } else {
        //     iterTmpIndex = find(pDest->vectorParents.begin(), pDest->vectorParents.end(), indexSrc);
        //     if (iterTmpIndex != pDest->vectorParents.end()) {
        //         *iterTmpIndex = indexParent; // 父节点->Src 链接 重置为 父节点->Dest
        //     } else {
        //         pDest->vectorParents.push_back(indexParent); // 新建链接 父节点->Dest
        //     }
        // }
    }

    // 将 Src 节点的子节点列表 插入 Dest 节点的子节点列表, 排除重复
    // 将 Src 节点的子节点们 链接到 Dest 节点, 排除重复
    for (unsigned long i = 0; i < pSrc->vectorChildren.size(); ++i) {
        NODE_INDEX indexChild = pSrc->vectorChildren[i];
        SEQ_PAR_NODE_BASE* pChild = NULL;
        if (indexChild.type == NODE_TYPE::SEQUENTIAL) {
            pChild = (SEQ_PAR_NODE_BASE*)(&vectorSEQ[indexChild.index]);
        } else if (indexChild.type == NODE_TYPE::PARALLEL) {
            pChild = (SEQ_PAR_NODE_BASE*)(&vectorOMP[indexChild.index]);
        } else {
            std::cout << "merge 错误: indexChild 类型是: " << indexChild.type << std::endl;
        }

        std::vector<NODE_INDEX>::iterator iterTmpIndex;

        // 处理 pChild->vectorParents
        iterTmpIndex = find(pChild->vectorParents.begin(), pChild->vectorParents.end(), indexDest);
        if (iterTmpIndex != pChild->vectorParents.end()) { // 如果 存在 Dest->子节点 链接
            iterTmpIndex = find(pChild->vectorParents.begin(), pChild->vectorParents.end(), indexSrc);
            if (iterTmpIndex != pChild->vectorParents.end()) {
                pChild->vectorParents.erase(iterTmpIndex); // 直接删除 Src->子节点 链接 即可
            }
        } else {
            iterTmpIndex = find(pChild->vectorParents.begin(), pChild->vectorParents.end(), indexSrc);
            if (iterTmpIndex != pChild->vectorParents.end()) {
                *iterTmpIndex = indexDest; // Src->子节点 链接 重置为 Dest->子节点
            }
        }

        // 处理 pDest->vectorChildren
        iterTmpIndex = find(pDest->vectorChildren.begin(), pDest->vectorChildren.end(), indexChild);
        if (iterTmpIndex == pDest->vectorChildren.end()) {  // 如果 子节点 没链接到 Dest
            pDest->vectorChildren.push_back(indexChild); // 新建链接 Dest->子节点
        }
        // if (iterTmpIndex != pDest->vectorChildren.end()) { // 如果 存在 Dest->子节点 链接
        //     iterTmpIndex = find(pDest->vectorChildren.begin(), pDest->vectorChildren.end(), indexSrc);
        //     if (iterTmpIndex != pDest->vectorChildren.end()) {
        //         pDest->vectorChildren.erase(iterTmpIndex); // 直接删除 Src->子节点 链接 即可
        //     }
        // } else {
        //     iterTmpIndex = find(pDest->vectorChildren.begin(), pDest->vectorChildren.end(), indexSrc);
        //     if (iterTmpIndex != pDest->vectorChildren.end()) {
        //         *iterTmpIndex = indexChild; // Src->子节点 链接 重置为 Dest->子节点
        //     } else {
        //         pDest->vectorChildren.push_back(indexChild); // 新建链接 Dest->子节点
        //     }
        // }
    }

    // 重置 Src 节点
    if (indexSrc.type == NODE_TYPE::SEQUENTIAL) {
        vectorSEQ[indexSrc.index].reset();
    } else if (indexSrc.type == NODE_TYPE::PARALLEL) {
        vectorOMP[indexSrc.index].reset();
    }
}

// 合并 SEQ 节点, 将 src 节点 合并到 dest 节点, 只将 src 节点空间重置 并不释放
void merge(int dest, int src, std::vector<SEQ_REGION> &vectorSEQ, Rewriter& Rewrite) {
    SEQ_REGION &SEQDest = vectorSEQ[dest];
    SEQ_REGION &SEQSrc = vectorSEQ[src];

    NODE_INDEX indexDest(NODE_TYPE::SEQUENTIAL, dest);
    NODE_INDEX indexSrc(NODE_TYPE::SEQUENTIAL, src);

    std::vector<NODE_INDEX>::iterator iterTmpIndex;

    // 从 dest 中 删除 src
    iterTmpIndex = find(SEQDest.vectorParents.begin(), SEQDest.vectorParents.end(), indexSrc);
    if (iterTmpIndex != SEQDest.vectorParents.end()) {
        SEQDest.vectorParents.erase(iterTmpIndex);
    }
    iterTmpIndex = find(SEQDest.vectorChildren.begin(), SEQDest.vectorChildren.end(), indexSrc);
    if (iterTmpIndex != SEQDest.vectorChildren.end()) {
        SEQDest.vectorChildren.erase(iterTmpIndex);
    }

    // 不考虑 dest 和 src 之间的环路, 所以加上下边的代码
    // 从 src 中 删除 dest
    iterTmpIndex = find(SEQSrc.vectorParents.begin(), SEQSrc.vectorParents.end(), indexDest);
    if (iterTmpIndex != SEQSrc.vectorParents.end()) {
        SEQSrc.vectorParents.erase(iterTmpIndex);
    }
    iterTmpIndex = find(SEQSrc.vectorChildren.begin(), SEQSrc.vectorChildren.end(), indexDest);
    if (iterTmpIndex != SEQSrc.vectorChildren.end()) {
        SEQSrc.vectorChildren.erase(iterTmpIndex);
    }

    // 将 src 节点的父节点列表 插入 dest 节点的父节点列表, 排除重复
    // 将 src 节点的父节点们 链接到 dest 节点, 排除重复
    for (unsigned long i = 0; i < SEQSrc.vectorParents.size(); ++i) {
        NODE_INDEX indexParent = SEQSrc.vectorParents[i];
        SEQ_REGION &SEQParent = vectorSEQ[indexParent.index];

        // 处理 SEQParent.vectorChildren
        iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexDest);
        if (iterTmpIndex != SEQParent.vectorChildren.end()) { // 如果 父节点 已经链接到 dest
            iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexSrc);
            if (iterTmpIndex != SEQParent.vectorChildren.end()) {
                SEQParent.vectorChildren.erase(iterTmpIndex); // 直接删除 父节点到 src 的链接 即可
            }
        } else {
            iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexSrc);
            if (iterTmpIndex != SEQParent.vectorChildren.end()) {
                *iterTmpIndex = indexDest; // 父节点->src 链接 重置为 父节点->dest
            }
        }

        // 处理 SEQDest.vectorParents
        iterTmpIndex = find(SEQDest.vectorParents.begin(), SEQDest.vectorParents.end(), indexParent);
        if (iterTmpIndex != SEQDest.vectorParents.end()) { // 如果 父节点 已经链接到 dest
            iterTmpIndex = find(SEQDest.vectorParents.begin(), SEQDest.vectorParents.end(), indexSrc);
            if (iterTmpIndex != SEQDest.vectorParents.end()) {
                SEQDest.vectorParents.erase(iterTmpIndex); // 直接删除 父节点到 src 的链接 即可
            }
        } else {
            iterTmpIndex = find(SEQDest.vectorParents.begin(), SEQDest.vectorParents.end(), indexSrc);
            if (iterTmpIndex != SEQDest.vectorParents.end()) {
                *iterTmpIndex = indexParent; // 父节点->src 链接 重置为 父节点->dest
            } else {
                SEQDest.vectorParents.push_back(indexParent); // 新建链接 父节点->dest
            }
        }
    }

    // 将 src 节点的子节点列表 插入 dest 节点的子节点列表, 排除重复
    // 将 src 节点的子节点们 链接到 dest 节点, 排除重复
    for (unsigned long i = 0; i < SEQSrc.vectorChildren.size(); ++i) {
        NODE_INDEX indexChild = SEQSrc.vectorChildren[i];
        SEQ_REGION &SEQChild = vectorSEQ[indexChild.index];

        // 处理 SEQChild.vectorParents
        iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexDest);
        if (iterTmpIndex != SEQChild.vectorParents.end()) { // 如果 存在 dest->子节点 链接
            iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexSrc);
            if (iterTmpIndex != SEQChild.vectorParents.end()) {
                SEQChild.vectorParents.erase(iterTmpIndex); // 直接删除 src->子节点 链接 即可
            }
        } else {
            iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexSrc);
            if (iterTmpIndex != SEQChild.vectorParents.end()) {
                *iterTmpIndex = indexDest; // src->子节点 链接 重置为 dest->子节点
            }
        }

        // 处理 SEQDest.vectorChildren
        iterTmpIndex = find(SEQDest.vectorChildren.begin(), SEQDest.vectorChildren.end(), indexChild);
        if (iterTmpIndex != SEQDest.vectorChildren.end()) { // 如果 存在 dest->子节点 链接
            iterTmpIndex = find(SEQDest.vectorChildren.begin(), SEQDest.vectorChildren.end(), indexSrc);
            if (iterTmpIndex != SEQDest.vectorChildren.end()) {
                SEQDest.vectorChildren.erase(iterTmpIndex); // 直接删除 src->子节点 链接 即可
            }
        } else {
            iterTmpIndex = find(SEQDest.vectorChildren.begin(), SEQDest.vectorChildren.end(), indexSrc);
            if (iterTmpIndex != SEQDest.vectorChildren.end()) {
                *iterTmpIndex = indexChild; // src->子节点 链接 重置为 dest->子节点
            } else {
                SEQDest.vectorChildren.push_back(indexChild); // 新建链接 dest->子节点
            }
        }
    }

    // 重置 src 节点
    SEQSrc.reset();

    // 重新设置 dest 的父节点的范围, 防止其与 dest 有重叠
    for (unsigned long i = 0; i < SEQDest.vectorParents.size(); ++i) {
        NODE_INDEX indexParent = SEQDest.vectorParents[i];
        SEQ_REGION &SEQParent = vectorSEQ[indexParent.index];
        SourceLocation TmpLoc;

        // 如果父节点范围中包含 dest 起始
        if (SEQParent.SEQRange.BeginOffset < SEQDest.SEQRange.BeginOffset &&
            SEQDest.SEQRange.BeginOffset <= SEQParent.SEQRange.EndOffset &&
            SEQParent.SEQRange.EndOffset < SEQDest.SEQRange.EndOffset) {
            TmpLoc = SEQDest.SEQRange.BeginLoc.getLocWithOffset(-1);
            SEQParent.SEQRange.SetEndLoc(TmpLoc, Rewrite);
        }

        // 如果父节点范围中包含 dest 结束
        if (SEQDest.SEQRange.BeginOffset < SEQParent.SEQRange.BeginOffset &&
            SEQParent.SEQRange.BeginOffset <= SEQDest.SEQRange.EndOffset &&
            SEQDest.SEQRange.EndOffset < SEQParent.SEQRange.EndOffset) {
            TmpLoc = SEQDest.SEQRange.EndLoc.getLocWithOffset(1);
            SEQParent.SEQRange.SetBeginLoc(TmpLoc, Rewrite);
        }
    }

    // 重新设置 dest 的子节点的范围, 防止其与 dest 有重叠
    for (unsigned long i = 0; i < SEQDest.vectorParents.size(); ++i) {
        NODE_INDEX indexChild = SEQDest.vectorParents[i];
        SEQ_REGION &SEQChild = vectorSEQ[indexChild.index];
        SourceLocation TmpLoc;

        // 如果父节点范围中包含 dest 起始
        if (SEQChild.SEQRange.BeginOffset < SEQDest.SEQRange.BeginOffset &&
            SEQDest.SEQRange.BeginOffset <= SEQChild.SEQRange.EndOffset &&
            SEQChild.SEQRange.EndOffset < SEQDest.SEQRange.EndOffset) {
            TmpLoc = SEQDest.SEQRange.BeginLoc.getLocWithOffset(-1);
            SEQChild.SEQRange.SetEndLoc(TmpLoc, Rewrite);
        }

        // 如果父节点范围中包含 dest 结束
        if (SEQDest.SEQRange.BeginOffset < SEQChild.SEQRange.BeginOffset &&
            SEQChild.SEQRange.BeginOffset <= SEQDest.SEQRange.EndOffset &&
            SEQDest.SEQRange.EndOffset < SEQChild.SEQRange.EndOffset) {
            TmpLoc = SEQDest.SEQRange.EndLoc.getLocWithOffset(1);
            SEQChild.SEQRange.SetBeginLoc(TmpLoc, Rewrite);
        }
    }
}