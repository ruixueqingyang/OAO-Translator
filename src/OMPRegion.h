/*******************************************************************************
Copyright(C), 2010-2018, 瑞雪轻飏
     FileName: OMPRegion.h
       Author: 瑞雪轻飏
      Version: //版本
Creation Date: //创建日期
  Description: //用于主要说明此程序文件完成的主要功能
               //与其他模块或函数的接口、输出值、取值范围、
               //含义及参数间的控制、顺序、独立及依赖关系
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#ifndef OMP_REGION_H
#define OMP_REGION_H

#include "Clang_LLVM_Inc.h"
#include "SrcRange.h"
#include "VarInfo.h"
#include "SeqParNodeBase.h"

using namespace clang;

class OMP_REGION : public SEQ_PAR_NODE_BASE{
public:
    //std::string  Name;
 	OMPExecutableDirective* ptrDecl; // 并行域的定义的地址
    Stmt*   pBodyStmt; // 并行域的主循环体 Stmt 的地址, 该地址不为 NULL, 则说明在主体中
    std::string  KindName; // 并行域的种类
    int indexFunc; // 如果这个值非负, 说明该节点是一个函数节点, indexFunc 指向 vectorFunc 中的某个元素
    int indexEntry; // 表示并行域开始节点的 index
    int indexExit; // 表示并行域结束点的 index, 只有开始节点的这个变量有意义

    MY_SOURCE_RANGE DirectorRange; // OMP指令在原码中的位置范围, 从 #pragma 之后开始算起
    MY_SOURCE_RANGE OMPRange; // 并行域在源码中的位置范围信息, 不算OMP指令
    SourceRange ReplaceRange; // "# pragma omp ... parallel for" 这个字符串需要被替换, 这里是替换的范围
    
    std::vector<VARIABLE_INFO>   vectorLocal; // 局部变量列表, 定义在并行域内的变量, 只记录并跟踪 指针/引用 类型的变量
    std::vector<unsigned int>   vectorFuncRef; // 并行域中调用函数的列表, 是函数列表中某项的 index

    std::vector<Stmt*>  forParallelStmt; //保存并行域的for循环指针
    std::string  scalar; // 加入标量"# pragma omp ... parallel for map(tofrom: ..)"
    
    void reset(){
        SourceLocation TmpLoc;
        ptrDecl = NULL;
        pBodyStmt = NULL;
        KindName = "NULL";
        scalar = "NULL";
        indexFunc = -1;
        indexEntry = -1;
        indexExit = -1;
        DirectorRange.reset();
        OMPRange.reset();
        ReplaceRange.setBegin(TmpLoc);
        ReplaceRange.setEnd(TmpLoc);
        vectorLocal.clear();
        vectorFuncRef.clear();
        vectorParents.clear();
        vectorChildren.clear();
        vectorVarRef.clear();
        vectorTrans.clear();
        isComplete = false;
        JumpStmt = "NULL";
        TerminatorStmt = NULL;
    }

    OMP_REGION(OMPParallelForDirective *S, Rewriter& Rewrite) : SEQ_PAR_NODE_BASE(){

        ptrDecl = (OMPExecutableDirective*)S;
        pBodyStmt = NULL;
        KindName = S->getStmtClassName(); // 并行域的种类
        scalar = "NULL";
        indexFunc = -1;
        indexEntry = -1;
        indexExit = -1;

        Stmt *SubStmt = (Stmt*)(*(S->child_begin())); // 这个子节点中保存着下边实际可执行的代码的范围信息, 例如for循环整体的范围信息。

        SourceLocation TmpBeginLoc, TmpEndLoc;

        TmpBeginLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
        TmpEndLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getEndLoc());
        DirectorRange.init(TmpBeginLoc, TmpEndLoc, Rewrite); // OMP 指令在源代码中的范围

        TmpBeginLoc = Rewrite.getSourceMgr().getExpansionLoc(SubStmt->getBeginLoc());
        TmpEndLoc = Rewrite.getSourceMgr().getExpansionLoc(SubStmt->getEndLoc());
        OMPRange.init(TmpBeginLoc, TmpEndLoc, Rewrite); // OMP 代码部分(大括号中的)的范围

        //ReplaceRange.setBegin(DirectorRange.BeginLoc.getLocWithOffset(-1*(DirectorRange.BeginCol)+1));
        ReplaceRange.setBegin(DirectorRange.BeginLoc);
        if( ptrDecl->getNumClauses() >= 1 ){
            OMPClause* pOMPClause = ptrDecl->getClause(0);
            SourceLocation EndLoc = pOMPClause->getBeginLoc().getLocWithOffset(-1);
            ReplaceRange.setEnd(EndLoc);
        }else{
            ReplaceRange.setEnd(DirectorRange.EndLoc);
        }
    }

    const OMP_REGION& operator=(const OMP_REGION& in_OMP){
        
        (SEQ_PAR_NODE_BASE&)(*this) = (SEQ_PAR_NODE_BASE&)in_OMP;

        init(in_OMP);

        return in_OMP;
    }

    OMP_REGION(const OMP_REGION& in_OMP) : SEQ_PAR_NODE_BASE(in_OMP){
        init(in_OMP);
    }

    void init(const OMP_REGION& in_OMP){
        ptrDecl = in_OMP.ptrDecl;
        pBodyStmt = in_OMP.pBodyStmt;
        KindName = in_OMP.KindName;
        scalar = in_OMP.scalar;
        indexFunc = in_OMP.indexFunc;
        indexEntry = in_OMP.indexEntry;
        indexExit = in_OMP.indexExit;
        DirectorRange = in_OMP.DirectorRange;
        OMPRange = in_OMP.OMPRange;
        ReplaceRange = in_OMP.ReplaceRange;
        vectorLocal = in_OMP.vectorLocal;
        vectorFuncRef = in_OMP.vectorFuncRef;
    }

    OMP_REGION() : SEQ_PAR_NODE_BASE(){
        init();
    }

    void init(){
        // Name = "NULL";
        ptrDecl = NULL;
        pBodyStmt = NULL;
        KindName = "NULL";
        scalar = "NULL";
        indexFunc = -1;
        indexEntry = -1;
        indexExit = -1;
    }
};

#endif // OMP_REGION_H
