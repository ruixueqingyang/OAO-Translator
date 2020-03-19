/***************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: SequentialRegion.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20181216
  Description: 串行域类
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
***************************************************************************/

#ifndef SEQUENTIAL_REGION_H
#define SEQUENTIAL_REGION_H

#include "Clang_LLVM_Inc.h"
#include "SrcRange.h"
#include "VarInfo.h"
#include "SeqParNodeBase.h"

using namespace clang;

class SEQ_REGION : public SEQ_PAR_NODE_BASE{
public:
    //std::vector<unsigned int>   vectorVarDecl; // 并行域中定义变量的列表, 是变量列表中某项的 index
    //std::vector<VAR_REF_LIST>        vectorVarRef; // 并行域中引用外部变量的列表,  VAR_REF 定义在 SeqParNodeBase.h
    int indexFunc; // 如果这个值非负, 说明该节点是一个函数节点, indexFunc 指向 vectorFunc 中的某个元素
    CallExpr* pFuncCall;
    MY_SOURCE_RANGE SEQRange;
    MY_SOURCE_RANGE StmtRange;
    std::string FuncName;

    void init(const CFGBlock& in_Block, Rewriter& Rewrite){

        std::cout << "SEQ_REGION::init: 当前节点 ID = " << in_Block.getBlockID() << std::endl;

        if(4<=in_Block.getBlockID() && in_Block.getBlockID()<=10){
            std::cout << "SEQ_REGION::init: 当前节点 ID = " << in_Block.getBlockID() << std::endl;
        }

        // 写入 parents 节点信息
        if(!in_Block.pred_empty()){
            vectorParents.clear(); // 先清空下
            for(CFGBlock::const_pred_iterator I = in_Block.pred_begin(); I != in_Block.pred_end(); ++I){
                CFGBlock* B = *I;
                // 处理可达性
                bool Reachable = true;
                if (!B) {
                    Reachable = false;
                    B = I->getPossiblyUnreachableBlock();
                }

                if (B) { // 写入 parent 的索引
                    vectorParents.emplace_back(NODE_TYPE::SEQUENTIAL, B->getBlockID());
                    if (!Reachable)
                        std::cout << "(Unreachable) CFGBlock 不可达" << std::endl;
                }
                else {
                    std::cout << "CFGBlock 不存在" << std::endl;
                }
            }
        }

        // 写入 children 节点信息
        if(!in_Block.succ_empty()){
            vectorChildren.clear(); // 先清空下
            for(CFGBlock::const_succ_iterator I = in_Block.succ_begin(); I != in_Block.succ_end(); ++I){
                CFGBlock* B = *I;
                // 处理可达性
                bool Reachable = true;
                if (!B) {
                    Reachable = false;
                    B = I->getPossiblyUnreachableBlock();
                }

                if (B) { // 写入 children 的索引
                    vectorChildren.emplace_back(NODE_TYPE::SEQUENTIAL, B->getBlockID());
                    if (!Reachable)
                        std::cout << "(Unreachable) CFGBlock 不可达" << std::endl;
                }
                else {
                    std::cout << "CFGBlock 不存在" << std::endl;
                }
            }
        }

        SourceRange CFGBlockRange = getCFGBlockRange(in_Block, Rewrite);
        // 写入 block 在源代码中的位置信息
        SEQRange.init( CFGBlockRange.getBegin(), CFGBlockRange.getEnd(), Rewrite );

        std::cout << "SEQ_REGION::init: BeginOffset = " << SEQRange.BeginOffset << std::endl;
        std::cout << "SEQ_REGION::init: EndOffset = " << SEQRange.EndOffset << std::endl;
    }

    void reset(){
        indexFunc = -1;
        pFuncCall = NULL;
        SEQRange.reset();
        StmtRange.reset();
        FuncName = "NULL";
        vectorParents.clear();
        vectorChildren.clear();
        vectorVarRef.clear();
        vectorTrans.clear();
        isComplete = false;
        JumpStmt = "NULL";
        TerminatorStmt = NULL;
    }

    SEQ_REGION(const CFGBlock &in_Block, Rewriter& Rewrite) : SEQ_PAR_NODE_BASE(){
        init();
        init(in_Block, Rewrite);
    }
    SEQ_REGION(Stmt* ptrBegin, Stmt* ptrEnd, Rewriter& Rewrite) : SEQ_PAR_NODE_BASE(){
        init();

        // wfr 20190718 修补 Stmt等 的结尾位置偏移, 因为 clang AST 中的原始位置可能不准, 例如不包括 ";"
        SourceLocation CallEndLoc = fixStmtLoc(ptrBegin->getEndLoc(), Rewrite);

        SEQRange.init(ptrBegin->getBeginLoc(), CallEndLoc, Rewrite);
    }

    const SEQ_REGION& operator=(const SEQ_REGION& in_SEQ){

        (SEQ_PAR_NODE_BASE&)(*this) = (SEQ_PAR_NODE_BASE&)in_SEQ;

        init(in_SEQ);

        return in_SEQ;
    }

    SEQ_REGION(const SEQ_REGION& in_SEQ) : SEQ_PAR_NODE_BASE(in_SEQ){

        init(in_SEQ);
    }

    SEQ_REGION() : SEQ_PAR_NODE_BASE(){
        init();
    }

    void init(const SEQ_REGION& in_SEQ){
        indexFunc = in_SEQ.indexFunc;
        pFuncCall = in_SEQ.pFuncCall;
        SEQRange = in_SEQ.SEQRange;
        StmtRange = in_SEQ.StmtRange;
        FuncName = in_SEQ.FuncName;
    }

    void init(){
        indexFunc = -1;
        pFuncCall = NULL;
        FuncName = "NULL";
    }

    SourceRange getCFGBlockRange(const CFGBlock& Block, Rewriter& Rewrite){
        SourceLocation EmptyLoc;
        SourceLocation BlockBeginLoc;
        BlockBeginLoc = BlockBeginLoc.getLocWithOffset(0x7FFFFFFF); // 先将开始偏移设成最大值, 便于之后的比较
        SourceLocation BlockEndLoc;

        if(Block.empty()){
            std::cout << "getCFGBlockRange 警告: CFGBlock 是空的" << std::endl;
        }

        for(CFGBlock::const_iterator iter=Block.begin(); iter!=Block.end(); ++iter){

            SourceLocation TmpBeginLoc, TmpEndLoc;
            const Stmt* pStmt; pStmt = NULL;
            // getCFGBlockLoc(TmpBeginLoc, TmpEndLoc, *iter);
            // wfr 20190711 获得 CFGElement 中包含的AST节点的 Stmt类型 的 指针
            pStmt = getCFGElementStmtPtr(*iter);
            if(pStmt == NULL){
                continue;
            }
            // wfr 20190711 递归获得一个 Stmt 覆盖的源码的起止范围
            IterGetStmtLoc(BlockBeginLoc, BlockEndLoc, pStmt, Rewrite);
            TmpBeginLoc = Rewrite.getSourceMgr().getExpansionLoc(TmpBeginLoc);
            TmpEndLoc = Rewrite.getSourceMgr().getExpansionLoc(TmpEndLoc);
        }

        if(BlockEndLoc<BlockBeginLoc){
            BlockBeginLoc = BlockEndLoc;
        }

        if(BlockBeginLoc==EmptyLoc || BlockEndLoc==EmptyLoc){
            std::cout << "getCFGBlockRange 警告: 获得 CFGBlock 在源代码中的范围失败" << std::endl;
            //std::cout << "getCFGBlockRange 警告: EmptyLoc.BlockBeginLoc() = " << EmptyLoc.getRawEncoding() << std::endl;
            //std::cout << "getCFGBlockRange 警告: BlockBeginLoc.BlockBeginLoc() = " << BlockBeginLoc.getRawEncoding() << std::endl;
            //std::cout << "getCFGBlockRange 警告: BlockEndLoc.BlockBeginLoc() = " << BlockEndLoc.getRawEncoding() << std::endl;
        }

        // wfr 20190718 修补 Stmt等 的结尾位置偏移, 因为 clang AST 中的原始位置可能不准, 例如不包括 ";"
        SourceLocation FixedEndLoc = fixStmtLoc(BlockEndLoc, Rewrite);

        SourceRange BlockRange(BlockBeginLoc, FixedEndLoc);

        return BlockRange;
    }

    // wfr 20190711 递归获得一个 Stmt 覆盖的源码的起止范围, 先根遍历
    void IterGetStmtLoc(SourceLocation& BeginLoc, SourceLocation& EndLoc, const Stmt* pStmt, Rewriter& Rewrite){
        SourceLocation EmptyLoc;
        SourceLocation TmpBeginLoc, TmpEndLoc;

        TmpBeginLoc = Rewrite.getSourceMgr().getExpansionLoc(pStmt->getBeginLoc());
        TmpEndLoc = Rewrite.getSourceMgr().getExpansionLoc(pStmt->getEndLoc());

        std::cout << "IterGetStmtLoc: Stmt 节点类型名称 " << pStmt->getStmtClassName() << std::endl;
        std::cout << "IterGetStmtLoc: BeginOffset = " << Rewrite.getSourceMgr().getFileOffset(TmpBeginLoc) << std::endl;
        std::cout << "IterGetStmtLoc: EndOffset = " << Rewrite.getSourceMgr().getFileOffset(TmpEndLoc) << std::endl;
            
        if(TmpBeginLoc!=EmptyLoc && TmpBeginLoc<BeginLoc){
            BeginLoc = TmpBeginLoc;
        }

        if(EndLoc<TmpEndLoc){
            EndLoc = TmpEndLoc;
        }

        for(Stmt::const_child_iterator iter=pStmt->child_begin(); iter!=pStmt->child_end(); ++iter){
            if((*iter)!=NULL){
                IterGetStmtLoc(BeginLoc, EndLoc, *iter, Rewrite);
            }
        }

        return;
    }

    // wfr 20190711 获得 CFGElement 中包含的AST节点的 Stmt类型 的 指针
    const Stmt* getCFGElementStmtPtr(const CFGElement& Element){
        const Stmt* pStmt;
        switch(Element.getKind())
        {
            case CFGElement::Kind::Initializer:
                //std::cout << "getCFGBlockRange: 1" << std::endl;
                pStmt = NULL;
                break;
            case CFGElement::Kind::ScopeBegin: // ScopeBegin 和 ScopeEnd 可能有问题, 因为有个成员函数是 getVarDecl() 不知道是什么作用
                //std::cout << "getCFGBlockRange: 2" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGScopeBegin>().getTriggerStmt() );
                break;
            case CFGElement::Kind::ScopeEnd:
                //std::cout << "getCFGBlockRange: 3" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGScopeEnd>().getTriggerStmt() );
                break;
            case CFGElement::Kind::NewAllocator:
                //std::cout << "getCFGBlockRange: 4" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGNewAllocator>().getAllocatorExpr() );
                break;
            case CFGElement::Kind::LifetimeEnds:
                //std::cout << "getCFGBlockRange: 5" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGLifetimeEnds>().getTriggerStmt() );
                break;
            case CFGElement::Kind::LoopExit:
                //std::cout << "getCFGBlockRange: 6" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGLoopExit>().getLoopStmt() );
                break;
            case CFGElement::Kind::Statement:
            //case CFGElement::Kind::STMT_BEGIN:
                //std::cout << "getCFGBlockRange: 7" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGStmt>().getStmt() );
                break;
            case CFGElement::Kind::Constructor: // 这几种情况找不到在源文件中的位置, 因此需要在附近的 CFGElement 中搜索, 直到找到在源文件中的位置
            case CFGElement::Kind::CXXRecordTypedCall:
            //case CFGElement::Kind::STMT_END:
            case CFGElement::Kind::MemberDtor:
                //std::cout << "getCFGBlockRange: 8" << std::endl;
                pStmt = NULL;
                break;
            case CFGElement::Kind::AutomaticObjectDtor:
            //case CFGElement::Kind::DTOR_BEGIN:
                //std::cout << "getCFGBlockRange: 9" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGAutomaticObjDtor>().getTriggerStmt() );
                break;
            case CFGElement::Kind::DeleteDtor:
                //std::cout << "getCFGBlockRange: 10" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGDeleteDtor>().getDeleteExpr() );
                break;
            case CFGElement::Kind::BaseDtor:
                //std::cout << "getCFGBlockRange: 11" << std::endl;
                pStmt = NULL;
                break;
            case CFGElement::Kind::TemporaryDtor:
            //case CFGElement::Kind::DTOR_END:
                //std::cout << "getCFGBlockRange: 12" << std::endl;
                pStmt = dyn_cast<Stmt>( Element.castAs<CFGTemporaryDtor>().getBindTemporaryExpr() );
                break;
            default:
                //std::cout << "getCFGBlockRange: 13" << std::endl;
                pStmt = NULL;
                break;
        }
        return pStmt;
    }
};

#endif // SEQUENTIAL_REGION_H
