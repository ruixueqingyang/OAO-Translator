/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: OAOSubFuncs.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190325
  Description: 实现 OAOASTVisitor 和 ASTConsumer 的一些成员函数
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

#include "OAORewriter.h"

using namespace clang;

// wfr 20190817 推导从同步状态约束 src 到同步状态约束 dest 的转换函数
STATE_CONSTR OAOASTVisitor::InferStTransFunc(STATE_CONSTR& src, STATE_CONSTR& dest){
    // 计算从 src 到 dest 所需的状态转移函数
    unsigned int ZERO = ~( (src.ZERO ^ dest.ZERO) & (~dest.ZERO) );
    unsigned int ONE = (src.ONE ^ dest.ONE) & dest.ONE;
    STATE_CONSTR StTransFunc(ZERO, ONE);
    return StTransFunc;
}

// wfr 20190809 获得 某函数member/全局member(不包括并行域中定义的member) 的 index 如果变量还不存在就存入
int  OAOASTVisitor::getMemberIndex(Decl* ptrClass, FieldDecl* pMemberDecl, unsigned int indexFunc){
    int indexMember = -1;
    int indexClass = -1;
    int indexGblMember = -1;
    int indexGblClass = -1;

    // 先在当前函数的 vectorVar 中搜索
    // 有就返回 indexMember
    std::string MemberName = pMemberDecl->getNameAsString();
    indexMember = MemberIsWhichVar(ptrClass, MemberName, indexFunc);
    if(indexMember>=0){
        return indexMember;
    }

    indexGblClass = WhichGblVar(ptrClass);
    indexClass = WhichVar(ptrClass, indexFunc);
    indexGblMember = MemberIsWhichGlobal(ptrClass, MemberName);

    if(indexGblClass<0 && indexClass<0){ // 没找到域所属的类, 认为出错
        std::cout << "getMemberIndex 错误: 没找到变量" << std::endl;
        exit(1);
    }else if(indexGblClass<0 && indexClass>=0){ // 不是全局变量, 是 indexFunc 的内部变量
        indexMember = saveMemberInfo(pMemberDecl, indexClass, indexFunc);
        return indexMember;
    }else if(indexGblClass>=0){ // 是全局变量
        if(indexGblMember<0){ // 在 vectorGblVar 中保存 域 信息
            indexGblMember = saveGlobalMemberInfo(pMemberDecl, indexGblClass);
        }
        if(indexClass<0){ // 在 vectorVar 中保存 类 信息
            vectorFunc[indexFunc].vectorVar.push_back(vectorGblVar[indexGblClass]);
            vectorFunc[indexFunc].vectorVar.back().vectorInsert.clear();
            vectorFunc[indexFunc].vectorVar.back().vectorReplace.clear();
            vectorFunc[indexFunc].vectorVar.back().indexClass = -1;
            vectorFunc[indexFunc].vectorVar.back().indexRoot = -1;
            vectorFunc[indexFunc].vectorVar.back().ptrRoot = NULL;
            vectorFunc[indexFunc].vectorVar.back().UsedInOMP = false;
            vectorFunc[indexFunc].vectorVar.back().indexDefNode.init();
            vectorFunc[indexFunc].vectorVar.back().indexLastRefNode.init();

            indexClass = vectorFunc[indexFunc].vectorVar.size() - 1;
        }
        if(indexMember<0){ // 在 vectorVar 中保存 域 信息
            vectorFunc[indexFunc].vectorVar.push_back(vectorGblVar[indexGblMember]);
            vectorFunc[indexFunc].vectorVar.back().vectorInsert.clear();
            vectorFunc[indexFunc].vectorVar.back().vectorReplace.clear();
            vectorFunc[indexFunc].vectorVar.back().indexClass = indexClass;
            vectorFunc[indexFunc].vectorVar.back().indexRoot = -1;
            vectorFunc[indexFunc].vectorVar.back().ptrRoot = NULL;
            vectorFunc[indexFunc].vectorVar.back().UsedInOMP = false;
            vectorFunc[indexFunc].vectorVar.back().indexDefNode.init();
            vectorFunc[indexFunc].vectorVar.back().indexLastRefNode.init();

            indexMember = vectorFunc[indexFunc].vectorVar.size() - 1;
        }
        return indexMember;
    }else{
        std::cout << "getMemberIndex 错误: 未知错误" << std::endl;
        exit(1);
    }
}

// wfr 20190808 获得 某函数中变量/全局变量(不包括并行域中定义的变量) 的 index 如果变量还不存在就存入
int  OAOASTVisitor::getVarIndex(Decl* pVarDecl, unsigned int indexFunc){

    int indexVar = -1;

    // 先在当前函数的 vectorVar 中搜索, 有就返回 index
    indexVar = WhichVar(pVarDecl, indexFunc);
    if(indexVar>=0){
        return indexVar;
    }

    // 再在全局 vectorGblVar 中搜索, 没有就报错, 有就
    // 1. 将变量信息也 存入 vectorVar
    // 2. 返回 vectorVar 中项的 index
    indexVar = WhichGblVar(pVarDecl);

    if(indexVar<0){
        if((unsigned long)pVarDecl< (unsigned long)vectorFunc[0].ptrDecl){
            return -1;
        }else{
            std::cout << "getVarIndex 错误: 没找到变量" << std::endl;
            exit(1);
        }
    }

    vectorFunc[indexFunc].vectorVar.push_back(vectorGblVar[indexVar]);
    vectorFunc[indexFunc].vectorVar.back().vectorInsert.clear();
    vectorFunc[indexFunc].vectorVar.back().vectorReplace.clear();
    vectorFunc[indexFunc].vectorVar.back().indexClass = -1;
    vectorFunc[indexFunc].vectorVar.back().indexRoot = -1;
    vectorFunc[indexFunc].vectorVar.back().ptrRoot = NULL;
    vectorFunc[indexFunc].vectorVar.back().UsedInOMP = false;
    vectorFunc[indexFunc].vectorVar.back().indexDefNode.init();
    vectorFunc[indexFunc].vectorVar.back().indexLastRefNode.init();

    indexVar = vectorFunc[indexFunc].vectorVar.size() - 1;

    return indexVar;
}

// wfr 20190809 返回全局变量在 vactorGblVar 中的 index, 没找到就返回 -1
int  OAOASTVisitor::WhichGblVar(Decl* ptrDecl){
    for (unsigned long i = 0; i < vectorGblVar.size(); ++i) {
        if (vectorGblVar[i].ptrDecl == ptrDecl) {
            return i;
        }
    }
    return -1;
}

// wfr 20190806 通过变量类型字符串, 分析获得变量类型
VAR_TYPE OAOASTVisitor::getVarType(std::string& TypeName){
    VAR_TYPE VarType = VAR_TYPE::VAR_UNINIT;
    if(TypeName.size()>0 && TypeName.back()=='&'){ // 引用 REF
        unsigned int pos = TypeName.size()-2;
        while(pos>=0){
            if(TypeName[pos]!=' '){
                if(TypeName[pos]=='*'){
                    VarType = VAR_TYPE::PTR;
                }
                break;
            }
            --pos;
        }
        if(VarType==VAR_TYPE::VAR_UNINIT){
            VarType = VAR_TYPE::REF;
        }
    }else if(TypeName.size()>0 && TypeName.back()=='*'){
        if(TypeName.size()>5 && TypeName.substr(0,5)=="const"){ // 指针指向的区域只读 PTR_CONST
            VarType = VAR_TYPE::PTR_CONST;
        }else{ // 指针 PTR
            VarType = VAR_TYPE::PTR;
        }
    }else if(TypeName.size()>6 && TypeName.substr(TypeName.size()-6,6)=="*const"){
        if(TypeName.substr(0,5)=="const"){ // 指针指向的区域只读 PTR_CONST
            VarType = VAR_TYPE::PTR_CONST;
        }else{ // 指针 PTR
            VarType = VAR_TYPE::PTR;
        }
    }else{ // 认为是拷贝 COPY
        VarType = VAR_TYPE::COPY; // 所有变量都处理, 因为可能出现 变量取地址作为函数参数的情况
    }

    return VarType;
}

// wfr 20190726 检查 同步状态 是否 满足约束
bool OAOASTVisitor::CheckConstrSatisfaction(STATE_CONSTR& state, STATE_CONSTR& constraint){
    DEFINE_ST_REQ_ST_TRANS
    if(constraint==StReqUninit || state==StReqDiverg){
        std::cout << "CheckConstrSatisfaction 错误: constraint 非法" << std::endl;
        exit(1);
    }
    if(state==StReqUninit){
        std::cout << "CheckConstrSatisfaction 警告: state 未初始化" << std::endl;
    }

    // wfr 20191218 处理 StReqDiverg 情况
    if(state == StReqDiverg && constraint != StReqNone){
        return false;
    }

    // 计算从 state 到 constraint 所需的状态转移函数
    unsigned int ZERO = ~( (state.ZERO ^ constraint.ZERO) & (~constraint.ZERO) );
    unsigned int ONE = (state.ONE ^ constraint.ONE) & constraint.ONE;
    STATE_CONSTR StTransFunc(ZERO, ONE);

    if(StTransFunc==StTransNone){
        return true;
    }else{
        return false;
    }

    // if(constraint==StReqNone){
    //     return true;
    // }else if(constraint==StReqHostOnly){
    //     if(state==constraint){
    //         return true;
    //     }
    // }else if(constraint==StReqInDevice){
    //     if( state==StReqInDevice || state==StReqHostNew || state==StReqDeviceNew){
    //         return true;
    //     }
    // }else if(state==constraint){
    //     return true;
    // }

    // return false;
}

// wfr 20190726 求两个状态约束/需求的交集, 不存在 就返回 StReqUninit
STATE_CONSTR OAOASTVisitor::ReqIntersection(STATE_CONSTR& StReq0, STATE_CONSTR& StReq1){
    DEFINE_ST_REQ_ST_TRANS
    if( StReq0==StReqUninit || StReq0==StReqDiverg 
        || StReq1==StReqUninit || StReq1==StReqDiverg )
    {
        return StReqUninit;
    }else if(StReq0==StReqNone){
        return StReq1;
    }else if(StReq1==StReqNone){
        return StReq0;
    }else if(StReq0==StReqHostOnly || StReq1==StReqHostOnly){
        if(StReq0==StReq1){
            return StReqHostOnly;
        }else{
            return StReqUninit;
        }
    }else if(StReq0==StReqSync){
        return StReqSync;
    }else if(StReq1==StReqSync){
        return StReqSync;
    }else if(StReq0==StReq1){
        return StReq0;
    }else{
        return StReqUninit;
    }
}

// wfr 20190716 执行同步状态装换
SYNC_BITS OAOASTVisitor::ExeStTrans(const STATE_CONSTR &StTransFunc, const SYNC_BITS &SyncBits) {
    SYNC_BITS NewSyncBits = (SYNC_BITS)(SyncBits & SYNC_BITS::BITS_MASK);
    NewSyncBits = (SYNC_BITS)((SyncBits & StTransFunc.ZERO) | StTransFunc.ONE);
    return NewSyncBits;
}

// wfr 20190716 执行同步状态装换
STATE_CONSTR OAOASTVisitor::ExeStTrans(const STATE_CONSTR &StTransFunc, const STATE_CONSTR &StReq) {
    DEFINE_ST_REQ_ST_TRANS
    // wfr 20191218 先判断不是 Diverg
    if(StTransDiverg == StTransFunc || StReqDiverg == StReq){
        return StReqDiverg;
    }
    STATE_CONSTR NewStReq = StReq;
    NewStReq.ZERO = ((NewStReq.ZERO & StTransFunc.ZERO) | StTransFunc.ONE);
    NewStReq.ONE = ((NewStReq.ONE & StTransFunc.ZERO) | StTransFunc.ONE);
    return NewStReq;
}

// wfr 20190716 检查 StReq 是否被初始化了
void OAOASTVisitor::CheckStReq(STATE_CONSTR &in) {
    STATE_CONSTR Uninit(ST_REQ_UNINIT);
    if (in == Uninit) {
        std::cout << "CheckStReq 错误: StReq 未初始化" << std::endl;
        exit(1);
    }
}

// wfr 20190716 检查 StTrans 是否被初始化了
void OAOASTVisitor::CheckStTrans(STATE_CONSTR &in) {
    STATE_CONSTR Uninit(ST_TRANS_UNINIT);
    if (in == Uninit) {
        std::cout << "CheckStTrans 错误: StTrans 未初始化" << std::endl;
        exit(1);
    }
}

// wfr 20190716 对于在 OMP中 调用函数 的情况, 对 StReq 和 StTransFunc 进行转换
void OAOASTVisitor::ModifyFuncCallInOMP(STATE_CONSTR &StReq, STATE_CONSTR &StTransFunc) {
    DEFINE_ST_REQ_ST_TRANS

    if (StReq == StReqHostNew || StReq==StReqNone) {
        StReq.init(ST_REQ_DEVICE_NEW);
    } else if (StReq == StReqHostOnly) {
        std::cout << "ModifyFuncCallInOMP 错误: OMP 域中有非法 的变量入口需求, 变量不能只在 device 中" << std::endl;
        exit(1);
    } else {
        std::cout << "ModifyFuncCallInOMP 错误: OMP 域中有其他非法 的变量入口需求" << std::endl;
        exit(1);
    }

    if (StTransFunc == StTransHostRead) {
        StTransFunc.init(ST_TRANS_DEVICE_READ);
    } else if (StTransFunc == StTransHostWrite) {
        StTransFunc.init(ST_TRANS_DEVICE_WRITE);
    } else if (StTransFunc == StTransHostFree) {
        std::cout << "ModifyFuncCallInOMP 错误: OMP 域中 调用的函数中有非法 free(), 不能在 target 中释放 关联指针" << std::endl;
        exit(1);
    } else {
        // std::cout << "ModifyFuncCallInOMP 错误: OMP 域中 调用的函数 的等效 StTransFunc 非法" << std::endl;
        // exit(1);
        StTransFunc.init(ST_TRANS_DEVICE_WRITE);
    }

    return;
}

// wfr 20190613 获得被调函数定义地址
FunctionDecl *OAOASTVisitor::getCalleeFuncPtr(CallExpr *pCallExpr) {
    FunctionDecl *pFuncDecl = pCallExpr->getDirectCallee(); // 被调用函数的定义的地址
    if (pFuncDecl == NULL) {
        Stmt *pChildBegin = *(pCallExpr->child_begin()); // 获得第一个子节点的指针
        if (isa<UnresolvedLookupExpr>(pChildBegin)) {    // 如果是 UnresolvedLookupExpr 类型的节点
            std::cout << "getCalleeFuncPtr: 是 UnresolvedLookupExpr" << std::endl;
            UnresolvedLookupExpr *pUnresolved = (UnresolvedLookupExpr *)pChildBegin;
            std::string TmpName = pUnresolved->getName().getAsString();
            if (TmpName == "free") {
                pFuncDecl = (FunctionDecl *)(*(pUnresolved->decls_begin()));
                std::cout << "getCalleeFuncPtr: 获得未解析的 free 函数的定义地址" << std::endl;
            }
        }
    }

    if (pFuncDecl == NULL) {
        std::cout << "getCalleeFuncPtr 警告: 未能获得被调函数定义地址" << std::endl;
        // exit(1);
    }

    return pFuncDecl;
}

// wfr 20190725 推断 while/for循环 整体的 入口状态需求 和 等效的状态转换函数
int  OAOASTVisitor::InferLoopInterfaceState(STATE_CONSTR& LoopStReq, STATE_CONSTR& LoopStTrans, 
    bool& LoopNeedAnalysis, NODE_INDEX indexLoopEntry, int indexVar, FUNC_INFO& Func, 
    bool& outStTransHostWrite, bool& outStTransDeviceWrite,
    bool& outStReqHostNew, bool& outStReqDeviceNew)
{
    DEFINE_ST_REQ_ST_TRANS
    SEQ_PAR_NODE_BASE *pBaseLoopEntry = getNodeBasePtr(Func, indexLoopEntry);
    SEQ_REGION *pSEQLoopEntry = NULL;
    OMP_REGION *pOMPLoopEntry = NULL;
    if (indexLoopEntry.type == NODE_TYPE::SEQUENTIAL) {
        pSEQLoopEntry = (SEQ_REGION *)pBaseLoopEntry;
    } else {
        pOMPLoopEntry = (OMP_REGION *)pBaseLoopEntry;
    }
    std::vector<VARIABLE_INFO> &vectorVar = Func.vectorVar;
    std::vector<SEQ_REGION> &vectorSEQ = Func.vectorSEQ;
    std::vector<OMP_REGION> &vectorOMP = Func.vectorOMP;
    VARIABLE_INFO &Var = vectorVar[indexVar]; // 当前要处理的变量

    // 需要一个栈来保存先深路径, 栈中每个项需要一个2元素数组保存路径上的同步状态
    // wfr 20190322 这里要保存的是路径到目前为止的叠加状态, 不仅仅是当前节点的状态
    class DEPTH_NODE : public NODE_INDEX {
    public:
        int Next; // 下一个要搜索的子节点
        // 表示一条路径上 累积/叠加的 入口状态需求 和 出口状态约束
        STATE_CONSTR AccumEnStReq, AccumExStConstr;
        // 表示 AccumEnStReq 是否完成了叠加
        // 遇到写操作 或 AccumEnStReq==StReqDiverg/StReqHostOnly/StTransSync
        // 则认为在当前路径上 AccumEnStReq 完成了叠加
        bool AccumEnStReqDone;

        DEPTH_NODE(const NODE_INDEX &in) : NODE_INDEX(in) {
            Next = 0;
            AccumEnStReqDone = false;
        }

        bool operator==(const NODE_INDEX &in) {
            if (in.type == type && in.index == index) {
                return true;
            }
            return false;
        }
    };
    std::vector<DEPTH_NODE> vectorRoute;

    LoopStReq = StReqDiverg; // 循环整体 的 入口同步状态需求
    LoopStTrans = StTransDiverg; // 循环整体 的 等效状态转换函数
    // LoopNeedAnalysis = false; // 表示循环内部是否需要分析

    bool HasStTransHostWrite = false;
    bool HasStTransDeviceWrite = false;
    bool HasStReqHostNew = false;
    bool HasStReqDeviceNew = false;

    STATE_CONSTR BodyStReq = StReqNone; // 循环体(即"{}"内的部分) 的 入口同步状态需求
    STATE_CONSTR BodyStTrans = StTransNone; // 循环体(即"{}"内的部分) 的 等效状态转换函数

    STATE_CONSTR LoopHeadStReq = StReqNone; // 循环头节点 的 入口同步状态需求
    STATE_CONSTR LoopHeadStTrans = StTransNone; // 循环头节点 的 等效状态转换函数

    // 0. 分析 循环头节点, 得到 LoopHeadStReq, LoopHeadStTrans
    std::vector<VAR_REF_LIST>::iterator iterVarRef;
    iterVarRef = find(pBaseLoopEntry->vectorVarRef.begin(), pBaseLoopEntry->vectorVarRef.end(), indexVar);
    if (iterVarRef != pBaseLoopEntry->vectorVarRef.end()) {
        CheckStReq(iterVarRef->StReq);
        CheckStTrans(iterVarRef->StTransFunc);
        LoopHeadStReq = iterVarRef->StReq;
        LoopHeadStTrans = iterVarRef->StTransFunc;
    }

    // 1. 对循环体(即"{}"内的部分) 进行保守分析, 得到 BodyStReq, BodyStTrans
    if(pBaseLoopEntry->vectorChildren.size()!=2){
        std::cout << "InferLoopInterfaceState 错误: 循环头节点的子节点数量 != 2" << std::endl;
        exit(1);
    }

    NODE_INDEX indexBodyFirst = pBaseLoopEntry->LoopBody; // 循环体中的第一个节点的 NODE_INDEX
    SEQ_PAR_NODE_BASE* pBodyFirstBase = getNodeBasePtr(Func, indexBodyFirst);
    NODE_INDEX indexBodyExit; // 循环体出口节点的 index
    SEQ_PAR_NODE_BASE* pBodyExitBase; // 循环体出口节点的 指针

    // 始化先深栈
    vectorRoute.clear();
    // 先推断入口同步状态
    vectorRoute.emplace_back(indexBodyFirst); // 入口节点入队

    // 初始化 BodyStReq 和 BodyStTrans
    // wfr 20190801 遇到嵌套的子循环, 递归调用处理
    if(pBodyFirstBase->LoopBody.index>=0){
        InferLoopInterfaceState(vectorRoute[0].AccumEnStReq, vectorRoute[0].AccumExStConstr, LoopNeedAnalysis, indexBodyFirst, indexVar, Func,
        HasStTransHostWrite, HasStTransDeviceWrite, HasStReqHostNew, HasStReqDeviceNew);
        // wfr 20190817
        // BodyStReq = vectorRoute[0].AccumEnStReq;
        // BodyStTrans = vectorRoute[0].AccumExStConstr;
    }else{
        iterVarRef = find(pBodyFirstBase->vectorVarRef.begin(), pBodyFirstBase->vectorVarRef.end(), indexVar);
        if (iterVarRef != pBodyFirstBase->vectorVarRef.end()) {
            STATE_CONSTR TmpStReq = iterVarRef->StReq;
            STATE_CONSTR TmpStTrans = iterVarRef->StTransFunc;
            CheckStReq(TmpStReq);
            CheckStTrans(TmpStTrans);
            vectorRoute[0].AccumEnStReq = TmpStReq;
            vectorRoute[0].AccumExStConstr = ExeStTrans(TmpStTrans, TmpStReq);
            // wfr 20191223 判断是否有 host/device 的读/写操作
            if((TmpStReq.ONE & StReqHostNew.ONE) > 0){
                HasStReqHostNew = true;
            }
            if((TmpStReq.ONE & StReqDeviceNew.ONE) > 0){
                HasStReqDeviceNew = true;
            }
            if((TmpStTrans.ONE & StTransHostWrite.ONE) > 0){
                HasStTransHostWrite = true;
            }
            if((TmpStTrans.ONE & StTransDeviceWrite.ONE) > 0){
                HasStTransDeviceWrite = true;
            }
            // wfr 20190817
            // BodyStReq = vectorRoute[0].AccumEnStReq;
            // BodyStTrans = vectorRoute[0].AccumExStConstr;
            if(TmpStTrans.ONE>0){ // 遇到写操作
                vectorRoute[0].AccumEnStReqDone = true;
            }else if(TmpStReq==StReqSync || TmpStReq==StReqDiverg || TmpStReq==StReqHostOnly){
                vectorRoute[0].AccumEnStReqDone = true;
            }
        }else{
            vectorRoute[0].AccumEnStReq = StReqNone;
            vectorRoute[0].AccumExStConstr = StReqNone;
        }
    }

    while (!vectorRoute.empty()) { // 先深栈非空则进入循环, 处理栈顶的节点
        // 如果 LoopBody 的 BodyStReq 和 BodyStTrans 已经出现分歧, 则不用再继续搜索, 直接跳出循环即可
        if (BodyStReq == StReqDiverg && BodyStTrans==StTransDiverg) {
            vectorRoute.clear();
            break;
        }
        int indexParent;
        NODE_INDEX indexParentNode;
        indexParent = vectorRoute.size() - 1; // vectorRoute 中最后一个节点的 index
        indexParentNode = vectorRoute[indexParent]; // 最后一个节点的 NODE_INDEX
        SEQ_PAR_NODE_BASE* pParentBase = getNodeBasePtr(Func, indexParentNode);    // 最后一个节点的 指针

        if ((unsigned long)vectorRoute[indexParent].Next < pParentBase->vectorChildren.size()) {
            int indexChild;
            NODE_INDEX indexChildNode;
            indexChildNode = pParentBase->vectorChildren[vectorRoute[indexParent].Next]; // 下一个节点的 NODE_INDEX
            vectorRoute[indexParent].Next++;                                           // 自增指向下一个子节点
            if(indexChildNode == pParentBase->LoopBody){ // wfr 20190726 不处理循环体, 因为在这之前, 下边已经递归调用当前函数整体处理了嵌套的循环
                continue;
            }
            SEQ_PAR_NODE_BASE* pChildBase = getNodeBasePtr(Func, indexChildNode); // 最后子节点的 指针

            if (!pChildBase) {
                std::cout << "InferLoopInterfaceState 错误: 节点没有初始化" << std::endl;
                exit(1);
            }

            // 到了当前循环的出口, 不处理当前循环的头节点
            // wfr 20190725 要在这里更新 BodyStReq 和 BodyStTrans
            if(indexChildNode==indexLoopEntry){
                
                // wfr 20190725 要在这里更新 BodyStReq
                if(BodyStReq==StReqNone){ // wfr 20190817
                    BodyStReq = vectorRoute[indexParent].AccumEnStReq;
                }else if(BodyStReq!=StReqDiverg){
                    if( vectorRoute[indexParent].AccumEnStReq==StReqDiverg 
                        || vectorRoute[indexParent].AccumEnStReq==StReqHostOnly 
                        || vectorRoute[indexParent].AccumEnStReq==StReqSync )
                    {
                        if(BodyStReq!=vectorRoute[indexParent].AccumEnStReq){
                            BodyStReq = StReqDiverg;
                        }
                    }

                    BodyStReq.ZERO &= vectorRoute[indexParent].AccumEnStReq.ZERO;
                    BodyStReq.ONE |= vectorRoute[indexParent].AccumEnStReq.ONE;
                }

                // wfr 20190725 要在这里更新 BodyStTrans
                if(BodyStTrans==StReqNone){
                    BodyStTrans=vectorRoute[indexParent].AccumExStConstr;
                }else if(vectorRoute[indexParent].AccumExStConstr == StTransNone
                        || vectorRoute[indexParent].AccumExStConstr == StTransUninit){
                    // wfr 20191218 这种情况 不更新 BodyStTrans
                }else if(BodyStTrans!=vectorRoute[indexParent].AccumExStConstr){
                    BodyStTrans=StTransDiverg;
                }

                // wfr 20190731 写入循环出口节点的出口状态约束
                pParentBase->ExitStConstr = vectorRoute[indexParent].AccumExStConstr;
                
                continue;
            }
            // ghn 20191109 循环外的子节点，加一个bool判断
            bool flag = false;
            // wfr 20190512 这里判断子节点是否在循环体中, 即不处理循环外的节点
            if (indexChildNode.type == NODE_TYPE::SEQUENTIAL) {
                SEQ_REGION *pSEQChild = (SEQ_REGION *)pChildBase;
                if (pSEQChild->SEQRange.EndOffset < pBaseLoopEntry->LoopBodyRange.BeginOffset ||
                    pBaseLoopEntry->LoopBodyRange.EndOffset < pSEQChild->SEQRange.BeginOffset) {
                // if (pSEQChild->SEQRange.BeginOffset < pBaseLoopEntry->LoopBodyRange.BeginOffset ||
                //     pBaseLoopEntry->LoopBodyRange.EndOffset < pSEQChild->SEQRange.EndOffset) {
                    std::cout << "InferLoopInterfaceState 警告: 遇到循环外的节点" << std::endl;
                    //exit(1);
                }else{
                    flag = true;
                }
            }
            if (indexChildNode.type == NODE_TYPE::PARALLEL) {
                OMP_REGION *pOMPChild = (OMP_REGION *)pChildBase;
                if (pOMPChild->OMPRange.EndOffset < pBaseLoopEntry->LoopBodyRange.BeginOffset ||
                    pBaseLoopEntry->LoopBodyRange.EndOffset < pOMPChild->DirectorRange.BeginOffset) {
                // if (pOMPChild->DirectorRange.BeginOffset < pBaseLoopEntry->LoopBodyRange.BeginOffset ||
                //     pBaseLoopEntry->LoopBodyRange.EndOffset < pOMPChild->OMPRange.EndOffset) {
                    std::cout << "InferLoopInterfaceState 警告: 遇到循环外的节点" << std::endl;
                    //exit(1);
                }else{
                    flag = true;
                }
            }

            STATE_CONSTR ChildStReq(StReqUninit);
            STATE_CONSTR ChildStTrans(StTransUninit);

            if(flag == true){
                // wfr 20190726 遇到嵌套的子循环, 递归调用处理
                if(pChildBase->LoopBody.index>=0){
                    InferLoopInterfaceState(ChildStReq, ChildStTrans, LoopNeedAnalysis, indexChildNode, indexVar, Func,
                    HasStTransHostWrite, HasStTransDeviceWrite, HasStReqHostNew, HasStReqDeviceNew);
                }

                vectorRoute.emplace_back(indexChildNode); // 子节点入队
                indexChild = vectorRoute.size() - 1;      // vectorRoute 中最后一个节点的 index

                if(ChildStReq==StReqUninit){
                    std::vector<VAR_REF_LIST>::iterator iterVarRef;
                    iterVarRef = find(pChildBase->vectorVarRef.begin(), pChildBase->vectorVarRef.end(), indexVar);
                    if (iterVarRef != pChildBase->vectorVarRef.end()) {
                        ChildStReq = iterVarRef->StReq;
                        ChildStTrans = iterVarRef->StTransFunc;
                        CheckStReq(ChildStReq);
                        CheckStTrans(ChildStTrans);
                    }else{
                        ChildStReq = StReqNone;
                        ChildStTrans = StTransNone;
                    }
                }

                // wfr 20191223 判断是否有 host/device 的读/写操作
                if((ChildStReq.ONE & StReqHostNew.ONE) > 0){
                    HasStReqHostNew = true;
                }
                if((ChildStReq.ONE & StReqDeviceNew.ONE) > 0){
                    HasStReqDeviceNew = true;
                }
                if((ChildStTrans.ONE & StTransHostWrite.ONE) > 0){
                    HasStTransHostWrite = true;
                }
                if((ChildStTrans.ONE & StTransDeviceWrite.ONE) > 0){
                    HasStTransDeviceWrite = true;
                }

                // wfr 20190725 先分析 AccumEnStReq
                // wfr 20190725 如果 当前路径的 AccumEnStReq 没分析完成
                if(vectorRoute[indexParent].AccumEnStReqDone==false){
                    if(ChildStReq==StReqSync || ChildStReq==StReqDiverg || ChildStReq==StReqHostOnly){
                        vectorRoute[indexChild].AccumEnStReq = ChildStReq;
                        vectorRoute[indexChild].AccumEnStReqDone = true;
                    }else{
                        vectorRoute[indexChild].AccumEnStReq.ZERO = vectorRoute[indexParent].AccumEnStReq.ZERO & ChildStReq.ZERO;
                        vectorRoute[indexChild].AccumEnStReq.ONE = vectorRoute[indexParent].AccumEnStReq.ONE | ChildStReq.ONE;
                    }

                    // wfr 20190725 如果是 StReqSync 或 有写操作
                    if(vectorRoute[indexChild].AccumEnStReq==StReqSync || ChildStTrans.ONE>0){
                        vectorRoute[indexChild].AccumEnStReqDone = true;
                    }
                }else{ // wfr 20190725 当前路径已经分析完成, 不再分析, 直接继承父节点信息
                    vectorRoute[indexChild].AccumEnStReqDone = true;
                    vectorRoute[indexChild].AccumEnStReq = vectorRoute[indexParent].AccumEnStReq;
                }

                // wfr 20190725 再分析 AccumExStConstr
                vectorRoute[indexChild].AccumExStConstr = ExeStTrans(ChildStReq, vectorRoute[indexParent].AccumExStConstr);
                vectorRoute[indexChild].AccumExStConstr = ExeStTrans(ChildStTrans, vectorRoute[indexChild].AccumExStConstr);
            }
        } else { // 说明到了出口节点, 进行回退
            vectorRoute.pop_back(); // 出栈当前子节点
        }
    }

    // wfr 20191223 有 host读device写 或 device读host写, 则需要进入循环分析, 需要在循环中插入 运行时API
    if(HasStReqHostNew==true && HasStTransDeviceWrite==true){
        LoopNeedAnalysis = true;
    }
    if(HasStReqDeviceNew==true && HasStTransHostWrite==true){
        LoopNeedAnalysis = true;
    }
    
    // wfr 20200108 更新参数 将 是否有以下四种 读/写 操作 的 flag 传出去
    outStReqHostNew |= HasStReqHostNew;
    outStReqDeviceNew |= HasStReqDeviceNew;
    outStTransHostWrite |= HasStTransHostWrite;
    outStTransDeviceWrite |= HasStTransDeviceWrite;

    // 2. 协同分析 得到 LoopStReq, LoopStTrans
    bool flag = true; // 标识变量, 表示是否出错, 没错就继续处理
    bool LoopStReqflag = false; // wfr 20190726 表示是否能得到协同的 LoopStReq
    bool LoopStTransflag = false; // wfr 20190726 表示是否能得到协同的 LoopStTrans
    LoopStReq = ReqIntersection(LoopHeadStReq, BodyStReq);
    if(LoopStReq!=StReqUninit){
        flag = true;
    }

    STATE_CONSTR HeadExConstr;
    if(flag==true){
        // wfr 20191218 先判断不是 StReqDiverg
        if(LoopHeadStTrans != StReqDiverg){
            HeadExConstr = ExeStTrans(LoopHeadStTrans, LoopStReq);
            flag = CheckConstrSatisfaction(HeadExConstr, BodyStReq);
        }else{
            flag = false;
        }
    }

    STATE_CONSTR BodyExConstr;
    if(flag==true){
        // wfr 20191218 先判断不是 StReqDiverg
        if(BodyStTrans != StTransDiverg){
            BodyExConstr = ExeStTrans(BodyStTrans, HeadExConstr);
            flag = CheckConstrSatisfaction(BodyExConstr, LoopStReq);
        }else{
            flag = false;
        }
        if(flag==true){
            LoopStReqflag = true;
        }else{
            LoopStReq = LoopHeadStReq;
        }
    }

    STATE_CONSTR HeadExConstr0;
    STATE_CONSTR HeadExConstr1;
    if(flag==true){
        // wfr 20190726 使用 BodyExConstr的转换结果 和 HeadExConstr 协商
        HeadExConstr0 = HeadExConstr;
        HeadExConstr1 = ExeStTrans(LoopHeadStTrans, BodyExConstr);
    }else{
        HeadExConstr0 = ExeStTrans(LoopHeadStTrans, LoopHeadStReq);
        HeadExConstr1 = ExeStTrans(BodyStTrans, BodyStReq);
        HeadExConstr1 = ExeStTrans(LoopHeadStTrans, HeadExConstr1);
    }
    
    // wfr 20190727 这里可能有问题
    if(LoopHeadStTrans==StTransNone && HasStTransHostWrite==false && HasStTransDeviceWrite==false){
        LoopStTrans = StTransNone;
        LoopStTransflag = true;
    }else if(LoopHeadStReq==StReqNone && LoopHeadStTrans==StTransNone){
        LoopStTrans = HeadExConstr1;
        LoopStTransflag = true;
    }else{
        LoopStTrans = HeadExConstr0;
        LoopStTransflag = true;
    }

    if(LoopNeedAnalysis==false){
        if(LoopStReqflag!=true || LoopStTransflag!=true){
            LoopNeedAnalysis=true;
        }
    }

    return 0;
}

// wfr 20190415 遍历 vectorSEQ, 找到某个 OMP外部变量 定义在哪个 SEQ 中
NODE_INDEX OAOASTVisitor::FindDefNode(const FUNC_INFO &Func, const VARIABLE_INFO &Var) {
    NODE_INDEX indexSEQ(NODE_TYPE::SEQUENTIAL, -1);
    unsigned int DefOffset = Var.EndOffset;

    for (unsigned long i = 0; i < Func.vectorSEQ.size(); ++i) {
        if (Func.vectorSEQ[i].SEQRange.BeginOffset <= DefOffset && DefOffset <= Func.vectorSEQ[i].SEQRange.EndOffset) {
            indexSEQ.index = i;
            return indexSEQ;
        }
    }

    return indexSEQ;
}

// wfr 20190409 获得一个 Scope 的 开始节点/结束节点 的 NODE_INDEX
void OAOASTVisitor::ScopeBeginEndNode(NODE_INDEX &indexOut, int indexFunc, MY_SOURCE_RANGE Scope,
                                      SCOPE_BEGIN_END flag) {

    // 与 Scope 开始/结束位置 的距离, 如果 开始/结束位置 不是正好在一个节点内部,
    // 那么就比较距离, 距离最小则认为是 开始/结束节点
    unsigned long delta = Scope.EndOffset - Scope.BeginOffset;

    if (flag == SCOPE_BEGIN_END::END) { // 如果要找 Scope 的 结束节点
        for (int i = 0; i < (int)vectorFunc[indexFunc].vectorSEQ.size(); i++) {
            if (vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset <= Scope.EndOffset &&
                Scope.EndOffset <= vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset) {
                indexOut.init(NODE_TYPE::SEQUENTIAL, i);
                return;
            } else if (Scope.BeginOffset < vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset &&
                       vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset < Scope.EndOffset &&
                       delta > Scope.EndOffset - vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset) {
                delta = Scope.EndOffset - vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset;
                indexOut.init(NODE_TYPE::SEQUENTIAL, i);
            }
        }

        for (int i = 0; i < (int)vectorFunc[indexFunc].vectorOMP.size(); i++) {
            if (vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset <= Scope.EndOffset &&
                Scope.EndOffset <= vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset) {
                indexOut.init(NODE_TYPE::PARALLEL, i);
                return;
            } else if (Scope.BeginOffset < vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset &&
                       vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset < Scope.EndOffset &&
                       delta > Scope.EndOffset - vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset) {
                delta = Scope.EndOffset - vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset;
                indexOut.init(NODE_TYPE::PARALLEL, i);
            }
        }
    } else { // 如果要找 Scope 的 开始节点
        for (int i = 0; i < (int)vectorFunc[indexFunc].vectorSEQ.size(); i++) {
            if (vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset <= Scope.BeginOffset &&
                Scope.BeginOffset <= vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset) {
                indexOut.init(NODE_TYPE::SEQUENTIAL, i);
                return;
            } else if (Scope.BeginOffset < vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset &&
                       vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset < Scope.EndOffset &&
                       delta > vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset - Scope.BeginOffset) {
                delta = vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset - Scope.BeginOffset;
                indexOut.init(NODE_TYPE::SEQUENTIAL, i);
            }
        }

        for (int i = 0; i < (int)vectorFunc[indexFunc].vectorOMP.size(); i++) {
            if (vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset <= Scope.BeginOffset &&
                Scope.BeginOffset <= vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset) {
                indexOut.init(NODE_TYPE::PARALLEL, i);
                return;
            } else if (Scope.BeginOffset < vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset &&
                       vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset < Scope.EndOffset &&
                       delta > vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset - Scope.BeginOffset) {
                delta = vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset - Scope.BeginOffset;
                indexOut.init(NODE_TYPE::PARALLEL, i);
            }
        }
    }

    return;
}

// wfr 20190330 初始化 VARIABLE_INFO
void OAOASTVisitor::InitVar(VARIABLE_INFO &Var, VarDecl *ptr, int indexFunc,
                            NODE_INDEX indexDefNode) { // : MY_SOURCE_RANGE(ptr, Rewrite)
    Var.Name = ptr->getNameAsString();
    Var.ptrDecl = (Decl *)ptr; // 变量的定义的地址
    // Var.pDeclStmt = pDeclStmtStack.back(); // 保存 插入指针区域长度语句 位置 的信息
    if (isa<ParmVarDecl>(ptr)) {
        SourceRange DeclRange = ((ParmVarDecl *)ptr)->getSourceRange();
        Var.DeclRange.init(DeclRange.getBegin(), DeclRange.getEnd().getLocWithOffset(Var.Name.size()), Rewrite);
    } else {
        Var.DeclRange.init(DeclStmtStack.back().BeginLoc, DeclStmtStack.back().EndLoc, Rewrite);
    }

    // 获得变量类型
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    Var.TypeName = QualType::getAsString(ptr->getType().split(),
                                         PrintPolicy); // 变量的类型：int、int*、float*等等

    Var.Type = VAR_TYPE::VAR_UNINIT;
    if (Var.TypeName.back() == '&') { // 引用 REF
        Var.Type = VAR_TYPE::REF;
    }
    if (Var.TypeName.back() == '*') {
        if (Var.TypeName.substr(0, 5) == "const") { // 指针指向的区域只读 PTR_CONST
            Var.Type = VAR_TYPE::PTR_CONST;
        } else { // 指针 PTR
            Var.Type = VAR_TYPE::PTR;
        }
    }
    if (Var.TypeName.substr(Var.TypeName.size() - 6, 6) == "*const") {
        if (Var.TypeName.substr(0, 5) == "const") { // 指针指向的区域只读 PTR_CONST
            Var.Type = VAR_TYPE::PTR_CONST;
        } else { // 指针 PTR
            Var.Type = VAR_TYPE::PTR;
        }
    } else {                       // 认为是拷贝 COPY
        Var.Type = VAR_TYPE::COPY; // 所有变量都处理, 因为可能出现
                                   // 变量取地址作为函数参数的情况
    }

    Var.indexDefNode = indexDefNode;
    Var.Scope = CompoundStack.back();

    Var.isClass = false;
    Var.isMember = false;
    Var.isArrow = 0;
    Var.ptrClass = NULL;
    Var.indexClass = -1;
    Var.RootName = "NULL";
    Var.indexRoot = -1;
    Var.ptrRoot = NULL;
    Var.UsedInOMP = false;
    Var.Rename = Var.Name;   // 写入变量全名称
    Var.FullName = Var.Name; // 写入变量全名称
    Var.ArrayLen = "NULL";
    // ArrayLenCalculate = "NULL";
    // isMapedWhileRewriting = false;

    if (Var.TypeName.back() == '*') {
        Var.ArrayLen = Var.Name; // 写入变量长度变量的名称
        Var.ArrayLen += "_LEN_"; // 写入变量长度变量的名称
        if (isa<ParmVarDecl>(ptr)) {
            // Var.vectorInsert.emplace_back(Var.DeclRange.EndLoc, ", int ", Rewrite);
            // Var.vectorInsert.back().Code += Var.ArrayLen;
        } else {
            // Var.vectorInsert.emplace_back(Var.DeclRange.EndLoc, "; int ", Rewrite);
            // Var.vectorInsert.back().Code += Var.ArrayLen;

            // 如果在声明的同时赋值了, 进行指针跟踪, 并更新指针区域大小
            // 如果不在则说明我对 TraverseVarDecl 中逻辑理解有误
            if (ASTStack.back().Name == "VarDecl" && ASTStack.back().OperandState[0] == true) {

                if (ASTStack.back().Operand[0] >= 0) {               // 如果指针定义时被使用其他指针赋值
                    Var.indexRoot = (int)ASTStack.back().Operand[0]; // indexRoot
                                                                     // 指向用来给当前变量赋值的变量
                    Var.RootName = vectorFunc[indexFunc].vectorVar[Var.indexRoot].Name;
                    Var.ptrRoot = vectorFunc[indexFunc].vectorVar[Var.indexRoot].ptrDecl;

                    // 更新指针指向的区域的大小
                    // Var.vectorInsert.back().Code += " = ";
                    // Var.vectorInsert.back().Code += vectorFunc[indexFunc].vectorVar[Var.indexRoot].ArrayLen;
                } else if (ASTStack.back().vectorInfo[0] == "malloc") { // 如果指针定义时指向 malloc 的空间
                    // 更新指针指向的区域的大小
                    // Var.vectorInsert.back().Code += " = malloc_usable_size( ";
                    // Var.vectorInsert.back().Code += Var.FullName;
                    // Var.vectorInsert.back().Code += " ) / sizeof( ";
                    // Var.vectorInsert.back().Code += Var.TypeName;
                    // Var.vectorInsert.back().Code.pop_back(); // 删掉最后的 “ * ”
                    // Var.vectorInsert.back().Code += ")";     // 至此写完获得数组长度的语句
                } else {
                }
            }
        }
    } else if (Var.TypeName.back() == ']') {
        // 获得多维数组的元素总数, 写入 ArrayLen 域中
        Var.getArrayLen();
    } else {
    }
}

// wfr 20190222 在当前函数中分离出 OMP 节点, 被 TraverseOMPParallelForDirective
// 调用
NODE_INDEX OAOASTVisitor::SplitOMPNode(Stmt *S) {
    OMPParallelForDirective *pParallelFor = NULL;
    if (isa<OMPParallelForDirective>(S)) {
        pParallelFor = (OMPParallelForDirective *)S;
    } else {
        std::cout << "不能处理的OMP指令类型" << std::endl;
        exit(1);
    }
    
    int indexFuncCurrent = FuncStack.back().indexFunc; // 当前代码所在的函数
    FUNC_INFO &FuncCurrent = vectorFunc[indexFuncCurrent];
    std::vector<SEQ_REGION> &vectorSEQ = FuncCurrent.vectorSEQ;
    std::vector<OMP_REGION> &vectorOMP = FuncCurrent.vectorOMP;

    // 检查排除以下异常情况
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int BeginOffset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    // OMP 节点在哪个 OMP 节点中
    int TmpIndex = InWhichOMP(indexFuncCurrent, BeginOffset);
    if (TmpIndex >= 0) {
        std::cout << "SplitOMPNode 错误: 并行域中嵌套并行域, 不能处理" << std::endl;
        exit(1);
    }

    // 0. 新建 OMP 节点
    NODE_INDEX indexOMPNode; // OMP 节点的 index
    vectorOMP.emplace_back(pParallelFor, this->Rewrite); // 新建 OMP 节点
    OMP_REGION& OMP = vectorOMP.back();
    indexOMPNode.init(NODE_TYPE::PARALLEL, vectorOMP.size() - 1);
    OMP.indexEntry = indexOMPNode.index; // 新建的 OMP 节点是 offloading 域的头结点,
                                                                   // 写入头结点的index
    OMP.indexExit = indexOMPNode.index; // 新建的 OMP 节点是 offloading 域的头结点,
                                                                  // 写入出口结点的index
    bool NeedNextLoop = false;
    std::vector<NODE_INDEX>::iterator iterTmpIndex;
    NODE_INDEX indexFront;   // 拆分出的前面的节点的 index
    NODE_INDEX indexBack;    // 拆分出的后面的节点的 index

    // 1. 遍历 vectorSEQ, 清除被 OMP 包含的 SEQ, 对于与 OMP 部分重叠的 SEQ 修改 SEQ 的范围,
    //    对于包含 OMP 的 SEQ 拆分出 前后两个 SEQ; 遍历直到没有需要修改的 SEQ, 分 4 种情况处理
    // OMP 包含 SEQ
    // SEQ 包含 OMP
    // 部分重叠, SEQ->OMP
    // 部分重叠, OMP->SEQ
    for (int i = 0; i < (int)vectorSEQ.size(); ++i)
    {
        if(i == 0){ // 循环完一遍, 重置标志变量
            NeedNextLoop = false;
        }

        NODE_INDEX indexOrigin(NODE_TYPE::SEQUENTIAL, i);
        SEQ_REGION& SEQOrigin = vectorSEQ[i];

        // 按照 SEQ/OMP 节点的位置关系, 分 4 中情况处理
        // 0. OMP 包含 SEQ, SEQOrigin 合并到 OMP
        if (OMP.DirectorRange.BeginOffset <= SEQOrigin.SEQRange.BeginOffset &&
            SEQOrigin.SEQRange.EndOffset <= OMP.OMPRange.EndOffset)
        {
            NeedNextLoop = true;
            merge(indexOMPNode, indexOrigin, vectorSEQ, vectorOMP);

        // 1. SEQ 包含 OMP, 前后拆分出两个 SEQ 节点, 需要新建一个 SEQ 节点
        }else if (SEQOrigin.SEQRange.BeginOffset < OMP.DirectorRange.BeginOffset &&
                    OMP.OMPRange.EndOffset < SEQOrigin.SEQRange.EndOffset)
        {
            NeedNextLoop = true; // 设置标识, 需要再一次循环

            deleteEdge2Self(indexOrigin, vectorSEQ); // wfr 20191225 删除节点指向自身的边

            // 新建后半个节点, 设置后半个节点的范围, 以及 parents/children
            vectorSEQ.emplace_back();
            indexBack.init(NODE_TYPE::SEQUENTIAL, vectorSEQ.size() - 1);
            SEQ_REGION& SEQBack = vectorSEQ[indexBack.index];
            indexFront = indexOrigin;
            SEQ_REGION& SEQFront = vectorSEQ[indexFront.index];
            SEQBack.vectorParents.push_back(indexOMPNode);
            SEQBack.vectorChildren = SEQFront.vectorChildren;
            TmpLoc = SEQFront.SEQRange.EndLoc;
            SEQBack.SEQRange.SetEndLoc(TmpLoc, this->Rewrite);
            TmpLoc = OMP.OMPRange.EndLoc.getLocWithOffset(1);
            SEQBack.SEQRange.SetBeginLoc(TmpLoc, this->Rewrite);

            // 将子节点们链接到 后继 节点
            for (unsigned long i = 0; i < SEQBack.vectorChildren.size(); ++i) {
                SEQ_PAR_NODE_BASE *pNodeBase = getNodeBasePtr(FuncCurrent, SEQBack.vectorChildren[i]);
                std::vector<NODE_INDEX>& vectorParents = pNodeBase->vectorParents;
                std::vector<NODE_INDEX>::iterator iterNodeIndex;
                iterNodeIndex = find(vectorParents.begin(), vectorParents.end(), indexFront);
                if (iterNodeIndex != vectorParents.end()) {
                    *iterNodeIndex = indexBack;
                }
            }

            // 设置 OMP 节点的 parents/children
            // wfr 20191225 不重复才填入
            iterTmpIndex = find(OMP.vectorParents.begin(), OMP.vectorParents.end(), indexFront);
            if (iterTmpIndex == OMP.vectorParents.end()) {
                OMP.vectorParents.push_back(indexFront);
            }
            OMP.vectorChildren.push_back(indexBack);

            // 设置前半个节点的范围, 以及 children
            SEQFront.vectorChildren.clear(); // 先清除, 再重新写入
            SEQFront.vectorChildren.push_back(indexOMPNode);
            TmpLoc = OMP.DirectorRange.BeginLoc.getLocWithOffset(-1);
            SEQFront.SEQRange.SetEndLoc(TmpLoc, this->Rewrite);

            // 修补 SEQFront 和 SEQBack 节点的范围, 消除 空格/回车/换行/tab 等空白字符
            SEQFront.SEQRange.FixRange(4,Rewrite);
            SEQBack.SEQRange.FixRange(4,Rewrite);

        // 2. 部分重叠, SEQ->OMP, 前边拆分出一个 SEQ 节点, 不需要新建 SEQ 节点
        }else if (SEQOrigin.SEQRange.BeginOffset < OMP.DirectorRange.BeginOffset && 
                    OMP.DirectorRange.BeginOffset < SEQOrigin.SEQRange.EndOffset &&
                    SEQOrigin.SEQRange.EndOffset <= OMP.OMPRange.EndOffset)
        {
            NeedNextLoop = true;
            deleteEdge2Self(indexOrigin, vectorSEQ); // wfr 20191225 删除节点指向自身的边

            indexFront = indexOrigin;
            SEQ_REGION& SEQFront = vectorSEQ[indexFront.index];

            // 设置 OMP 节点的 parents
            // wfr 20191225 不重复才填入
            iterTmpIndex = find(OMP.vectorParents.begin(), OMP.vectorParents.end(), indexFront);
            if (iterTmpIndex == OMP.vectorParents.end()) {
                OMP.vectorParents.push_back(indexFront);
            }

            // 将子节点们链接到 OMP 节点
            for (unsigned long i = 0; i < SEQFront.vectorChildren.size(); ++i) {
                NODE_INDEX indexChild = SEQFront.vectorChildren[i];
                SEQ_REGION& SEQChild = vectorSEQ[indexChild.index];

                // 处理 SEQChild.vectorParents
                iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexOMPNode);
                if (iterTmpIndex != SEQChild.vectorParents.end()) { // 如果 存在 indexOMPNode->子节点 链接
                    iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexFront);
                    if (iterTmpIndex != SEQChild.vectorParents.end()) {
                        SEQChild.vectorParents.erase(iterTmpIndex); // 直接删除 indexFront->子节点 链接 即可
                    }
                } else {
                    iterTmpIndex = find(SEQChild.vectorParents.begin(), SEQChild.vectorParents.end(), indexFront);
                    if (iterTmpIndex != SEQChild.vectorParents.end()) {
                        *iterTmpIndex = indexOMPNode; // indexFront->子节点 链接 重置为 indexOMPNode->子节点
                    }
                }

                // 处理 OMP.vectorChildren
                iterTmpIndex = find(OMP.vectorChildren.begin(), OMP.vectorChildren.end(), indexChild);
                if (iterTmpIndex == OMP.vectorChildren.end() && indexChild != indexOMPNode) {  // 如果 子节点 没链接到 indexOMPNode
                    OMP.vectorChildren.push_back(indexChild); // 新建链接 indexOMPNode->子节点
                }
            }

            // 处理前驱节点的 范围 和 后向链路
            TmpLoc = OMP.DirectorRange.BeginLoc.getLocWithOffset(-1);
            SEQFront.SEQRange.SetEndLoc(TmpLoc, this->Rewrite);

            SEQFront.vectorChildren.clear();
            SEQFront.vectorChildren.push_back(indexOMPNode);

            // 修补 SEQFront 节点的范围, 消除 空格/回车/换行/tab 等空白字符
            SEQFront.SEQRange.FixRange(4,Rewrite);
            
        // 3. 部分重叠, OMP->SEQ, 后边拆分出一个 SEQ 节点, 不需要新建 SEQ 节点
        }else if (OMP.DirectorRange.BeginOffset <= SEQOrigin.SEQRange.BeginOffset &&
                    SEQOrigin.SEQRange.BeginOffset < OMP.OMPRange.EndOffset &&
                    OMP.OMPRange.EndOffset < SEQOrigin.SEQRange.EndOffset)
        {
            NeedNextLoop = true;
            deleteEdge2Self(indexOrigin, vectorSEQ); // wfr 20191225 删除节点指向自身的边

            indexBack = indexOrigin;
            SEQ_REGION& SEQBack = vectorSEQ[indexBack.index];

            // 设置 OMP 节点的 children
            // wfr 20191225 不重复才填入
            iterTmpIndex = find(OMP.vectorChildren.begin(), OMP.vectorChildren.end(), indexBack);
            if (iterTmpIndex == OMP.vectorChildren.end()) {
                OMP.vectorChildren.push_back(indexBack);
            }

            // 将父节点们链接到 OMP 节点
            for (unsigned long i = 0; i < SEQBack.vectorParents.size(); ++i) {
                NODE_INDEX indexParent = SEQBack.vectorParents[i];
                SEQ_REGION& SEQParent = vectorSEQ[indexParent.index];

                // 处理 SEQParent.vectorChildren
                iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexOMPNode);
                if (iterTmpIndex != SEQParent.vectorChildren.end()) { // 如果 父节点 已经链接到 indexOMPNode
                    iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexBack);
                    if (iterTmpIndex != SEQParent.vectorChildren.end()) {
                        SEQParent.vectorChildren.erase(iterTmpIndex); // 直接删除 父节点到 indexBack 的链接 即可
                    }
                } else {
                    iterTmpIndex = find(SEQParent.vectorChildren.begin(), SEQParent.vectorChildren.end(), indexBack);
                    if (iterTmpIndex != SEQParent.vectorChildren.end()) {
                        *iterTmpIndex = indexOMPNode; // 父节点->indexBack 链接 重置为 父节点->indexOMPNode
                    }
                }

                // 处理 OMP.vectorParents
                iterTmpIndex = find(OMP.vectorParents.begin(), OMP.vectorParents.end(), indexParent);
                if (iterTmpIndex == OMP.vectorParents.end() && indexParent != indexOMPNode) { // 如果 父节点 没链接到 indexOMPNode
                    OMP.vectorParents.push_back(indexParent); // 新建链接 父节点->indexOMPNode
                }
            }

            // 处理后继节点的 范围 和 前向链路
            TmpLoc = OMP.OMPRange.EndLoc.getLocWithOffset(1);
            SEQBack.SEQRange.SetBeginLoc(TmpLoc, this->Rewrite);

            SEQBack.vectorParents.clear();
            SEQBack.vectorParents.push_back(indexOMPNode);

            // 修补 SEQBack 节点的范围, 消除 空格/回车/换行/tab 等空白字符
            SEQBack.SEQRange.FixRange(4,Rewrite);

        }
        
        if (i == vectorSEQ.size()-1 && NeedNextLoop == true)
        {
            i = -1;
        }
    }

    // deleteEdge2Self(indexOMPNode, vectorOMP);

    std::cout << "SplitOMPNode: indexOMPNode =  " << std::dec << indexOMPNode.index << ", type = " << indexOMPNode.type
              << std::endl;
    std::cout << "SplitOMPNode: OMP BeginOffset = " << vectorOMP[indexOMPNode.index].DirectorRange.BeginOffset
              << ", EndOffset = " << vectorOMP[indexOMPNode.index].OMPRange.EndOffset << std::endl;

    return indexOMPNode;
}

// ghn 20200301 AST OMP节点中ForStmt的深度遍历
void OAOASTVisitor::DeepTraversal(Stmt* pParent, OMP_REGION& OMP){
    // ghn 20200228 判断是否有for循环，以及获取多层for循环的指针
    if(pParent != NULL){
//         if(isa<Stmt>(pParent)){
//             std::cout << "Stmt: " << pParent->getStmtClassName() << std::endl;
//         }else{
//             std::cout << "other: " << pParent << std::endl;
//         }

        if(pParent->child_begin() != pParent->child_end()){
            for (Stmt::child_iterator child = pParent->child_begin(); child != pParent->child_end(); child++) {
                if(*child != NULL && isa<ForStmt>(*child)){
                    std::cout << "DeepTraversal: " << (*child)->getStmtClassName() << std::endl;
                    OMP.forParallelStmt.push_back(*child);
                }
                DeepTraversal((Stmt*)*child, OMP);
            }
        }
    }
}

// ghn 20200303 ForStmt首个BinaryOperator的遍历，找到循环变量的DeclRef
void OAOASTVisitor::ForStmtFirstTraversal(Stmt* pBinary, std::vector<std::string>& declRefe){
    if(pBinary != NULL){
        if(pBinary->child_begin() != pBinary->child_end()){
            for(Stmt::child_iterator child = pBinary->child_begin(); child != pBinary->child_end(); child++) {
                if(*child != NULL && isa<DeclRefExpr>(*child)){
                    std::cout << "ForStmtFirstTraversal: " << (*child)->getStmtClassName() << std::endl;
                    DeclRefExpr* dee = (DeclRefExpr*)(*child);
                    NamedDecl *na = dee->getFoundDecl();
                    std::string conditionV = na->getNameAsString();
                    std::cout << "ForStmtFirstTraversal: 循环变量为" << conditionV << std::endl;
                    declRefe.push_back(conditionV);
                }
                ForStmtFirstTraversal((Stmt*)*child, declRefe);
            }
        }
    }
}
// wfr 20190214 保存函数定义和引用相关信息
// 该函数在 TraverseCallExpr 中被调用, 处理函数引用
// 拆分一个函数节点(顺序节点的特殊情况)
// 对于已经处理好的函数, 读取函数对于 参数的同步状态要求 以及 结束时同步状态,
// 标记函数节点为完成状态 对于 vectorFunc 中没有的函数, 新建 FUNC_INFO 项,
// 初步初始化 对于 vectorFunc 存在 但 没处理完成的 FUNC_INFO, 不做处理,
// 等待最后的 类似链接的阶段处理
NODE_INDEX OAOASTVisitor::SplitFuncNode(CallExpr *S) {

    FunctionDecl *pFuncDecl = getCalleeFuncPtr(S); // 获得被调函数定义地址
    std::string CalleeName = pFuncDecl->getNameAsString();

    NODE_INDEX indexFront;
    NODE_INDEX indexFuncNode;
    NODE_INDEX indexBack;

    std::cout << "SplitFuncNode: 进入" << std::endl;

    // FunctionDecl *pFuncDecl = dyn_cast<FunctionDecl>(S->getDecl());
    int indexFunc;                                 // 被调用的函数的 index
    if (pFuncDecl->getPrimaryTemplate() != NULL) { // 如果是函数模板的实例化 的 调用
        indexFunc = WhichFuncDecl(pFuncDecl->getPrimaryTemplate()->getTemplatedDecl());
    } else {
        indexFunc = WhichFuncDecl(pFuncDecl);
    }
    if (indexFunc < 0) {
        std::cout << "SplitFuncNode 警告: 被调函数未找到" << std::endl;
        return indexFuncNode;
        // exit(1);
    }

    std::cout << "SplitFuncNode: 分离函数调用 " << vectorFunc[indexFunc].Name << std::endl;

    int indexFuncCurrent = FuncStack.back().indexFunc; // 当前代码所在的函数
    FUNC_INFO &FuncCurrent = vectorFunc[indexFuncCurrent];
    std::vector<SEQ_REGION> &vectorSEQ = vectorFunc[indexFuncCurrent].vectorSEQ;
    std::vector<OMP_REGION> &vectorOMP = vectorFunc[indexFuncCurrent].vectorOMP;

    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int BeginOffset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    // 如果是串行节点中调用的函数
    if ((indexFront.index = InWhichSEQ(indexFuncCurrent, BeginOffset)) >= 0) {

        // 设置节点索引是 串行节点
        indexFront.type = NODE_TYPE::SEQUENTIAL;
        indexFuncNode.type = NODE_TYPE::SEQUENTIAL;
        indexBack.type = NODE_TYPE::SEQUENTIAL;

        // ASTStack.emplace_back((Stmt *)S, "CallExpr", -1,
        // vectorFunc[indexFunc].vectorParm.size()); 拆分 函数节点:
        // 一个串行节点从中间拆分出一个函数节点, 以及前半个节点和后半个节点,
        // 一共前后相继的三个节点 先 新建 函数SEQ节点
        vectorSEQ.emplace_back(S, S, this->Rewrite);
        indexFuncNode.index = vectorSEQ.size() - 1;
        vectorSEQ[indexFuncNode.index].indexFunc = indexFunc;
        vectorSEQ[indexFuncNode.index].pFuncCall = S;

        if (vectorSEQ[indexFront.index].SEQRange.BeginOffset < vectorSEQ[indexFuncNode.index].SEQRange.BeginOffset &&
            vectorSEQ[indexFuncNode.index].SEQRange.EndOffset <
              vectorSEQ[indexFront.index].SEQRange.EndOffset) { // 前驱 / 后继 节点 都有

            // 设置 函数节点 parents
            vectorSEQ[indexFuncNode.index].vectorParents.clear();
            vectorSEQ[indexFuncNode.index].vectorParents.push_back(indexFront);
            if (CalleeName == "free") {
                vectorSEQ[indexFuncNode.index].FuncName = "free";
            }

            // 新建后半个节点, 设置后半个节点的范围, 以及 parents/children
            vectorSEQ.emplace_back();
            indexBack.index = vectorSEQ.size() - 1;
            vectorSEQ[indexBack.index].vectorParents.clear();
            vectorSEQ[indexBack.index].vectorParents.push_back(indexFuncNode);
            vectorSEQ[indexBack.index].vectorChildren = vectorSEQ[indexFront.index].vectorChildren;
            vectorSEQ[indexBack.index].SEQRange.SetEndLoc(vectorSEQ[indexFront.index].SEQRange.EndLoc, this->Rewrite);
            TmpLoc = vectorSEQ[indexFuncNode.index].SEQRange.EndLoc.getLocWithOffset(1);
            vectorSEQ[indexBack.index].SEQRange.SetBeginLoc(TmpLoc, this->Rewrite);

            // 将子节点们链接到 后继 节点
            for (unsigned long i = 0; i < vectorSEQ[indexBack.index].vectorChildren.size(); ++i) {
                SEQ_PAR_NODE_BASE *pNodeBase =
                  getNodeBasePtr(FuncCurrent, vectorSEQ[indexBack.index].vectorChildren[i]);
                std::vector<NODE_INDEX> &vectorParents = pNodeBase->vectorParents;
                std::vector<NODE_INDEX>::iterator iterNodeIndex;
                iterNodeIndex = find(vectorParents.begin(), vectorParents.end(), indexFront);
                if (iterNodeIndex != vectorParents.end()) {
                    *iterNodeIndex = indexBack;
                }
            }

            // 设置 函数节点 children
            vectorSEQ[indexFuncNode.index].vectorChildren.clear();
            vectorSEQ[indexFuncNode.index].vectorChildren.push_back(indexBack);

            // 设置前半个节点的范围, 以及 children
            vectorSEQ[indexFront.index].vectorChildren.clear(); // 先清除, 再重新写入
            vectorSEQ[indexFront.index].vectorChildren.emplace_back(indexFuncNode);
            TmpLoc = vectorSEQ[indexFuncNode.index].SEQRange.BeginLoc.getLocWithOffset(-1);
            vectorSEQ[indexFront.index].SEQRange.SetEndLoc(TmpLoc, this->Rewrite);

        } else if (vectorSEQ[indexFront.index].SEQRange.BeginOffset ==
                     vectorSEQ[indexFuncNode.index].SEQRange.BeginOffset &&
                   vectorSEQ[indexFuncNode.index].SEQRange.EndOffset <
                     vectorSEQ[indexFront.index].SEQRange.EndOffset) { // 没有前驱节点

            indexBack = indexFront;

            // 设置 函数调用 节点的 parents/children
            vectorSEQ[indexFuncNode.index].vectorParents = vectorSEQ[indexFront.index].vectorParents;
            vectorSEQ[indexFuncNode.index].vectorChildren.clear();
            vectorSEQ[indexFuncNode.index].vectorChildren.push_back(indexBack);

            // 将父节点们链接到 函数调用 节点
            for (unsigned long i = 0; i < vectorSEQ[indexFuncNode.index].vectorParents.size(); ++i) {
                SEQ_PAR_NODE_BASE *pNodeBase =
                  getNodeBasePtr(FuncCurrent, vectorSEQ[indexFuncNode.index].vectorParents[i]);
                std::vector<NODE_INDEX> &vectorChildren = pNodeBase->vectorChildren;
                std::vector<NODE_INDEX>::iterator iterNodeIndex;
                iterNodeIndex = find(vectorChildren.begin(), vectorChildren.end(), indexFront);
                if (iterNodeIndex != vectorChildren.end()) {
                    *iterNodeIndex = indexFuncNode;
                }
            }

            // 处理后继节点的 范围 和 前向链路
            TmpLoc = vectorSEQ[indexFuncNode.index].SEQRange.EndLoc.getLocWithOffset(1);
            vectorSEQ[indexBack.index].SEQRange.SetBeginLoc(TmpLoc, this->Rewrite);

            vectorSEQ[indexBack.index].vectorParents.clear();
            vectorSEQ[indexBack.index].vectorParents.push_back(indexFuncNode);

        } else if (vectorSEQ[indexFront.index].SEQRange.BeginOffset <
                     vectorSEQ[indexFuncNode.index].SEQRange.BeginOffset &&
                   vectorSEQ[indexFuncNode.index].SEQRange.EndOffset ==
                     vectorSEQ[indexFront.index].SEQRange.EndOffset) { // 没有后继节点

            indexBack = indexFuncNode;

            // 设置 函数调用 节点的 parents/children
            vectorSEQ[indexFuncNode.index].vectorParents.clear();
            vectorSEQ[indexFuncNode.index].vectorParents.push_back(indexFront);
            vectorSEQ[indexFuncNode.index].vectorChildren = vectorSEQ[indexFront.index].vectorChildren;

            // 将子节点们链接到 函数调用 节点
            for (unsigned long i = 0; i < vectorSEQ[indexFuncNode.index].vectorChildren.size(); ++i) {
                SEQ_PAR_NODE_BASE *pNodeBase =
                  getNodeBasePtr(FuncCurrent, vectorSEQ[indexFuncNode.index].vectorChildren[i]);
                std::vector<NODE_INDEX> &vectorParents = pNodeBase->vectorParents;
                std::vector<NODE_INDEX>::iterator iterNodeIndex;
                iterNodeIndex = find(vectorParents.begin(), vectorParents.end(), indexFront);
                if (iterNodeIndex != vectorParents.end()) {
                    *iterNodeIndex = indexFuncNode;
                }
            }

            // 处理前驱节点的 范围 和 后向链路
            TmpLoc = vectorSEQ[indexFuncNode.index].SEQRange.BeginLoc.getLocWithOffset(-1);
            vectorSEQ[indexFront.index].SEQRange.SetEndLoc(TmpLoc, this->Rewrite);

            vectorSEQ[indexFront.index].vectorChildren.clear();
            vectorSEQ[indexFront.index].vectorChildren.push_back(indexFuncNode);

        } else if (vectorSEQ[indexFront.index].SEQRange.BeginOffset ==
                     vectorSEQ[indexFuncNode.index].SEQRange.BeginOffset &&
                   vectorSEQ[indexFuncNode.index].SEQRange.EndOffset ==
                     vectorSEQ[indexFront.index].SEQRange.EndOffset) { // 前驱 / 后继节点 都没有

            // 这里 前驱节点(indexFront) 中只有一条函数调用语句, 不用新建节点, 在
            // 前驱节点 中存入 函数调用信息即可
            vectorSEQ[indexFront.index].indexFunc = indexFunc;
            vectorSEQ[indexFront.index].pFuncCall = S;

            // 同时移除刚刚新建的 函数调用节点(indexFuncNode)
            vectorSEQ.erase(vectorSEQ.begin() + indexFuncNode.index);

            indexFuncNode = indexFront;
            indexBack = indexFront;

        } else {
            std::cout << "SplitFuncNode 错误: CallExpr 语句 与 SEQ 代码块 "
                         "之间的位置关系有问题"
                      << std::endl;
            exit(1);
        }

        // wfr 20191223 这里分析并存入函数调用所在行的 起始和结束位置, 以便之后插入运行时API
        SourceLocation EmptyLoc;
        SourceLocation StmtBeginLoc;
        SourceLocation StmtEndLoc;
        SourceLocation SEQBeginLoc = vectorSEQ[indexFuncNode.index].SEQRange.BeginLoc.getLocWithOffset(1);
        SourceLocation SEQEndLoc = vectorSEQ[indexFuncNode.index].SEQRange.EndLoc.getLocWithOffset(-2);
        int MaxOffset = 256; // wfr 20191223 最多只 向前/后 检查 MaxOffset 个字符
        
        // 确定 StmtBeginLoc
        if(SEQBeginLoc!=EmptyLoc){
            int Offset = 0;
            FileID MyFileID = Rewrite.getSourceMgr().getFileID(SEQBeginLoc);
            // StringRef filename = Rewrite.getSourceMgr().getFilename(BeginLoc);
            const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(MyFileID);
            // 获得源代码的字符串 buffer
            const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
            const char* start = buffer->getBufferStart();
            const char* cursor = start + Rewrite.getSourceMgr().getFileOffset(SEQBeginLoc); // 游标设置到节点开始位置

            for(int i=0; i<MaxOffset; i++){ // 最多只从开始位置向前检查 MaxOffset 个字符
                unsigned int character = (unsigned int)(*(cursor-i));

                if(character==0x0A){ // 是 换行
                    StmtBeginLoc = SEQBeginLoc.getLocWithOffset((int)1-i);
                    break;
                }
            }
        }

        // 确定 StmtEndLoc
        if(SEQEndLoc!=EmptyLoc){
            int Offset = 0;
            FileID MyFileID = Rewrite.getSourceMgr().getFileID(SEQEndLoc);
            // StringRef filename = Rewrite.getSourceMgr().getFilename(BeginLoc);
            const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(MyFileID);
            // 获得源代码的字符串 buffer
            const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
            const char* start = buffer->getBufferStart();
            const char* cursor = start + Rewrite.getSourceMgr().getFileOffset(SEQEndLoc); // 游标设置到节点开始位置

            for(int i=0; i<MaxOffset; i++){ // 最多只从结束位置向后检查 MaxOffset 个字符
                unsigned int character = (unsigned int)(*(cursor+i));

                if(character==0x0A){ // 是 换行
                    StmtEndLoc = SEQEndLoc.getLocWithOffset(i-1);
                    break;
                }
            }
        }
        // wfr 20191223 设置函数调用所在行的 起始和结束位置
        vectorSEQ[indexFuncNode.index].StmtRange.init(StmtBeginLoc, StmtEndLoc, Rewrite);

        // 更新 ASTStack 中的 indexNode
        // ASTStack.back().indexNode.type = NODE_TYPE::SEQUENTIAL;
        // ASTStack.back().indexNode.index = indexFuncNode.index;

        std::cout << "SplitFuncNode: indexFront =  " << std::dec << indexFront.index << ", type = " << indexFront.type
                  << std::endl;
        std::cout << "SplitFuncNode: indexFuncNode =  " << std::dec << indexFuncNode.index
                  << ", type = " << indexFuncNode.type << std::endl;
        std::cout << "SplitFuncNode: indexBack =  " << std::dec << indexBack.index << ", type = " << indexBack.type
                  << std::endl;

        // 如果是并行节点中调用的函数
    } else if ((indexFront.index = InWhichOMP(indexFuncCurrent, BeginOffset)) >= 0) {

        // 因为没考虑周全, 现在还不支持拆分 OMP 节点, 临时处理下

        indexFront.type = NODE_TYPE::PARALLEL;

        // vectorOMP[indexFront.index].vectorVarRef.emplace_back();
        // vectorOMP[indexFront.index].vectorVarRef.back().pFuncCall = S;
        // vectorOMP[indexFront.index].vectorVarRef.back().indexCallee = indexFunc;

        // ASTStack.back().init(S, "CallExpr", -1, vectorFunc[indexFunc].vectorParm.size());

        // // 更新 ASTStack 中的 indexNODE
        // ASTStack.back().indexNode.type = NODE_TYPE::PARALLEL;
        // ASTStack.back().indexNode.index = indexFront.index;
        // ASTStack.back().indexVarRef = vectorOMP[indexFront.index].vectorVarRef.size() - 1;

        return indexFront;

    } else {
        std::cout << "警告: CallExpr 不在 SEQ/OMP 节点中" << std::endl;
    }

    return indexFuncNode;
}

// wfr 20190325 保存某个变量的一次引用
void OAOASTVisitor::saveVarRefInfo(std::vector<VAR_REF_LIST> &vectorVarRef, int indexVar,
                                   VAR_REF::MEMBER_TYPE in_MemberType, STATE_CONSTR in_StReq,
                                   STATE_CONSTR in_StTransFunc, MY_SOURCE_RANGE &in_SrcRange) {
    // 判断是否引用了当前变量, 还没引用过就新建项
    std::vector<VAR_REF_LIST>::iterator iterVarRef;
    // 找到引用列表
    iterVarRef = find(vectorVarRef.begin(), vectorVarRef.end(), indexVar);
    if (iterVarRef == vectorVarRef.end()) { // 如果还没记录该变量的引用, 则新建项
        vectorVarRef.emplace_back(indexVar);
        iterVarRef = vectorVarRef.end() - 1;
    }

    // 取出引用列表
    std::vector<VAR_REF> &RefList = iterVarRef->RefList;

    // 找到插入位置, 插入位置应该按照在源代码中出现的先后顺序排序,
    // 如果没有分支的话也是执行顺序
    std::vector<VAR_REF>::iterator iterRef = RefList.end();
    for (; iterRef != RefList.begin(); --iterRef) {
        if (iterRef->SrcRange.EndOffset < in_SrcRange.BeginOffset) {
            break;
        }
    }

    // 写入读写信息
    RefList.emplace(iterRef, in_MemberType, in_StReq, in_StTransFunc, in_SrcRange);
}

// wfr 20190324 判断一个 Offset 在哪个 SEQ/OMP 中, 根据源代码文本顺序
void OAOASTVisitor::InWhichSEQOMP(NODE_INDEX &indexNode, unsigned int indexFunc, unsigned int Offset) {
    int index;
    if ((index = InWhichSEQ(indexFunc, Offset)) >= 0) {
        indexNode.index = index;
        indexNode.type = NODE_TYPE::SEQUENTIAL;
    } else if ((index = InWhichOMP(indexFunc, Offset)) >= 0) {
        indexNode.index = index;
        indexNode.type = NODE_TYPE::PARALLEL;
    } else {
        indexNode.index = -1;
        indexNode.type = NODE_TYPE::NODE_UNINIT;
    }
}

// wfr 20190719 处理节点入口处的变量状态转换相关问题:
// 1. 从 src(即前驱节点的出口状态约束) 到 dest(即当前节点的入口状态约束) 是否需要插入 OAODataTrans函数
// 2. destReal(即实际的入口状态约束) 是什么
int OAOASTVisitor::InferEnteryTrans(STATE_CONSTR &StTrans, STATE_CONSTR &destReal, STATE_CONSTR src,
                                    STATE_CONSTR dest) {
    DEFINE_ST_REQ_ST_TRANS
    if (src == StReqUninit || dest == StReqUninit) {
        std::cout << "InferEnteryTrans 错误: 非法的 src/dest 类型" << std::endl;
        exit(1);
    }
    bool isSatisfied = false;

    if (src == StReqSync && dest != StReqHostOnly) {
        isSatisfied = true;
    } else if (src == StReqDeviceNew && dest == StReqInDevice) {
        isSatisfied = true;
    } else if (src == dest) {
        isSatisfied = true;
    } else if (dest == StReqNone || dest == StReqDiverg) {
        isSatisfied = true;
    } else if (src == StReqDiverg) {
        if (dest == StReqNone)
        {
            isSatisfied = true;
        }
    } else {
        isSatisfied = CheckConstrSatisfaction(src, dest);
    }

    if (isSatisfied == true) {
        StTrans = StTransNone;
        destReal = src;
    } else {
        StTrans = dest;
        if (src == StReqDiverg) {
            destReal = dest;
        } else {
            destReal = ExeStTrans(dest, src);
        }
    }

    return 0;
}

// wfr 20190719 处理节点入口处的变量状态转换相关问题:
// 1. ExitStConstr(即出口状态约束) 是什么
// 2. 出口处是否插入 OMPWrite/SEQWrite函数, 由 ExitStTrans 标识
int OAOASTVisitor::InferExitTrans(STATE_CONSTR &ExitStTrans, STATE_CONSTR &ExitStConstr, STATE_CONSTR EntryStConstr,
                                  STATE_CONSTR StTrans) {
    DEFINE_ST_REQ_ST_TRANS
    if (EntryStConstr == StReqUninit || StTrans == StTransUninit) {
        std::cout << "InferExitTrans 错误: 非法的 EntryStConstr/StTrans 类型" << std::endl;
        exit(1);
    }

    // 解决两个问题：
    // 出口状态约束是什么
    // 出口处是否 调用 OMPWrite/SEQWrite

    // 出口状态约束是什么
    if (StTrans == StTransDiverg) {
        ExitStConstr = StReqDiverg;
    }else if(EntryStConstr == StReqDiverg && StTrans != StTransDiverg){
        ExitStConstr = StTrans;
    }else {
        ExitStConstr = ExeStTrans(StTrans, EntryStConstr);
    }

    // 出口处是否 调用 OMPWrite/SEQWrite
    if (StTrans == StTransDiverg) {
        ExitStTrans = StTransNone;
    } else if (StTrans.ONE == 0) {
        ExitStTrans = StTransNone;
    } else if (ExitStConstr==EntryStConstr) {
        ExitStTrans = StTransNone;
    } else {
        ExitStTrans = StTrans;
    }

    return 0;
}

// wfr 20190318 获得多维数组的元素的个数, 即数组长度
int OAOASTVisitor::getArrayLen(std::string &ArrayLen, const std::string &TypeName) {
    if (TypeName.back() != ']') {
        std::cout << "getArrayLen 错误：不是数组类型" << std::endl;
        exit(1);
    }

    ArrayLen = "";

    // 找到第一对中括号的位置
    std::size_t LeftPos, RightPos;
    LeftPos = TypeName.find_first_of('[');
    RightPos = TypeName.find_first_of(']');

    while (LeftPos != std::string::npos && RightPos != std::string::npos) {

        // 获取 中括号中的内容, 相乘, 得到数组元素总数
        ArrayLen += TypeName.substr(LeftPos + 1, RightPos - LeftPos - 1);
        ArrayLen += "*";

        LeftPos = TypeName.find_first_of('[', LeftPos + 1);
        RightPos = TypeName.find_first_of(']', RightPos + 1);
    }

    ArrayLen.pop_back(); // 移除最后的 '*'

    return 0;
}

// wfr 20190305 判断 indexNode 是指向 SEQ/OMP, 获得对应节点的 SEQ_PAR_NODE_BASE*
// 类型的指针
SEQ_PAR_NODE_BASE *OAOASTVisitor::getNodeBasePtr(const FUNC_INFO &Func, const NODE_INDEX &indexNode) {
    assert(indexNode.index >= 0 && "getNodeBasePtr错误: indexNode.index<0");
    assert((indexNode.type == NODE_TYPE::SEQUENTIAL || indexNode.type == NODE_TYPE::PARALLEL) &&
           "getNodeBasePtr错误: indexNode.type 类型未知");
    if (indexNode.type == NODE_TYPE::SEQUENTIAL) {
        return (SEQ_PAR_NODE_BASE *)(&(Func.vectorSEQ[indexNode.index]));
    } else if (indexNode.type == NODE_TYPE::PARALLEL) {
        return (SEQ_PAR_NODE_BASE *)(&(Func.vectorOMP[indexNode.index]));
    } else {
        return NULL;
    }
}

// wfr 20190213 判断 某个Stmt 是否在 某个OMP 中
bool OAOASTVisitor::IsInOMP(Stmt *S, unsigned int indexFunc, unsigned int indexOMP) {
    std::cout << "IsInOMP: 0" << std::endl;
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int Offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    std::cout << "IsInOMP: 1" << std::endl;
    if (vectorFunc[indexFunc].vectorOMP[indexOMP].DirectorRange.BeginOffset < Offset &&
        Offset < vectorFunc[indexFunc].vectorOMP[indexOMP].OMPRange.EndOffset) {
        return true;
    } else {
        return false;
    }
}
bool OAOASTVisitor::IsInOMP(Decl *S, unsigned int indexFunc, unsigned int indexOMP) {

    int indexLocal = WhichLocalVar(S, indexFunc, indexOMP);

    if (indexLocal >= 0) {
        return true;
    } else {
        return false;
    }
}
bool OAOASTVisitor::IsInOMP(Expr *S, unsigned int indexFunc, unsigned int indexOMP) {

    std::cout << "IsInOMP: 6" << std::endl;

    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int Offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);

    if (vectorFunc[indexFunc].vectorOMP[indexOMP].DirectorRange.BeginOffset < Offset &&
        Offset < vectorFunc[indexFunc].vectorOMP[indexOMP].OMPRange.EndOffset) {
        return true;
    } else {
        return false;
    }
}

// 检查 member 是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
int OAOASTVisitor::MemberIsWhichLocal(Decl *ptrClass, std::string Name, unsigned int indexFunc, unsigned int indexOMP) {
    std::vector<VARIABLE_INFO> &vectorLocal = vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal;
    for (int i = 0; i < (int)vectorLocal.size(); i++) {
        if (vectorLocal[i].isMember && vectorLocal[i].ptrClass == ptrClass && vectorLocal[i].Name == Name) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// wfr 20190807 变量是哪个全局 member
int  OAOASTVisitor::MemberIsWhichGlobal(Decl *ptrClass, std::string Name){
    for (int i = 0; i < (int)vectorGblVar.size(); i++) {
        if (vectorGblVar[i].isMember && vectorGblVar[i].ptrClass == ptrClass && vectorGblVar[i].Name == Name) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 检查 member 是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
int OAOASTVisitor::MemberIsWhichVar(Decl *ptrClass, std::string Name, FUNC_INFO &Func) {
    std::vector<VARIABLE_INFO> &vectorVar = Func.vectorVar;
    for (int i = 0; i < (int)vectorVar.size(); i++) {
        if (vectorVar[i].isMember && vectorVar[i].ptrClass == ptrClass && vectorVar[i].Name == Name) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 检查 member 是否在变量列表中, 以及是哪个表项, 返回 vectorVar 表项的 index
int OAOASTVisitor::MemberIsWhichVar(Decl *ptrClass, std::string Name, unsigned int indexFunc) {
    return MemberIsWhichVar(ptrClass, Name, vectorFunc[indexFunc]);
}

// 变量的定义是否在 vectorLocal 列表中, 以及是哪个表项,返回 vectorLocal 表项的
// index, 不在就返回 -1
int OAOASTVisitor::WhichLocalVar(Decl *ptrDecl, unsigned int indexFunc, unsigned int indexOMP) {
    for (int i = 0; i < (int)vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal.size(); i++) {
        if (vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal[i].ptrDecl == ptrDecl) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// wfr 20190729 判断一个偏移在哪个函数中
int  OAOASTVisitor::InWhichFunc(unsigned int Offset){
    for(unsigned long i=0; i<vectorFunc.size(); i++){
        if (vectorFunc[i].BeginOffset <= Offset &&
            Offset <= vectorFunc[i].EndOffset)
        {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 该函数用来判断一个偏移是否在OMP中, 以及在哪个OMP中, 返回 vectorOMP 表项的
// index 或 -1
int OAOASTVisitor::InWhichOMP(unsigned int indexFunc, unsigned int Offset) {
    for (int i = 0; i < (int)vectorFunc[indexFunc].vectorOMP.size(); i++) {
        if (vectorFunc[indexFunc].vectorOMP[i].DirectorRange.BeginOffset <= Offset &&
            Offset <= vectorFunc[indexFunc].vectorOMP[i].OMPRange.EndOffset) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 该函数用来判断一个偏移是否在SEQ中, 以及在哪个SEQ中, 返回 vectorOMP 表项的
// index 或 -1
int OAOASTVisitor::InWhichSEQ(unsigned int indexFunc, unsigned int Offset) {
    for (int i = 0; i < (int)vectorFunc[indexFunc].vectorSEQ.size(); i++) {
        if (vectorFunc[indexFunc].vectorSEQ[i].SEQRange.BeginOffset <= Offset &&
            Offset <= vectorFunc[indexFunc].vectorSEQ[i].SEQRange.EndOffset) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 该函数用来判断一个函数是否在 vectorFunc 中, 以及是哪个 FUNC_INFO , 返回
// vectorFunc 表项的 index 或 -1
int OAOASTVisitor::WhichFuncDefine(FunctionDecl *pFuncDecl) {
    FunctionDecl *pTarget;
    if (pFuncDecl->getPrimaryTemplate()) {
        pTarget = pFuncDecl->getPrimaryTemplate()->getTemplatedDecl(); // 获得函数模版的地址
    } else {
        pTarget = pFuncDecl;
    }

    for (int i = 0; i < (int)vectorFunc.size(); i++) {
        if (vectorFunc[i].ptrDefine == pTarget) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 判断一个 函数的定义 是否在 vectorFunc 中, 在则返回 index, 不在则返回 -1
int OAOASTVisitor::WhichFuncDecl(FunctionDecl *pFuncDecl) {
    FunctionDecl *pTarget;
    if (pFuncDecl->getPrimaryTemplate()) {
        pTarget = pFuncDecl->getPrimaryTemplate()->getTemplatedDecl(); // 获得函数模版的地址
    } else {
        pTarget = pFuncDecl;
    }
    for (int i = 0; i < (int)vectorFunc.size(); i++) {
        if (vectorFunc[i].ptrDecl == pTarget) {
            return i;
        }
    }
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

// 变量的定义是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
int OAOASTVisitor::WhichVar(Decl *ptrDecl, unsigned int indexFunc) {
    
    for (size_t i = 0; i < vectorFunc[indexFunc].vectorVar.size(); i++) {
        if (vectorFunc[indexFunc].vectorVar[i].ptrDecl == ptrDecl) {
            return i; // wfr 20190809 如果在当前函数的变量列表中找到就直接返回 index
        }
    }
    
    return -1; // 没找到就返回 -1, 因此调用者需要通过这个判断是否找到
}

bool OAOASTVisitor::TraverseUnaryAddrOf(UnaryOperator *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << std::endl;
        std::cout << "TraverseUnaryAddrOf: 处理" << std::endl;

        // 将 OperandID 指向的操作数的访问方式改成 PTR, 表示引用指针
        if (!ASTStack.empty() && ASTStack.back().OperandID >= 0 &&
            ASTStack.back().AccessType[ASTStack.back().OperandID] == ASTStackNode::PTR_ARY::PTR_UNINIT) {
            ASTStack.back().AccessType[ASTStack.back().OperandID] = ASTStackNode::PTR_ARY::PTR;
        }
    }

    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromUnaryAddrOf(S)) {
                return false;
            }
        } while (false);
    }
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getSubExpr()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getSubExpr())))) {
            return false;
        }
    } while (false);
    return true;
}

// wfr 20190403
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)
// 改成了:
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()))
// 去掉了参数 Queue, 即使用默认的 Queue=nullptr, 这样 TraverseStmt
// 才不会直接返回
bool OAOASTVisitor::TraverseUnaryPostInc(UnaryOperator *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << std::endl;
        std::cout << "TraverseUnaryPostInc: 处理" << std::endl;

        std::string OpName = UnaryOperator::getOpcodeStr(S->getOpcode());
        ASTStack.emplace_back((Stmt *)S, OpName, 0, 1);
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getSubExpr());
        ASTStack.back().NextOperandID[0] = -1;
    }

    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromUnaryPostInc(S)) {
                return false;
            }
        } while (false);
    }
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getSubExpr()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getSubExpr())))) {
            return false;
        }
    } while (false);
    return true;
}

bool OAOASTVisitor::TraverseUnaryPreInc(UnaryOperator *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << std::endl;
        std::cout << "TraverseUnaryPreInc: 处理" << std::endl;

        std::string OpName = UnaryOperator::getOpcodeStr(S->getOpcode());
        ASTStack.emplace_back((Stmt *)S, OpName, 0, 1);
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getSubExpr());
        ASTStack.back().NextOperandID[0] = -1;
    }

    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromUnaryPreInc(S)) {
                return false;
            }
        } while (false);
    }
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getSubExpr()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getSubExpr())))) {
            return false;
        }
    } while (false);
    return true;
}

bool OAOASTVisitor::TraverseUnaryPostDec(UnaryOperator *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << std::endl;
        std::cout << "TraverseUnaryPostDec: 处理" << std::endl;

        Stmt::child_iterator FierstChild = S->children().begin();

        assert(isa<Stmt>(*(FierstChild)) && "TraverseUnaryPostDec");

        Expr *TmpPtr = S->getSubExpr();
        std::cout << "TraverseUnaryPostDec: S->getSubExpr() = " << std::hex << TmpPtr << std::endl;

        std::string OpName = UnaryOperator::getOpcodeStr(S->getOpcode());
        std::cout << "TraverseUnaryPostDec: a, Queue = " << std::hex << Queue << std::endl;
        ASTStack.emplace_back((Stmt *)S, OpName, 0, 1);
        std::cout << "TraverseUnaryPostDec: b, OpName = " << OpName << std::endl;
        TmpPtr = S->getSubExpr();
        std::cout << "TraverseUnaryPostDec: b" << std::endl;
        // ASTStack.back().SubStmts[0] = (Stmt *)(S->getSubExpr());
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getSubExpr());
        std::cout << "TraverseUnaryPostDec: c, Queue = " << std::hex << Queue << std::endl;
        ASTStack.back().NextOperandID[0] = -1;
    }

    std::cout << "TraverseUnaryPostDec: 0, Queue = " << std::hex << Queue << std::endl;

    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromUnaryPostDec(S)) {
                return false;
            }
        } while (false);
    }

    std::cout << "TraverseUnaryPostDec: 1" << std::endl;

    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getSubExpr()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getSubExpr())))) {
            return false;
        }
    } while (false);

    std::cout << "TraverseUnaryPostDec: 2" << std::endl;

    return true;
}

bool OAOASTVisitor::TraverseUnaryPreDec(UnaryOperator *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << std::endl;
        std::cout << "TraverseUnaryPreDec: 处理" << std::endl;

        std::string OpName = UnaryOperator::getOpcodeStr(S->getOpcode());
        ASTStack.emplace_back((Stmt *)S, OpName, 0, 1);
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getSubExpr());
        ASTStack.back().NextOperandID[0] = -1;
    }

    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromUnaryPreDec(S)) {
                return false;
            }
        } while (false);
    }
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getSubExpr()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getSubExpr())))) {
            return false;
        }
    } while (false);
    return true;
}

// wfr 20190403
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)
// 改成了:
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()))
// 去掉了参数 Queue, 即使用默认的 Queue=nullptr, 这样 TraverseStmt
// 才不会直接返回
#define TRAVERSE_BINARY_OPERATOR(OP_NAME)                                                                              \
    bool OAOASTVisitor::TraverseBin##OP_NAME(BinaryOperator *S, DataRecursionQueue *Queue) {                           \
        std::string OpName = BinaryOperator::getOpcodeStr(S->getOpcode());                                             \
        if (BeginFlag == true && FuncStack.empty() == false) {                                                         \
            std::cout << "TraverseBin" << #OP_NAME << ": 操作符 " << OpName << ", 地址是 " << std::hex << S            \
                      << std::endl;                                                                                    \
            ASTStack.emplace_back((Stmt *)S, OpName, 1, 2);                                                            \
            ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getLHS());                                            \
            ASTStack.back().NextOperandID[0] = -1;                                                                     \
            ASTStack.back().SubStmts[1] = static_cast<Stmt *>(S->getRHS());                                            \
            ASTStack.back().NextOperandID[1] = 0;                                                                      \
            LangOptions MyLangOptions;                                                                                 \
            PrintingPolicy PrintPolicy(MyLangOptions);                                                                 \
            std::string TypeName;                                                                                      \
            TypeName = QualType::getAsString(S->getType().split(), PrintPolicy);                                       \
            if (TypeName.back() == '*') {                                                                              \
                ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::PTR;                                            \
                ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::PTR;                                            \
            } else {                                                                                                   \
                ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::ARRAY;                                          \
                ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::ARRAY;                                          \
            }                                                                                                          \
        }                                                                                                              \
        if (!getDerived().shouldTraversePostOrder()) {                                                                 \
            do {                                                                                                       \
                if (!getDerived().WalkUpFromBin##OP_NAME(S)) {                                                         \
                    return false;                                                                                      \
                }                                                                                                      \
            } while (false);                                                                                           \
        }                                                                                                              \
        do {                                                                                                           \
            if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                           \
                                               decltype(&OAOASTVisitor::TraverseStmt)>::value                          \
                    ? static_cast<typename std::conditional<                                                           \
                        has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                     \
                                                     decltype(&OAOASTVisitor::TraverseStmt)>::value,                   \
                        OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)                                          \
                        .TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)                                         \
                    : getDerived().TraverseStmt(static_cast<Stmt *>(S->getRHS())))) {                                  \
                return false;                                                                                          \
            }                                                                                                          \
        } while (false);                                                                                               \
        do {                                                                                                           \
            if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                           \
                                               decltype(&OAOASTVisitor::TraverseStmt)>::value                          \
                    ? static_cast<typename std::conditional<                                                           \
                        has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                     \
                                                     decltype(&OAOASTVisitor::TraverseStmt)>::value,                   \
                        OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)                                          \
                        .TraverseStmt(static_cast<Stmt *>(S->getLHS()), Queue)                                         \
                    : getDerived().TraverseStmt(static_cast<Stmt *>(S->getLHS())))) {                                  \
                return false;                                                                                          \
            }                                                                                                          \
        } while (false);                                                                                               \
        return true;                                                                                                   \
    }

// wfr 20190403
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)
// 改成了:
// (*this).TraverseStmt(static_cast<Stmt *>(S->getRHS()))
// 去掉了参数 Queue, 即使用默认的 Queue=nullptr, 这样 TraverseStmt
// 才不会直接返回
bool OAOASTVisitor::TraverseBinAssign(BinaryOperator *S, DataRecursionQueue *Queue) {

    //这里设置标识变量, 告知后续的 AST 遍历, 是在解析 BinaryOperator
    std::string OpName = BinaryOperator::getOpcodeStr(S->getOpcode()); // 获得字符串格式的赋值操作符 "="
    if (BeginFlag == true && FuncStack.empty() == false) {
        std::cout << "TraverseBinAssign: 操作符 " << OpName << ", 地址是 " << std::hex << S << std::endl;
        ASTStack.emplace_back((Stmt *)S, OpName, 1, 2);
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getLHS());
        ASTStack.back().NextOperandID[0] = -1;
        ASTStack.back().SubStmts[1] = static_cast<Stmt *>(S->getRHS());
        ASTStack.back().NextOperandID[1] = 0;
        // 如果是指针类型, 则在 ASTStack.back() 中标记变量类型为指针
        LangOptions MyLangOptions;
        PrintingPolicy PrintPolicy(MyLangOptions);
        std::string TypeName;
        TypeName = QualType::getAsString(S->getType().split(),
                                         PrintPolicy); // 变量的类型：int、int*、float*等等
        if (TypeName.back() == '*') { // 如果是指针类型, 则在 ASTStack.back() 中标记变量类型为指针
            ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::PTR;
            ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::PTR;
        } else {
            ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::ARRAY;
            ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::ARRAY;
        }
    }
    if (!getDerived().shouldTraversePostOrder()) {
        do {
            if (!getDerived().WalkUpFromBinAssign(S)) {
                return false;
            }
        } while (false);
    }

    // 这里改变两个操作数的访问顺序, 原来是先左后右, 改成先右后左,
    // 为了进行变量跟踪
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getRHS())))) {
            return false;
        }
    } while (false);
    do {
        if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                           decltype(&OAOASTVisitor::TraverseStmt)>::value
                ? static_cast<typename std::conditional<
                    has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                 decltype(&OAOASTVisitor::TraverseStmt)>::value,
                    OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                    .TraverseStmt(static_cast<Stmt *>(S->getLHS()), Queue)
                : getDerived().TraverseStmt(static_cast<Stmt *>(S->getLHS())))) {
            return false;
        }
    } while (false);

    return true;
}

// TypeName = QualType::getAsString(S->getType().split(), PrintPolicy);
#define TRAVERSE_COMPOUND_OPERATOR(OP_NAME)                                                                            \
    bool OAOASTVisitor::TraverseBin##OP_NAME(CompoundAssignOperator *S, DataRecursionQueue *Queue) {                   \
        std::string OpName = BinaryOperator::getOpcodeStr(S->getOpcode());                                             \
        if (BeginFlag == true && FuncStack.empty() == false) {                                                         \
            std::cout << std::endl;                                                                                    \
            std::cout << "TraverseBin" << #OP_NAME << ": 操作符 " << OpName << ", 地址是 " << std::hex << S            \
                      << std::endl;                                                                                    \
            ASTStack.emplace_back((Stmt *)S, OpName, 1, 2);                                                            \
            ASTStack.back().SubStmts[0] = static_cast<Stmt *>(S->getLHS());                                            \
            ASTStack.back().NextOperandID[0] = -1;                                                                     \
            ASTStack.back().SubStmts[1] = static_cast<Stmt *>(S->getRHS());                                            \
            ASTStack.back().NextOperandID[1] = 0;                                                                      \
            LangOptions MyLangOptions;                                                                                 \
            PrintingPolicy PrintPolicy(MyLangOptions);                                                                 \
            std::string TypeName;                                                                                      \
            TypeName = QualType::getAsString(S->getComputationResultType().split(), PrintPolicy);                      \
            if (TypeName.back() == '*') {                                                                              \
                ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::PTR;                                            \
                ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::PTR;                                            \
            } else {                                                                                                   \
                ASTStack.back().AccessType[0] = ASTStackNode::PTR_ARY::ARRAY;                                          \
                ASTStack.back().AccessType[1] = ASTStackNode::PTR_ARY::ARRAY;                                          \
            }                                                                                                          \
        }                                                                                                              \
        if (!getDerived().shouldTraversePostOrder()) {                                                                 \
            do {                                                                                                       \
                if (!getDerived().WalkUpFromBinAddAssign(S)) {                                                         \
                    return false;                                                                                      \
                }                                                                                                      \
            } while (false);                                                                                           \
        }                                                                                                              \
        do {                                                                                                           \
            if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                           \
                                               decltype(&OAOASTVisitor::TraverseStmt)>::value                          \
                    ? static_cast<typename std::conditional<                                                           \
                        has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                     \
                                                     decltype(&OAOASTVisitor::TraverseStmt)>::value,                   \
                        OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)                                          \
                        .TraverseStmt(static_cast<Stmt *>(S->getRHS()), Queue)                                         \
                    : getDerived().TraverseStmt(static_cast<Stmt *>(S->getRHS())))) {                                  \
                return false;                                                                                          \
            }                                                                                                          \
        } while (false);                                                                                               \
        do {                                                                                                           \
            if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                           \
                                               decltype(&OAOASTVisitor::TraverseStmt)>::value                          \
                    ? static_cast<typename std::conditional<                                                           \
                        has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),                     \
                                                     decltype(&OAOASTVisitor::TraverseStmt)>::value,                   \
                        OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)                                          \
                        .TraverseStmt(static_cast<Stmt *>(S->getLHS()), Queue)                                         \
                    : getDerived().TraverseStmt(static_cast<Stmt *>(S->getLHS())))) {                                  \
                return false;                                                                                          \
            }                                                                                                          \
        } while (false);                                                                                               \
        return true;                                                                                                   \
    }

TRAVERSE_COMPOUND_OPERATOR(MulAssign)
TRAVERSE_COMPOUND_OPERATOR(DivAssign)
TRAVERSE_COMPOUND_OPERATOR(RemAssign)
TRAVERSE_COMPOUND_OPERATOR(AddAssign)
TRAVERSE_COMPOUND_OPERATOR(SubAssign)
TRAVERSE_COMPOUND_OPERATOR(ShlAssign)
TRAVERSE_COMPOUND_OPERATOR(ShrAssign)
TRAVERSE_COMPOUND_OPERATOR(AndAssign)
TRAVERSE_COMPOUND_OPERATOR(XorAssign)
TRAVERSE_COMPOUND_OPERATOR(OrAssign)

bool OAOASTVisitor::TraverseStmt(Stmt *S, DataRecursionQueue *Queue) {

    if (BeginFlag == true && vectorFunc.empty() == false) {
        // std::cout << "TraverseStmt: 进入, S = " << std::hex << S << ", Queue = "
        // << std::hex << Queue << std::endl;
    }

    if (!S)
        return true;
    if (Queue) {
        Queue->push_back({S, false});
        return true;
    }
    SmallVector<llvm::PointerIntPair<Stmt *, 1, bool>, 8> LocalQueue;
    LocalQueue.push_back({S, false});
    while (!LocalQueue.empty()) {
        auto &CurrSAndVisited = LocalQueue.back();
        Stmt *CurrS = CurrSAndVisited.getPointer();
        bool Visited = CurrSAndVisited.getInt();

        // 在这里更新 OperandID
        std::vector<Stmt *>::iterator iterpStmt;
        if (!ASTStack.empty()) {
            // std::cout << "TraverseStmt: 操作符 " << ASTStack.back().Name <<
            // std::endl;
            iterpStmt = find(ASTStack.back().SubStmts.begin(), ASTStack.back().SubStmts.end(), CurrS);
            if (iterpStmt != ASTStack.back().SubStmts.end()) {
                ASTStack.back().OperandID = iterpStmt - ASTStack.back().SubStmts.begin();
                // std::cout << "TraverseStmt: 操作符 " << ASTStack.back().Name
                //   << " 的 OperandID = " << ASTStack.back().OperandID << std::endl;
            }
        }

        if (Visited) {
            LocalQueue.pop_back();
            if (!getDerived().dataTraverseStmtPost(CurrS)) {
                return false;
            }
            if (getDerived().shouldTraversePostOrder()) {
                // std::cout << "TraverseStmt: PostVisitStmt, CurrS = " << std::hex <<
                // CurrS << std::endl;
                if (!getDerived().PostVisitStmt(CurrS)) {

                    // 在这里出栈 ASTStackNode
                    if (!ASTStack.empty() && ASTStack.back().pOperator == CurrS) {
                        // std::cout << "TraverseStmt: 出栈操作符 " << ASTStack.back().Name
                        // << std::endl;
                        ASTStack.pop_back();
                    }

                    return false;
                }
                // 在这里出栈 ASTStackNode
                if (!ASTStack.empty() && ASTStack.back().pOperator == CurrS) {
                    // std::cout << "TraverseStmt: 出栈操作符 " << ASTStack.back().Name <<
                    // std::endl;
                    ASTStack.pop_back();
                }
            }
            continue;
        }
        if (getDerived().dataTraverseStmtPre(CurrS)) {
            CurrSAndVisited.setInt(true);
            size_t N = LocalQueue.size();
            // std::cout << "TraverseStmt: dataTraverseNode, CurrS = " << std::hex <<
            // CurrS << std::endl;
            if (!getDerived().dataTraverseNode(CurrS, &LocalQueue)) {
                return false;
            }
            std::reverse(LocalQueue.begin() + N, LocalQueue.end());
        } else {
            LocalQueue.pop_back();
        }
    }

    // std::cout << "TraverseStmt: 退出" << std::endl;
    return true;
}

// wfr 20190420 在第一次访问 OMP 主体的时候, 保存 主体地址
bool OAOASTVisitor::TraverseForStmt(ForStmt *S, DataRecursionQueue *Queue) {

    if (!OMPStack.empty()) { // 如果正在处理 OMP
        OMP_REGION &OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
        if (OMP.pBodyStmt == NULL) { // 如果还没找到 主体地址
            // 存入主体地址, 可以根据主体地址判断 变量定义/引用 是否在主体中, 不在主体中不处理
            OMP.pBodyStmt = S;
        }
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromForStmt(S)) {
            return false;
        }
    }
    if (ShouldVisitChildren) {
        for (Stmt *SubStmt : getDerived().getStmtChildren(S)) {
            if (!(has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                               decltype(&OAOASTVisitor::TraverseStmt)>::value
                    ? static_cast<typename std::conditional<
                        has_same_member_pointer_type<decltype(&RecursiveASTVisitor::TraverseStmt),
                                                     decltype(&OAOASTVisitor::TraverseStmt)>::value,
                        OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this)
                        .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue)
                    : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)))) {
                return false;
            }
        }
    }
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromForStmt(S)) {
            return false;
        }
    }
    return ReturnValue;
}

// wfr 20190420 在离开 OMP 主体的时候, 将 主体地址 置为 NULL
bool OAOASTVisitor::VisitForStmt(ForStmt *S) {
    if (!OMPStack.empty()) { // 如果正在处理 OMP
        OMP_REGION &OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
        // 如果和保存的 主体地址 相同, 则说明要完成整个 主体 的处理了, 因为 VisitXXX 这种函数是在最后才调用的
        if (OMP.pBodyStmt == S) {
            OMP.pBodyStmt = NULL; // 将 主体地址 置为 NULL, 说明要离开 主体 范围了
        }
    }
    return true;
}
