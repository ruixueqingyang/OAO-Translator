/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: OAORewriter.cpp
       Author: 瑞雪轻飏
      Version: 0.02
Creation Date: 20181122
  Description: 实现修改的 OAOASTVisitor 和 ASTConsumer
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1. main: 主函数
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期, 修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

/***   OAORewriter.cpp   ******************************************************
 * Usage:
 * OAORewriter.bin <options> <file>.c
 * where <options> allow for parameters to be passed to the preprocessor
 * such as -DFOO to define FOO.
 *
 * Generated as output <file>_out.c
 *
 * Note: This tutorial uses the CompilerInstance object which has as one of
 * its purposes to create commonly used Clang types.
 *****************************************************************************/
#include "OAORewriter.h"

using namespace clang;

// wfr 20190729 将 delete[] 替换成 OAODelete
bool OAOASTVisitor::TraverseCXXDeleteExpr(CXXDeleteExpr *S, DataRecursionQueue *Queue) {

    SourceLocation DeleteBegin = S->getBeginLoc();
    unsigned int DeleteBeginOffset = Rewrite.getSourceMgr().getFileOffset(DeleteBegin);
    int indexFunc = InWhichFunc(DeleteBeginOffset);
    bool ASTStackPushed = false;

    // 对 delete[] 进行整体替换, 举例:
    // delete[] array 整体替换成 OAODelete((void*)array)
    // delete[] array; 改成 OAODelete((void*)array);

    // if( BeginFlag==true && indexFunc>=0 && S->getOperatorDelete()!=NULL
    //     && S->getOperatorDelete()->getNameAsString()=="operator delete[]")
    if( BeginFlag==true && indexFunc>=0 )
    {
        FileID Fileid = Rewrite.getSourceMgr().getFileID(DeleteBegin);
        // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
        // const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
        // const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
        const llvm::MemoryBuffer* buffer = Rewrite.getSourceMgr().getBuffer(Fileid);
        const char* start = buffer->getBufferStart();
        size_t size = buffer->getBufferSize();

        std::string parm;
        size_t index = DeleteBeginOffset; --index;
        size_t DeleteEndOffset = 0;
        bool FindParmBegin = false;
        // bool FindParm = false;

        while(index<size){
            ++index;
            if(start[index]==' '){
                continue;
            }else if(FindParmBegin==false){
                if(start[index]==']'){
                    FindParmBegin = true;
                }
            }else if(FindParmBegin==true){
                char tmp = start[index];
                if( tmp=='_' || ('0'<=tmp&&tmp<='9') || ('a'<=tmp&&tmp<='z')
                    || ('A'<=tmp&&tmp<='Z') )
                {
                    parm += tmp;
                    DeleteEndOffset = index;
                }else{
                    break;
                }
            }
        }

        if(parm.size()<=1){
            std::cout << "VisitCXXDeleteExpr 错误: 没找到 delete[] 的参数" << std::endl;
            exit(1);
        }

        SourceRange TmpRange;
        TmpRange.setBegin(DeleteBegin);
        TmpRange.setEnd( DeleteBegin.getLocWithOffset(DeleteEndOffset-DeleteBeginOffset) );

        const char* tmp = &start[DeleteBeginOffset];
        std::string code(tmp, (DeleteEndOffset-DeleteBeginOffset+1));
        code += "; ";
        code += OAO_DELETE_NAME; code += "( (void*)(";
        code += parm; code += ") )";
        
        vectorFunc[indexFunc].vectorReplace.emplace_back(TmpRange, code, Rewrite);

        // wfr 20190729 下边保存引用信息
        // 这里对 ASTStack 压栈, 表示之后遇到的第一个变量引用是 delete[] 的参数, 后边需要出栈
        ASTStack.emplace_back((Stmt *)S, "delete[]", 0, 1);
        ASTStack.back().SubStmts[0] = static_cast<Stmt *>(*(S->child_begin()));
        ASTStack.back().NextOperandID[0] = -1;
        ASTStackPushed = true;
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromCXXDeleteExpr(S)) {
            if(ASTStackPushed){
                ASTStack.pop_back();
            }
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
                    : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt))))
            {
                return false;
            }
        }
    }
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromCXXDeleteExpr(S))
        {
            return false;
        }
    }
    return ReturnValue;
}

// wfr 20190729 将 new 替换成 OAONew
bool OAOASTVisitor::VisitCXXNewExpr(CXXNewExpr *S) {

    if(S->getOperatorNew()==NULL){
        return true;
    }

    SourceLocation NewBegin = S->getBeginLoc();
    unsigned int NewBeginOffset = Rewrite.getSourceMgr().getFileOffset(NewBegin);
    int indexFunc = InWhichFunc(NewBeginOffset);

    if(BeginFlag==false){
        return true;
    }

    if(indexFunc<0){
        std::cout << "VisitCXXNewExpr 错误: 当前函数的 indexFunc<0" << std::endl;
        exit(1);
    }

    // 对 new[] 进行整体替换, 举例:
    // new T[N] 整体替换成 (T*)OAONew(sizeof(T)*N)
    // return new T[N]; 改成 return (T*)OAONew(sizeof(T)*N);

    if(S->getOperatorNew()->getNameAsString()!="operator new[]"){
        return true; // 如果不是 new[] 就返回
    }

    FileID Fileid = Rewrite.getSourceMgr().getFileID(NewBegin);
    // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
    // const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
    // const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
    const llvm::MemoryBuffer* buffer = Rewrite.getSourceMgr().getBuffer(Fileid);
    const char* start = buffer->getBufferStart();
    size_t size = buffer->getBufferSize();

    std::string type;
    std::string length;
    unsigned int index = NewBeginOffset + 3; --index;
    unsigned int NewEndOffset = 0;
    bool FindType = false;
    bool FindLength = false;

    while(index<size){
        ++index;
        if(start[index]==' '){
            continue;
        }else if(FindType==false){
            if(start[index]!='['){
                type += start[index];
            }else{
                FindType=true;
            }
        }else if(FindLength==false){
            if(start[index]!=']'){
                length += start[index];
            }else{
                NewEndOffset = index;
                FindLength=true;
            }
        }else if(FindType==true && FindLength==true){
            if(start[index]=='['){
                std::cout << "VisitCXXNewExpr 警告: 使用 new 为多级指针分配空间, 不能处理" << std::endl;
                return true;
            }else{
                break;
            }
        }else{
            std::cout << "VisitCXXNewExpr 警告: 出现未知情况, 不能处理" << std::endl;
            return true;
        }
    }

    if(FindType==false || FindLength==false){
        std::cout << "VisitCXXNewExpr 警告: 类型 或 长度 未解析成功, 不能处理" << std::endl;
        return true;
    }

    SourceRange TmpRange;
    TmpRange.setBegin(NewBegin);
    TmpRange.setEnd( NewBegin.getLocWithOffset(NewEndOffset-NewBeginOffset) );

    const char* tmp = &start[NewBeginOffset];
    std::string code0(tmp, (NewEndOffset-NewBeginOffset+1));

    std::string code = "("; code += type; code += "*)";
    code += OAO_NEW_NAME; code += "( (void*)("; code += code0; code += "), sizeof(";
    code += type; code += "), ("; code += length; code += ") )";
    
    vectorFunc[indexFunc].vectorReplace.emplace_back(TmpRange, code, Rewrite);

    return true;
}

// wfr 20190430 将 return 节点的范围保存到当前节点
bool OAOASTVisitor::VisitReturnStmt(ReturnStmt *S){
    if(BeginFlag==true && FuncStack.empty()==false){
        int indexFunc = FuncStack.back().indexFunc;
        NODE_INDEX indexNode;
        SourceLocation BeginLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
        SourceLocation EndLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getEndLoc());
        InWhichSEQOMP(indexNode, indexFunc, Rewrite.getSourceMgr().getFileOffset(BeginLoc));
        SEQ_PAR_NODE_BASE* pNodeBase = getNodeBasePtr(vectorFunc[indexFunc], indexNode);

        pNodeBase->ReturnRange.init(BeginLoc, EndLoc, Rewrite);
    }
    return true;
}

// wfr 20180318 将 DeclStmt *S 写入 pDeclStmtStack, 这个包含着: 插入指针区域长度变量的 位置 的信息
bool OAOASTVisitor::TraverseDeclStmt( DeclStmt *S, DataRecursionQueue *Queue){

    // 将 DeclStmt *S 写入 pDeclStmtStack, 这个包含着: 插入指针区域长度变量的 位置 的信息
    // pDeclStmtStack.push_back(S);
    DeclStmtStack.emplace_back(S->getBeginLoc(), S->getEndLoc(), Rewrite);

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()){
        do { 
            if (!getDerived().WalkUpFromDeclStmt(S)){
                return false;
            }
        } while (false);
    }

    for (auto *I : S->decls()) { 
        do {
            if (!getDerived().TraverseDecl(I)){
                DeclStmtStack.pop_back();
                return false;
            }
        } while (false); 
    }
    ShouldVisitChildren = false; 
    
    if (ShouldVisitChildren) { 
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) {
            do {
                if (!(has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)))){
                    DeclStmtStack.pop_back();
                    return false;
                }
            } while (false);
        }
    }
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()){
        do {
            if (!getDerived().WalkUpFromDeclStmt(S)){
                return false;
            }
        } while (false);
    }
    return ReturnValue;
}
bool OAOASTVisitor::VisitDeclStmt(DeclStmt *S) { DeclStmtStack.pop_back(); return true; }

// wfr 20190307 为了获得变量作用域, 要建立 CompoundStmt 栈, 栈顶即为当前所在的 "{ }" 的范围, 即为变量的作用域
bool OAOASTVisitor::TraverseCompoundStmt( CompoundStmt *S, DataRecursionQueue *Queue) {

    // CompoundStmt 入栈
    CompoundStack.emplace_back(S->getBeginLoc(), S->getEndLoc(), this->Rewrite);
    SourceLocation TmpLoc;
    // 存入函数的 {} 的范围
    if(BeginFlag==true && FuncStack.empty()==false && vectorFunc[FuncStack.back().indexFunc].CompoundRange.BeginLoc==TmpLoc){
        vectorFunc[FuncStack.back().indexFunc].CompoundRange.init(S->getBeginLoc(), S->getEndLoc(), Rewrite); // 当前代码所在的函数
    }

	bool ShouldVisitChildren = true;
	bool ReturnValue = true;
	if (!getDerived().shouldTraversePostOrder()) do { if (!getDerived().WalkUpFromCompoundStmt(S)) return false; } while (false); 
	
	if (ShouldVisitChildren) { 
		for (Stmt * SubStmt : getDerived().getStmtChildren(S)) { 
			do { 
				if (!(has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)))) 
				{
                    // CompoundStmt 出栈
                    CompoundStack.pop_back();
                    return false;
                }
			} while (false); 
		} 
	} 
	
	if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()) do { if (!getDerived().WalkUpFromCompoundStmt(S)) return false; } while (false); 
    return ReturnValue;
}
// CompoundStmt 出栈
bool OAOASTVisitor::VisitCompoundStmt(CompoundStmt *S) { CompoundStack.pop_back(); return true; }

// wfr 20190319 当一个函数的返回值是指针类型的时候, 认为在函数中分配了内存, 并返回内存首地址
bool OAOASTVisitor::VisitCallExpr(CallExpr *S){
    int indexCaller;
    // int indexCallee;
    NODE_INDEX indexNode;
    //FunctionDecl* pFuncDecl = S->getDirectCallee(); // 被调用函数的定义的地址

    if(BeginFlag==true && FuncStack.empty()==false){
        indexCaller = FuncStack.back().indexFunc; // 当前代码所在的函数
    }else{
        return true;
    }

    if(indexCaller<0){
        std::cout << "VisitCallExpr 错误: 当前函数的 indexCaller<0" << std::endl;
        exit(1);
    }

    if(ASTStack.empty()){
        std::cout << "VisitCallExpr 警告: ASTStack 为空" << std::endl;
        return true;
    }
   
    std::string ASTOpName = ASTStack.back().Name; // AST 操作符的名称字符串
    int OperandID = ASTStack.back().OperandID; // 操作符的 第几个操作数

    // 这里获得函数返回值类型的 字符串
    QualType CallReturnType = S->getCallReturnType(OAOASTContext);
    SplitQualType T_split = CallReturnType.split();
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    std::string ReturnTypeName = QualType::getAsString(T_split, PrintPolicy);

    return true;
}

// wfr 20190210 在该函数中处理函数调用, 拆分出函数节点
// 调用该函数前, 应该有对应的 FUNC_INFO, 否则出错
// 将 CallExpr 压入 ASTStack
// 对于已经处理好的函数, 读取函数对于 参数的同步状态要求 以及 结束时同步状态, 标记函数节点为完成状态
// 对于 vectorFunc 中没有的函数, 报错
// 对于 vectorFunc 存在 但 没处理完成的 FUNC_INFO, 将拆分出来的节点设置未完成
bool OAOASTVisitor::TraverseCallExpr(CallExpr *S, DataRecursionQueue *Queue){
    // wfr 20190210
    //int ChildID = 0;
    bool flag = false;
    if(BeginFlag==true && FuncStack.empty()==false){

        std::cout << "TraverseCallExpr: 进入" << std::endl;
        
        int indexFuncCurrent = FuncStack.back().indexFunc; // 当前代码所在的函数
        NODE_INDEX indexNode; // 当前函数调用语句所在的 Node
        if(indexFuncCurrent>=0){
            SourceLocation SrcLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
            InWhichSEQOMP(indexNode, indexFuncCurrent, Rewrite.getSourceMgr().getFileOffset(SrcLoc));
        }
        
        int indexFuncCallee = -1; // 被调用函数的 index
        FunctionDecl* pFuncDecl = getCalleeFuncPtr(S); // 被调用函数的定义的地址
        
        if(pFuncDecl == NULL){
                std::cout << "TraverseCallExpr 警告: 被调函数不是 FunctionDecl 类型" << std::endl;
                exit(1);
        }else{
            if(pFuncDecl->getPrimaryTemplate()!=NULL){ // 如果是函数模板的实例化 的 调用
                indexFuncCallee = WhichFuncDecl(pFuncDecl->getPrimaryTemplate()->getTemplatedDecl());
            }else{
                indexFuncCallee = WhichFuncDecl(pFuncDecl);
            }
        }

        std::string CalleeName = pFuncDecl->getNameAsString();
        std::cout << "TraverseCallExpr: 被调函数名 " << std::dec << CalleeName << std::endl;

        // wfr 20190720 在这里将:
        // 1. malloc 替换为 OAOMalloc
        // 2. free 替换为 OAOFree
        // 3. 保存 被调函数的 indexFuncCallee 到当前函数的 vectorCallee
        int indexCaller = FuncStack.back().indexFunc; // 当前代码所在的函数
        SourceLocation BeginLoc = S->getBeginLoc();
        SourceRange TmpRange;
        TmpRange.setBegin( Rewrite.getSourceMgr().getExpansionLoc( BeginLoc ) );
        if(CalleeName=="malloc"){
            TmpRange.setEnd( Rewrite.getSourceMgr().getExpansionLoc( BeginLoc.getLocWithOffset(5) ) );
            vectorFunc[indexCaller].vectorReplace.emplace_back(TmpRange, OAO_MALLOC_NAME, Rewrite);
        }else if (CalleeName=="free"){
            TmpRange.setEnd( Rewrite.getSourceMgr().getExpansionLoc( BeginLoc.getLocWithOffset(3) ) );
            vectorFunc[indexCaller].vectorReplace.emplace_back(TmpRange, OAO_FREE_NAME, Rewrite);
        }else{
            if(indexFuncCallee>=0){
                vectorFunc[indexFuncCurrent].vectorCallee.push_back(indexFuncCallee);
            }
        }

        if(indexFuncCallee<0 && CalleeName=="free"){ // 这里处理 free 函数, 将 free 函数存入 vectorFunc
            std::cout << "TraverseCallExpr: 处理 free 函数" << std::endl;

            // 新建函数节点
            vectorFunc.emplace_back(pFuncDecl);
            indexFuncCallee = vectorFunc.size()-1;
            vectorFunc[indexFuncCallee].Name = "free";
            vectorFunc[indexFuncCallee].NumParams = 1;
            
            // 新建变量节点
            ParmVarDecl* pParm = pFuncDecl->getParamDecl(0); // 获得第0个形参定义的地址, 也应该是唯一的形参
            // ParmVarDecl* pParm  = NULL;
            NODE_INDEX indexDefNode(NODE_TYPE::SEQUENTIAL, -1);
            vectorFunc[indexFuncCallee].vectorVar.emplace_back(pParm, Rewrite);
            InitVar(vectorFunc[indexFuncCallee].vectorVar.back(), (VarDecl*)pParm, indexFuncCallee, indexDefNode);

            // 新建形参节点
            STATE_CONSTR TmpStReq(ST_REQ_HOST_ONLY);
            STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_FREE);
            vectorFunc[indexFuncCallee].vectorParm.emplace_back(0, vectorFunc[indexFuncCallee].vectorVar.back().Type, 
                                                                SYNC_STATE::HOST_ONLY, SYNC_STATE::HOST_ONLY, 
                                                                TmpStReq, TmpStTransFunc);

            // 设置分析完成标识
            vectorFunc[indexFuncCallee].isComplete = true;
        }

        std::cout << "TraverseCallExpr: indexFuncCurrent = " << std::dec << indexFuncCurrent << std::endl;
        std::cout << "TraverseCallExpr: indexNode.index = " << std::dec << indexNode.index << std::endl;
        std::cout << "TraverseCallExpr: 被调函数 indexFuncCallee = " << std::dec << indexFuncCallee << std::endl;

        if(!ASTStack.empty()){
            std::string ASTOpName = ASTStack.back().Name; // AST 操作符的名称字符串
            int OperandID = ASTStack.back().OperandID; // 操作符的 第几个操作数

            // 这里获得函数返回值类型的 字符串
            QualType CallReturnType = S->getCallReturnType(OAOASTContext);
            SplitQualType T_split = CallReturnType.split();
            LangOptions MyLangOptions;
            PrintingPolicy PrintPolicy(MyLangOptions);
            std::string ReturnTypeName = QualType::getAsString(T_split, PrintPolicy);

            // wfr 20190319 当一个函数的返回值是指针类型的时候, 认为在函数中分配了内存, 并返回内存首地址
            // if( ReturnTypeName.back() == '*' && 0<=OperandID && OperandID<(int)ASTStack.back().Operand.size()
            //     && ASTStack.back().OperandState[OperandID] == false){
            // wfr 20190720 认为 只有 malloc 分配内存
            if( CalleeName=="malloc" && 0<=OperandID && OperandID<(int)ASTStack.back().Operand.size()
                && ASTStack.back().OperandState[OperandID] == false){
                ASTStack.back().Operand[OperandID] = -1;
                ASTStack.back().vectorInfo[OperandID] = "malloc";
                ASTStack.back().OperandState[OperandID] = true;

                std::cout << "TraverseCallExpr: 找到 malloc" << std::endl;
            }
        }

        // 新建 ASTStack 项
        if(indexFuncCurrent>=0 && indexNode.index>=0 && indexFuncCallee>=0){

            std::cout << "TraverseCallExpr: 分离函数调用" << std::endl;

            flag = true;
            
            ASTStack.emplace_back((Stmt*)S, "CallExpr", -1, S->getNumArgs());
            indexNode = SplitFuncNode(S);
            if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                // 保存信息到 ASTStack
                ASTStack.back().indexNode = indexNode;
                ASTStack.back().indexFunc = indexFuncCallee;

                // 完善函数调用节点的信息
                SEQ_REGION& SEQ = vectorFunc[indexFuncCurrent].vectorSEQ[indexNode.index];
                SEQ.pFuncCall = S;
                SEQ.indexFunc = indexFuncCallee;
                // int NumParams = vectorFunc[indexFuncCallee].NumParams; // wfr 20191221 形参个数
                int NumParams = vectorFunc[indexFuncCallee].vectorParm.size(); // wfr 20191221 形参个数
                // wfr 20191221 给实参列表 vectorVarRef 分配空间
                SEQ.vectorVarRef.resize(NumParams);
            }else{ // 并行节点没有拆分, 要特殊处理, 将并行域中的函数调用信息 存入 vectorVarRef
                vectorFunc[indexFuncCallee].UsedInOMP = true; // 设置标识, 表示在并行域中被调用过

                OMP_REGION& OMP = vectorFunc[indexFuncCurrent].vectorOMP[indexNode.index];

                // 完善函数调用的信息
                OMP.vectorVarRef.emplace_back();
                OMP.vectorVarRef.back().pFuncCall = S;
                OMP.vectorVarRef.back().indexCallee = indexFuncCallee;
                // int NumParams = vectorFunc[indexFuncCallee].NumParams; // wfr 20191221 形参个数
                int NumParams = vectorFunc[indexFuncCallee].vectorParm.size(); // wfr 20191221 形参个数
                // wfr 20191221 给实参列表 vectorVarRef 分配空间
                OMP.vectorVarRef.back().RefList.clear();
                OMP.vectorVarRef.back().RefList.resize(NumParams);
                OMP.vectorVarRef.back().vectorArgIndex.clear();
                OMP.vectorVarRef.back().vectorArgIndex.resize(NumParams, -1);

                // 保存信息到 ASTStack
                ASTStack.back().init(S, "CallExpr", -1, NumParams);
                ASTStack.back().indexNode = indexNode;
                ASTStack.back().indexFunc = indexFuncCallee;
                ASTStack.back().indexVarRef = OMP.vectorVarRef.size() - 1;
                
            }
            ASTStack.back().OperandID = -1; // 用来跳过第一个子节点, 因为第一个子节点是函数引用, 不是函数参数, 所以跳过
        }
    }
    
    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromCallExpr(S)){
            if(flag){
                ASTStack.pop_back();
            }
            return false;
        }
    }
    if(ShouldVisitChildren){
        for(Stmt * SubStmt : getDerived().getStmtChildren(S)){
            if(!(has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)))) 
            {
                return false;
            }
            if(flag == true){

                // 如果 Queue != NULL 就要先把所有直接子节点入队, 然后再处理, 所以这里需要在 SubStmts 保存一份除去函数引用的 其他直接子节点
                // 这样就可以在 TraverseStmt 中 根据 SubStmts 来判断处理的是函数的第几个实参, 从而设置 OperandID
                if(ASTStack.back().OperandID>=0){
                    ASTStack.back().SubStmts[ASTStack.back().OperandID] = SubStmt;
                }

                // 处理完第一个子节点再 ++, 就可以跳过第一个子节点
                // 因为第一个子节点是函数引用, 不是函数参数, 所以跳过
                ASTStack.back().OperandID++;
            }
        }
    }
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromCallExpr(S)){
            if(flag){
                ASTStack.pop_back();
            }
            return false;
        }
    }

    return ReturnValue;
}

// 在调用该函数之前, 应该已经调用了 VisitVarDecl 函数, 函数参数信息应该已经被存入 vectorVar
// 在调用该函数之前, FuncStack 中栈顶应该保存了当前函数的信息, 可以从中获取 indexFunc
bool OAOASTVisitor::VisitParmVarDecl(ParmVarDecl *D){
    int indexFunc = -1;
    if(BeginFlag==true && FuncStack.empty()==false){
        indexFunc = FuncStack.back().indexFunc;
    }else{
        return true;
    }

    if(indexFunc<0){
        std::cout << "VisitParmVarDecl 错误: 当前函数的 indexFunc<0" << std::endl;
        exit(1);
    }

    FUNC_INFO& Func = vectorFunc[indexFunc];
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    std::vector<FUNC_PARM_VAR_INFO>& vectorParm = Func.vectorParm;

    std::vector<VARIABLE_INFO>& vectorDeclVar = Func.vectorDeclVar;
    std::vector<FUNC_PARM_VAR_INFO>& vectorDeclParm = Func.vectorDeclParm;

    if(vectorFunc[indexFunc].ptrDefine != NULL){

        // 判断 vectorVar 尾项是不是当前函数的当前参数
        if(vectorVar.empty()){
            return true; // wfr 20180814 应该是遇到了函数声明处的参数变量定义, 不处理
        }else if(vectorVar.back().ptrDecl != D){
            std::cout << "VisitParmVarDecl 错误: vectorVar 尾项不是当前函数的当前参数" << std::endl;
            exit(1);
        }

        // 获得函数参数信息
        // 在此之前, 该变量刚刚被存入 vectorVar, 即是其中的最后一项
        int indexVar = vectorVar.size() - 1;
        VAR_TYPE ParmType = vectorVar.back().Type;

        // 保存信息: 参数变量定义在哪个 node 中
        vectorVar[indexVar].indexDefNode = vectorFunc[indexFunc].MapEntry;
        // 保存变量作用域信息
        if(!CompoundStack.empty()){
            vectorVar[indexVar].Scope = CompoundStack.back();
        }else{
            std::cout << "VisitParmVarDecl 错误: CompoundStack 空" << std::endl;
            exit(1);
        }
        
        // 保存函数参数信息
        vectorParm.emplace_back(indexVar, ParmType);

    // 这里是函数定义的情况
    }else if(vectorFunc[indexFunc].ptrDecl!=NULL && vectorFunc[indexFunc].ptrDecl!=vectorFunc[indexFunc].ptrDefine){
        
        // 判断 vectorVar 尾项是不是当前函数的当前参数
        if(vectorDeclVar.empty()){
            return true; // wfr 20180814 应该是遇到了函数声明处的参数变量定义, 不处理
        }else if(vectorDeclVar.back().ptrDecl != D){
            std::cout << "VisitParmVarDecl 错误: vectorDeclVar 尾项不是当前函数的当前参数" << std::endl;
            exit(1);
        }
        
        // 获得函数参数信息
        // 在此之前, 该变量刚刚被存入 vectorDeclVar, 即是其中的最后一项
        int indexVar = vectorDeclVar.size() - 1;
        VAR_TYPE ParmType = vectorDeclVar.back().Type;

        // 保存函数参数信息
        vectorDeclParm.emplace_back(indexVar, ParmType);
    }else{
        std::cout << "VisitParmVarDecl 错误: ptrDefine 和 ptrDecl 都为 NULL" << std::endl;
        exit(1);
    }

    return true;
}

bool OAOASTVisitor::TraverseFunctionDecl(FunctionDecl *D){

    std::string FuncName = D->getNameAsString();
    if(FuncName=="main"){
        std::cout << "TraverseFunctionDecl: 找到 main 函数声明" << std::endl;
        BeginFlag = true;
    }

    int indexFunc;
    FunctionDecl *Prev = D->getPreviousDecl();
    //std::cout << "TraverseFunctionDecl: D = 0x" << std::hex << (void*)D << std::endl;
    //std::cout << "TraverseFunctionDecl: Prev = 0x" << std::hex << (void*)Prev << std::endl;
    bool  NeedProcess = false; // 表示遇到的函数是否需要处理

    // 不是模版函数 或者 是函数模版
    if( BeginFlag == true &&
        ( (D->getDescribedFunctionTemplate()==NULL && D->getPrimaryTemplate()==NULL) || 
          (D->getDescribedFunctionTemplate()!=NULL && D->getDescribedFunctionTemplate()->getTemplatedDecl()==D) )
    ){
        std::cout << "TraverseFunctionDecl: 找到函数 " << D->getNameAsString() << std::endl;
        // 入栈 CompoundStmt
        std::cout << "TraverseFunctionDecl: 建立 CompoundStackNode" << std::endl;
        CompoundStack.emplace_back(D->getBeginLoc(), D->getEndLoc(), this->Rewrite);
        std::cout << "TraverseFunctionDecl: 建立 CompoundStackNode 完成" << std::endl;

        if(D->hasBody()){ // 大体上说, 这里用来判断这里是不是一个函数定义, 还有两个函数可供选择, 且越来越严格 isThisDeclarationADefinition / isDefined
            //SourceRange FuncSourceRange = D->getSourceRange();
            std::cout << "TraverseFunctionDecl: 有 Body" << std::endl;
            Stmt* FuncBodyStmt = D->getBody();
            
            // 下边建立 CFG
            // CFG
            std::cout << "TraverseFunctionDecl: 建立 CFG" << std::endl;
            std::unique_ptr<CFG> pFuncCFG = CFG::buildCFG(D, FuncBodyStmt, &OAOASTContext, CFG::BuildOptions());
            std::cout << "TraverseFunctionDecl: 建立 CFG 完成" << std::endl;
            //pFuncCFG->print(llvm::errs(), LangOptions(), true);
            // 上边建立 CFG

            if (Prev){ // 为真, 则说明当前函数在之前被声明过, 这里是定义/实现
                indexFunc = WhichFuncDecl(Prev); // vectorFunc 中搜索函数的声明
                if(indexFunc < 0){
                    std::cout << "TraverseFunctionDecl 错误: 找到函数定义, 但是没找到之前的函数声明" << std::endl;
                    exit(1);
                }else{
                    // 初始化 FUNC_INFO, 建立最初的所有串行节点
                    vectorFunc[indexFunc].init(Prev, D, this->Rewrite, *pFuncCFG);
                }
            }else{ // 这种情况是 声明同时定义
                std::cout << "TraverseFunctionDecl: 使用 CFG 初始化函数节点" << std::endl;
                std::string FuncName = D->getNameAsString();
                if(FuncName == "main"){
                    indexFunc = 0; // 如果是 main 函数, 就放到第0个
                    vectorFunc.emplace(vectorFunc.begin(), D, D, this->Rewrite, *pFuncCFG);
                }else{
                    vectorFunc.emplace_back(D, D, this->Rewrite, *pFuncCFG);
                    indexFunc = vectorFunc.size()-1;
                }
                std::cout << "TraverseFunctionDecl: 使用 CFG 初始化函数节点完成" << std::endl;
            }
        }else{
            std::cout << "TraverseFunctionDecl: 无 Body" << std::endl;
            std::string FuncName = D->getNameAsString();
            if(FuncName == "main"){
                indexFunc = 0; // 如果是 main 函数, 就放到第0个
                vectorFunc.emplace(vectorFunc.begin(), D);
            }else{
                vectorFunc.emplace_back(D);
                indexFunc = vectorFunc.size()-1;
            }
        }

        // 入栈当前函数信息
        NeedProcess = true;
        FuncStack.emplace_back(indexFunc, D);
        // 遍历函数的参数列表, 获得参数变量信息
        // 之后遍历建立的 CFG 获得变量引用情况
        std::cout << "TraverseFunctionDecl: 新建了 FuncStack 节点" << std::endl;
    }

	//bool ShouldVisitChildren = true;
	bool ReturnValue = true;
	if (!getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromFunctionDecl(D)){
            if(NeedProcess){
                // 这里将当前函数信息出栈
                FuncStack.pop_back();
                // 出栈 CompoundStmt
                CompoundStack.pop_back();
                std::cout << "TraverseFunctionDecl: 退出" << std::endl;
            }
            return false;
        }
    }
	
    //ShouldVisitChildren = false;
    ReturnValue = TraverseFunctionHelper(D);

	//if (ReturnValue && ShouldVisitChildren){
    //    if (!getDerived().TraverseDeclContextHelper(dyn_cast<DeclContext>(D))) return false;
    //}
	if (ReturnValue && getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromFunctionDecl(D)){
            if(NeedProcess){
                // 这里将当前函数信息出栈
                FuncStack.pop_back();
                // 出栈 CompoundStmt
                CompoundStack.pop_back();
                std::cout << "TraverseFunctionDecl: 退出" << std::endl;
            }
            return false;
        }
    }
	
    if(NeedProcess){
        // 这里将当前函数信息出栈
        FuncStack.pop_back();
        // 出栈 CompoundStmt
        CompoundStack.pop_back();
        std::cout << "TraverseFunctionDecl: 退出" << std::endl;
    }
    return ReturnValue;
}

bool OAOASTVisitor::TraverseDeclRefExpr(DeclRefExpr *S, DataRecursionQueue *Queue){

    if(BeginFlag==true && FuncStack.empty()==false){

        // wfr 20190420 如果在处理 OMP, 且不在 主体 中, 就直接返回
        if(!OMPStack.empty()){
            OMP_REGION& OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
            if(OMP.pBodyStmt==NULL){
                std::cout << "TraverseDeclRefExpr: 不处理 OMP 主体外的引用" << std::endl;
                return true;
            }
        }

        std::cout << "TraverseDeclRefExpr: 进入" << std::endl;
        
        std::string DeclKindName = S->getDecl()->getDeclKindName();
        if(!ASTStack.empty() && (DeclKindName == "Var" || DeclKindName == "ParmVar") ){ // 如果 ASTStack 非空 且 声明的是一个变量
            // 获得类型
            LangOptions MyLangOptions;
            PrintingPolicy PrintPolicy(MyLangOptions);
            std::string TypeName;
            TypeName = QualType::getAsString( S->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等

            if( TypeName.back()=='*' && ASTStack.back().OperandID>=0 && 
                ASTStack.back().AccessType[ASTStack.back().OperandID]==ASTStackNode::PTR_ARY::PTR_UNINIT){ // 如果变量是指针类型, 且没经过初始化
                ASTStack.back().AccessType[ASTStack.back().OperandID]=ASTStackNode::PTR_ARY::PTR;
            }
        }
        std::cout << "TraverseDeclRefExpr: 调用 OAOVisitDeclRefExpr" << std::endl;
        // 调用自定义的 Visit 函数
        OAOVisitDeclRefExpr(S);
        std::cout << "TraverseDeclRefExpr: 调用 OAOVisitDeclRefExpr 完成" << std::endl;
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromDeclRefExpr(S)) { return false; }
    }
    if (!getDerived().TraverseNestedNameSpecifierLoc(S->getQualifierLoc())) { return false; }
    if (!getDerived().TraverseDeclarationNameInfo(S->getNameInfo())) { return false; }


    if (!getDerived().TraverseTemplateArgumentLocsHelper(S->getTemplateArgs(), S->getNumTemplateArgs())) { return false; }    
    
    if (ShouldVisitChildren) {
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) {
            if (!(has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt))))
            {
                return false;
            }
        }
    }
    
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromDeclRefExpr(S)){
            return false;
        }
    }
    
    return ReturnValue;
}

bool OAOASTVisitor::TraverseMemberExpr( MemberExpr *S, DataRecursionQueue *Queue) {

    if(BeginFlag==true && FuncStack.empty()==false){

        // wfr 20190420 如果在处理 OMP, 且不在 主体 中, 就直接返回
        if(!OMPStack.empty()){
            OMP_REGION& OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
            if(OMP.pBodyStmt==NULL){
                std::cout << "TraverseMemberExpr: 不处理 OMP 主体外的引用" << std::endl;
                return true;
            }
        }

        std::cout << "TraverseMemberExpr: 进入" << std::endl;
        if(!ASTStack.empty()){ // 如果 ASTStack 非空
            // 获得类型
            LangOptions MyLangOptions;
            PrintingPolicy PrintPolicy(MyLangOptions);
            std::string TypeName;
            TypeName = QualType::getAsString( S->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等

            if( TypeName.back()=='*' && ASTStack.back().OperandID>=0 && 
                ASTStack.back().AccessType[ASTStack.back().OperandID]==ASTStackNode::PTR_ARY::PTR_UNINIT)
            { // 如果变量是指针类型, 且没经过初始化
                ASTStack.back().AccessType[ASTStack.back().OperandID]=ASTStackNode::PTR_ARY::PTR;
            }
        }
        std::cout << "TraverseMemberExpr: 调用 OAOVisitMemberExpr" << std::endl;
        // 调用自定义的 Visit 函数
        OAOVisitMemberExpr(S);
        std::cout << "TraverseMemberExpr: 调用 OAOVisitMemberExpr 完成" << std::endl;
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromMemberExpr(S)) {
            return false;
        }
    }
    
    if (!getDerived().TraverseNestedNameSpecifierLoc(S->getQualifierLoc())) {return false;}
    if (!getDerived().TraverseDeclarationNameInfo(S->getMemberNameInfo())) {return false;}
    if (!getDerived().TraverseTemplateArgumentLocsHelper(S->getTemplateArgs(), S->getNumTemplateArgs())) {return false;}
    
    if (ShouldVisitChildren) {
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) { 
            if (!(has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type < decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type> (*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)) ))
            {
                return false;
            }
        }
    }

    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()){ 
        if (!getDerived().WalkUpFromMemberExpr(S)) {
            return false;
        }
    }
    return ReturnValue;
}

bool OAOASTVisitor::TraverseImplicitCastExpr( ImplicitCastExpr *S, DataRecursionQueue *Queue)
{
    if(BeginFlag==true && FuncStack.empty()==false){
        std::cout << "TraverseImplicitCastExpr: 处理" << std::endl;
        if(!ASTStack.empty()){ // 如果 ASTStack 非空
            std::string CastKindName = S->getCastKindName();
            if(CastKindName.back()=='*' && ASTStack.back().OperandID>=0 && 
                ASTStack.back().AccessType[ASTStack.back().OperandID]==ASTStackNode::PTR_ARY::PTR_UNINIT)
            { // 如果是指针类型
                ASTStack.back().AccessType[ASTStack.back().OperandID]=ASTStackNode::PTR_ARY::PTR;
            }
        }
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromImplicitCastExpr(S)){
            return false;
        }
    }
    if (ShouldVisitChildren) {
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) {
            if (!(
            has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt))
            )){
                return false;
            }
        }
    }
    
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromImplicitCastExpr(S)){
            return false;
        }
    }
    return ReturnValue;
}

bool OAOASTVisitor::TraverseArraySubscriptExpr( ArraySubscriptExpr *S, DataRecursionQueue *Queue ) {
    if(BeginFlag==true && FuncStack.empty()==false){
        // 将 OperandID 指向的操作数的访问方式改成 ARRAY, 表示引用指针指向的数组的元素
        if(!ASTStack.empty()){
            if(ASTStack.back().OperandID>=0 && ASTStack.back().AccessType[ASTStack.back().OperandID]==ASTStackNode::PTR_ARY::PTR_UNINIT){
                ASTStack.back().AccessType[ASTStack.back().OperandID]=ASTStackNode::PTR_ARY::ARRAY;
            }
        }
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromArraySubscriptExpr(S)) {
            return false;
        }
    }
    
    
    if (ShouldVisitChildren) {
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) {
            if (!(
            has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this) .TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt))
            )) {
                return false;
            }
        }
    }
    
    
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()) {
        if (!getDerived().WalkUpFromArraySubscriptExpr(S)) {
            return false;
        }
    }
    return ReturnValue;
}

bool OAOASTVisitor::TraverseVarDecl(VarDecl *D){

    if(BeginFlag==true && FuncStack.empty()==false){

        // wfr 20190420 如果在处理 OMP, 且不在 主体 中, 就直接返回
        if(!OMPStack.empty()){
            OMP_REGION& OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
            if(OMP.pBodyStmt==NULL){
                std::cout << "TraverseVarDecl: 不处理 OMP 主体外的定义" << std::endl;
                return true;
            }
        }

        std::cout << std::endl;
        std::cout << "TraverseVarDecl: 进入" << std::endl;
        //MyDumper.VisitVarDecl(S);

        //这里设置标识变量, 告知后续的 AST 遍历, 是在解析 VarDecl
        std::string OpName = "VarDecl";
        ASTStack.emplace_back((VarDecl*)D, OpName, 0, 1);
    }

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    if (!getDerived().shouldTraversePostOrder()){
        std::cout << "Do not shouldTraversePostOrder" << std::endl;
        if (!getDerived().WalkUpFromVarDecl(D)){
            if(BeginFlag==true && FuncStack.empty()==false){
                // 这里设置标识变量,  告知后续的 AST 遍历, 不是在解析 VarDecl
                std::cout << "jump out of TraverseVarDecl from 1st WalkUpFromVarDecl" << std::endl;
                ASTStack.pop_back();
            }
            return false;
        }
    }
    
    if (!getDerived().TraverseVarHelper(D)){
        if(BeginFlag==true && FuncStack.empty()==false){
            // 这里设置标识变量,  告知后续的 AST 遍历, 不是在解析 VarDecl
            std::cout << "jump out of TraverseVarDecl after TraverseVarHelper" << std::endl;
            ASTStack.pop_back();
        }
        return false;
    }
    
    if (ReturnValue && ShouldVisitChildren){
        if (!getDerived().TraverseDeclContextHelper(dyn_cast<DeclContext>(D))){
            if(BeginFlag==true && FuncStack.empty()==false){
                // 这里删除标识变量,  告知后续的 AST 遍历, 不是在解析 VarDecl
                std::cout << "jump out of TraverseVarDecl after TraverseVarHelper" << std::endl;
                ASTStack.pop_back();
            }
            return false;
        }
    }

    if (ReturnValue && getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromVarDecl(D)){
            if(BeginFlag==true && FuncStack.empty()==false){
                // 这里删除标识变量,  告知后续的 AST 遍历, 不是在解析 VarDecl
                std::cout << "jump out of TraverseVarDecl from 2nd WalkUpFromVarDecl" << std::endl;
                ASTStack.pop_back();
            }
            return false;
        }
    }

    if(BeginFlag==true && FuncStack.empty()==false){
        // 这里删除标识变量,  告知后续的 AST 遍历, 不是在解析 VarDecl
        ASTStack.pop_back();
        std::cout << "TraverseVarDecl: 退出" << std::endl;
    }
    return ReturnValue;
}

// wfr 20190324 有三种信息需要 插入 / 修改 源代码:translation
// 1. 每个节点中 vectorTrans 中的 待插入 的信息
// 2. offloading 指令
// 3. 每个变量的 VARIABLE_INFO 中的 待插入 的信息
// 在此函数中改写源文件
bool OAOASTVisitor::translation(){
    DEFINE_ST_REQ_ST_TRANS

    // 0. 在文件头部插入 #include <malloc.h>
    SourceLocation TmpLoc = vectorFunc[0].BeginLoc;
    int offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    TmpLoc = TmpLoc.getLocWithOffset(-1*offset);
    std::string buffer = "#include <malloc.h>\n";
    buffer += "#include \"RunTime.h\"\n";
    // buffer += "DEFINE_ST_REQ_ST_TRANS\n";
    buffer += "STATE_CONSTR StConstrTarget;\n";
    Rewrite.InsertText(TmpLoc, buffer, true, true);

    // wfr 20190729 将 delete[] 替换成 OAODelete; 将 new 替换成 OAONew
    for(unsigned long i=0; i<GlobalReplace.size(); ++i){
        Rewrite.ReplaceText(GlobalReplace[i].Range, GlobalReplace[i].Code);
    }

    // wfr 20190811 插入全局变量相关代码
    for(unsigned long iVar = 0; iVar < vectorGblVar.size(); ++iVar){
        VARIABLE_INFO& Var = vectorGblVar[iVar];
        std::cout << "translation: 处理全局变量 " << Var.Rename << std::endl;
        std::cout << "translation: vectorInsert.size() = " << Var.vectorInsert.size() << std::endl;
        for(unsigned long j=0; j<Var.vectorInsert.size(); ++j){
            CODE_INSERT& InsertInfo = Var.vectorInsert[j];
            std::cout << "translation: 变量 " << Var.Rename << " 的第 " << std::dec << j << " 个插入位置" << std::endl;
            std::cout << "translation: vectorInsert[j].Code.size()  = " << InsertInfo.Code.size()  << std::endl;
            Rewrite.InsertText(InsertInfo.InsertLoc, InsertInfo.Code, true, false);
        }
        std::cout << "translation: vectorReplace.size() = " << Var.vectorReplace.size() << std::endl;
        for(unsigned long j=0; j<Var.vectorReplace.size(); ++j){ // wfr 20190429 在这里 插入 变量重命名替换, 替换 OMP 中引用的 外部 类的域
            CODE_REPLACE& ReplaceInfo = Var.vectorReplace[j];
            std::cout << "translation: 变量 " << Var.Rename << " 的第 " << std::dec << j << " 个替换位置" << std::endl;
            std::cout << "translation: vectorReplace[j].Code.size()  = " << ReplaceInfo.Code.size()  << std::endl;
            Rewrite.ReplaceText(ReplaceInfo.Range, ReplaceInfo.Code);
        }
    }

    for(unsigned long iFunc=0; iFunc<vectorFunc.size(); ++iFunc){ // 循环处理每个函数

        FUNC_INFO& Func = vectorFunc[iFunc];
        std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
        std::vector<SEQ_REGION>& vectorSEQ = Func.vectorSEQ;
        std::vector<OMP_REGION>& vectorOMP = Func.vectorOMP;
        std::vector<CODE_REPLACE>& vectorReplace = Func.vectorReplace;

        std::cout << "translation: 处理函数 " << Func.Name << std::endl;

        
        SEQ_PAR_NODE_BASE* pNodeBase;

        // 0. 插入 FUNC_INFO::vectorReplace 中的信息
        for(unsigned long i=0; i<vectorReplace.size(); ++i){
            Rewrite.ReplaceText(vectorReplace[i].Range, vectorReplace[i].Code);
        }

        // 1. 插入 每个变量的 VARIABLE_INFO 中的 待插入 的信息
        for(unsigned long iVar = 0; iVar < vectorVar.size(); ++iVar){
            VARIABLE_INFO& Var = vectorVar[iVar];
            std::cout << "translation: 处理变量 " << Var.Rename << std::endl;
            std::cout << "translation: vectorInsert.size() = " << Var.vectorInsert.size() << std::endl;
            for(unsigned long j=0; j<Var.vectorInsert.size(); ++j){
                CODE_INSERT& InsertInfo = Var.vectorInsert[j];
                std::cout << "translation: 变量 " << Var.Rename << " 的第 " << std::dec << j << " 个插入位置" << std::endl;
                std::cout << "translation: vectorInsert[j].Code.size()  = " << InsertInfo.Code.size()  << std::endl;
                Rewrite.InsertText(InsertInfo.InsertLoc, InsertInfo.Code, true, false);
            }
            std::cout << "translation: vectorReplace.size() = " << Var.vectorReplace.size() << std::endl;
            for(unsigned long j=0; j<Var.vectorReplace.size(); ++j){ // wfr 20190429 在这里 插入 变量重命名替换, 替换 OMP 中引用的 外部 类的域
                CODE_REPLACE& ReplaceInfo = Var.vectorReplace[j];
                std::cout << "translation: 变量 " << Var.Rename << " 的第 " << std::dec << j << " 个替换位置" << std::endl;
                std::cout << "translation: vectorReplace[j].Code.size()  = " << ReplaceInfo.Code.size()  << std::endl;
                Rewrite.ReplaceText(ReplaceInfo.Range, ReplaceInfo.Code);
            }
        }

        // 2.0. 插入 offloading 指令的后半个 "}"
        // for(unsigned long iOMP=0; iOMP<vectorOMP.size(); ++iOMP){
        //     OMP_REGION& OMP = vectorOMP[iOMP];
        //     if(OMP.indexEntry!=(int)iOMP){ // 如果不是 并行域起始节点 就跳过当前节点
        //         continue;
        //     }
        //     Rewrite.InsertText(OMP.OMPRange.EndLoc.getLocWithOffset(1), "\n}\n", true, false);
        // }

        // 2. 插入 vectorTrans 中的信息
        // 循环处理每个 SEQ/OMP 节点
        for(unsigned long NodeAccumulation=0; NodeAccumulation < (vectorSEQ.size()+vectorOMP.size()); ++NodeAccumulation){
            if(NodeAccumulation<vectorSEQ.size()){
                pNodeBase = (SEQ_PAR_NODE_BASE*)( &(vectorSEQ[NodeAccumulation]) );
            }else{
                pNodeBase = (SEQ_PAR_NODE_BASE*)( &(vectorOMP[NodeAccumulation-vectorSEQ.size()]) );
            }
            std::vector<TRANS>& vectorTrans = pNodeBase->vectorTrans;

            for(unsigned long iTrans=0; iTrans<vectorTrans.size(); ++iTrans){ // 循环处理整个 vectorTrans
                if(vectorTrans[iTrans].isWritten==true){ // 如果已经被写入源代码了, 就跳过当前项
                    continue;
                }
                SourceLocation iInsertLocation = vectorTrans[iTrans].InsertLocation;

                // 每个字符串保存不同类型的信息
                std::string VarRename = "\n"; unsigned long VarRenameLen = VarRename.size();
                // std::string CodeDataTrans = "\n"; unsigned long CodeDataTransLen = CodeDataTrans.size();
                // std::string CodeStTrans = "\n"; unsigned long CodeStTransLen = CodeStTrans.size();
                std::string CodeTrans = "\n"; unsigned long CodeTransLen = CodeTrans.size();
                
                // 遍历整个 vectorTrans, 找到所有同一插入位置的信息, 这些信息放在一起插入
                for(unsigned long jTrans=iTrans; jTrans<vectorTrans.size(); ++jTrans){
                    
                    SourceLocation jInsertLocation = vectorTrans[jTrans].InsertLocation;

                    // 如果插入位置相同
                    if(Rewrite.getSourceMgr().getFileOffset(iInsertLocation)==Rewrite.getSourceMgr().getFileOffset(jInsertLocation)){
                        VARIABLE_INFO& Var = vectorVar[vectorTrans[jTrans].index];
                        vectorTrans[jTrans].isWritten=true;

                        if(Var.isMember==true){
                            VarRename += Var.Rename;
                            VarRename += " = ";
                            VarRename += Var.FullName;
                            VarRename += ";\n";
                        }

                        STATE_CONSTR StConstrTarget = vectorTrans[jTrans].StConstrTarget;
                        std::string CodeConstrTarget = "StConstrTarget.init(";
                        CodeConstrTarget += std::to_string(StConstrTarget.ZERO);
                        CodeConstrTarget += ", ";
                        CodeConstrTarget += std::to_string(StConstrTarget.ONE);
                        CodeConstrTarget += ")";

                        if(vectorTrans[jTrans].type==TRANS_TYPE::DATA_TRANS){
                            CodeTrans += OAO_DATA_TRANS;
                            CodeTrans += "( ";
                            CodeTrans += Var.Rename;
                            CodeTrans += ", ";
                            CodeTrans += CodeConstrTarget;
                            CodeTrans += " );\n";
                        }else if(vectorTrans[jTrans].type==TRANS_TYPE::STATE_TRANS){
                            CodeTrans += OAO_ST_TRANS;
                            CodeTrans += "( ";
                            CodeTrans += Var.Rename;
                            CodeTrans += ", ";
                            CodeTrans += CodeConstrTarget;
                            CodeTrans += " );\n";
                        }else{
                            std::cout << "translation 错误: 传输类型错误" << std::endl;
                            exit(1);
                        }
                    }
                }

                // wfr 20190721 写入指针变量重命名
                if(VarRenameLen < VarRename.size()){
                    if(Rewrite.InsertText(iInsertLocation, VarRename, true, true)){
                        std::cout << "translation: VarRename, iInsertLocation 位置不可写" << std::endl;
                    }
                }

                // wfr 20190721 写入变量传输/状态转换指令
                if(CodeTransLen < CodeTrans.size()){
                    if(Rewrite.InsertText(iInsertLocation, CodeTrans, true, true)){
                        std::cout << "translation: CodeTrans, iInsertLocation 位置不可写" << std::endl;
                    }
                }
            }
        }


        // 3. 插入 offloading 指令, 即原来CPU上的 OpenMP 指令 替换成 支持 offloading 的指令
        for(unsigned long iOMP=0; iOMP<vectorOMP.size(); ++iOMP){
            OMP_REGION& OMP = vectorOMP[iOMP];
            if(OMP.indexEntry!=(int)iOMP){ // 如果不是 并行域起始节点 就跳过当前节点
                continue;
            }
            // std::string buffer = "#pragma omp target teams num_teams(20) thread_limit(512)\n"; // 以后再改 ？？？
            // std::string buffer = "#pragma omp target teams distribute parallel for\n{\n"; // 以后再改 ？？？
            // buffer += "#pragma omp distribute parallel for\n{\n"; // 以后再改 ？？？
            // SourceRange range(OMP.DirectorRange.BeginLoc.getLocWithOffset(-1*(OMP.DirectorRange.BeginCol)+1), OMP.DirectorRange.EndLoc.getLocWithOffset(0));
            // Rewrite.ReplaceText(range, buffer);

            std::string buffer;
            //ghn 20191015 有标量被修改，需要用map语句
            if(OMP.scalar != "NULL"){
                std::string name = OMP.scalar;
                buffer = "#pragma omp target teams distribute parallel for map(tofrom: " + name + ")";
            }else{
                buffer = "#pragma omp target teams distribute parallel for "; // 以后再改 ？？？
            }
            Rewrite.ReplaceText(OMP.ReplaceRange, buffer);
            
            // Rewrite.InsertText(OMP.DirectorRange.EndLoc, "\n{\n", true, false);
        }


        
    }
    
    return true;
}

// wfr 20190209 保存 OMP 中 LocalClass 信息
bool OAOASTVisitor::saveLocalClassInfo(VarDecl* pClassDecl, int indexFunc, int indexOMP){ // 保存 OMP 中 LocalClass 信息
    // 这里新建表项, 并从 offset 获得直观的代码范围参数, 即行号列号,以及 SourceLocation 表示的代码位置
    vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal.emplace_back(pClassDecl, this->Rewrite);
    VARIABLE_INFO& TempVar = vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal.back();

    // 这里初始化其他信息
    TempVar.Name = pClassDecl->getNameAsString();
    TempVar.isClass = true;
    TempVar.ptrDecl = pClassDecl; // 变量的定义的地址
    //TempVar.pDeclStmt = pDeclStmtStack.back(); // 保存 插入指针区域长度语句 位置 的信息
    TempVar.DeclRange.init(DeclStmtStack.back().BeginLoc, DeclStmtStack.back().EndLoc, Rewrite);
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    TempVar.TypeName = QualType::getAsString( pClassDecl->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等

    // 判断定义在哪个OMP域中
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pClassDecl->getBeginLoc());
    unsigned int Offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    NODE_INDEX  indexNode;
    InWhichSEQOMP(indexNode, indexFunc, Offset);
    // 保存信息: 变量定义在哪个 node
    TempVar.indexDefNode = indexNode;

    return true;
}

// wfr 20190209 保存 OMP 中 LocalMember 信息
//int OAOASTVisitor::saveLocalMemberInfo(MemberExpr* S, int indexLocalClass, int indexFunc, int indexOMP){
int OAOASTVisitor::saveLocalMemberInfo(FieldDecl* pMemberDecl, int indexLocalClass, int indexFunc, int indexOMP){
    //FieldDecl* pMemberDecl = S->getMemberDecl(); // 获得域定义的指针, 这个指针指向类的类型定义中的域定义, 而不是指向实例的定义
    // 这里新建表项, 并从 offset 获得直观的代码范围参数, 即行号列号,以及 SourceLocation 表示的代码位置
    std::vector<VARIABLE_INFO>& vectorVar = vectorFunc[indexFunc].vectorVar;
    std::vector<VARIABLE_INFO>& vectorLocal = vectorFunc[indexFunc].vectorOMP[indexOMP].vectorLocal;
    vectorLocal.emplace_back(pMemberDecl, this->Rewrite);
    VARIABLE_INFO& TempMember = vectorLocal.back();
    int indexLocalMember = vectorLocal.size()-1;
    // vectorLocal[indexLocalClass].vectorIndexMember.push_back(vectorLocal.size()-1); // 将 member 的 index 写入 class

    // member 的 offset 和 SourceLocation 没有意义就不再赋值；因为类实例定义的时候没有反映出 member
    
    TempMember.Name = pMemberDecl->getNameAsString();
    TempMember.ptrClass = vectorLocal[indexLocalClass].ptrDecl; // 如果是类的成员, 这里是类实例定义的地址
    //TempMember.pDeclStmt = vectorLocal[indexLocalClass].pDeclStmt; // 保存 插入指针区域长度语句 位置 的信息
    TempMember.DeclRange = vectorLocal[indexLocalClass].DeclRange;
    TempMember.indexClass = indexLocalClass; // 如果是类的成员, 这里是类的 VARIABLE_INFO 的 index
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    TempMember.TypeName = QualType::getAsString( pMemberDecl->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等
    TempMember.isMember = true;
    //TempMember.isArrow = S->isArrow();
    if(vectorLocal[indexLocalClass].TypeName.back()=='*'){
        TempMember.isArrow = true;
    }
    // 保存信息: 变量定义在哪个 node
    TempMember.indexDefNode = vectorLocal[indexLocalClass].indexDefNode;
    // 保存变量作用域信息
    TempMember.Scope = vectorLocal[indexLocalClass].Scope;

    std::string TypeName = TempMember.TypeName;
    // wfr 20190806 通过变量类型字符串, 分析获得变量类型
    VAR_TYPE VarType = getVarType(TypeName);
    TempMember.Type = VarType;

    if(vectorLocal[indexLocalClass].indexRoot>=0){ // 说明是个指针, 需要进行变量跟踪
        int indexClass = vectorLocal[indexLocalClass].indexRoot;
        VarDecl* pClassDecl = (VarDecl*)vectorVar[indexClass].ptrDecl;
        // 获得 member 变量信息
        int indexMember = getMemberIndex(pClassDecl, pMemberDecl, indexFunc);

        if( VarType==VAR_TYPE::PTR || VarType==VAR_TYPE::PTR_CONST 
            || VarType==VAR_TYPE::CLASS_PTR || VarType==VAR_TYPE::CLASS_PTR_CONST ){
            TempMember.indexRoot = vectorVar[indexMember].indexRoot;
            if(TempMember.indexRoot>=0){
                TempMember.RootName = vectorVar[TempMember.indexRoot].Name;
                TempMember.ptrRoot = vectorVar[TempMember.indexRoot].ptrDecl;
            }
        }else{
            TempMember.indexRoot = indexMember;
            if(indexMember>=0){
                TempMember.RootName = vectorVar[indexMember].Name;
                TempMember.ptrRoot = vectorVar[indexMember].ptrDecl;
            }
        }
    }
    
    return indexLocalMember;
}

// wfr 20190807 保存全局变量中的子域
int  OAOASTVisitor::saveGlobalMemberInfo(FieldDecl* pMemberDecl, int indexGblClass){

    vectorGblVar.emplace_back(pMemberDecl, this->Rewrite);
    VARIABLE_INFO& TempMember = vectorGblVar.back();
    int indexMember = vectorGblVar.size()-1;
        
    // member 的 offset 和 SourceLocation 没有意义就不再赋值；因为类实例定义的时候没有反映出 member

    vectorGblVar[indexGblClass].isClass = true;

    TempMember.isGlobal = true;
    TempMember.Name = pMemberDecl->getNameAsString();
    TempMember.ptrClass = vectorGblVar[indexGblClass].ptrDecl; // 如果是类的成员, 这里是类实例定义的地址
    //TempMember.pDeclStmt = vectorVar[indexGblClass].pDeclStmt; // 保存 插入指针区域长度语句 位置 的信息
    TempMember.DeclRange = vectorGblVar[indexGblClass].DeclRange;
    TempMember.indexClass = indexGblClass; // 如果是类的成员, 这里是类的 VARIABLE_INFO 的 index
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    TempMember.TypeName = QualType::getAsString( pMemberDecl->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等
    TempMember.isMember = true;
    //TempMember.isArrow = S->isArrow();
    if( vectorGblVar[indexGblClass].Type==VAR_TYPE::PTR || vectorGblVar[indexGblClass].Type==VAR_TYPE::PTR_CONST
        || vectorGblVar[indexGblClass].Type==VAR_TYPE::CLASS_PTR || vectorGblVar[indexGblClass].Type==VAR_TYPE::CLASS_PTR_CONST ){
        TempMember.isArrow = true;
    }
    // 保存信息: 变量定义在哪个 node
    TempMember.indexDefNode = vectorGblVar[indexGblClass].indexDefNode;
    // 保存变量作用域信息
    TempMember.Scope = vectorGblVar[indexGblClass].Scope;

    std::string TypeName = TempMember.TypeName;
    // wfr 20190806 通过变量类型字符串, 分析获得变量类型
    VAR_TYPE Type = getVarType(TypeName);
    TempMember.Type = Type;

    if( Type==VAR_TYPE::CLASS_PTR || Type==VAR_TYPE::CLASS_PTR_CONST 
        || Type==VAR_TYPE::PTR || Type==VAR_TYPE::PTR_CONST 
    ){
        std::cout << "saveGlobalMemberInfo 错误: 程序中有全局指针, 不能安全处理" << std::endl;
        exit(1);
    }

    // 下边设置指针的 全变量名 和 指针区域长度变量
    TempMember.FullName = vectorGblVar[indexGblClass].FullName; // 写入变量全名称
    TempMember.FullName += (TempMember.isArrow ? "->" : ".");
    TempMember.FullName += TempMember.Name;
    TempMember.Rename = vectorGblVar[indexGblClass].Name; // 写入变量长度变量的重命名
    TempMember.Rename += (TempMember.isArrow ? "__" : "_");
    TempMember.Rename += TempMember.Name;
    if(TempMember.TypeName.back()=='*'){ // 写入变量长度变量的名称
        TempMember.ArrayLen = TempMember.Rename;
        TempMember.ArrayLen += "_LEN_";
    }

    VARIABLE_INFO& TempVar = vectorGblVar[indexGblClass];
    std::string RenameCode = "";
    std::string RunTimeCode = "";
    SourceLocation TmpInsertLoc = TempVar.DeclRange.EndLoc;

    // Member 如果是 指针/数组, 需要重命名, 即建立新变量(这里不赋值, 实际使用的时候再将Member地址赋值给新变量)
    if(TempMember.TypeName.back()=='*' || TempMember.TypeName.back()==']' || TempMember.isArrow==true){
        if(TempMember.TypeName.back()=='*'){ // 如果域是指针, 下边声明 指针区域长度变量

            // 变量重命名
            RenameCode += "; ";
            RenameCode += TempMember.TypeName;
            RenameCode += ' ';
            RenameCode += TempMember.Rename;

        }else if(TempMember.TypeName.back()==']'){

            std::string ElementType = TempMember.TypeName.substr(0, TempMember.TypeName.find('[')); // 取出 '[' 之前的字符串

            // 变量重命名
            RenameCode += "; ";
            RenameCode += ElementType; // 取出 '[' 之前的字符串
            RenameCode += "* "; // 修改成指针类型
            RenameCode += TempMember.Rename;

            // 获得多维数组的元素总数, 写入 ArrayLen 域中
            TempMember.getArrayLen();
            
            RunTimeCode += "\n";
            RunTimeCode += OAO_ARRAY_NAME;
            RunTimeCode += "( (void*)(";
            RunTimeCode += TempMember.FullName;
            RunTimeCode += "), sizeof(";
            RunTimeCode += TempMember.TypeName;
            RunTimeCode += "), sizeof(";
            RunTimeCode += ElementType;
            RunTimeCode += ") );\n";
        }else{
            // 变量重命名
            RenameCode += "; ";
            RenameCode += TempMember.TypeName;
            RenameCode += ' ';
            RenameCode += TempMember.Rename;
        }

        // wfr 20190807 插入重命名代码
        std::vector<CODE_INSERT>::iterator iterCodeInsert;
        iterCodeInsert = find(TempVar.vectorInsert.begin(), TempVar.vectorInsert.end(), TmpInsertLoc);
        if(iterCodeInsert==TempVar.vectorInsert.end()){ // 如果没找到该插入位置（在源代码中的位置）,  就新建该插入位置
            TempVar.vectorInsert.emplace_back(TmpInsertLoc, "", Rewrite);
            iterCodeInsert = TempVar.vectorInsert.end()-1;
        }
        iterCodeInsert->Code += RenameCode;
    }

    return indexMember;
}

// wfr 20190209 保存 OMP 外部 Member 信息
//int OAOASTVisitor::saveMemberInfo(MemberExpr* S, int indexClass, int indexFunc){
int OAOASTVisitor::saveMemberInfo(FieldDecl* pMemberDecl, int indexClass, FUNC_INFO& Func){
    //ValueDecl* pMemberDecl = S->getMemberDecl(); // 获得域定义的指针, 这个指针指向类的类型定义中的域定义, 而不是指向实例的定义
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    std::vector<FUNC_PARM_VAR_INFO>& vectorParm = Func.vectorParm;

    vectorVar[indexClass].isClass = true;

    vectorVar.emplace_back(pMemberDecl, this->Rewrite);
    VARIABLE_INFO& TempMember = vectorVar.back();
    int indexMember = vectorVar.size()-1;
    // vectorVar[indexClass].vectorIndexMember.push_back(indexMember); // 将 member 的 index 写入 class
        
    // member 的 offset 和 SourceLocation 没有意义就不再赋值；因为类实例定义的时候没有反映出 member

    TempMember.Name = pMemberDecl->getNameAsString();
    TempMember.ptrClass = vectorVar[indexClass].ptrDecl; // 如果是类的成员, 这里是类实例定义的地址
    //TempMember.pDeclStmt = vectorVar[indexClass].pDeclStmt; // 保存 插入指针区域长度语句 位置 的信息
    TempMember.DeclRange = vectorVar[indexClass].DeclRange;
    TempMember.indexClass = indexClass; // 如果是类的成员, 这里是类的 VARIABLE_INFO 的 index
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    TempMember.TypeName = QualType::getAsString( pMemberDecl->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等
    TempMember.isMember = true;
    //TempMember.isArrow = S->isArrow();
    if( vectorVar[indexClass].Type==VAR_TYPE::PTR || vectorVar[indexClass].Type==VAR_TYPE::PTR_CONST
        || vectorVar[indexClass].Type==VAR_TYPE::CLASS_PTR || vectorVar[indexClass].Type==VAR_TYPE::CLASS_PTR_CONST ){
        TempMember.isArrow = true;
    }
    // 保存信息: 变量定义在哪个 node
    TempMember.indexDefNode = vectorVar[indexClass].indexDefNode;
    // 保存变量作用域信息
    TempMember.Scope = vectorVar[indexClass].Scope;

    std::string TypeName = TempMember.TypeName;
    // wfr 20190806 通过变量类型字符串, 分析获得变量类型
    VAR_TYPE Type = getVarType(TypeName);
    TempMember.Type = Type;

    // 下边设置指针的 全变量名 和 指针区域长度变量
    TempMember.FullName = vectorVar[indexClass].FullName; // 写入变量全名称
    TempMember.FullName += (TempMember.isArrow ? "->" : ".");
    TempMember.FullName += TempMember.Name;
    TempMember.Rename = vectorVar[indexClass].Name; // 写入变量长度变量的重命名
    TempMember.Rename += (TempMember.isArrow ? "__" : "_");
    TempMember.Rename += TempMember.Name;
    if(TempMember.TypeName.back()=='*'){ // 写入变量长度变量的名称
        TempMember.ArrayLen = TempMember.Rename;
        TempMember.ArrayLen += "_LEN_";
    }

    // 这里判断是不是函数的参数, 分类讨论
    bool isParm = false;
    std::vector<FUNC_PARM_VAR_INFO>::iterator iterParm;
    iterParm = find(vectorParm.begin(), vectorParm.end(), indexClass);
    if(iterParm != vectorParm.end()) {
        isParm=true;
    }

    if(isParm==true){ // 如果类是函数参数
        // 将 member 存入 vectorParm
        vectorParm.emplace_back(indexMember, TempMember.Type);

        // 如果 member 是指针类型, 在函数参数列表中写入指针区域长度变量
        if(TempMember.TypeName.back()=='*'){
            // 什么都不干
        }else if(TempMember.TypeName.back()==']'){
            // 获得多维数组的元素总数, 写入 ArrayLen 域中
            TempMember.getArrayLen();
        }else{}

        // Member 如果是 指针/数组, 需要重命名, 即建立新变量(这里不赋值, 实际使用的时候再将Member地址赋值给新变量)
        if(TempMember.TypeName.back()=='*' || TempMember.TypeName.back()==']' || TempMember.isArrow==true){
            VARIABLE_INFO& TempVar = vectorVar[indexClass];
            std::string Code = "";
            if(TempMember.TypeName.back()==']'){
                Code = TempMember.TypeName.substr(0, TempMember.TypeName.rfind('[')); // 取出 '[' 之前的字符串
                Code += '*'; // 修改成指针类型
            }else{
                Code = TempMember.TypeName;
            }
            Code += " ";
            Code += TempMember.Rename;
            Code += " = ";
            Code += TempMember.FullName;
            Code += "; ";

            SourceLocation TmpInsertLoc = Func.CompoundRange.BeginLoc.getLocWithOffset(1);
            std::vector<CODE_INSERT>::iterator iterCodeInsert;
            iterCodeInsert = find(TempVar.vectorInsert.begin(), TempVar.vectorInsert.end(), TmpInsertLoc);
            if(iterCodeInsert==TempVar.vectorInsert.end()){ // 如果没找到该插入位置（在源代码中的位置）,  就新建该插入位置
                TempVar.vectorInsert.emplace_back(TmpInsertLoc, "\n", Rewrite);
                iterCodeInsert = TempVar.vectorInsert.end()-1;
            }
            iterCodeInsert->Code += Code;
        }

    }else{ // 如果类不是函数参数

        VARIABLE_INFO& TempVar = vectorVar[indexClass];
        std::string Code = "";
        SourceLocation TmpInsertLoc = TempVar.DeclRange.EndLoc;

        // Member 如果是 指针/数组, 需要重命名, 即建立新变量(这里不赋值, 实际使用的时候再将Member地址赋值给新变量)
        if(TempMember.TypeName.back()=='*' || TempMember.TypeName.back()==']' || TempMember.isArrow==true){
            if(TempMember.TypeName.back()=='*'){ // 如果域是指针, 下边声明 指针区域长度变量

                // 变量重命名
                Code += "; ";
                Code += TempMember.TypeName;
                Code += ' ';
                Code += TempMember.Rename;

                // 变量长度
                // Code += "; int ";
                // Code += TempMember.ArrayLen;

                // wfr 20190327 ??? 处理不了在构造函数中对指针变量赋值的情况
            }else if(TempMember.TypeName.back()==']'){

                std::string ElementType = TempMember.TypeName.substr(0, TempMember.TypeName.find('[')); // 取出 '[' 之前的字符串

                // 变量重命名
                Code += "; ";
                Code += ElementType; // 取出 '[' 之前的字符串
                Code += "* "; // 修改成指针类型
                Code += TempMember.Rename;

                // 获得多维数组的元素总数, 写入 ArrayLen 域中
                TempMember.getArrayLen();
                
                Code += "; ";
                Code += OAO_ARRAY_NAME;
                Code += "( (void*)(";
                Code += TempMember.FullName;
                Code += "), sizeof(";
                Code += TempMember.TypeName;
                Code += "), sizeof(";
                Code += ElementType;
                Code += ") )";
            }else{
                // 变量重命名
                Code += "; ";
                Code += TempMember.TypeName;
                Code += ' ';
                Code += TempMember.Rename;

                // 变量长度
                // Code += "; int ";
                // Code += TempMember.ArrayLen;

                // wfr 20190327 ??? 处理不了在构造函数中对指针变量赋值的情况
            }

            std::vector<CODE_INSERT>::iterator iterCodeInsert;
            iterCodeInsert = find(TempVar.vectorInsert.begin(), TempVar.vectorInsert.end(), TmpInsertLoc);
            if(iterCodeInsert==TempVar.vectorInsert.end()){ // 如果没找到该插入位置（在源代码中的位置）,  就新建该插入位置
                TempVar.vectorInsert.emplace_back(TmpInsertLoc, "", Rewrite);
                iterCodeInsert = TempVar.vectorInsert.end()-1;
            }
            iterCodeInsert->Code += Code;
        }

        if(vectorVar[indexClass].indexRoot>=0){ // 说明是个指针, 需要进行变量跟踪
            int indexRootClass = vectorVar[indexClass].indexRoot;
            VarDecl* pRootClassDcl = (VarDecl*)vectorVar[indexRootClass].ptrDecl;
            int indexFunc = -1;
            for(unsigned long i=0; i<vectorFunc.size(); ++i){
                if(Func.ptrDefine==vectorFunc[i].ptrDefine){
                    indexFunc = i;
                }
            }
            // 获得 member 变量信息
            int indexRootMember = getMemberIndex(pRootClassDcl, pMemberDecl, indexFunc);

            // saveMemberInfo 中对 vectorVar 新建了项, 因此再使用 TempMember(引用) 会有问题
            // 所以下边 应该使用 indexMember
            VAR_TYPE VarType = TempMember.Type;
            if( VarType==VAR_TYPE::PTR || VarType==VAR_TYPE::PTR_CONST 
                || VarType==VAR_TYPE::CLASS_PTR || VarType==VAR_TYPE::CLASS_PTR_CONST ){
                vectorVar[indexMember].indexRoot = vectorVar[indexRootMember].indexRoot;
                if(vectorVar[indexMember].indexRoot>=0){
                    vectorVar[indexMember].RootName = vectorVar[vectorVar[indexMember].indexRoot].Name;
                    vectorVar[indexMember].ptrRoot = vectorVar[vectorVar[indexMember].indexRoot].ptrDecl;
                }
            }else{
                vectorVar[indexMember].indexRoot = indexRootMember;
                if(indexRootMember>=0){
                    vectorVar[indexMember].RootName = vectorVar[indexRootMember].Name;
                    vectorVar[indexMember].ptrRoot = vectorVar[indexRootMember].ptrDecl;
                }
            }
        }

    }

    return indexMember;
}

// wfr 20190209 保存 OMP 外部 Member 信息
//int OAOASTVisitor::saveMemberInfo(MemberExpr* S, int indexClass, int indexFunc){
int OAOASTVisitor::saveMemberInfo(FieldDecl* pMemberDecl, int indexClass, int indexFunc){

    return saveMemberInfo(pMemberDecl, indexClass, vectorFunc[indexFunc]);
}

// `-MemberExpr 0x564962ed3600 <col:26, col:35> 'float *' lvalue .v2 0x564962ecd410
// wfr 20181124 这里处理类/结构体的引用
bool OAOASTVisitor::OAOVisitMemberExpr(MemberExpr *S) {

    // 以下处理引用变量的情况
    int indexFuncCurrent = -1; // 表示当前代码在哪个函数

    // 不在任何函数中, 直接退出
    if(BeginFlag==true && FuncStack.empty()==false){
        indexFuncCurrent = FuncStack.back().indexFunc; // 当前代码所在的函数
    }else{
        return true;
    }

    // wfr 以下是调试输出语句
    std::cout << "OAOVisitMemberExpr: 进入" << std::endl;
    std::cout << "Field 名: " << S->getMemberDecl()->getNameAsString() << std::endl;
    if(S->getMemberDecl()->getNameAsString()=="const_len2"){
        std::cout << "找到 const_len2" << std::endl;
    }
    std::cout << "MemberExpr 地址: 0x" << std::hex << (void*)S << std::endl;

    FUNC_INFO& FuncCurrent = vectorFunc[indexFuncCurrent]; // 获得当前函数, 方便后边使用, 简化程序
    std::vector<VARIABLE_INFO>& vectorVar = FuncCurrent.vectorVar;

    // 判断在哪个 SEQ/OMP 节点中, 没找到则退出; 获得当前串/并行节点指针, 方便后边使用, 简化程序
    NODE_INDEX indexNode;
    SEQ_REGION* pSEQRegion = NULL;
    OMP_REGION* pOMPRegion = NULL;
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int Offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    InWhichSEQOMP(indexNode, indexFuncCurrent, Offset);
    if(indexNode.type == NODE_TYPE::NODE_UNINIT){
        return true;
    }else if(indexNode.type == NODE_TYPE::SEQUENTIAL){
        pSEQRegion = &FuncCurrent.vectorSEQ[indexNode.index];
    }else{
        pOMPRegion = &FuncCurrent.vectorOMP[indexNode.index];
    }

    // wfr 以下是调试输出语句
    std::cout << "找到 MemberExpr 所在的 SEQRegion/OMPRegion" << std::endl;

    // 再判断该member的类的定义是否在OMP中, 在则返回
    // 需要获得member的类的实例的定义的指针/引用, 即逐层深入查找 DeclRefExpr
    Stmt* pParent = dyn_cast<Stmt>(S);
    Stmt* pChild;
    size_t NumLayer = 0; // 记录子节点的层数
    bool Array = false;  //是否为结构体数组
    while(true){
        if(NumLayer<4){
            if(pParent->children().begin()!=pParent->children().end()){
                // 获得指向子节点的指针
                pChild = *(pParent->children().begin()); // 如果没有子节点这两句不知道会不会出错 !!!
                if(pChild!=NULL){ // 如果子节点存在
                    //ghn 20190928 判断是否是结构体数组
                    if(isa<ArraySubscriptExpr>(pChild)){
                        std::cout << "ArraySubscriptExpr 此为结构体数组" << std::endl;
                        Array = true;
                        // break;
                    }
                    if(isa<DeclRefExpr>(pChild)){
                        break;
                    }
                    pParent = pChild; // 迭代
                    if(!isa<ImplicitCastExpr>(pChild)){
                        // 除了隐式类型转换以外的情况, 层数自增
                        NumLayer++;
                    }
                }else{
                    break;
                }
            }else{
                break;
            }
        }else{
            std::cout << "OAOVisitMemberExpr 错误: MemberExpr 的子节点层数超过 4, 目前不能安全处理。" << std::endl;
            exit(1);
        }
    }

    if(!isa<DeclRefExpr>(pChild)){
        std::cout << "OAOVisitMemberExpr 错误: 没找到 DeclRefExpr" << std::endl;
        exit(1);
    }

    DeclRefExpr* pClassDeclRefExpr = dyn_cast<DeclRefExpr>(pChild); // 获得 MemberExpr 的子节点的地址, 子节点是类实例的声明的引用
    VarDecl* pClassDecl = dyn_cast<VarDecl>(pClassDeclRefExpr->getDecl());// 获得类实例定义的地址
    std::cout << "OAOVisitMemberExpr: pClassDecl = " << std::hex << pClassDecl << std::endl;
    std::cout << "OAOVisitMemberExpr: indexFuncCurrent = " << std::dec << indexFuncCurrent << std::endl;
    std::cout << "OAOVisitMemberExpr: indexNode.index = " << std::dec << indexNode.index << std::endl;
    std::cout << "强转完成" << std::endl;
    //SourceLocation ClassDeclBeginLoc = pClassDecl->getBeginLoc(); // 类实例定义在源文件中的位置
    std::cout << "获得位置" << std::endl;
    //unsigned int ClassBeginOffset = Rewrite.getSourceMgr().getFileOffset(ClassDeclBeginLoc); // 类实例定义在源文件中的偏移

    std::string MemberName = "NULL";
    // std::string TypeName = "NULL";

    // 如果是 OMP local 变量, 使用这两个
    int indexLocalClass = -1;
    VARIABLE_INFO* pLocalClass = NULL;
    int indexLocalMember = -1;
    VARIABLE_INFO* pLocalMember = NULL;
    
    // 如果是OMP外部变量, 使用这两个
    int indexClass = -1;
    VARIABLE_INFO* pClass = NULL;
    int indexMember = -1;
    VARIABLE_INFO* pMember = NULL;
    
    // indexLocalMember 的赋值源变量, 应该是一个外部指针变量
    int indexVar = -1;
    VARIABLE_INFO* pVar = NULL;
    ValueDecl* pMemberDecl = S->getMemberDecl(); // 获得域定义的指针, 这个指针指向类的类型定义中的域定义, 而不是指向实例的定义
    MemberName = pMemberDecl->getNameAsString();
    size_t TempOffset = 0; // 类的子域名的长度, 即 member 名字的长度
    MY_SOURCE_RANGE TempSrcRange; // 生成 member 引用在源码中的位置信息 ？？？

    std::cout << "OAOVisitMemberExpr: 0" << std::endl;

    // 获得 member 及 相关 信息, member 不存在则存入
    if(indexNode.type == NODE_TYPE::PARALLEL){
        std::cout << "OAOVisitMemberExpr: a" << std::endl;
        // 判断该变量的定义是否在当前OMP中
        if(IsInOMP(static_cast<Decl*>(pClassDecl), indexFuncCurrent, indexNode.index)){
            std::cout << "OAOVisitMemberExpr: b" << std::endl;
            // 获得 OMP节点 Local Class 信息
            indexLocalClass = WhichLocalVar(pClassDecl, indexFuncCurrent, indexNode.index);
            if(indexLocalClass<0){
                std::cout << "OAOVisitMemberExpr 错误: LocalClass 变量未找到" << std::endl;
                exit(1);
            }
            pLocalClass = &pOMPRegion->vectorLocal[indexLocalClass];
            pLocalClass->isClass = true; // 设置标识, 表示该变量是类
            // 更新信息: 变量的最后一次引用是在哪个 node
            pLocalClass->indexLastRefNode = indexNode;

            std::cout << "OAOVisitMemberExpr: 1" << std::endl;

            // 获得 local member 信息
            indexLocalMember = MemberIsWhichLocal(pClassDecl, MemberName, indexFuncCurrent, indexNode.index);
            if(indexLocalMember<0 && isa<FieldDecl>(S->getMemberDecl())){ // 不存在就存入
                // 保存 OMP 中 LocalMember 信息
                indexLocalMember = saveLocalMemberInfo((FieldDecl*)(S->getMemberDecl()), indexLocalClass, indexFuncCurrent, indexNode.index);
            }

            std::cout << "OAOVisitMemberExpr: 2" << std::endl;

            pLocalMember = &pOMPRegion->vectorLocal[indexLocalMember];
            // 更新信息: 变量的最后一次引用是在哪个 node
            pLocalMember->indexLastRefNode = indexNode;
            // 这里进行变量跟踪的第一步, 获得对应的 indexVar
            indexVar = pOMPRegion->vectorLocal[indexLocalMember].indexRoot; // 获取 源外部变量 的 index
            if(indexVar>=0){
                // ghn 20191111 更新indexVar
                if(FuncCurrent.vectorVar[indexMember].indexRoot >= 0){
                    indexVar = FuncCurrent.vectorVar[indexMember].indexRoot;
                }
                pVar = &FuncCurrent.vectorVar[indexVar];
                // 更新信息: 变量的最后一次引用是在哪个 node
                pVar->indexLastRefNode = indexNode;
            }
        }else{
            std::cout << "OAOVisitMemberExpr: c" << std::endl;
            // 获得 class 变量信息
            indexClass = getVarIndex(pClassDecl, indexFuncCurrent);
            if(indexClass<0){
                if((unsigned long)pClassDecl < (unsigned long)vectorFunc[0].ptrDecl){
                    std::cout << "OAOVisitMemberExpr 警告: 变量 " << pClassDecl->getNameAsString() 
                        << " 定义在 main() 函数之前, 不处理" << std::endl;
                    return true;
                }else{
                    std::cout << "OAOVisitMemberExpr 错误: 变量 " << pClassDecl->getNameAsString() 
                        << " 未找到" << std::endl;
                    exit(1);
                }
            }
            pClass = &FuncCurrent.vectorVar[indexClass];
            pClass->isClass = true; // 设置标识, 表示该变量是类
            // 更新信息: 变量的最后一次引用是在哪个 node
            pClass->indexLastRefNode = indexNode;

            std::cout << "OAOVisitMemberExpr: 3" << std::endl;

            //ghn 20190928 如果是结构体数字，那么只修改isClass属性，而不做其他修改
            if(Array){
                indexMember = indexClass;
                pClass->isClass = false; // 结构体数组不作为类，作为一个整体
            }else if(isa<FieldDecl>(S->getMemberDecl())){    // 获得 member 变量信息
                // 保存 OMP 外部 Member 信息
                indexMember = getMemberIndex(pClassDecl, (FieldDecl*)(S->getMemberDecl()), indexFuncCurrent);
            }else{
                std::cout << "OAOVisitMemberExpr 警告: MemberExpr 不是 FieldDecl 类型, 不处理" << std::endl;
                return true;
            }
            pMember = &FuncCurrent.vectorVar[indexMember];

            std::cout << "OAOVisitMemberExpr: 4" << std::endl;

            // 更新信息: 变量的最后一次引用是在哪个 node
            pMember->indexLastRefNode = indexNode;
            TempOffset = FuncCurrent.vectorVar[indexMember].Name.length();
            TempSrcRange.init(S->getBeginLoc(), S->getEndLoc(), TempOffset-1, this->Rewrite);

            indexVar = FuncCurrent.vectorVar[indexMember].indexRoot; // 获取 源外部变量 的 index
            if(indexVar>=0){
                pVar = &FuncCurrent.vectorVar[indexVar];
                // 更新信息: 变量的最后一次引用是在哪个 node
                pVar->indexLastRefNode = indexNode;
            }

            // wfr 20190429 应该在这里 插入 变量重命名替换, 替换 OMP 中引用的 外部 类的域
            // wfr 20190806 只有指针的时候才重命名
            // ghn 20190928 Array判断不是结构体数组才需要进入，结构体数组作为整体处理
            if(!Array && (pClass->Type==VAR_TYPE::PTR || pMember->Type==VAR_TYPE::PTR)){
                SourceRange TmpSourceRange(TempSrcRange.BeginLoc, TempSrcRange.EndLoc);
                vectorVar[indexVar>=0 ? indexVar : indexMember].vectorReplace.emplace_back(TmpSourceRange, vectorVar[indexVar>=0 ? indexVar : indexMember].Rename, Rewrite);
            }
        }
    }else{
        std::cout << "OAOVisitMemberExpr: d" << std::endl;
        // 获得 class 变量信息
        indexClass = getVarIndex(pClassDecl, indexFuncCurrent);
        if(indexClass<0){
            if((unsigned long)pClassDecl < (unsigned long)vectorFunc[0].ptrDecl){
                std::cout << "OAOVisitMemberExpr 警告: 变量 " << pClassDecl->getNameAsString() 
                    << " 定义在 main() 函数之前, 不处理" << std::endl;
                return true;
            }else{
                std::cout << "OAOVisitMemberExpr 错误: 变量 " << pClassDecl->getNameAsString() 
                    << " 未找到" << std::endl;
                exit(1);
            }
        }
        pClass = &FuncCurrent.vectorVar[indexClass];
        pClass->isClass = true; // 设置标识, 表示该变量是类
        // 更新信息: 变量的最后一次引用是在哪个 node
        pClass->indexLastRefNode = indexNode;

        std::cout << "OAOVisitMemberExpr: indexClass = " << indexClass << std::endl;

        //ghn 20190928 如果是结构体数组，那么只修改isClass属性，而不做其他修改
        if(Array){
            indexMember = indexClass;
            pClass->isClass = false; // 结构体数组不作为类，作为一个整体
        }else if(isa<FieldDecl>(S->getMemberDecl())){    // 获得 member 变量信息
            // 保存 OMP 外部 Member 信息
            indexMember = getMemberIndex(pClassDecl, (FieldDecl*)(S->getMemberDecl()), indexFuncCurrent);
        }else{
            std::cout << "OAOVisitMemberExpr 警告: MemberExpr 不是 FieldDecl 类型, 不处理" << std::endl;
            return true;
        }

        std::cout << "OAOVisitMemberExpr: indexMember = " << indexMember << std::endl;

        pMember = &FuncCurrent.vectorVar[indexMember];
        // 更新信息: 变量的最后一次引用是在哪个 node
        pMember->indexLastRefNode = indexNode;
        TempOffset = FuncCurrent.vectorVar[indexMember].Name.length();
        TempSrcRange.init(S->getBeginLoc(), S->getEndLoc(), TempOffset-1, this->Rewrite);

        indexVar = FuncCurrent.vectorVar[indexMember].indexRoot; // 获取 源外部变量 的 index
        if(indexVar>=0){
            pVar = &FuncCurrent.vectorVar[indexVar];
            // 更新信息: 变量的最后一次引用是在哪个 node
            pVar->indexLastRefNode = indexNode;
        }
    }

    std::cout << "获得 member 及 相关 信息" << std::endl;

    // wfr 20190421 设置 UsedInOMP 标识, 表示变量在 OMP 中被引用
    if(indexNode.type == NODE_TYPE::PARALLEL && indexLocalClass>=0 && pOMPRegion->vectorLocal[indexLocalClass].indexRoot>=0){ // 说明该外部变量在 OMP 中被使用了
        FuncCurrent.vectorVar[pOMPRegion->vectorLocal[indexLocalClass].indexRoot].UsedInOMP = true; // 设置标识
    }
    if(indexNode.type == NODE_TYPE::PARALLEL && indexVar>=0){ // 说明该外部变量在 OMP 中被使用了
        FuncCurrent.vectorVar[indexVar].UsedInOMP = true; // 设置标识
    }
    if(indexNode.type == NODE_TYPE::PARALLEL && indexMember>=0){ // 说明该外部变量在 OMP 中被使用了
        FuncCurrent.vectorVar[indexMember].UsedInOMP = true; // 设置标识
    }

    // 上边完成将 member 和 类实例 定义 写入 vectorVar/vectorLocal
    // 下边根据 操作符种类不同, 是第几个操作数 进行不同处理

    std::string ASTOpName = "UNINIT"; // AST 操作符的名称字符串
    int OperandID = 999; // 操作符的 第几个操作数
    ASTStackNode::PTR_ARY AccessType;
    if(!ASTStack.empty()){
        ASTOpName = ASTStack.back().Name; // AST 操作符的名称字符串
        OperandID = ASTStack.back().OperandID; // 操作符的 第几个操作数
        AccessType = ASTStack.back().AccessType[OperandID];
    }

    std::cout << "OAOVisitMemberExpr: OperandName " << ASTOpName << std::endl;
    std::cout << "OAOVisitMemberExpr: OperandID " << OperandID << std::endl;

    // 这里 忽略 多级指针的非函数参数引用
    // 获得类型
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    std::string TypeName, DoublePtr;
    TypeName = QualType::getAsString( S->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等
    if(TypeName.size()>=2){
        DoublePtr = TypeName.substr(TypeName.size()-2, 2);
        if(ASTOpName!="CallExpr" && DoublePtr=="**"){
            if(!ASTStack.empty()){
                ASTStack.back().OperandState[OperandID] = true;
            }
            std::cout << "OAOVisitMemberExpr: 忽略 多级指针的非函数参数引用" << std::endl;
            return true;
        }
    }

    // wfr 20190730 处理 delete[] 的参数引用
    if(ASTOpName == "delete[]" && ASTStack.back().OperandState[OperandID] == false){
        if(OperandID!=0){
            std:: cout << "OAOVisitMemberExpr 错误: delete[] 操作只有一个操作数, OperandID 越界, OperandID = " << OperandID << std::endl;
            exit(1);
        }
        if(indexNode.type == NODE_TYPE::SEQUENTIAL){
            // 看作是对 vectorVar[indexVar] 保存的外部指针的数据域的释放操作
            STATE_CONSTR TmpStReq(ST_REQ_HOST_ONLY);
            STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_FREE);
            saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
        }else{
            if(indexMember>=0){ // 外部 member
                // 看作是对 vectorVar[indexVar] 保存的外部指针的数据域的释放操作
                STATE_CONSTR TmpStReq(ST_REQ_HOST_ONLY);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_FREE);
                saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
            }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                std:: cout << "OAOVisitMemberExpr 错误: 不能处理的情况: 在并行域中释放外部指针" << std::endl;
                exit(1);
            }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
            }else{
                std::cout << "OAOVisitMemberExpr 错误: 运算符 delete[] OperandID == 0 处理错误" << std::endl;
                exit(1);
            }
        }              
        ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识

    // 处理函数调用时的参数
    }else if(ASTOpName == "CallExpr" && ASTStack.back().OperandState[OperandID] == false){
        std::cout << "OAOVisitMemberExpr: CallExpr" << std::endl;
        
        // 获得实参在源代码中的范围
        CallExpr* pFuncCall = static_cast<CallExpr*>(ASTStack.back().pOperator);
        if(OperandID==(int)(pFuncCall->getNumArgs())-1){
            TempSrcRange.init(pFuncCall->getArg(OperandID)->getBeginLoc(), pFuncCall->getEndLoc(), Rewrite);
        }else{
            TempSrcRange.init(pFuncCall->getArg(OperandID)->getBeginLoc(), pFuncCall->getArg(OperandID+1)->getBeginLoc(), Rewrite);
        }

        if(indexNode.type == NODE_TYPE::SEQUENTIAL){ // 如果是串行节点, 串行节点是新建的/独立的函数调用节点, 使用 vectorVarRef 中的一项 保存一个实参
            // wfr 201912221 处理 OperandID 越界
            if(pSEQRegion->vectorVarRef.size() <= OperandID){
                std::cout << "OAOVisitMemberExpr 错误: OperandID 超出 实参列表 vectorVarRef 范围 " << std::endl;
                exit(1);
            }
            // pSEQRegion->vectorVarRef.emplace(pSEQRegion->vectorVarRef.begin()+OperandID, (indexVar>=0 ? indexVar : indexMember));
            pSEQRegion->vectorVarRef[OperandID].init( (indexVar>=0 ? indexVar : indexMember) );
            STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
            STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
            // pSEQRegion->vectorVarRef.back().RefList.emplace_back((VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
            pSEQRegion->vectorVarRef[OperandID].RefList.emplace_back((VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
        }else{ // 如果是并行节点, 不是新建的/独立的函数调用节点, 使用 vectorVarRef 中的一项 保存所有实参
            if(ASTStack.back().indexFunc>=0 && ASTStack.back().indexVarRef>=0){
                int indexVarRef = ASTStack.back().indexVarRef;
                if( OperandID >= pOMPRegion->vectorVarRef[indexVarRef].vectorArgIndex.size() 
                    || OperandID >= pOMPRegion->vectorVarRef[indexVarRef].RefList.size() ){
                    std::cout << "OAOVisitMemberExpr 错误: OperandID 超出 实参列表 vectorArgIndex / RefList 范围 " << std::endl;
                    exit(1);
                }
                pOMPRegion->vectorVarRef[indexVarRef].vectorArgIndex[OperandID] = (indexVar>=0 ? indexVar : indexMember);
                // pOMPRegion->vectorVarRef[indexVarRef].vectorArgIndex.push_back( (indexVar>=0 ? indexVar : indexMember) ); // 这里的 indexVar 可能小于 0, 表示引用的是 OMP 内的局部变量, 可以不考虑
                STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                pOMPRegion->vectorVarRef[indexVarRef].RefList[OperandID].init((VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
                // pOMPRegion->vectorVarRef[indexVarRef].RefList.emplace_back((VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
            }else{
                std::cout << "OAOVisitMemberExpr 错误: 遇到 OMP 时, 保存 函数信息的 VAR_REF_LIST 有问题" << std::endl;
                exit(1);
            }
        }
        ASTStack.back().OperandState[OperandID] = true;

    // 处理定义变量的同时进行赋值的情况
    }else if(ASTOpName == "VarDecl"){ // 处理定义变量的同时进行赋值的情况
        std::cout << "OAOVisitMemberExpr: VarDecl" << std::endl;
        if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
            if(indexLocalMember>=0){
                ASTStack.back().Operand[0] = indexVar;
            }else if(indexMember>=0){
                ASTStack.back().Operand[0] = (indexVar>=0 ? indexVar : indexMember);
            }
        }else if( AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
            if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
            }else{
                if(indexMember>=0){ // 外部 member
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                    STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                    saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                    STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                    saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                    // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                }else{
                    std::cout << "OAOVisitMemberExpr错误: OMP 节点 VarDecl 异常" << std::endl;
                    exit(1);
                }
            }
        }else{
            //std::cout << "操作符\"VarDecl\" OperandID == 0 引用类型错误" << std::endl;
        }
        ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识

    // 处理运算符 =
    }else if(ASTOpName == "="){
        std::cout << "OAOVisitMemberExpr: =" << std::endl;
        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){ // 如果是等号左边的操作数
            if(AccessType==ASTStackNode::PTR_ARY::PTR){ // 引用指针
                
                if(indexLocalMember>=0){ // 处理局部指针的跟踪
                    if(ASTStack.back().OperandState[1]==false){
                        std::cout << "没找到赋值源指针, 指针跟踪失败!" << std::endl;
                        exit(1);
                    }else{
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pLocalMember->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pLocalMember->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pLocalMember->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }
                }else{ // 这里更新外部指针指向区域大小
                    if(ASTStack.back().vectorInfo[1]=="malloc"){

                    }else if(ASTStack.back().Operand[1]>=0){
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pMember->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pMember->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pMember->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }else{}
                }
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 VarDecl 异常" << std::endl;
                        exit(1);
                    }
                }
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == true){ //ghn 20191008 如果等号左边的操作数，多级嵌套
            if(AccessType==ASTStackNode::PTR_ARY::PTR){ // 引用指针
                if(indexLocalMember>=0){ // 处理局部指针的跟踪
                    if(ASTStack.back().OperandState[1]==false){
                        std::cout << "没找到赋值源指针, 指针跟踪失败!" << std::endl;
                        exit(1);
                    }else{
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pLocalMember->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pLocalMember->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pLocalMember->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }
                }else{ // 这里更新外部指针指向区域大小
                    if(ASTStack.back().vectorInfo[1]=="malloc"){

                    }else if(ASTStack.back().Operand[1]>=0){
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pMember->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pMember->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pMember->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }else{}
                }
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 VarDecl 异常" << std::endl;
                        exit(1);
                    }
                }
            }else{}
        }else if(OperandID == 1){
            if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
                if(indexLocalMember>=0){
                    ASTStack.back().Operand[1] = indexVar;
                }else if(indexMember>=0){
                    ASTStack.back().Operand[1] = (indexVar>=0 ? indexVar : indexMember);
                }
            }else if( AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 操作符\"=\" OperandID == 1 处理错误" << std::endl;
                        exit(1);
                    }
                }
            }else{
                //std::cout << "操作符\"=\" OperandID == 1 引用类型错误" << std::endl;
            }
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else{
            std:: cout << "= 操作只有两个操作数, OperandID 越界, OperandID = " << OperandID << std::endl;
            exit(1);
        }

    // 处理复合赋值运算符
    }else if(ASTOpName == "+=" || ASTOpName == "-=" || ASTOpName == "*=" || ASTOpName == "/=" || ASTOpName == "%=" ||
             ASTOpName == "<<=" || ASTOpName == ">>=" || ASTOpName == "&=" || ASTOpName == "^=" || ASTOpName == "|=")
    {
        std::cout << "OAOVisitMemberExpr: 符合复制运算符" << std::endl;
        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){
            if( AccessType==ASTStackNode::PTR_ARY::PTR ){ // 引用指针
                // 目前只处理局部指针的跟踪
                if(indexLocalMember>=0){
                    if(ASTStack.back().OperandState[1]==false){
                        std::cout << "没找到赋值源指针, 指针跟踪失败!" << std::endl;
                        exit(1);
                    }else{
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pLocalMember->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pLocalMember->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pLocalMember->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }
                }
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 复合赋值操作符 OperandID == 0 处理错误" << std::endl;
                        exit(1);
                    }
                }
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID == 1){
            if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
                if(indexLocalMember>=0){
                    ASTStack.back().Operand[1] = indexVar;
                }else if(indexMember>=0){
                    ASTStack.back().Operand[1] = (indexVar>=0 ? indexVar : indexMember);
                }
            }else if( AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 复合赋值操作符 OperandID == 1 处理错误" << std::endl;
                        exit(1);
                    }
                }
            }else{
                //std::cout << "复合赋值操作符 OperandID == 1 引用类型错误" << std::endl;
            }
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else{
            std:: cout << "复合赋值 操作只有两个操作数, OperandID 越界, 错误!" << std::endl;
            exit(1);
        }
    }else if(ASTOpName == "++" || ASTOpName == "--"){
        std::cout << "OAOVisitMemberExpr: ++/--" << std::endl;
        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){
            if( AccessType==ASTStackNode::PTR_ARY::PTR ){ // 引用指针
                // 这里不需要指针跟踪
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    if(indexMember>=0){ // 外部 member
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                        // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                        // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
                    }else{
                        std::cout << "OAOVisitMemberExpr错误: OMP 节点 运算符 ++/-- OperandID == 0 处理错误" << std::endl;
                        exit(1);
                    }
                }              
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else{
            std:: cout << "++/-- 操作只有一个操作数, OperandID 越界, 错误!" << std::endl;
            exit(1);
        }
    }else{ // 这里处理不是上述运算符的情况
        std::cout << "OAOVisitMemberExpr: 其他" << std::endl;
        if(indexNode.type == NODE_TYPE::SEQUENTIAL){
            // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
            // wfr 20190508 REF_TYPE::READ_WRITE 改为 REF_TYPE::READ
            STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
            STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
            saveVarRefInfo(pSEQRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
        }else{
            // wfr 20190508 REF_TYPE::READ_WRITE 改为 REF_TYPE::READ
            if(indexMember>=0){ // 外部 member
                // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                saveVarRefInfo(pOMPRegion->vectorVarRef, (indexVar>=0 ? indexVar : indexMember), (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
            }else if(indexLocalMember>=0 && indexVar>=0){ // local member 引用了外部数据
                // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读写操作
                STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, (VAR_REF::MEMBER_TYPE)S->isArrow(), TmpStReq, TmpStTransFunc, TempSrcRange);
            }else if(indexLocalMember>=0 && indexVar<0){ // local member 没有引用外部数据
                // 这种情况是 OMP 内部变量之间的引用, 不涉及外部变量, 不用处理
            }else{
                std::cout << "OAOVisitMemberExpr错误: OMP 节点 其他引用情况 处理错误" << std::endl;
                exit(1);
            }
        }
    }
    
    return true;
}

// wfr 20181124 保存变量引用读写信息
bool OAOASTVisitor::OAOVisitDeclRefExpr(DeclRefExpr *S) {
    if(BeginFlag==false){
        return true;
    }
    
    std::string DeclKindName = S->getDecl()->getDeclKindName();
    if( const NamedDecl *ND = dyn_cast<NamedDecl>(S->getDecl()) ){
        std::cout << "OAOVisitDeclRefExpr: 声明名称 " << ND->getNameAsString() << std::endl;
    }else{
        std::cout << "OAOVisitDeclRefExpr: 没找到声明名称" << std::endl;
    }

    // std::string DeclName = S->getDecl()->getDeclKindName();
    // 先处理引用函数的情况
    if( DeclKindName == "Function" ){ // 如果声明的类型是函数声明, 说明是 CallExpr 语句的第一个子节点, 即函数调用, 直接返回
        return true;
    }else if( DeclKindName != "Var" && DeclKindName != "ParmVar" ){
        std::cout << "OAOVisitDeclRefExpr 警告: 既不是引用函数, 也不是引用变量, 而是: " << DeclKindName << std::endl;
        return true; // 如果声明的类型不是变量声明就返回
    }

    // 以下处理引用变量的情况
    int indexFuncCurrent = -1; // 表示当前代码在哪个函数

    // 不在任何函数中, 直接退出
    if(FuncStack.empty()){
        return true;
    }else{
        indexFuncCurrent = FuncStack.back().indexFunc; // 当前代码所在的函数
        if(indexFuncCurrent<0){
            std::cout << "OAOVisitDeclRefExpr 错误: 当前函数的 indexFuncCurrent<0" << std::endl;
            exit(1);
        }
    }

    std::cout << "OAOVisitDeclRefExpr: 进入" << std::endl;

    FUNC_INFO& FuncCurrent = vectorFunc[indexFuncCurrent]; // 获得当前函数, 方便后边使用, 简化程序

    std::cout << "OAOVisitDeclRefExpr: 0" << std::endl;

    // 判断在哪个 SEQ/OMP 节点中, 没找到则退出; 获得当前串/并行节点指针, 方便后边使用, 简化程序
    NODE_INDEX indexNode;
    SEQ_REGION* pSEQRegion = NULL;
    OMP_REGION* pOMPRegion = NULL;
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(S->getBeginLoc());
    unsigned int Offset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
    InWhichSEQOMP(indexNode, indexFuncCurrent, Offset);
    if(indexNode.type == NODE_TYPE::NODE_UNINIT){
        // std::cout << "OAOVisitDeclRefExpr: NODE_UNINIT" << std::endl;
        // std::cout << "OAOVisitDeclRefExpr: Offset = " << std::hex << Offset << std::endl;
        return true;
    }else if(indexNode.type == NODE_TYPE::SEQUENTIAL){
        pSEQRegion = &FuncCurrent.vectorSEQ[indexNode.index];
    }else{
        pOMPRegion = &FuncCurrent.vectorOMP[indexNode.index];
    }

    std::cout << "OAOVisitDeclRefExpr: 1" << std::endl;

    // 获得引用的变量的 index, 没找到则认为出错, 退出
    // 先获得该变量的定义的地址
    VarDecl *ptrDecl = dyn_cast<VarDecl>(S->getDecl());
    int indexVar = -1;
    int indexLocal = -1;
    VARIABLE_INFO* pVar = NULL;
    VARIABLE_INFO* pLocal = NULL;
    // 获得当前变量的地址, 方便后边使用, 简化程序
    if(indexNode.type == NODE_TYPE::PARALLEL){
        std::cout << "OAOVisitDeclRefExpr: 2" << std::endl;
        // 判断该变量的定义是否在当前OMP中
        if(IsInOMP(ptrDecl, indexFuncCurrent, indexNode.index)){
        // if(IsInOMP(static_cast<Stmt*>(ptrDecl->getInit()), indexFuncCurrent, indexNode.index)){
            std::cout << "OAOVisitDeclRefExpr: 3" << std::endl;
            // 获得 OMP节点 中局部变量的 index
            indexLocal = WhichLocalVar(ptrDecl, indexFuncCurrent, indexNode.index);
            if(indexLocal<0){
                std::cout << "OAOVisitDeclRefExpr 出错: Local 变量未找到" << std::endl;
                exit(1);
            }
            pLocal = &pOMPRegion->vectorLocal[indexLocal];
            // 更新信息: 变量的最后一次引用是在哪个 node
            pLocal->indexLastRefNode = indexNode;

            // 这里进行变量跟踪的第一步, 获得对应的 indexVar
            indexVar = vectorFunc[indexFuncCurrent].vectorOMP[indexNode.index].vectorLocal[indexLocal].indexRoot; // 获取 源外部变量 的 index
            if(indexVar>=0){
                // ghn 20191111 更新indexVar
                if(vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot >= 0){
                    indexVar = vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot;
                }
                pVar = &FuncCurrent.vectorVar[indexVar];
                // 更新信息: 变量的最后一次引用是在哪个 node
                pVar->indexLastRefNode = indexNode;
            }
        }else{
            std::cout << "OAOVisitDeclRefExpr: 4" << std::endl;
            indexVar = getVarIndex(ptrDecl, indexFuncCurrent);
            if(indexVar<0){
                if((unsigned long)ptrDecl < (unsigned long)vectorFunc[0].ptrDecl){
                    std::cout << "OAOVisitDeclRefExpr 警告: 变量 " << ptrDecl->getNameAsString() 
                        << " 定义在 main() 函数之前, 不处理" << std::endl;
                    return true;
                }else{
                    std::cout << "OAOVisitDeclRefExpr 错误: 变量 " << ptrDecl->getNameAsString() 
                        << " 未找到" << std::endl;
                    exit(1);
                }
            }
            // ghn 20191111 更新indexVar
            if(vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot >= 0){
                indexVar = vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot;
            }
            pVar = &FuncCurrent.vectorVar[indexVar];
            // 更新信息: 变量的最后一次引用是在哪个 node
            pVar->indexLastRefNode = indexNode;
        }
    }else{
        std::cout << "OAOVisitDeclRefExpr: 变量定义地址是 " << std::hex << ptrDecl << std::endl;
        std::cout << "OAOVisitDeclRefExpr: indexFuncCurrent = " << indexFuncCurrent << std::endl;
        // 获得函数中变量的 index
        indexVar = getVarIndex(ptrDecl, indexFuncCurrent);
        if(indexVar<0){
            if((unsigned long)ptrDecl < (unsigned long)vectorFunc[0].ptrDecl){
                std::cout << "OAOVisitDeclRefExpr 警告: 变量 " << ptrDecl->getNameAsString() 
                    << " 定义在 main() 函数之前, 不处理" << std::endl;
                return true;
            }else{
                std::cout << "OAOVisitDeclRefExpr 错误: 变量 " << ptrDecl->getNameAsString() 
                    << " 未找到" << std::endl;
                exit(1);
            }
        }
        std::cout << "OAOVisitDeclRefExpr: 5" << std::endl;
        // ghn 20191111 更新indexVar
        if(vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot >= 0){
            indexVar = vectorFunc[indexFuncCurrent].vectorVar[indexVar].indexRoot;
        }
        pVar = &FuncCurrent.vectorVar[indexVar];
        // 更新信息: 变量的最后一次引用是在哪个 node
        pVar->indexLastRefNode = indexNode;
    }

    // ghn 20191109 如果引用外部变量，将当前变量的引用替换为外部变量
    // if(indexVar >= 0){
    //     if(vectorFunc[indexFuncCurrent].vectorVar[indexNode.index].indexRoot >= 0){
    //         indexVar = vectorFunc[indexFuncCurrent].vectorVar[indexNode.index].indexRoot;
    //     }
    // }

    // 至此完成 indexVar / indexLocal 的初始化

    std::cout << "OAOVisitDeclRefExpr: 至此完成 indexVar / indexLocal 的初始化" << std::endl;

    size_t TempOffset = 0; // 类的子域名的长度, 即 member 名字的长度
    MY_SOURCE_RANGE TempSrcRange;// Var 引用在源码中的位置信息
    if(indexVar>=0){
        // S 的 开始位置/结束位置 是一样, 需要加个 偏移
        TempOffset = vectorFunc[indexFuncCurrent].vectorVar[indexVar].Name.length(); // 计算偏移
        TempSrcRange.init(S->getBeginLoc(), S->getEndLoc(), TempOffset, this->Rewrite); // 生成 Var 引用在源码中的位置信息 ？？？ -1 不知道对不对
    }

    // wfr 20190421 设置 UsedInOMP 标识, 表示变量在 OMP 中被引用
    if(indexNode.type == NODE_TYPE::PARALLEL && indexVar>=0){ // 说明该外部变量在 OMP 中被使用了
        FuncCurrent.vectorVar[indexVar].UsedInOMP = true; // 设置标识
    }

    // 下面将引用信息写入 vectorVarRef
    // 分情况: 函数参数, 需要变量跟踪, 不需要变量跟踪的直接引用

    std::string ASTOpName="UNINIT"; // AST 操作符的名称字符串
    int OperandID = 999; // 操作符的 第几个操作数
    ASTStackNode::PTR_ARY AccessType;
    if(!ASTStack.empty()){
        ASTOpName = ASTStack.back().Name; // AST 操作符的名称字符串
        OperandID = ASTStack.back().OperandID; // 操作符的 第几个操作数
        AccessType = ASTStack.back().AccessType[OperandID];
        std::cout << "OAOVisitDeclRefExpr: pOperator = " << std::hex << ASTStack.back().pOperator << std::endl;
        if(ASTOpName=="CallExpr" && ASTStack.back().pOperator){
            FunctionDecl* pFuncDecl = getCalleeFuncPtr((CallExpr*)ASTStack.back().pOperator);
            std::string CalleeName = pFuncDecl->getNameAsString();
            std::cout << "OAOVisitDeclRefExpr: CallExpr 被调函数名 " << CalleeName << std::endl;
        }
    }
    std::cout << "OAOVisitDeclRefExpr: 完成变量信息收集" << std::endl;
    std::cout << "OAOVisitDeclRefExpr: ASTOpName = \"" << ASTOpName << "\"" << std::endl;
    std::cout << "OAOVisitDeclRefExpr: OperandID = " << std::dec << OperandID << std::endl;
    
    // std::cout << "OAOVisitDeclRefExpr: pDecl = " << std::hex << ASTStack.back().pDecl << std::endl;
    // std::cout << "OAOVisitDeclRefExpr: Operand.size() = " << std::dec << ASTStack.back().Operand.size() << std::endl;
    // std::cout << "OAOVisitDeclRefExpr: ASTStack.size() = " << std::dec << ASTStack.size() << std::endl;

    // 这里 忽略 多级指针的非函数参数引用
    // 获得类型
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    std::string TypeName, DoublePtr;
    TypeName = QualType::getAsString( S->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等
    if(TypeName.size()>=2){
        DoublePtr = TypeName.substr(TypeName.size()-2, 2);
        if(ASTOpName!="CallExpr" && DoublePtr=="**"){
            if(!ASTStack.empty()){
                ASTStack.back().OperandState[OperandID] = true;
            }
            std::cout << "OAOVisitDeclRefExpr: 忽略 多级指针的非函数参数引用" << std::endl;
            return true;
        }
    }

    // wfr 20190730 处理 delete[] 的参数引用
    if(ASTOpName == "delete[]" && ASTStack.back().OperandState[OperandID] == false){
        std::cout << "OAOVisitDeclRefExpr: delete[]" << std::endl;

        if(OperandID!=0){
            std:: cout << "delete[] 操作只有一个操作数, OperandID 越界, OperandID = " << OperandID << std::endl;
            exit(1);
        }

        if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
            // 看作是对 vectorVar[indexVar] 保存的外部指针的数据域的释放操作
            if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                STATE_CONSTR TmpStReq(ST_REQ_HOST_ONLY);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_FREE);
                saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
            }else{
                std:: cout << "OAOVisitDeclRefExpr 错误: 不能处理的情况: 在并行域中释放外部指针" << std::endl;
                exit(1);
            }
        }else if(indexLocal>=0 && indexVar<0){ // 局部指针指向OMP内分配的数据区域
            // 这种情况什么都不做
        }else{
            std::cout << "OAOVisitDeclRefExpr 没找到变量???" << std::endl;
        }                
        ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识

    // 处理函数调用时的参数
    }else if(ASTOpName == "CallExpr" && ASTStack.back().OperandState[OperandID] == false){ // 这里将 实参的 index 存入 OMP/SEQ 的不同结构中, 以便后续分析使用
        
        std::cout << "OAOVisitDeclRefExpr: CallExpr" << std::endl;
        
        // 获得实参在源代码中的范围
        CallExpr* pFuncCall = static_cast<CallExpr*>(ASTStack.back().pOperator);
        if(OperandID==(int)(pFuncCall->getNumArgs())-1){
            TempSrcRange.init(pFuncCall->getArg(OperandID)->getBeginLoc(), pFuncCall->getEndLoc(), Rewrite);
        }else{
            TempSrcRange.init(pFuncCall->getArg(OperandID)->getBeginLoc(), pFuncCall->getArg(OperandID+1)->getBeginLoc(), Rewrite);
        }

        if(indexNode.type == NODE_TYPE::SEQUENTIAL){ // 如果是串行节点, 串行节点是新建的/独立的函数调用节点, 使用 vectorVarRef 中的一项 保存一个实参
            // wfr 201912221 处理 OperandID 越界
            if(pSEQRegion->vectorVarRef.size() <= OperandID){
                std::cout << "OAOVisitDeclRefExpr 错误: OperandID 超出 实参列表 vectorVarRef 范围 " << std::endl;
                exit(1);
            }
            pSEQRegion->vectorVarRef[OperandID].init(indexVar);
            STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
            STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
            pSEQRegion->vectorVarRef[OperandID].RefList.emplace_back(VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
        }else{ // 如果是并行节点, 不是新建的/独立的函数调用节点, 使用 vectorVarRef 中的一项 保存所有实参
            if(ASTStack.back().indexFunc>=0 && ASTStack.back().indexVarRef>=0){
                int indexVarRef = ASTStack.back().indexVarRef;
                if( OperandID >= pOMPRegion->vectorVarRef[indexVarRef].vectorArgIndex.size() 
                    || OperandID >= pOMPRegion->vectorVarRef[indexVarRef].RefList.size() ){
                    std::cout << "OAOVisitDeclRefExpr 错误: OperandID 超出 实参列表 vectorArgIndex / RefList 范围 " << std::endl;
                    exit(1);
                }
                pOMPRegion->vectorVarRef[indexVarRef].vectorArgIndex[OperandID] = indexVar; // 这里的 indexVar 可能小于 0, 表示引用的是 OMP 内的局部变量, 可以不考虑
                STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                pOMPRegion->vectorVarRef[indexVarRef].RefList[OperandID].init(VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
                // pOMPRegion->vectorVarRef[indexVarRef].RefList.emplace_back(VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange); // 保存引用位置
            }else{
                std::cout << "OAOVisitMemberExpr 错误: 遇到 OMP 时, 保存 函数信息的 VAR_REF_LIST 有问题" << std::endl;
                exit(1);
            }
        }
        ASTStack.back().OperandState[OperandID] = true;

    // 处理定义变量的同时进行赋值的情况
    }else if(ASTOpName == "VarDecl"){

        if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
            // 需要将指针跟踪结果存入 ASTStack
            //if(ASTStack.back().OperandState[0] == false){
                ASTStack.back().Operand[OperandID] = indexVar;
                std::cout << "OAOVisitDeclRefExpr: VarDecl 引用的 indexVar = " << ASTStack.back().Operand[OperandID] << std::endl;
            //}
        }else if( AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
            if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的读操作
                if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                    saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                }else{
                    STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                    saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                }
            }
        }else{
        }
        ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识

    // 处理运算符 =
    }else if(ASTOpName == "="){
        
        std::cout << "OAOVisitDeclRefExpr: =" << std::endl;
        
        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){ // 如果是等号左边的操作数
            if(AccessType==ASTStackNode::PTR_ARY::PTR){ // 引用指针
                std::cout << "OAOVisitDeclRefExpr: 对指针变量赋值" << std::endl;
                if(indexLocal>=0){ // 处理局部指针的跟踪
                    std::cout << "OAOVisitDeclRefExpr: 处理局部指针的跟踪" << std::endl;
                    if(ASTStack.back().OperandState[1]==false){
                        std::cout << "没找到赋值源指针, 指针跟踪失败!" << std::endl;
                        exit(1);
                    }else{
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pLocal->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pLocal->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pLocal->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }
                }else{ // 这里更新外部指针指向区域大小
                    std::cout << "OAOVisitDeclRefExpr: 处理外部指针的跟踪" << std::endl;
                    if(ASTStack.back().vectorInfo[1]=="malloc"){
                    }else if(ASTStack.back().Operand[1]>=0){
                        // ghn 20191205 如果等号右边是指针，则赋值
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        if(FuncCurrent.vectorVar[indexRoot].Type != VAR_TYPE::PTR){
                            std::cout << "OAOVisitDeclRefExpr: 警告，等号右边不是指针！" << std::endl;
                        }else{
                            pVar->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                            if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                                pVar->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                                pVar->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                            }
                        }
                    }else{}
                }
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                        STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                        saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else{
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }
                }else if(indexLocal>=0 && indexVar<0){ // 局部指针指向OMP内分配的数据区域
                    // 这种情况什么都不做
                }else{
                    // std::cout << "VisitDeclRefExpr 没找到变量???" << std::endl;
                }                
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID == 1){
            if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
                // 需要将指针跟踪结果存入 ASTStack
                //if(ASTStack.back().OperandState[1] == false){
                    ASTStack.back().Operand[1] = indexVar;
                //}
            }else if( AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
                if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                        STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                        saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else{
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }
                }
            }else{
                //std::cout << "操作符\"=\" OperandID == 1 引用类型错误" << std::endl;
            }
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID>1){
            // std:: cout << "= 操作只有两个操作数, OperandID 越界, OperandID = " << OperandID << std::endl;
            exit(1);
        }

    // 处理复合赋值运算符
    }else if(ASTOpName == "+=" || ASTOpName == "-=" || ASTOpName == "*=" || ASTOpName == "/=" || ASTOpName == "%=" ||
             ASTOpName == "<<=" || ASTOpName == ">>=" || ASTOpName == "&=" || ASTOpName == "^=" || ASTOpName == "|=")
    {
        // std::cout << "OAOVisitDeclRefExpr: 复合赋值" << std::endl;

        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){
            if( AccessType==ASTStackNode::PTR_ARY::PTR ){ // 引用指针
                // 目前只处理局部指针的跟踪
                if(indexLocal>=0){
                    if(ASTStack.back().OperandState[1]==false){
                        std::cout << "没找到赋值源指针, 指针跟踪失败!" << std::endl;
                        exit(1);
                    }else{
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pLocal->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pLocal->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pLocal->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }
                }else{ // 这里更新外部指针指向区域大小
                    if(ASTStack.back().vectorInfo[1]=="malloc"){
                    }else if(ASTStack.back().Operand[1]>=0){
                        int indexRoot = (int)ASTStack.back().Operand[1];
                        pVar->indexRoot = indexRoot; // indexRoot 指向用来给当前变量赋值的外部变量
                        if(indexRoot>=0){ // 只有使用外部变量赋值的时候 才跟踪
                            pVar->RootName = FuncCurrent.vectorVar[indexRoot].Name;
                            pVar->ptrRoot = FuncCurrent.vectorVar[indexRoot].ptrDecl;
                        }
                    }else{}
                }
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
                if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                        STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                        saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else{
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }
                }else if(indexLocal>=0 && indexVar<0){ // 局部指针指向OMP内分配的数据区域
                }else{
                }                
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID == 1){
            if(AccessType==ASTStackNode::PTR_ARY::PTR && ASTStack.back().OperandState[OperandID] == false){ // 引用指针
                // 需要将指针跟踪结果存入 ASTStack
                //if(ASTStack.back().OperandState[1] == false){
                    ASTStack.back().Operand[1] = indexVar;
                //}
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY){ // 引用数组元素
                if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                        STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                        saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else{
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }
                }
            }else{
                //std::cout << "复合赋值操作符 OperandID == 1 时 引用类型错误" << std::endl;
            }
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID>1){
            // std:: cout << "复合赋值 操作只有两个操作数, OperandID 越界, 错误!" << std::endl;
            // std:: cout << "OperandID = " << OperandID << std::endl;
            // std:: cout << "操作数个数 = " << ASTStack.back().Operand.size() << std::endl;
            exit(1);
        }

    // 处理自增自减运算符
    }else if(ASTOpName == "++" || ASTOpName == "--"){
        
        std::cout << "OAOVisitDeclRefExpr: ++/--" << std::endl;
        
        if(OperandID == 0 && ASTStack.back().OperandState[OperandID] == false){
            if( AccessType==ASTStackNode::PTR_ARY::PTR ){ // 引用指针
                // 这里不需要指针跟踪
            }else if(AccessType==ASTStackNode::PTR_ARY::ARRAY ){ // 引用数组元素
                if(indexVar>=0){ // 局部指针指向外部指针的数据区域 或 引用外部指针的数据区域
                    // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
                    if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                        STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_WRITE);
                        saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }else{
                        STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                        STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_WRITE);
                        saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
                    }
                }else if(indexLocal>=0 && indexVar<0){ // 局部指针指向OMP内分配的数据区域
                    // 这种情况什么都不做
                }else{
                    std::cout << "VisitDeclRefExpr 没找到变量???" << std::endl;
                }                
            }else{}
            ASTStack.back().OperandState[OperandID] = true; // 写入处理完成标识
        }else if(OperandID>0){
            std:: cout << "++/-- 操作只有一个操作数, OperandID 越界, OperandID = " << OperandID << std::endl;
            exit(1);
        }
    }else{ // 这里处理不是上述运算符的情况
        
        // std::cout << "OAOVisitDeclRefExpr: 其他" << std::endl;
        
        if(indexVar>=0){ // 如果使用了外部变量, 为了保守, 都认为是 READ_WRITE
            // 看作是对 vectorVar[indexVar]保存的外部指针的数据域的写操作
            // wfr 20190508 REF_TYPE::READ_WRITE 改为 REF_TYPE::READ
            if(indexNode.type == NODE_TYPE::SEQUENTIAL){
                STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                saveVarRefInfo(pSEQRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
            }else{
                STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc(ST_TRANS_DEVICE_READ);
                saveVarRefInfo(pOMPRegion->vectorVarRef, indexVar, VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, TempSrcRange);
            }
        }
    }
    
    // std::cout << "OAOVisitDeclRefExpr: 退出" << std::endl;
    return true;
}

// wfr 20190206
// 处理在函数中的变量定义, 如在OMP中则保存到OMP的 vectorLocal, 否则保存到FUNC_INFO的 vectorVar
// SEQ 节点中的变量定义都获取, 因为不能事先知道哪些变量被 OMP 节点引用了
// OMP 节点中的变量都获取, 因为要进行变量跟踪
bool OAOASTVisitor::VisitVarDecl(VarDecl *VD) {
    if(BeginFlag==false){
        return true;
    }

    // 先判断在哪个函数中, 不在就跳出, 应该可以通过判断函数堆栈来确定
    // wfr 20190811 从 函数栈 来说, 变量定义属于哪个函数
    int indexFunc = -1;

    // wfr 20190811 从代码行列范围上来说, 变量定义属于哪个函数
    SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(VD->getEndLoc());
    int indexFuncWithin = InWhichFunc(Rewrite.getSourceMgr().getFileOffset(TmpLoc));

    if(FuncStack.empty()==false){
        indexFunc = FuncStack.back().indexFunc;
        if(indexFunc<0){
            std::cout << "VisitVarDecl 错误: 当前函数的 indexFunc<0" << std::endl;
            exit(1);
        }else{
            ///std::cout << "VisitVarDecl: indexFunc = " << indexFunc << std::endl;
            // wfr 20190420 如果在处理 OMP, 且不在 主体 中, 就直接返回
            if(!OMPStack.empty()){
                OMP_REGION& OMP = vectorFunc[FuncStack.back().indexFunc].vectorOMP[OMPStack.back().index];
                if(OMP.pBodyStmt==NULL){
                    std::cout << "VisitVarDecl: 不处理 OMP 主体外的定义" << std::endl;
                    return true;
                }
            }
        }
    }else{
        // return true; // 如果不是在函数中就退出
    }
    
    // wfr 以下是调试输出语句
    std::cout << "进入函数 VisitVarDecl: " << std::endl;
    std::cout << "找到变量: " << VD->getNameAsString() << std::endl;

    NODE_INDEX  indexNode;
    if(indexFunc>=0){
        // 找到定义所在的节点
        if(isa<ParmVarDecl>(VD)){
            indexNode = vectorFunc[indexFunc].MapEntry;
        }else{
            SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(VD->getEndLoc());
            unsigned int BeginOffset = Rewrite.getSourceMgr().getFileOffset(TmpLoc);
            InWhichSEQOMP(indexNode, indexFunc, BeginOffset);
        }
        if(indexNode.type==NODE_TYPE::NODE_UNINIT && vectorFunc[indexFunc].ptrDefine!=NULL){
            std::cout << "VisitVarDecl: 没找到变量定义所在的节点" << std::endl;
            exit(1);
        }
    }
    


    std::cout << "VisitVarDecl: indexDefNode.type = " << std::dec << indexNode.type << ", indexDefNode.index = " << std::dec << indexNode.index << std::endl;

    // 获得变量类型
    LangOptions MyLangOptions;
    PrintingPolicy PrintPolicy(MyLangOptions);
    std::string TypeName;
    TypeName = QualType::getAsString( VD->getType().split(), PrintPolicy ); // 变量的类型: int, int*, float*等等

    // wfr 20190806 通过变量类型字符串, 分析获得变量类型
    VAR_TYPE VarType = getVarType(TypeName);

    std::cout << "变量类型: " << TypeName << std::endl;

    // wfr 20190811
    if(indexFunc<0 && indexFuncWithin<0){// 是全局变量
        if( VarType==VAR_TYPE::CLASS_PTR || VarType==VAR_TYPE::CLASS_PTR_CONST 
            || VarType==VAR_TYPE::PTR || VarType==VAR_TYPE::PTR_CONST 
        ){
            std::cout << "VisitVarDecl 警告: 程序中有全局指针, 不能安全处理" << std::endl;
            // exit(1);
            // return true;
        }
    }else if(indexFunc<0 && indexFuncWithin>=0){
        return true; // 是模板类的 AST 上的同一个变量的多个声明, 不重复处理
    }else if(indexFunc>=0 && indexFuncWithin<0){
        return true; // wfr 20190814 不处理函数声明处的参数定义
    }else{}

    // 新建 VARIABLE_INFO 类型节点, 处理定义同时赋值的情况, ptrRoot 使用 ASTStack.back().Operand[0]
    VARIABLE_INFO   TempVar(VD, this->Rewrite);
    //std::cout << "新建 VARIABLE_INFO 类型" << std::endl;
    TempVar.Name = VD->getNameAsString();
    TempVar.ptrDecl = VD; // 变量的定义的地址
    std::cout << "VisitVarDecl: 存入变量定义地址 " << std::hex << VD << std::endl;
    // 知道变量定义所在的 DeclStmt, 才能知道变量定义语句的结尾的 位置, 从而获得代码插入位置
    if(!DeclStmtStack.empty()){ // 对于函数参数变量不采集 DeclRange 信息, 因为 DeclStmtStack 为空
        TempVar.DeclRange.init(DeclStmtStack.back().BeginLoc, DeclStmtStack.back().EndLoc, Rewrite);
    }else{
        SourceLocation TmpBeginLoc = Rewrite.getSourceMgr().getExpansionLoc(VD->getSourceRange().getBegin());
        SourceLocation TmpEndLoc = Rewrite.getSourceMgr().getExpansionLoc(VD->getSourceRange().getEnd());
        TempVar.DeclRange.init(TmpBeginLoc, TmpEndLoc.getLocWithOffset(TempVar.Name.size()), Rewrite);
    }
    TempVar.TypeName = TypeName;
    TempVar.Type = VarType;
    // 保存信息: 变量定义在哪个 node 中
    if(indexNode.type!=NODE_TYPE::NODE_UNINIT){
        TempVar.indexDefNode = indexNode;
    }
    // 保存变量作用域信息
    if(!CompoundStack.empty()){
        TempVar.Scope = CompoundStack.back();
    }
    if(TypeName.size()>5 && TypeName.substr(0,5)=="class"){
        TempVar.isClass = true;
    }

    TempVar.FullName = TempVar.Name; // 写入变量全名称
    TempVar.Rename = TempVar.Name; // 写入变量重命名

    SourceLocation InsertLoc;
    if(TypeName.back()=='*'){

        TempVar.ArrayLen = TempVar.Name; // 写入变量长度变量的名称
        TempVar.ArrayLen += "_LEN_"; // 写入变量长度变量的名称
        if (isa<ParmVarDecl>(VD)){
            InsertLoc = TempVar.DeclRange.EndLoc;

        }else{
            InsertLoc = TempVar.DeclRange.EndLoc;
            
            // 如果在声明的同时赋值了, 进行指针跟踪, 并更新指针区域大小
            // 如果不在则说明我对 TraverseVarDecl 中逻辑理解有误
            if(!ASTStack.empty() && ASTStack.back().Name == "VarDecl" && ASTStack.back().OperandState[0] == true){
                std::cout << "VisitVarDecl: ASTStack.back().Operand.size() = " << ASTStack.back().Operand.size() << std::endl;
                if(ASTStack.back().Operand[0]>=0){ // 如果指针定义时被使用其他指针赋值
                    TempVar.indexRoot = (int)ASTStack.back().Operand[0]; // indexRoot 指向用来给当前变量赋值的变量
                    std::cout << "VisitVarDecl: indexRoot = " << TempVar.indexRoot << std::endl;
                    TempVar.RootName = vectorFunc[indexFunc].vectorVar[TempVar.indexRoot].Name;
                    TempVar.ptrRoot = vectorFunc[indexFunc].vectorVar[TempVar.indexRoot].ptrDecl;
                }else if(ASTStack.back().vectorInfo[0] == "malloc"){ // 如果指针定义时指向 malloc 的空间
                }else{}
                
            }
        }
    }else if(TypeName.back()==']'){
        // 获得多维数组的元素总数, 写入 ArrayLen 域中
        TempVar.getArrayLen();
        // wfr 20190721 插入运行时函数 OAOArrayInfo 保存静态定义的数组的信息
        if (isa<ParmVarDecl>(VD)==false && indexFunc>=0) { //　wfr 20190809 如果不是函数参数 且　不是全局变量
            InsertLoc = TempVar.DeclRange.EndLoc;
            std::string ElementType = TempVar.TypeName.substr(0, TempVar.TypeName.find('[')); // 取出 '[' 之前的字符串
            TempVar.vectorInsert.emplace_back(InsertLoc, "; ", Rewrite);
            TempVar.vectorInsert.back().Code += OAO_ARRAY_NAME;
            TempVar.vectorInsert.back().Code += "( (void*)(";
            TempVar.vectorInsert.back().Code += TempVar.FullName;
            TempVar.vectorInsert.back().Code += "), sizeof(";
            TempVar.vectorInsert.back().Code += TempVar.TypeName;
            TempVar.vectorInsert.back().Code += "), sizeof(";
            TempVar.vectorInsert.back().Code += ElementType;
            TempVar.vectorInsert.back().Code += ") )";
        }
    }else{}

    // wfr 20190207 将变量存入 vectorVar / vectorLocal
    if(indexFunc>=0){
        if(indexNode.type==NODE_TYPE::PARALLEL){ // 如果在 OMP 域中
            vectorFunc[indexFunc].vectorOMP[indexNode.index].vectorLocal.push_back(TempVar);
        }else{
            if(vectorFunc[indexFunc].ptrDefine!=NULL){
                vectorFunc[indexFunc].vectorVar.push_back(TempVar);
            }else if(vectorFunc[indexFunc].ptrDecl!=NULL && vectorFunc[indexFunc].ptrDecl!=vectorFunc[indexFunc].ptrDefine){
                vectorFunc[indexFunc].vectorDeclVar.push_back(TempVar);
            }else{
                std::cout << "VisitVarDecl: ptrDefine 和 ptrDecl 都是 NULL" << std::endl;
                exit(1);
            }
        }
    }else{
        vectorGblVar.push_back(TempVar);
        vectorGblVar.back().isGlobal = true;
    }

    std::cout << "VisitVarDecl: 变量名 " << TempVar.Name << std::endl;
    std::cout << "VisitVarDecl: 变量定义地址 " << std::hex << TempVar.ptrDecl << std::endl;
    
    return true;
}

// wfr 20190420 OMP 节点的整个处理过程的最后是调用这个函数, 因此在这里出栈 OMPStack
bool OAOASTVisitor::VisitOMPParallelForDirective(OMPParallelForDirective *D) {
    if(BeginFlag==true && FuncStack.empty()==false && !OMPStack.empty()){
        OMPStack.pop_back();
    }
    return true;
}

// wfr 20190222 处理 OMP 节点的入口
bool OAOASTVisitor::TraverseOMPParallelForDirective(OMPParallelForDirective *S, DataRecursionQueue *Queue) {

    bool ShouldVisitChildren = true;
    bool ReturnValue = true;
    
    if(BeginFlag==true && FuncStack.empty()==false){        
        // 拆分出 OMP 节点
        NODE_INDEX indexOMPNode = SplitOMPNode(S);
        OMPStack.push_back(indexOMPNode);
    }

    if (!getDerived().shouldTraversePostOrder()){ // 改为后序访问, 这里应该直接跳过
        if (!getDerived().WalkUpFromOMPParallelForDirective(S)){
            if(BeginFlag==true && FuncStack.empty()==false){OMPStack.pop_back();}
            return false;
        }
    }
    
    if (!getDerived().TraverseOMPExecutableDirective(S)){
        if(BeginFlag==true && FuncStack.empty()==false){OMPStack.pop_back();}
        return false;
    }
    
    if (ShouldVisitChildren) {
        for (Stmt * SubStmt : getDerived().getStmtChildren(S)) {
            if (!( has_same_member_pointer_type<decltype( &RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value ? static_cast<typename std::conditional< has_same_member_pointer_type< decltype(&RecursiveASTVisitor::TraverseStmt), decltype(&OAOASTVisitor::TraverseStmt)>::value, OAOASTVisitor &, RecursiveASTVisitor &>::type>(*this).TraverseStmt(static_cast<Stmt *>(SubStmt), Queue) : getDerived().TraverseStmt(static_cast<Stmt *>(SubStmt)) ))
            {
                if(BeginFlag==true && FuncStack.empty()==false){OMPStack.pop_back();}
                return false; 
            }
        }
    }
    
    if (!Queue && ReturnValue && getDerived().shouldTraversePostOrder()){
        if (!getDerived().WalkUpFromOMPParallelForDirective(S)){
            return false;
        }
    }
    return ReturnValue; 
}

// 分析处理 SEQ 节点, 遍历变量引用类型, 分析推导出 入口出口的变量同步状态
// 完成则返回 true, 未完成(例如函数引用且被引用函数未完成分析的情况)则返回 false
// 这里需要使用之前想好的推断规则
// wfr 20190321 遇到函数节点需要处理: 分析类(实参是类)的域的出口入口状态, 插入源代码(表示指针区域大小的实参)
bool OAOASTVisitor::AnalyzeSEQNode(int indexSEQ, FUNC_INFO& Func){
    // 在一个节点之中代码应该是串行的

    SEQ_REGION& SEQNode = Func.vectorSEQ[indexSEQ];
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    // 变量引用列表, 每个表项对应一个变量且也是一个 vector
    std::vector<VAR_REF_LIST>& vectorVarRef = SEQNode.vectorVarRef;

    int indexFuncCallee = SEQNode.indexFunc;

    // wfr 20190501 函数出口/入口 是 SEQ 节点 且 默认的 起始/结束位置 都是0, 这里填充 起始/结束位置
    if(indexSEQ==Func.MapEntry.index){
        SEQNode.SEQRange.init(Func.CompoundRange.BeginLoc.getLocWithOffset(1), Func.CompoundRange.BeginLoc.getLocWithOffset(1), Rewrite);
    }else if(indexSEQ==Func.MapExit.index){
        SEQNode.SEQRange.init(Func.CompoundRange.EndLoc, Func.CompoundRange.EndLoc, Rewrite);
    }else{}

    if(SEQNode.isComplete==true){ // 如果已经被分析, 返回 true
        // 对于串行域中的函数调用, 在处理变量引用时已经处理完成, 直接返回 isComplete 即可
        return true;
    }
    if(vectorVarRef.size()==0){
        SEQNode.isComplete = true;
        return true;
    }

    // 1. 如果是函数调用节点, 处理: 分析一般实参的入口出口状态, 分析类(实参是类)的域的入口出口状态, 插入源代码(表示指针区域大小的实参)
    if(indexFuncCallee>=0){ // 如果是函数调用节点
        FUNC_INFO& FuncCallee = vectorFunc[indexFuncCallee]; // 被调函数信息
        if(FuncCallee.isComplete==false){
            return false; // 被调用的函数还没被分析, 当前 SEQ 需要等到所有调用的函数都分析完成, 才能进行分析
        }

        std::cout << "AnalyzeSEQNode: 处理函数调用" << FuncCallee.Name << std::endl;

        std::vector<FUNC_PARM_VAR_INFO>& vectorCalleeParm = FuncCallee.vectorParm; // 被调函数参数列表
        std::vector<VARIABLE_INFO>& vectorCalleeVar = FuncCallee.vectorVar; // 被调函数变量列表
        std::vector<FUNC_PARM_VAR_INFO>& vectorCallerParm = Func.vectorParm; // 当前函数参数列表
        //int NumArgs = SEQNode.pFuncCall->getNumArgs();

        // wfr 20191219 如果所有实参都没有在

        // 循环处理每一个形参
        for(unsigned long iArg=0; iArg<vectorCalleeParm.size(); iArg++){
            FUNC_PARM_VAR_INFO& CalleeParm = vectorCalleeParm[iArg]; // 形参信息
            VARIABLE_INFO& CalleeVar = vectorCalleeVar[CalleeParm.indexVar]; // 形参信息
            SourceLocation InsertLoc;
            std::string Code;

            if(CalleeVar.isClass==false && CalleeVar.isMember==false){ // 如果形参不是类 也不是类的域
                // wfr 20191220 处理特殊情况
                if(iArg >= vectorVarRef.size() || iArg < 0){
                    std::cout << "AnalyzeSEQNode 警告: 索引 iArg 超出 vectorVarRef.size(), 跳过本形参" << FuncCallee.Name << std::endl;
                    continue;
                }
                if(vectorVarRef[iArg].index >= vectorVar.size() || vectorVarRef[iArg].index<0){
                    std::cout << "AnalyzeSEQNode 警告: 当前实参没找到, 可能不是变量是常数或是缺省值, 跳过本实参" << FuncCallee.Name << std::endl;
                    continue;
                }

                VARIABLE_INFO& Var = vectorVar[vectorVarRef[iArg].index]; // 实参信息

                // wfr 20190723
                std::vector<FUNC_PARM_VAR_INFO>::iterator iterCallerParm;
                iterCallerParm = find(vectorCallerParm.begin(), vectorCallerParm.end(), vectorVarRef[iArg].index);
                if(iterCallerParm!=vectorCallerParm.end()){ // 说明当前实参是 当前函数的入口参数
                    if(Func.UsedInOMP==true && FuncCallee.Name=="free"){
                        std::cout << "AnalyzeSEQNode 错误: 在并行域中被调用的函数中 不能对 入口参数 进行 free()" << std::endl;
                        exit(1);
                    }
                }

                std::cout << "AnalyzeSEQNode: 处理实参 " << Var.Name << std::endl;

                Var.UsedInOMP |= CalleeVar.UsedInOMP; // 变量是否在 OMP 中被使用, 有一次被使用就认为被使用了

                // 下边, 插入源代码: 表示 指针区域大小 的实参
                if(CalleeVar.TypeName.back()=='*'){ // 如果参数是指针, 插入指针区域大小 实参

                    // 保存实参引用信息
                    // 写入 入口出口同步状态
                    vectorVarRef[iArg].SyncStateBegin = CalleeParm.SyncStateBegin;
                    vectorVarRef[iArg].SyncStateEnd = CalleeParm.SyncStateEnd;
                    CheckStReq(CalleeParm.StReq);
                    CheckStTrans(CalleeParm.StTransFunc);
                    vectorVarRef[iArg].StReq = CalleeParm.StReq;
                    vectorVarRef[iArg].StTransFunc = CalleeParm.StTransFunc;
                    vectorVarRef[iArg].NeedInsertStTrans = FuncCallee.UsedInOMP;
                    // vectorVarRef[iArg].NeedInsertStTrans = false;
                    vectorVarRef[iArg].RefList[0].StReq = CalleeParm.StReq;
                    vectorVarRef[iArg].RefList[0].StTransFunc = CalleeParm.StTransFunc;
                }else{
                    // 保存实参引用信息
                    // 写入 入口出口同步状态
                    if(CalleeVar.Type==VAR_TYPE::REF){ // 如果传入的是简单引用
                        vectorVarRef[iArg].SyncStateBegin = CalleeParm.SyncStateBegin;
                        vectorVarRef[iArg].SyncStateEnd = CalleeParm.SyncStateEnd;
                        CheckStReq(CalleeParm.StReq);
                        CheckStTrans(CalleeParm.StTransFunc);
                        vectorVarRef[iArg].StReq = CalleeParm.StReq;
                        vectorVarRef[iArg].StTransFunc = CalleeParm.StTransFunc;
                        vectorVarRef[iArg].NeedInsertStTrans = FuncCallee.UsedInOMP;
                        // vectorVarRef[iArg].NeedInsertStTrans = false;
                        vectorVarRef[iArg].RefList[0].StReq = CalleeParm.StReq;
                        vectorVarRef[iArg].RefList[0].StTransFunc = CalleeParm.StTransFunc;
                    }else{
                        vectorVarRef[iArg].SyncStateBegin = SYNC_STATE::HOST_NEW;
                        vectorVarRef[iArg].SyncStateEnd = SYNC_STATE::HOST_NEW;
                        vectorVarRef[iArg].StReq.init(ST_REQ_HOST_NEW);
                        vectorVarRef[iArg].StTransFunc.init(ST_TRANS_HOST_READ);
                        vectorVarRef[iArg].NeedInsertStTrans = FuncCallee.UsedInOMP;
                        // vectorVarRef[iArg].NeedInsertStTrans = false;
                        vectorVarRef[iArg].RefList[0].StReq.init(ST_REQ_HOST_NEW);
                        vectorVarRef[iArg].RefList[0].StTransFunc.init(ST_TRANS_HOST_READ);
                    }
                }
            }else if(CalleeVar.isMember==true){ // 这里不是真正的形参, 而是 类 形参 的 子域
                // 当前 子域 所属的 类 在形参列表中的 index, 也是 这个类 对应的 实参类 在 原始实参列表 中的 index
                // 即 在 vectorVarRef 中的 index
                int indexCalleeClassParm = find(vectorCalleeParm.begin(), vectorCalleeParm.end(), CalleeVar.indexClass) - vectorCalleeParm.begin();

                // wfr 20191220 处理特殊情况
                if(indexCalleeClassParm >= vectorCalleeParm.size() || indexCalleeClassParm < 0){
                    std::cout << "AnalyzeSEQNode 错误: 没找到子域所属的形参" << std::endl;
                    exit(1);
                }
                if(indexCalleeClassParm >= vectorVarRef.size() || indexCalleeClassParm < 0){
                    std::cout << "AnalyzeSEQNode 警告: 索引 indexCalleeClassParm 超出 vectorVarRef.size(), 跳过本形参" << FuncCallee.Name << std::endl;
                    continue;
                }
                if(vectorVarRef[indexCalleeClassParm].index >= vectorVar.size() || vectorVarRef[indexCalleeClassParm].index < 0){
                    std::cout << "AnalyzeSEQNode 警告: 当前实参没找到, 可能不是变量是常数或是缺省值, 跳过本实参" << FuncCallee.Name << std::endl;
                    continue;
                }

                // 实参 类
                VARIABLE_INFO& Class = vectorVar[vectorVarRef[indexCalleeClassParm].index];
                MY_SOURCE_RANGE RefSrcRange = vectorVarRef[indexCalleeClassParm].RefList[0].SrcRange;


                // wfr 20190723
                std::vector<FUNC_PARM_VAR_INFO>::iterator iterCallerParm;
                iterCallerParm = find( vectorCallerParm.begin(), vectorCallerParm.end(), 
                                       vectorVarRef[indexCalleeClassParm].index);
                if(iterCallerParm!=vectorCallerParm.end()){ // 说明当前实参是 当前函数的入口参数
                    if(Func.UsedInOMP==true && FuncCallee.Name=="free"){
                        std::cout << "AnalyzeSEQNode 错误: 在并行域中被调用的函数中 不能对 入口参数 进行 free()" << std::endl;
                        exit(1);
                    }
                }

                FieldDecl* pMemberDecl;
                // wfr 20191107
                const Type* pType = ((VarDecl*)(Class.ptrDecl))->getType().getTypePtr();
                std::string name = ((VarDecl*)(Class.ptrDecl))->getType().getAsString();
                if(pType==NULL){
                    std::cout << "AnalyzeSEQNode 错误: 类类型定义指针为空" << std::endl;
                    exit(1);
                }
                CXXRecordDecl* pCXX = pType->getAsCXXRecordDecl();
                //CXXRecordDecl* previous = pCXX->getPreviousDecl();
                CXXRecordDecl::field_iterator iterField=pCXX->field_begin();
                for(; iterField!=pCXX->field_end(); ++iterField){
                    if(iterField->getNameAsString()==CalleeVar.Name){ // 通过 域 的 名称字符串, 找到域的定义地址
                        pMemberDecl = (*iterField);
                        break;
                    }
                }

                int indexFunc = -1;
                for(unsigned int i=0; i<vectorFunc.size(); ++i){
                    if(Func.ptrDefine == vectorFunc[i].ptrDefine){
                        indexFunc = i;
                    }
                }
                if(indexFunc == -1){
                    std::cout << "AnalyzeSEQNode 错误: 没找到对应函数定义" << std::endl;
                    exit(1);
                }

                int indexMember = -1;
                if(iterField == pCXX->field_end()){
                    // wfr 20191219 处理没找到对应域的情况, 这是因为出现了目前不能处理的多层结构体, 这里勉强打个补丁凑合下......
                    for(unsigned int i=0; i<vectorVarRef.size(); ++i){
                        // indexClass 相同
                        // name 相同
                        int indexArgClass = vectorVarRef[indexCalleeClassParm].index; // 该实参是类, 这里是该实参的 index
                        int indexArgVar = vectorVarRef[i].index; // 该类实参的 域 的 index
                        if( indexArgClass == vectorVar[indexArgVar].indexClass
                            && CalleeVar.Name == vectorVar[indexArgVar].Name)
                        {
                            indexMember = indexArgVar;
                            break;
                        }
                    }
                }else{
                    indexMember = getMemberIndex(Class.ptrDecl, (*iterField), indexFunc);
                }

                if(indexMember == -1){
                    std::cout << "AnalyzeSEQNode 警告: 有没处理的类中的域" << std::endl;
                    continue;
                    // std::cout << "AnalyzeSEQNode 错误: 没找到类中对应的域" << std::endl;
                    // exit(1);
                }
                VARIABLE_INFO& Member = vectorVar[indexMember]; // 实参 类 的 对应的 域

                Member.UsedInOMP |= CalleeVar.UsedInOMP; // 变量是否在 OMP 中被使用, 有一次被使用就认为被使用了

                // 保存实参引用信息
                // 写入 入口出口同步状态
                if(CalleeVar.Type==VAR_TYPE::PTR || CalleeVar.Type==VAR_TYPE::REF){ // 如果传入的是简单引用
                    CheckStReq(CalleeParm.StReq);
                    CheckStTrans(CalleeParm.StTransFunc);
                    vectorVarRef.emplace_back(indexMember, CalleeParm.SyncStateBegin, CalleeParm.SyncStateEnd,
                                                CalleeParm.StReq, CalleeParm.StTransFunc); // 新建引用项, 存入 入口出口状态
                    vectorVarRef.back().NeedInsertStTrans = FuncCallee.UsedInOMP;
                    // vectorVarRef.back().NeedInsertStTrans = false;
                    vectorVarRef.back().RefList.emplace_back(VAR_REF::MEMBER_TYPE::NO_MEMBER, CalleeParm.StReq, CalleeParm.StTransFunc, RefSrcRange);
                }else{
                    STATE_CONSTR TmpStReq(ST_REQ_HOST_NEW);
                    STATE_CONSTR TmpStTransFunc(ST_TRANS_HOST_READ);
                    vectorVarRef.emplace_back(indexMember, SYNC_STATE::HOST_NEW, SYNC_STATE::HOST_NEW, 
                                                TmpStReq, TmpStTransFunc); // 新建引用项, 存入 入口出口状态
                    vectorVarRef.back().NeedInsertStTrans = FuncCallee.UsedInOMP;
                    // vectorVarRef.back().NeedInsertStTrans = false;
                    vectorVarRef.back().RefList.emplace_back(VAR_REF::MEMBER_TYPE::NO_MEMBER, TmpStReq, TmpStTransFunc, RefSrcRange);
                }
            }else{
                // 形参是类, 类的信息都由其中的域来体现, 而其自身不用处理
            }
        }

        // wfr 20190805 将相同变量的引用合并, 例如 foo(v1[index[i]], v2[index[i]])
        // 这里 index 和 i 的引用就需要合并
        for(unsigned long i = 0; i< vectorVarRef.size(); ++i){
            if(vectorVarRef[i].index < 0) continue;
            for (unsigned long j = i+1; j< vectorVarRef.size(); ++j){
                // 如果保存的是同一个变量的引用信息
                if(vectorVarRef[i].index == vectorVarRef[j].index){
                    // 将 j 的信息保存到 i 中
                    vectorVarRef[i].RefList.emplace_back(vectorVarRef[j].RefList[0]);
                    // 删除 j
                    vectorVarRef.erase(vectorVarRef.begin()+j);
                    --j; // 因为删除了 j, 所以 -- 抵消 ++ 的效果
                }
            }
            
        }
    }


    // 2. 推导: 入口同步状态约束 StReq, 以及 状态转换函数 StTransFunc
    for(unsigned int i=0; i<vectorVarRef.size(); ++i){
        if(vectorVarRef[i].RefList.empty()){
            // wfr 20191220 修改逻辑, 遇到空表项就报警告并删除, 不再认为是错误
            std::cout << "AnalyzeSEQNode 警告: 某变量的引用列表为空, 删除此空表项" << std::endl;
            vectorVarRef.erase(vectorVarRef.begin()+i);
            --i;
            continue;
        }
        VAR_REF_LIST& VarRefList = vectorVarRef[i];

        // wfr 20190730 VarRefList.StReq 等于第一个 StReq, 应该只有 HostNew 和 HostOnly 两种情况
        VarRefList.StReq = VarRefList.RefList[0].StReq;
        VarRefList.StTransFunc.init(ST_TRANS_NONE);
        if(indexFuncCallee<0){ // wfr 20191225 如果不是函数调用节点, 就设为 true
            VarRefList.NeedInsertStTrans = true;
        }

        STATE_CONSTR TmpFunc(ST_TRANS_HOST_WRITE);

        for (unsigned int i = 0; i < VarRefList.RefList.size(); ++i)
        {
            VAR_REF& VarRef = VarRefList.RefList[i];
            CheckStTrans(VarRef.StTransFunc);
            VarRefList.StTransFunc.ZERO &= VarRef.StTransFunc.ZERO;
            VarRefList.StTransFunc.ONE |= VarRef.StTransFunc.ONE;  // 按位或 后赋值

            // wfr 20191222
            if( VarRef.StTransFunc == TmpFunc )
            { // 判断是否已经遇到了 写操作, 如果遇到则之后的操作都不用再处理了
                break;
            }
        }
    }
    // 设置分析完成标识
    SEQNode.isComplete = true;
    return true;
}

// 分析处理 OMP 节点, 遍历变量引用类型, 分析推导出 入口出口的变量同步状态
// 完成则返回 true, 未完成(例如函数引用且被引用函数未完成分析的情况)则返回 false
// 这里需要使用之前想好的推断规则
// wfr 20190320 当前 OMP 节点 不存在单独的函数节点, 函数引用信息保存在 vectorVarRef
// 需要先判断 vectorVarRef 中的节点是 变量的引用 还是 函数的引用
bool OAOASTVisitor::AnalyzeOMPNode(int indexOMP, FUNC_INFO& Func){
    // 在一个节点之中代码应该是串行的

    OMP_REGION& OMPNode = Func.vectorOMP[indexOMP];
    int indexFunc = WhichFuncDecl(Func.ptrDecl);
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    // 变量引用列表, 每个表项对应一个变量且也是一个 vector
    std::vector<VAR_REF_LIST>& vectorVarRef = OMPNode.vectorVarRef;

    if(OMPNode.isComplete==true){ // 如果已经被分析, 返回 true
        // 对于串行域中的函数调用, 在处理变量引用时已经处理完成, 直接返回 isComplete 即可
        return true;
    }
    if(vectorVarRef.size()==0){
        OMPNode.isComplete = true;
        return true;
    }

    
    OMPExecutableDirective* OMP_Ptr = OMPNode.ptrDecl;
    std::vector<std::string> private_ptr;
    OMPClause* pChild;
    Stmt* pPrivateChild;
    // wfr  20191111 已知 OMPParallelForDirective* OMP_Ptr
    int clausesNum = OMP_Ptr->getNumClauses();
    for(int i = 0; i < clausesNum; ++i){
        pChild = OMP_Ptr->getClause(i);
        // ghn 20191112 找到private,firstprivate,lastprivate变量，存入数组，在处理标量时不做处理
        if( isa<OMPPrivateClause>( (pChild) ) || isa<OMPFirstprivateClause>( (pChild) ) || isa<OMPLastprivateClause>( (pChild) )){
            OMPClause::child_range iterPrivate = pChild->children();
            for(OMPClause::child_iterator iter = iterPrivate.begin() ; iter != iterPrivate.end() ; iter++){
                pPrivateChild = dyn_cast<Stmt>(*iter);
                if( isa<DeclRefExpr>( (pPrivateChild) ) ){ 
                    std::cout << "OMPPrivateClause 提示: 存在private语句" << std::endl; 
                    DeclRefExpr* pName = dyn_cast<DeclRefExpr>(pPrivateChild);
                    std::string name = pName->getNameInfo().getAsString();    
                    private_ptr.push_back(name);
                }
            }
        }       
    } 
    
        // ghn 20200228 判断OMP节点内是否有for循环，以及获取多层for循环的指针
    Stmt* pParent = OMPNode.ptrDecl;
    Stmt* pChilds;
    //const Stmt* xx = pParallelFor->getBody();
    
    Stmt* forParall = nullptr;
    if(strcmp(pParent->getStmtClassName(),"OMPParallelForDirective") == 0){  //最顶层父节点是OMP节点
        while(true){
            if(pParent->child_begin()!=pParent->child_end()){
                //获得指向子节点的指针
                pChilds = *(pParent->child_begin());
                if(pChilds!=NULL){ // 如果子节点存在
                    //std::cout << "Stmt 父节点: " << pParent->getStmtClassName() << std::endl;
                    //std::cout << "Stmt 子节点: " << pChilds->getStmtClassName() <<     std::endl;
                    bool flag = false;
                    if(isa<CapturedStmt>(pParent)){
                        //std::cout << "CapturedStmt: " << pParent->getStmtClassName() << std::endl;
                        CapturedStmt *cap = (CapturedStmt *)pParent;
                        CapturedDecl* cons= cap->getCapturedDecl();
                        Stmt* first = cons->getBody(); 
                        //std::cout << "Stmt: " << first->getStmtClassName() << std::endl;
                        if(isa<ForStmt>(first)){
                            std::cout << "AnalyzeOMPNode: " << first->getStmtClassName() << std::endl;
                            OMPNode.forParallelStmt.push_back(first);
                            pParent = first;
                            forParall = first;
                            flag = true;
                        }
                    }
                    if(flag==false){
                        pParent = pChilds;
                    }
                }else{
                    break;
                }
            }else{
                break;
            }
        }
    }
    
    // ghn 20200301 
    // 使用递归函数遍历当前ForStmt的所有子孙节点
    // 目的是看是否存在多重循环,将所有OMP的ForStmt存入OMP节点内
    //if(forParall!=NULL){
    if(isa<ForStmt>(forParall)){
        DeepTraversal(forParall,OMPNode);
    }
    
    // ghn 20200302 将for循环的循环变量存入vector
    std::vector<std::string> declRefe;
    for(std::vector<Stmt*>::iterator iter = OMPNode.forParallelStmt.begin();iter != OMPNode.forParallelStmt.end();iter++){
        ForStmt *par = (ForStmt*)(*iter); 
        VarDecl *cv = par->getConditionVariable();
        if(cv == nullptr){
            std::cout << "AnalyzeOMPNode: ForStmt内部 循环变量为空" << std::endl; 
            // ghn 20200303 forStmt的子节点,找到第一个BinaryOperator
            Stmt* pPa = *iter;
            Stmt* bin;
            for(Stmt::child_iterator childd = pPa->child_begin(); childd != pPa->child_end(); childd++){
                // ghn 20200303 若int i 定义在循环内，第一个子节点不是BinaryOperator，需要循环查找
                if(*childd != NULL && isa<BinaryOperator>(*childd)){
                    bin = *childd;
                    break;
                }
            }
            if(isa<BinaryOperator>(bin)){
                ForStmtFirstTraversal(bin,declRefe);
            }
        }else{
            std::cout << "AnalyzeOMPNode: ForStmt内部 循环变量存在:" << std::endl; 
        }
    } 

    DEFINE_ST_REQ_ST_TRANS

    // 该函数的处理逻辑: 
    // 1. 先遍历 vectorVarRef 找出其中的函数引用, 处理这些函数引用, 从函数引用获得各个实参变量引用, 将这些实参引用存入 vectorVarRef 的对应项中
    // 2. 之后再 遍历 vectorVarRef 处理各个变量引用, 分析推导出 入口出口的变量同步状态

    // 1.下面先来处理函数引用, 函数引用需要处理:
    // 1.1. 在引用的地方 插入 表示指针区域大小的 实参
    // 1.2. 从形参的出入口状态, 推导实参引用类型, 在引用列表中的合适位置插入实参的引用
    for (unsigned int indexVarRef = 0; indexVarRef < vectorVarRef.size(); ++indexVarRef){
        // VAR_REF_LIST& VarRefList = vectorVarRef[indexVarRef]; // wfr 20191224 vector 重新分配空间会导致 VarRefList 指向旧空间
        if(vectorVarRef[indexVarRef].indexCallee<0){ // 这里只处理 没处理过的 函数引用, 如果不是就进入下一次循环
            continue;
        }

        FUNC_INFO& FuncCallee = vectorFunc[vectorVarRef[indexVarRef].indexCallee]; // 被调函数信息

        if(FuncCallee.isComplete==false){
            return false; // 被调用的函数还没被分析, 当前 OMP 需要等到所有调用的函数都分析完成, 才能进行分析
        }

        std::vector<FUNC_PARM_VAR_INFO>& vectorCalleeParm = FuncCallee.vectorParm; // 被调函数参数列表
        std::vector<VARIABLE_INFO>& vectorCalleeVar = FuncCallee.vectorVar; // 被调函数变量列表
         // wfr 20191224 vector 重新分配空间会导致 vectorArgIndex 指向旧空间
        // std::vector<int>& vectorArgIndex = VarRefList.vectorArgIndex; // 引用处的实参列表

        // 循环处理每一个形参
        for(unsigned long iArg=0; iArg<vectorCalleeParm.size(); iArg++){
            FUNC_PARM_VAR_INFO& CalleeParm = vectorCalleeParm[iArg]; // 形参信息
            VARIABLE_INFO& CalleeVar = vectorCalleeVar[CalleeParm.indexVar]; // 形参信息
            SourceLocation InsertLoc;
            std::string Code;

            int TmpIndex = vectorVarRef[indexVarRef].vectorArgIndex[iArg];
            if(TmpIndex<0){ // 说明当前实参引用的是 OMP 内部变量, 直接跳过
                continue;
            }

            // wfr 20190723
            if(FuncCallee.Name=="free"){
                std::cout << "AnalyzeOMPNode 错误: 不支持在并行域中 对外部变量进行 free()" << std::endl;
                exit(1);
            }

            if(CalleeVar.isClass==false && CalleeVar.isMember==false){ // 如果形参不是类 也不是类的域
                int TmpIndex = vectorVarRef[indexVarRef].vectorArgIndex[iArg];
                VARIABLE_INFO& Var = vectorVar[TmpIndex]; // 实参信息
                Var.UsedInOMP = true; // 这里是处理 OMP 中的函数调用, 所以认为该变量在 OMP 中被使用了

                MY_SOURCE_RANGE RefSrcRange = vectorVarRef[indexVarRef].RefList[iArg].SrcRange;
                // 插入引用
                // wfr 20190716 这里需要对 CalleeParm.StReq 和 CalleeParm.StTransFunc 进行转换
                // 因为 这里在 device 上调用 函数, 被调函数中 所有读写操作 都变成了 device上的 读写操作
                // 需要转换 入口同步状态需求 以及 状态转换函数
                // CheckStReq(CalleeParm.StReq);
                CheckStTrans(CalleeParm.StTransFunc);
                STATE_CONSTR TmpStReq = CalleeParm.StReq;
                // STATE_CONSTR TmpStReq(ST_REQ_DEVICE_NEW);
                STATE_CONSTR TmpStTransFunc = CalleeParm.StTransFunc;
                ModifyFuncCallInOMP(TmpStReq, TmpStTransFunc);
                // wfr 20191224 这里对 vectorVarRef 进行了 emplace_back
                saveVarRefInfo(vectorVarRef, TmpIndex, (VAR_REF::MEMBER_TYPE)Var.isArrow, 
                                TmpStReq, TmpStTransFunc, RefSrcRange);

            }else if(CalleeVar.isMember==true){ // 这里不是真正的形参, 而是 类 形参 的 子域
                // 当前 子域 所属的 类 在形参列表中的 index 以及 表项
                /*int indexCalleeClassParm = find(vectorVarRef.begin(), vectorVarRef.end(), CalleeVar.indexClass) - vectorVarRef.begin();
                FUNC_PARM_VAR_INFO& CalleeClassParm = vectorVarRef[indexCalleeClassParm];
                VARIABLE_INFO& CalleeClass =  vectorCalleeVar[CalleeVar.indexClass]; // 当前 子域 所属的 类 在参数列表中的表项*/
                int indexCalleeClassParm = find(vectorCalleeParm.begin(), vectorCalleeParm.end(), CalleeVar.indexClass) - vectorCalleeParm.begin();
                //FUNC_PARM_VAR_INFO& CalleeClassParm = vectorCalleeParm[indexCalleeClassParm];
                //VARIABLE_INFO& CalleeClass =  vectorCalleeVar[CalleeVar.indexClass]; // 当前 子域 所属的 类 在参数列表中的表项

                if(indexCalleeClassParm<0){
                    std::cout << "AnalyzeOMPNode 错误: 在形参列表中没找到该 类的 子域" << std::endl;
                    exit(1);
                }

                // wfr 20190802 实参是并行域内部变量, 不处理
                int TmpIndex = vectorVarRef[indexVarRef].vectorArgIndex[indexCalleeClassParm];
                if(TmpIndex<0){
                    continue;
                }

                // 找到当前函数中 对应的 域
                VARIABLE_INFO& Class = vectorVar[TmpIndex]; // 实参 类
                int indexMember = MemberIsWhichLocal(Class.ptrDecl , CalleeVar.Name, indexFunc, indexOMP); // 对应的 类 实参 的 域 的 index
                // 如果变量列表中不存在 对应的 域, 则新建项
                if(indexMember<0){
                    CXXRecordDecl* pCXX = ((VarDecl*)(Class.ptrDecl))->getType()->getAsCXXRecordDecl();
                    CXXRecordDecl::field_iterator iterField=pCXX->field_begin();
                    for(; iterField!=pCXX->field_end(); ++iterField){
                        if(iterField->getNameAsString()==CalleeVar.Name){ // 通过 域 的 名称字符串, 找到域的定义地址
                            break;
                        }
                    }
                    // 保存 OMP 中 LocalMember 信息
                    indexMember = saveLocalMemberInfo((*iterField), TmpIndex, indexFunc, indexOMP);
                }
                VARIABLE_INFO& Member = vectorVar[indexMember]; // 实参 类 的 对应的 域
                Member.UsedInOMP = true; // 这里是处理 OMP 中的函数调用, 所以认为该变量在 OMP 中被使用了

                MY_SOURCE_RANGE RefSrcRange = vectorVarRef[indexVarRef].RefList[indexCalleeClassParm].SrcRange;
                // 插入引用
                // InsertVarRef(*iterVarRef, RefType, (VAR_REF::MEMBER_TYPE)Member.isArrow, RefSrcRange);
                // wfr 20190716 这里需要对 CalleeParm.StReq 和 CalleeParm.StTransFunc 进行转换
                // 因为 这里在 device 上调用 函数, 被调函数中 所有读写操作 都变成了 device上的 读写操作
                // 需要转换 入口同步状态需求 以及 状态转换函数
                CheckStReq(CalleeParm.StReq);
                CheckStTrans(CalleeParm.StTransFunc);
                STATE_CONSTR TmpStReq = CalleeParm.StReq;
                STATE_CONSTR TmpStTransFunc = CalleeParm.StTransFunc;
                ModifyFuncCallInOMP(TmpStReq, TmpStTransFunc);
                saveVarRefInfo(vectorVarRef, indexMember, (VAR_REF::MEMBER_TYPE)Member.isArrow, 
                                TmpStReq, TmpStTransFunc, RefSrcRange);
            }else{
                // 形参是类, 类的信息都由其中的域来体现, 而其自身不用处理
            }
        }

        vectorVarRef[indexVarRef].indexCallee = -1; // 处理完当前函数引用后, 将 indexCallee 设置为 -1, 表示已经处理了, 防止重复处理
    }

    //SourceLocation ForLoc;
    // 2. 下面处理变量引用, 分析推导出: 入口同步状态约束 StReq, 以及 状态转换函数 StTransFunc
    for (unsigned int i = 0; i < vectorVarRef.size(); ++i)
    {
        VAR_REF_LIST& VarRefList = vectorVarRef[i];
        if(VarRefList.index<0){ // 说明是一个函数引用节点, 进下一次循环
            continue;
        }

        if(VarRefList.RefList.empty()) {
            std::cout << "AnalyzeOMPNode 错误: 某变量的引用列表为空" << std::endl;
            exit(1);
        }

        VarRefList.StReq.init(ST_REQ_DEVICE_NEW); // wfr 20190715 认为变量只要用到就要求最新, 因为无法准确判断 全写 操作
        VarRefList.StTransFunc.init(ST_TRANS_DEVICE_READ);
        VarRefList.NeedInsertStTrans = true;

        STATE_CONSTR TmpFunc(ST_TRANS_DEVICE_WRITE);

        for (unsigned int i = 0; i < VarRefList.RefList.size(); ++i)
        {
            VAR_REF& VarRef = VarRefList.RefList[i];
            CheckStTrans(VarRef.StTransFunc);
            VarRefList.StTransFunc.ZERO &= VarRef.StTransFunc.ZERO;
            VarRefList.StTransFunc.ONE |= VarRef.StTransFunc.ONE;

            if ( VarRefList.StTransFunc == TmpFunc )
            { // 判断是否已经遇到了 写操作, 如果遇到则之后的操作都不用再处理了
                //ghn 20191020 判断标量是否在并行域中被修改
                if( vectorVar[VarRefList.index].Type == COPY && vectorVar[VarRefList.index].isMember == false 
                    && vectorVar[VarRefList.index].TypeName.back()!=']' && vectorVar[VarRefList.index].TypeName.back()!='*'){
                    // ghn 20191112 已被表示为private的不处理
                    bool isPrivate = false;
                    std::vector<std::string>::iterator privateName;
                    for(privateName = private_ptr.begin(); privateName != private_ptr.end(); privateName++){
                        if(*privateName == vectorVar[VarRefList.index].Name){
                            isPrivate = true;
                        }
                    }
                    // ghn 20200304 排除循环变量（例如i,j）
                    bool isParallel = false;
                    for(std::vector<std::string>::iterator iter = declRefe.begin();iter != declRefe.end();iter++){
                        //std::string expectsString{ *iter };
                        std::string res = (std::string)(*iter);
                        char *strc1 = new char[strlen(res.c_str())+1];
                        strcpy(strc1, res.c_str());
                        char *strc2 = new char[strlen(vectorVar[VarRefList.index].Name.c_str())+1];
                        strcpy(strc2, vectorVar[VarRefList.index].Name.c_str());
                        if(strcmp(strc1,strc2) == 0){
                            isParallel = true;
                            break;
                        }
                    }
                    if(isPrivate == false && isParallel == false){
                        std::cout << "AnalyzeOMPNode: 标量被更新" << std::endl;
                        if(OMPNode.scalar == "NULL"){
                            OMPNode.scalar = vectorVar[VarRefList.index].Name;
                        }else{
                            OMPNode.scalar = OMPNode.scalar + ", " + vectorVar[VarRefList.index].Name;
                        }
                    }
                }
                break;
            }
            
        }
        
    }

    // 设置分析完成标识
    OMPNode.isComplete = true;
    return true;
}

// wfr 20190720 协商 入口同步状态; 用于 确定 for/while 循环 入口处的 变量入口状态
int OAOASTVisitor::CoSyncStateBegin(STATE_CONSTR& EntryStConstr, STATE_CONSTR& EntryStConstrBegin){
    DEFINE_ST_REQ_ST_TRANS
    if(EntryStConstrBegin==StReqUninit){
        std::cout << "CoSyncStateBegin 错误: 非法的 EntryStConstrBegin 类型" << std::endl;
        exit(1);
    }else if(EntryStConstr==StReqUninit || EntryStConstr==StReqNone){
        // wfr 20190720 如果 EntryStConstrBegin 是 StReqHostOnly,只有这里才能被用来赋值
        EntryStConstr = EntryStConstrBegin;
    }else{
        EntryStConstr.ONE |= EntryStConstrBegin.ONE;
    }

    return 0;
}

// 将当前节点的在变量作用域中的子节点入队
int OAOASTVisitor::EnqueueChildren(std::vector<NODE_INDEX>& vectorProcess, NODE_INDEX indexNode, int indexVar, FUNC_INFO& Func){

    SEQ_PAR_NODE_BASE* pParentBase;
    pParentBase = getNodeBasePtr(Func, indexNode);
    VARIABLE_INFO& Var = Func.vectorVar[indexVar];

    // 这里检查当前 Parent 的所有 Children 节点, 入队满足条件的 Children 节点, 设置 incChildren
    for(unsigned long i=0; i<pParentBase->vectorChildren.size(); ++i){
        // 子节点入队条件: SEQ/OMP节点与变量作用域有交集 && 不重复入队
        bool enqueue; enqueue = false; // 是否入队的标识
        NODE_INDEX indexChild;
        OMP_REGION* pOMP; pOMP = NULL;
        SEQ_REGION* pSEQ; pSEQ = NULL;
        SEQ_PAR_NODE_BASE* pBase; pBase = NULL;
        indexChild = pParentBase->vectorChildren[i];

        // 循环内部不需要处理, 不入队循环体入口节点
        if(pParentBase->LoopNeedAnalysis==false && indexChild==pParentBase->LoopBody){
            continue;
        }

        if(indexChild.type == NODE_TYPE::SEQUENTIAL){
            pSEQ = &Func.vectorSEQ[indexChild.index];
            pBase = (SEQ_PAR_NODE_BASE*)pSEQ;

            // 先初始化下, 即如果保存的不是当前变量的信息就会被初始化
            pBase->initSYNCInfo(indexVar);
            if(pBase->isProcessed==true){
                continue; // 如果子节点已经处理了就跳过
            }
            
            // SEQ/OMP节点与变量作用域有交集
            if( (pSEQ->SEQRange.BeginOffset<=Var.Scope.BeginOffset && Var.Scope.BeginOffset<=pSEQ->SEQRange.EndOffset) ||
                (pSEQ->SEQRange.BeginOffset<=Var.Scope.EndOffset && Var.Scope.EndOffset<=pSEQ->SEQRange.EndOffset) ||
                (Var.Scope.BeginOffset<=pSEQ->SEQRange.BeginOffset && pSEQ->SEQRange.EndOffset<=Var.Scope.EndOffset) )
            {
                enqueue = true;
            }

        }else if(indexChild.type == NODE_TYPE::PARALLEL){
            pOMP = &Func.vectorOMP[indexChild.index];
            pBase = (SEQ_PAR_NODE_BASE*)pOMP;

            // 先初始化下, 即如果保存的不是当前变量的信息就会被初始化
            pBase->initSYNCInfo(indexVar);
            if(pBase->isProcessed==true){
                continue; // 如果子节点已经处理了就跳过
            }
            
            // SEQ/OMP节点与变量作用域有交集
            if( (pOMP->DirectorRange.BeginOffset<=Var.Scope.BeginOffset && Var.Scope.BeginOffset<=pOMP->OMPRange.EndOffset) ||
                (pOMP->DirectorRange.BeginOffset<=Var.Scope.EndOffset && Var.Scope.EndOffset<=pOMP->OMPRange.EndOffset) ||
                (Var.Scope.BeginOffset<=pOMP->DirectorRange.BeginOffset && pOMP->OMPRange.EndOffset<=Var.Scope.EndOffset) )
            {
                enqueue = true;
            }
            
        }else{
            std::cout << "initBlockGroup 错误: 某个子节点没初始化" << std::endl;
            exit(1);
        }

        if(enqueue == false){ continue; } // 不满足 (完全在变量作用域中 || 有变量引用) , 跳过当前子节点

        // wfr 20190430 更新子节点 isMapped 状态
        pBase->isMapped = pParentBase->isMapped;

        // 判断当前子节点是否已经在 vectorChildren 中
        std::vector<NODE_INDEX>::iterator iterChild;
        iterChild = find(vectorProcess.begin(), vectorProcess.end(), indexChild);
        if(iterChild != vectorProcess.end()) { continue; } // 如果重复则跳过当前子节点

        if(enqueue == true){ // 满足子节点入队条件, 入队当前子节点
            pBase->initSYNCInfo(indexVar); // 初始化, 说明当前正在处理 indexVar 指向的变量
            vectorProcess.push_back(indexChild);
        }
    }
    return 0;
}

// 先深遍历串并行图, 推断 函数参数 的 接口同步状态:
// 1. 函数参数的入口同步需求
// 2. 函数参数的 等效状态转移函数 即 出口状态约束
// wfr 20190321 不处理 类, 因为 类 是通过 其中的 域 来体现同步状态的
int OAOASTVisitor::InferParmInterfaceState(FUNC_INFO& FuncCurrent){
    std::vector<SEQ_REGION>& vectorSEQ = FuncCurrent.vectorSEQ;
    std::vector<OMP_REGION>& vectorOMP = FuncCurrent.vectorOMP;
    std::vector<VARIABLE_INFO>& vectorVar = FuncCurrent.vectorVar;
    std::vector<FUNC_PARM_VAR_INFO>& vectorParm = FuncCurrent.vectorParm;

    DEFINE_ST_REQ_ST_TRANS

    // 需要一个栈来保存先深路径, 栈中每个项需要一个2元素数组保存路径上的同步状态
    // wfr 20190322 这里要保存的是路径到目前为止的叠加状态, 不仅仅是当前节点的状态
    class DEPTH_NODE : public NODE_INDEX{
    public:
        int Next; // 下一个要搜索的子节点
        STATE_CONSTR StReq, StTrans;

        DEPTH_NODE(const NODE_INDEX& in) : NODE_INDEX(in){
            Next = 0;
        }

        bool operator==(const NODE_INDEX& in){
            if(in.type==type && in.index==index){
                return true;
            }
            return false;
        }
    };

    // wfr 20190725 如果该函数在 OMP 中被调用
    if(FuncCurrent.UsedInOMP==true){

        // 每个循环处理一个函数参数, 保守分析
        for(unsigned long indexParm=0; indexParm<vectorParm.size(); ++indexParm){
            int indexVar = vectorParm[indexParm].indexVar;
            if(vectorVar[indexVar].isClass==true){ // wfr 20190321 不处理 类, 因为 类 是通过 其中的 域 来体现同步状态的
                continue;
            }
            vectorParm[indexParm].StReq=StReqHostNew; // 保守认为是 StReqHostNew
            vectorParm[indexParm].StTransFunc=StTransHostRead; // 先设为 StTransHostRead, 之后找有没有写操作

            if(!vectorOMP.empty()){
                std::cout << "InferParmInterfaceState 错误: 当前函数在 OMP 中被调用的, 且还含有并行域, 造成并行域嵌套" << std::endl;
                exit(1);
            }

            for(unsigned long iSEQ=0; iSEQ<vectorSEQ.size(); ++iSEQ){
                std::vector<VAR_REF_LIST>::iterator iterVarRef;
                iterVarRef = find(vectorSEQ[iSEQ].vectorVarRef.begin(), vectorSEQ[iSEQ].vectorVarRef.end(), indexVar);
                if(iterVarRef!=vectorSEQ[iSEQ].vectorVarRef.end()){
                    STATE_CONSTR TmpStReq = iterVarRef->StReq;
                    STATE_CONSTR TmpStTrans = iterVarRef->StTransFunc;
                    CheckStReq(TmpStReq);
                    CheckStTrans(TmpStTrans);
                    
                    if(TmpStReq!=StReqHostNew && TmpStReq!=StReqNone){
                        std::cout << "InferParmInterfaceState 错误: 当前函数在 OMP 中被调用的, 且 参数入口状态需求非法" << std::endl;
                        exit(1);
                    }

                    if(TmpStTrans==StTransHostWrite){
                        vectorParm[indexParm].StTransFunc=StTransHostWrite; // 保守地认为有写操作
                    }else if(TmpStTrans==StTransHostRead){
                        // 什么都不做
                    }else{
                        std::cout << "InferParmInterfaceState 错误: 当前函数在 OMP 中被调用的, 且 参数状态转换函数非法" << std::endl;
                        exit(1);
                    }
                }
            }
        }// 每个循环处理一个函数参数


    // 这里处理该函数没有在 OMP 中被调用的情况
    }else{

        std::vector<DEPTH_NODE> vectorRoute;

        // 每个循环处理一个函数参数
        for(unsigned long indexParm=0; indexParm<vectorParm.size(); ++indexParm){
            int indexVar = vectorParm[indexParm].indexVar;
            if(vectorVar[indexVar].isClass==true){ // wfr 20190321 不处理 类, 因为 类 是通过 其中的 域 来体现同步状态的
                continue;
            }

            // 先正序先深遍历, 推断参数的入口同步状态
            // 初始化入口同步状态
            STATE_CONSTR ParmStReq;
            // 始化先深栈
            vectorRoute.clear();
            // 先推断入口同步状态
            vectorRoute.emplace_back(FuncCurrent.MapEntry); // 入口节点入队
            SEQ_PAR_NODE_BASE* pParentNodeBase;
            SEQ_PAR_NODE_BASE* pChildNodeBase;
            pParentNodeBase = getNodeBasePtr(FuncCurrent, FuncCurrent.MapEntry); // 获得节点指针
            if(!pParentNodeBase){
                std::cout << "InferParmInterfaceState 错误: 函数入口节点没有初始化" << std::endl;
                exit(1);
            }
            // 初始化同步状态信息
            std::vector<VAR_REF_LIST>::iterator iterVarRef;
            iterVarRef = find(pParentNodeBase->vectorVarRef.begin(), pParentNodeBase->vectorVarRef.end(), indexVar);
            if(iterVarRef!=pParentNodeBase->vectorVarRef.end()){
                CheckStReq(iterVarRef->StReq);
                vectorParm[indexParm].StReq=iterVarRef->StReq;
                vectorRoute.pop_back(); // 第一个节点就可以获得入口约束, 不用再讨论后边的子节点
            }

            while(!vectorRoute.empty()){ // 先深栈非空则进入循环, 处理栈顶的节点
                // 如果入口同步状态要求已经出现了分歧, 不用继续讨论, 直接跳出
                if(vectorParm[indexParm].StReq==StReqDiverg){
                    vectorRoute.clear();
                    break;
                }
                int indexParent;
                NODE_INDEX indexParentNode;
                indexParent = vectorRoute.size() - 1; // vectorRoute 中最后一个节点的 index
                indexParentNode = vectorRoute[indexParent]; // 最后一个节点的 NODE_INDEX
                pParentNodeBase = getNodeBasePtr(FuncCurrent, indexParentNode); // 最后一个节点的 指针

                if((unsigned long)vectorRoute[indexParent].Next < pParentNodeBase->vectorChildren.size()){
                    int indexChild;
                    NODE_INDEX indexChildNode;
                    indexChildNode = pParentNodeBase->vectorChildren[vectorRoute[indexParent].Next]; // 下一个节点的 NODE_INDEX
                    vectorRoute[indexParent].Next++; // 自增指向下一个子节点
                    
                    // 这里要判断该子节点是不是回头路径
                    std::vector<DEPTH_NODE>::iterator iterDepthNode;
                    iterDepthNode = find(vectorRoute.begin(), vectorRoute.end(), indexChildNode);
                    if(iterDepthNode!=vectorRoute.end()){ // 说明当前子节点已经在栈中, 当前正在走回头路径
                        continue; // 不处理当前子节点
                    }

                    vectorRoute.emplace_back(indexChildNode); // 子节点入队
                    indexChild = vectorRoute.size() - 1; // vectorRoute 中最后一个节点的 index
                    pChildNodeBase = getNodeBasePtr(FuncCurrent, indexChildNode); // 最后子节点的 指针

                    if(!pChildNodeBase){
                        std::cout << "InferParmInterfaceState 错误: 节点没有初始化" << std::endl;
                        exit(1);
                    }
                    // 处理新压入先深路径的节点的信息
                    iterVarRef = find(pChildNodeBase->vectorVarRef.begin(), pChildNodeBase->vectorVarRef.end(), indexVar);
                    if(iterVarRef!=pChildNodeBase->vectorVarRef.end()){
                        CheckStReq(iterVarRef->StReq);
                        ParmStReq = iterVarRef->StReq;

                        if(vectorParm[indexParm].StReq==StReqUninit){
                            vectorParm[indexParm].StReq = ParmStReq;
                        }else if( vectorParm[indexParm].StReq!=StReqUninit 
                                && vectorParm[indexParm].StReq!=ParmStReq
                        ){
                            vectorParm[indexParm].StReq = StReqDiverg;
                        }else{}
                        vectorRoute.pop_back(); // 出栈当前子节点
                    }

                }else{ // 说明到了出口节点, 进行回退
                    vectorRoute.pop_back(); // 出栈当前子节点
                }
            }
            // 至此处理完入口同步状态

            if(vectorParm[indexParm].StReq==StReqUninit){
                vectorParm[indexParm].StReq.init(ST_REQ_NONE);
                std::cout << "InferParmInterfaceState 警告: vectorParm[indexParm].StReq 设为 ST_REQ_NONE" << std::endl;
            }


            // 上边获得入口同步状态约束
            // 下边获得等效状态转移函数


            // 再倒序(即从出口向入口方向)先深遍历, 推断参数的出口同步状态
            STATE_CONSTR ParmStTrans;
            // 初始化先深栈
            vectorRoute.clear();
            // 先推断出口同步状态
            vectorRoute.emplace_back(FuncCurrent.MapExit); //  出口节点入队
            pChildNodeBase = getNodeBasePtr(FuncCurrent, FuncCurrent.MapExit); // 获得节点指针
            if(!pChildNodeBase){
                std::cout << "InferParmInterfaceState 错误: 函数出口节点没有初始化" << std::endl;
                exit(1);
            }
            // 初始化同步状态信息
            iterVarRef = find(pChildNodeBase->vectorVarRef.begin(), pChildNodeBase->vectorVarRef.end(), indexVar);
            if(iterVarRef!=pChildNodeBase->vectorVarRef.end()){
                CheckStReq(iterVarRef->StReq);
                CheckStTrans(iterVarRef->StTransFunc);
                // 初始化参数的接口同步状态
                STATE_CONSTR TmpStReq = iterVarRef->StReq;
                STATE_CONSTR TmpStTrans = iterVarRef->StTransFunc;
                if( TmpStTrans==StTransHostWrite || TmpStTrans==StTransDeviceWrite 
                    || TmpStReq==StReqHostOnly ){
                    // 获得函数参数的 等效状态转移函数, 即函数参数的 出口状态约束

                    // wfr 20190817
                    STATE_CONSTR TmpExitConstr = ExeStTrans(TmpStTrans, TmpStReq);
                    vectorParm[indexParm].StTransFunc = InferStTransFunc(vectorParm[indexParm].StReq, TmpExitConstr);
                    // vectorParm[indexParm].StTransFunc = ExeStTrans(TmpStTrans, TmpStReq);


                    vectorRoute.pop_back(); // 倒数第一节点就有 写操作/free操作, 之前的节点都不用处理了, 该节点出栈, 之后的循环也就不会进入了
                }else if(TmpStTrans==StTransDiverg){
                    // 获得函数参数的 等效状态转移函数, 即函数参数的 出口状态约束
                    vectorParm[indexParm].StTransFunc = StTransDiverg;
                    vectorRoute.pop_back(); // 倒数第一节点就有 分歧, 之前的节点都不用处理了, 该节点出栈, 之后的循环也就不会进入了
                }else{
                    vectorRoute[0].StTrans = ExeStTrans(TmpStTrans, TmpStReq);
                }
            }

            while(!vectorRoute.empty()){ // 先深栈非空则进入循环, 处理栈顶的节点
                // 如果 状态转移函数已经出现分歧, 不用分析剩余节点, 直接退出
                if(vectorParm[indexParm].StTransFunc==StTransDiverg){
                    vectorRoute.clear();
                    break;
                }

                // 因为是倒序搜索, 即从子节点到父节点, 所以先获得子节点信息
                int indexChild;
                NODE_INDEX indexChildNode;
                indexChild = vectorRoute.size() - 1; // vectorRoute 中最后一个节点的 index
                indexChildNode.init(vectorRoute[indexChild].type, vectorRoute[indexChild].index); // 最后一个节点的 NODE_INDEX
                pChildNodeBase = getNodeBasePtr(FuncCurrent, indexChildNode); // 最后一个节点的 指针

                if((unsigned long)vectorRoute[indexChild].Next < pChildNodeBase->vectorParents.size()){

                    int indexParent;
                    NODE_INDEX indexParentNode;
                    indexParentNode = pChildNodeBase->vectorParents[vectorRoute[indexChild].Next]; // 一个父节点节点的 NODE_INDEX
                    vectorRoute[indexChild].Next++; // 自增指向下一个父节点
                    pParentNodeBase = getNodeBasePtr(FuncCurrent, indexParentNode); // 最后子节点的 指针

                    // 这里要判断该父节点是不是回头路径
                    std::vector<DEPTH_NODE>::iterator iterDepthNode;
                    iterDepthNode = find(vectorRoute.begin(), vectorRoute.end(), indexParentNode);
                    if(iterDepthNode!=vectorRoute.end()){ // 说明当前父节点已经在栈中, 当前正在走回头路径
                        continue; // 不处理当前父节点
                    }

                    vectorRoute.emplace_back(indexParentNode); // 子节点入队
                    indexParent = vectorRoute.size() - 1; // vectorRoute 中最后一个节点的 index

                    if(!pParentNodeBase){
                        std::cout << "InferParmInterfaceState 错误: 节点没有初始化" << std::endl;
                        exit(1);
                    }
                    // 初始化同步状态信息
                    iterVarRef = find(pParentNodeBase->vectorVarRef.begin(), pParentNodeBase->vectorVarRef.end(), indexVar);
                    if(iterVarRef!=pParentNodeBase->vectorVarRef.end()){ // 在当前节点中找到了当前变量的引用
                        CheckStReq(iterVarRef->StReq);
                        CheckStTrans(iterVarRef->StTransFunc);
                        STATE_CONSTR TmpStReq = iterVarRef->StReq;
                        STATE_CONSTR TmpStTrans = iterVarRef->StTransFunc;

                        if( TmpStTrans==StTransHostWrite || TmpStTrans==StTransDeviceWrite 
                            || TmpStReq==StReqHostOnly ){
                            TmpStTrans = ExeStTrans(TmpStTrans, TmpStReq);
                            if(vectorRoute[indexChild].StTrans!=StTransUninit){
                                vectorRoute[indexParent].StTrans = ExeStTrans(vectorRoute[indexChild].StTrans, TmpStTrans);
                            }else{
                                vectorRoute[indexParent].StTrans = TmpStTrans;
                            }

                            if(vectorParm[indexParm].StTransFunc==StTransUninit){
                                // vectorParm[indexParm].StTransFunc = vectorRoute[indexParent].StTrans;
                                // wfr 20190817
                                vectorParm[indexParm].StTransFunc = InferStTransFunc(vectorParm[indexParm].StReq, vectorRoute[indexParent].StTrans);
                            }else if(vectorParm[indexParm].StTransFunc!=vectorRoute[indexParent].StTrans){
                                vectorParm[indexParm].StTransFunc = StTransDiverg;
                            }else{}

                            vectorRoute.pop_back(); // 倒数第一节点就有 写操作/free操作/分歧, 之前的节点都不用处理了, 该节点出栈, 之后的循环也就不会进入了
                        }else if(TmpStTrans==StTransDiverg){
                            vectorParm[indexParm].StTransFunc = StTransDiverg;
                            vectorRoute.pop_back(); // 倒数第一节点就有 分歧, 之前的节点都不用处理了
                        }else{
                            TmpStTrans = ExeStTrans(TmpStTrans, TmpStReq);
                            if(vectorRoute[indexChild].StTrans!=StTransUninit){
                                vectorRoute[indexParent].StTrans = ExeStTrans(vectorRoute[indexChild].StTrans, TmpStTrans);
                            }else{
                                vectorRoute[indexParent].StTrans = TmpStTrans;
                            }
                            if(vectorRoute[indexParent].StTrans==StTransSync){ // 如果已经是同步状态
                                if(vectorParm[indexParm].StTransFunc==StReqUninit){ // 如果 函数参数的 等效状态转移函数 还未初始化
                                    // wfr 20190817
                                    vectorParm[indexParm].StTransFunc = InferStTransFunc(vectorParm[indexParm].StReq, vectorRoute[indexParent].StTrans);

                                // 如果 函数参数在已经处理过的先深路径上的 等效状态转移函数 与 当前先深路径整体的 等效状态转移函数 不相同
                                }else if(vectorParm[indexParm].StTransFunc!=vectorRoute[indexParent].StTrans){
                                    vectorParm[indexParm].StTransFunc = StTransDiverg; // 则认为出现分歧
                                }else{}
                                vectorRoute.pop_back(); // 出栈当前子节点
                            }
                        }
                    }else{
                        // 当前节点没有引用当前函数参数, indexParent节点 就继承 indexChild节点 的 StTrans
                        vectorRoute[indexParent].StTrans = vectorRoute[indexChild].StTrans;
                    }

                }else{ // 说明到了入口节点, 进行回退
                    if(vectorRoute.back().StTrans!=StReqUninit){ // 当前先深路径整体的 等效状态转移函数 不为空
                        if(vectorParm[indexParm].StTransFunc==StReqUninit){ // 如果 函数参数的 等效状态转移函数 还未初始化
                            // vectorParm[indexParm].StTransFunc = vectorRoute.back().StTrans; // 进行初始化
                            // wfr 20190817
                            vectorParm[indexParm].StTransFunc = InferStTransFunc(vectorParm[indexParm].StReq, vectorRoute.back().StTrans);

                        // 如果 函数参数在已经处理过的先深路径上的 等效状态转移函数 与 当前先深路径整体的 等效状态转移函数 不相同
                        }else if(vectorParm[indexParm].StTransFunc!=vectorRoute.back().StTrans){
                            vectorParm[indexParm].StTransFunc = StTransDiverg; // 则认为出现分歧
                        }else{}
                    }
                    vectorRoute.pop_back(); // 出栈当前子节点
                }
            } // while(!vectorRoute.empty()){ // 先深栈非空则进入循环, 处理栈顶的节点

            if(FuncCurrent.UsedInOMP==true){
                if( vectorParm[indexParm].StReq==StReqUninit 
                    || vectorParm[indexParm].StReq==StReqDiverg )
                {
                    vectorParm[indexParm].StReq=StReqHostNew;
                }
                if( vectorParm[indexParm].StTransFunc==StTransUninit 
                    || vectorParm[indexParm].StTransFunc==StTransDiverg )
                { // wfr 20190723 对于在并行域中被调用的函数, 出口状态约束不确定/有分歧的时候, 就认为在 host 中有写操作
                    vectorParm[indexParm].StTransFunc=StTransHostWrite;
                }
            }else{
                if(vectorParm[indexParm].StTransFunc==StReqUninit){
                    vectorParm[indexParm].StTransFunc = StTransDiverg; // 则认为出现分歧
                    std::cout << "InferParmInterfaceState 警告: vectorParm[indexParm].StTransFunc 设为 TmpStTransDiverg" << std::endl;
                }
            }
            // 至此处理完出口同步状态

            // 至此处理完一个参数的 入口 / 出口 同步状态
        } // 每个循环处理一个函数参数

    }

    return 0;
}

// wfr 20190309 该函数用来初始化 阻塞组, 即找到需要与当前节点(在当前变量的情况下)一起阻塞的节点
STATE_CONSTR OAOASTVisitor::CheckBlockGroup(std::vector<BLOCK_INFO>& vectorBlock, NODE_INDEX indexNode, int indexVar, FUNC_INFO& Func){
    vectorBlock.emplace_back();
    BLOCK_INFO& BlockGroup = vectorBlock.back();
    std::vector<SEQ_REGION>& vectorSEQ = Func.vectorSEQ;
    std::vector<OMP_REGION>& vectorOMP = Func.vectorOMP;
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    VARIABLE_INFO& VarInfo = vectorVar[indexVar];
    
    STATE_CONSTR CoExitStContr(ST_REQ_UNINIT); // 表示协同出口状态满足的同步状态约束
    DEFINE_ST_REQ_ST_TRANS // wfr 20190719 通过这个宏 定义所有的 StReq 和 StTrans 类型

    SourceLocation TmpLoc;

    OMP_REGION* pOMP; pOMP = NULL;
    SEQ_REGION* pSEQ; pSEQ = NULL;
    SEQ_PAR_NODE_BASE* pChildBase;
    pChildBase = getNodeBasePtr(Func, indexNode);
    pChildBase->initSYNCInfo(indexVar);
    // 先将当前要处理的子节点入队
    BlockGroup.Children.push_back(indexNode);
    std::vector<NODE_INDEX>& vectorParents = pChildBase->vectorParents;

    
    if(indexNode.type==NODE_TYPE::SEQUENTIAL){
        pSEQ=(SEQ_REGION*)pChildBase;
    }else if(indexNode.type==NODE_TYPE::PARALLEL){
        pOMP=(OMP_REGION*)pChildBase;
    }else{
        std::cout << "CheckBlockGroup 错误: vectorProcess 队列中的节点没初始化" << std::endl;
        exit(1);
    }
    // wfr 20190509 在这里判断当前节点是否是 for/while/if 的 循环体/分支体入口节点
    bool isLoopOrIfBodyEntry = false;
    SEQ_PAR_NODE_BASE* pParentBase;
    for(unsigned int i=0; i<vectorParents.size(); ++i){
        pParentBase = getNodeBasePtr(Func, vectorParents[i]);
        if( pParentBase->TerminatorStmt!= NULL
            && ( isa<ForStmt>(pParentBase->TerminatorStmt) || isa<WhileStmt>(pParentBase->TerminatorStmt) )
        ){
            if(indexNode==pParentBase->LoopBody){
                isLoopOrIfBodyEntry = true;
                break;
            }
        }else if( pParentBase->TerminatorStmt!= NULL && isa<IfStmt>(pParentBase->TerminatorStmt) ){
            if( indexNode.type==NODE_TYPE::SEQUENTIAL ){
                if( pParentBase->ThenRange.BeginOffset<=pSEQ->SEQRange.BeginOffset
                    && pSEQ->SEQRange.EndOffset<=pParentBase->ThenRange.EndOffset
                ){
                    isLoopOrIfBodyEntry = true;
                    break;
                }
                if( pParentBase->ElseRange.BeginOffset<=pSEQ->SEQRange.BeginOffset
                    && pSEQ->SEQRange.EndOffset<=pParentBase->ElseRange.EndOffset
                ){
                    isLoopOrIfBodyEntry = true;
                    break;
                }
            }
            if( indexNode.type==NODE_TYPE::PARALLEL ){
                if( pParentBase->ThenRange.BeginOffset<=pOMP->DirectorRange.BeginOffset
                    && pOMP->OMPRange.EndOffset<=pParentBase->ThenRange.EndOffset
                ){
                    isLoopOrIfBodyEntry = true;
                    break;
                }
                if( pParentBase->ElseRange.BeginOffset<=pOMP->DirectorRange.BeginOffset
                    && pOMP->OMPRange.EndOffset<=pParentBase->ElseRange.EndOffset
                ){
                    isLoopOrIfBodyEntry = true;
                    break;
                }
            }
        }
    }

    // wfr 20200114 对于需要进入分析的循环, 强制设置其头节点的出口约束是 StTransDiverg, 以使循环中的第一个节点之前可以插入数据传输语句
    // 因为第一轮和第二轮循环一致性状态可能不同, 这样就保证了循环进行到第二次时, 第一个节点的数据一致性
    if(isLoopOrIfBodyEntry && pParentBase->EntryStConstr!=StReqUninit){
        CoExitStContr = StTransDiverg;
        vectorBlock.pop_back(); // 不需要阻塞, 出队上边入队的信息
        return CoExitStContr;
    }
    // wfr 20190509 在这里判断当前节点如果是 for/while/if 的 循环体/分支体入口节点
    // 就 确定 for/while/if 节点 指向 循环体/分支体 的出口的 变量出口状态
    // 分支是在 for/while/if 节点 实现的, 即每执行完一次循环体都会回到 for/while/if 节点,
    // 重新判断, 如果不符合条件, 则跳至循环之后的节点
    // if(isLoopOrIfBodyEntry && pParentBase->EntryStConstr!=StReqUninit){
    //     std::vector<VAR_REF_LIST>::iterator iterVarRef;
    //     iterVarRef = find(pParentBase->vectorVarRef.begin(), pParentBase->vectorVarRef.end(), indexVar);
    //     if(iterVarRef!=pParentBase->vectorVarRef.end()) {
    //         STATE_CONSTR UnusedStTrans;
    //         InferExitTrans(UnusedStTrans, CoExitStContr, pParentBase->EntryStConstr, iterVarRef->StTransFunc);
    //     }else{
    //         CoExitStContr = pParentBase->EntryStConstr; // 如果 for/while/if 节点 没使用变量, 就使用 入口状态给出口状态赋值
    //     }
    //     vectorBlock.pop_back(); // 不需要阻塞, 出队上边入队的信息
    //     return CoExitStContr;
    // }

    // 父节点入队条件: SEQ/OMP节点与变量作用域有交集  && 不重复入队 && (出口约束未确定 || 与 CoExitStContr 相同)
    for (size_t iParent = 0; iParent < vectorParents.size(); ++iParent)
    {
        NODE_INDEX indexParent = vectorParents[iParent];
        bool enqueue; enqueue = false; // 是否入队的标识
        SEQ_PAR_NODE_BASE* pBase = NULL;

        // 如果父节点是一个 host 节点
        if(indexParent.type == NODE_TYPE::SEQUENTIAL){
            SEQ_REGION* pSEQ = &vectorSEQ[indexParent.index];
            pBase = (SEQ_PAR_NODE_BASE*)pSEQ;
            pBase->initSYNCInfo(indexVar);

            if( pChildBase->TerminatorStmt!= NULL
                && (isa<ForStmt>(pChildBase->TerminatorStmt) || isa<WhileStmt>(pChildBase->TerminatorStmt))
                && pChildBase->EntryStConstr != StReqUninit 
                && (pChildBase->LoopBodyRange.BeginOffset<=pSEQ->SEQRange.BeginOffset && pSEQ->SEQRange.EndOffset<=pChildBase->LoopBodyRange.EndOffset)
            ){
                continue; // wfr 20190801 如果父节点是循环体的出口节点 且 循环已经被分析完了, 就不考虑该节点
            }

            // SEQ/OMP节点与变量作用域有交集
            if( (pSEQ->SEQRange.BeginOffset<=VarInfo.Scope.BeginOffset && VarInfo.Scope.BeginOffset<=pSEQ->SEQRange.EndOffset) ||
                (pSEQ->SEQRange.BeginOffset<=VarInfo.Scope.EndOffset && VarInfo.Scope.EndOffset<=pSEQ->SEQRange.EndOffset) ||
                (VarInfo.Scope.BeginOffset<=pSEQ->SEQRange.BeginOffset && pSEQ->SEQRange.EndOffset<=VarInfo.Scope.EndOffset) ||
                (pSEQ->SEQRange.BeginLoc==TmpLoc && pSEQ->SEQRange.EndLoc==TmpLoc) )
            {
                if(pBase->ExitStConstr == StReqUninit){
                    enqueue = true;
                }else if(pBase->ExitStConstr == StReqDiverg){
                    CoExitStContr = StReqDiverg;
                    break;
                }else if(CoExitStContr == StReqUninit){
                    CoExitStContr = pBase->ExitStConstr;
                    enqueue = true;
                }else if(CoExitStContr != pBase->ExitStConstr){
                    CoExitStContr = StReqDiverg;
                    break;
                }else{
                    enqueue = true;
                }
            }

        // 如果父节点是一个 device 节点
        }else if(indexParent.type == NODE_TYPE::PARALLEL){
            OMP_REGION* pOMP = &vectorOMP[indexParent.index];
            pBase = (SEQ_PAR_NODE_BASE*)pOMP;
            pBase->initSYNCInfo(indexVar);

            if( pChildBase->TerminatorStmt!= NULL
                && (isa<ForStmt>(pChildBase->TerminatorStmt) || isa<WhileStmt>(pChildBase->TerminatorStmt))
                && pChildBase->EntryStConstr != StReqUninit 
                && (pChildBase->LoopBodyRange.BeginOffset<=pOMP->DirectorRange.BeginOffset && pOMP->OMPRange.EndOffset<=pChildBase->LoopBodyRange.EndOffset)
            ){
                continue; // wfr 20190801 如果父节点是循环体的出口节点 且 循环已经被分析完了, 就不考虑该节点
            }

            // SEQ/OMP节点与变量作用域有交集
            if( (pOMP->DirectorRange.BeginOffset<=VarInfo.Scope.BeginOffset && VarInfo.Scope.BeginOffset<=pOMP->OMPRange.EndOffset) ||
                (pOMP->DirectorRange.BeginOffset<=VarInfo.Scope.EndOffset && VarInfo.Scope.EndOffset<=pOMP->OMPRange.EndOffset) ||
                (VarInfo.Scope.BeginOffset<=pOMP->DirectorRange.BeginOffset && pOMP->OMPRange.EndOffset<=VarInfo.Scope.EndOffset) )
            {
                if(pBase->ExitStConstr == StReqUninit){
                    enqueue = true;
                }else if(pBase->ExitStConstr == StReqDiverg){
                    CoExitStContr = StReqDiverg;
                    break;
                }else if(CoExitStContr == StReqUninit){
                    CoExitStContr = pBase->ExitStConstr;
                    enqueue = true;
                }else if(CoExitStContr != pBase->ExitStConstr){
                    CoExitStContr = StReqDiverg;
                    break;
                }else{
                    enqueue = true;
                }
            }
        }else{
            std::cout << "CheckBlockGroup 错误: 某个父节点没初始化" << std::endl;
            exit(1);
        }

        if(enqueue == false){ continue; } // 不入队, 跳过当前父节点

        // 判断当前父节点是否已经在 vectorParents 中
        std::vector<NODE_INDEX>::iterator iterParent;
        iterParent = find(BlockGroup.Parents.begin(), BlockGroup.Parents.end(), indexParent);
        if(iterParent != BlockGroup.Parents.end()) { continue; } // 如果重复则跳过当前父节点

        if(enqueue == true){ // 满足父节点入队条件, 入队当前父节点
            pBase->initSYNCInfo(indexVar); // 初始化, 说明当前正在处理 indexVar 指向的变量
            BlockGroup.Parents.push_back(indexParent);
        }
    }

    if(vectorParents.size()==0 && CoExitStContr==StReqUninit){
        if(pChildBase->EntryStConstr != StReqUninit){
            CoExitStContr = pChildBase->EntryStConstr;
        }else{
            std::cout << "CheckBlockGroup 错误: 节点没有父节点, 且节点 EntryStConstr == StReqUninit" << std::endl;
            exit(1);
        }
    }

    // wfr 20190719 循环要整体处理, 类似于函数, 这里将还没整体处理的循环设为阻塞
    // 如果是循环入口 且 当前处理的变量在循环入口处的 入口状态没确定 就阻塞, 在阻塞中确定 变量 在循环入口的 入口状态
    if( pChildBase->TerminatorStmt!= NULL
        && ( isa<ForStmt>(pChildBase->TerminatorStmt) || isa<WhileStmt>(pChildBase->TerminatorStmt) )
        && pChildBase->EntryStConstr == StReqUninit
    ){
        CoExitStContr = StReqUninit;
        std::cout << "CheckBlockGroup: 循环入口节点变量入口状态未确定, 需要阻塞; type = " << std::dec << indexNode.type << ", index = " << std::dec << indexNode.index << std::endl;
    }


    // 下面判断是否需要阻塞, 需要则处理阻塞相关的操作
    if( CoExitStContr != StReqUninit ){ // 如果是 HOST_ONLY/HOST_NEW/DEVICE_NEW/SYNC 则不需要阻塞, 直接返回 协同出口状态

        // 设置所有父节点的协商标识, 表示父节点都经过协商了, 不用再检验是否阻塞
        for(unsigned long i=0; i<BlockGroup.Parents.size(); ++i){
            SEQ_PAR_NODE_BASE* pBase;
            pBase = getNodeBasePtr(Func, BlockGroup.Parents[i]);
            pBase->isNegotiated = true;
        }

        vectorBlock.pop_back(); // 不需要阻塞, 出队上边入队的信息
        return CoExitStContr;
    }

    // 下边进行阻塞相关的操作

    std::cout << "CheckBlockGroup: 遇到阻塞组" << std::endl;
    std::cout << "阻塞组父节点 ID: ";
    // 打印下阻塞组的 父节点 / 子节点 的 ID
    for(unsigned long i=0; i<BlockGroup.Parents.size(); ++i){
        std::cout << std::dec << BlockGroup.Parents[i].index << ", ";
    }
    std::cout << std::endl;

    std::cout << "阻塞组子节点 ID: ";
    // 打印下阻塞组的 父节点 / 子节点 的 ID
    for(unsigned long i=0; i<BlockGroup.Children.size(); ++i){
        std::cout << std::dec << BlockGroup.Children[i].index << ", ";
    }
    std::cout << std::endl;

    // 为子节点设置阻塞标识
    for(unsigned long i=0; i<BlockGroup.Children.size(); ++i){
        SEQ_PAR_NODE_BASE* pBase;
        pBase = getNodeBasePtr(Func, BlockGroup.Children[i]);
        pBase->isBlocked = true;
    }
    
    return  CoExitStContr;
}

// 在 AnalyzeAllFunctions 以及 VisitFunctionDecl 函数中 调用该函数, 分析一个具体的函数
// wfr 20190321 注意 HOST_ONLY, 由于free的存在, 节点的入口也可能是 HOST_ONLY, 出口也可能是 HOST_ONLY
bool OAOASTVisitor::AnalyzeOneFunction(int indexFunc){
    FUNC_INFO& FuncCurrent = vectorFunc[indexFunc]; // 获得当前函数节点
    std::vector<SEQ_REGION>& vectorSEQ = FuncCurrent.vectorSEQ;
    std::vector<OMP_REGION>& vectorOMP = FuncCurrent.vectorOMP;
    std::vector<VARIABLE_INFO>& vectorVar = FuncCurrent.vectorVar;
    std::vector<FUNC_PARM_VAR_INFO>& vectorParm = FuncCurrent.vectorParm;

    // 所有分析都完成(为真), 整个函数分析才完成(为真)
    // 之后要和每个分析结果做 & 运算, 所以先赋值为 1
    FuncCurrent.isComplete = true;

    // wfr 20190723
    if(FuncCurrent.UsedInOMP==true && vectorOMP.size()>0){
        std::cout << "AnalyzeOneFunction 错误: 当前函数中含有并行域, 且在其他并行域中被调用; 并行域不能嵌套"<< std::endl;
    }

    std::cout << "AnalyzeOneFunction: 处理函数 " << FuncCurrent.Name << std::endl;

    // 遍历 SEQ 节点, 推导变量在节点内的同步状态信息
    for(unsigned long i=0; i<vectorSEQ.size(); i++){
        // 所有分析都完成(为真), 整个函数分析才完成(为真)
        FuncCurrent.isComplete &= AnalyzeSEQNode(i, FuncCurrent);
        if(FuncCurrent.isComplete == false){
            return false;
        }
    }

    // 遍历 OMP 节点, 推导变量在节点内的同步状态信息
    for(unsigned long i=0; i<vectorOMP.size(); i++){
        // 所有分析都完成(为真), 整个函数分析才完成(为真)
        FuncCurrent.isComplete &= AnalyzeOMPNode(i, FuncCurrent);
        if(FuncCurrent.isComplete == false){
            return false;
        }
    }

    // wfr 20190809 在 main函数 开头插入保存全局数组信息的 运行时API代码
    if(indexFunc==0){
        SourceLocation MainInsertLoc = vectorFunc[0].CompoundRange.BeginLoc.getLocWithOffset(1);
        for(unsigned long i=0; i<vectorGblVar.size(); ++i){
            std::string RunTimeCode = "";
            if(vectorGblVar[i].TypeName.back()==']'){
                std::string ElementType = vectorGblVar[i].TypeName.substr(0, vectorGblVar[i].TypeName.find('[')); // 取出 '[' 之前的字符串
                RunTimeCode += "\n";
                RunTimeCode += OAO_ARRAY_NAME;
                RunTimeCode += "( (void*)(";
                RunTimeCode += vectorGblVar[i].FullName;
                RunTimeCode += "), sizeof(";
                RunTimeCode += vectorGblVar[i].TypeName;
                RunTimeCode += "), sizeof(";
                RunTimeCode += ElementType;
                RunTimeCode += ") );";

                std::vector<CODE_INSERT>::iterator iterCodeInsert;
                iterCodeInsert = find(vectorGblVar[i].vectorInsert.begin(), vectorGblVar[i].vectorInsert.end(), MainInsertLoc);
                if(iterCodeInsert==vectorGblVar[i].vectorInsert.end()){ // 如果没找到该插入位置（在源代码中的位置）,  就新建该插入位置
                    vectorGblVar[i].vectorInsert.emplace_back(MainInsertLoc, "", Rewrite);
                    iterCodeInsert = vectorGblVar[i].vectorInsert.end()-1;
                }
                iterCodeInsert->Code += RunTimeCode;
            }
        }
    }

    // 先深遍历串并行图, 推断 函数参数 的 入口/出口 同步状态约束
    InferParmInterfaceState(FuncCurrent);

    // wfr 20190723 在并行域中被调用的函数不能插入运行时, 要整体保守分析
    if(FuncCurrent.UsedInOMP==true){
        return true;
    }

    // wfr 20190718 通过这个宏 定义所有的 StReq 和 StTrans 类型
    DEFINE_ST_REQ_ST_TRANS

    // 对每一个变量, 遍历串并行域图, 在合适的地方加入 OMP指令 以及 变量传输/同步指令
    int ParmCount = 0;
    for(unsigned long indexVar=0; indexVar<vectorVar.size(); ++indexVar){
        // 变量没在 OMP 中被引用, 这种变量不用管, 直接进行下一次循环, 处理下一个变量
        // 变量是 类, 也不讨论, 因为对类的处理是从类的各个域入手, 不是整个类
        // 变量 不是 指针 / 数组, 直接进行下一次循环, 处理下一个变量
        if( vectorVar[indexVar].UsedInOMP==false || vectorVar[indexVar].isClass==true ||
            (vectorVar[indexVar].TypeName.back()!='*' && vectorVar[indexVar].TypeName.back()!=']') )
        {
            continue;
        }

        std::cout << "AnalyzeOneFunction: 处理变量 " << vectorVar[indexVar].FullName << std::endl;

        bool isParm; isParm = false;
        STATE_CONSTR StReq;
        // STATE_CONSTR StTrans;
        std::vector<FUNC_PARM_VAR_INFO>::iterator iterParm;
        // 只处理一种情况: 变量在OMP中被使用
        if((unsigned long)ParmCount<vectorParm.size()){ // 说明函数的参数没处理完
            // 遍历 vectorParm, 匹配 indexVar, 先看看是不是函数参数
            iterParm = find(vectorParm.begin(), vectorParm.end(), indexVar);
            if(iterParm!=vectorParm.end()){ // 找到了一个函数参数
                ParmCount++; // 已处理参数个数计数 +1
                isParm = true; // 设置标志变量, 说明在处理函数参数
                StReq = iterParm->StReq; // 函数参数满足的 入口同步约束
                // StTrans = iterParm->StTransFunc; // 函数参数满足的 出口同步约束

            }else{ // 不是函数参数
                StReq = StReqHostOnly; // 函数参数满足的 入口同步约束
                // StTrans = StTransUninit; // 函数参数满足的 出口同步约束
            }
        }else{ // 不是函数参数, 因为所有函数参数已经处理完了
            if( (vectorVar[indexVar].Type==VAR_TYPE::PTR || vectorVar[indexVar].Type==VAR_TYPE::PTR_CONST || vectorVar[indexVar].Type==VAR_TYPE::CLASS_PTR || vectorVar[indexVar].Type==VAR_TYPE::CLASS_PTR_CONST) 
                && vectorVar[indexVar].indexRoot>=0){
                continue; // 不处理指向其他变量的指针
            }
            if(vectorVar[indexVar].isGlobal==true){
                StReq = StReqNone; // 函数参数满足的 入口同步约束
            }else{
                StReq = StReqHostOnly; // 函数参数满足的 入口同步约束
            }
            // StTrans = StTransUninit; // 函数参数满足的 出口同步约束
        }
        
        // 一个队列（正在处理队列）存储先广搜索的当前需要处理的节点的 index
        std::vector<NODE_INDEX> vectorProcess; vectorProcess.clear();
        NODE_INDEX indexDefNode; // 先找到当前变量的定义在哪个节点, 从该节点开始分析
        // wfr 20190421 这里先检查是否为空, 类的域的 ptrDecl 为空, 而 isa 参数不能是空指针
        if( (vectorVar[indexVar].ptrDecl!=NULL && isa<ParmVarDecl>(vectorVar[indexVar].ptrDecl))
            || vectorVar[indexVar].isGlobal==true
        ){
            indexDefNode = FuncCurrent.MapEntry;
        }else{
            int indexClass = vectorVar[indexVar].indexClass;
            if(indexClass>=0){ // 如果是 Member 就去找 类 的定义地址
                if(vectorVar[indexClass].ptrDecl!=NULL && isa<ParmVarDecl>(vectorVar[indexClass].ptrDecl)){
                    indexDefNode = FuncCurrent.MapEntry;
                }else{
                    indexDefNode = FindDefNode(FuncCurrent, vectorVar[indexClass]);
                }
            }else{
                indexDefNode = FindDefNode(FuncCurrent, vectorVar[indexVar]);
            }
        }

        std::cout << "AnalyzeOneFunction: indexDefNode.index = " << std::dec << indexDefNode.index << std::endl;
        if(indexDefNode.index>=0){
            SEQ_PAR_NODE_BASE* pDefNodeBase = getNodeBasePtr(FuncCurrent, indexDefNode);
            pDefNodeBase->initSYNCInfo(indexVar); // 初始化, 说明当前正在处理 indexVar 指向的变量
            // pDefNodeBase->EntryStTrans = StTransUninit;
            pDefNodeBase->EntryStConstr = StReq;
            vectorProcess.emplace_back(indexDefNode); // 变量定义所在的节点入队
        }else{
            std::cout << "AnalyzeOneFunction 错误: 没找到变量 " << vectorVar[indexVar].Name << " 的定义所在的节点" << std::endl;
            exit(1);
        }

        // 一个队列（阻塞信息队列）存储阻塞组信息: 一个阻塞涉及的父节点们和子节点们, 父节点们需要进行出口状态协商。
        std::vector<BLOCK_INFO> vectorBlock; vectorBlock.clear();

        while(true){ // 在一次 while 循环中: 处理一个没阻塞的节点 / 处理一个阻塞
            int indexProcess = -1; // 未阻塞节点在 vectorProcess 中的 index
            // std::vector<NODE_INDEX>::iterator iterProcess;
            //int indexBlock = -1; // 需要处理的阻塞在 vectorBlock 中的 index
            NODE_INDEX indexNode; // 节点 在 vectorOMP/vectorSEQ 中的 index
            OMP_REGION* pOMP; pOMP = NULL;
            SEQ_REGION* pSEQ; pSEQ = NULL;
            SEQ_PAR_NODE_BASE* pNodeBase; pNodeBase = NULL;

            // 从头扫描 vectorProcess, 获得第一个没阻塞的节点
            indexProcess = -1;
            // iterProcess = vectorProcess.begin();
            for(unsigned long i=0; i<vectorProcess.size(); ++i){
                indexNode = vectorProcess[i]; // SEQ/OMP 节点的 index
                // 如果没阻塞就给 indexProcess 赋值
                pNodeBase = getNodeBasePtr(FuncCurrent, indexNode);
                if(pNodeBase->isBlocked==false){ // 如果没被阻塞
                    indexProcess=i;
                    if(indexNode.type==NODE_TYPE::SEQUENTIAL){
                        pSEQ=(SEQ_REGION*)pNodeBase;
                    }else if(indexNode.type==NODE_TYPE::PARALLEL){
                        pOMP=(OMP_REGION*)pNodeBase;
                    }else{
                        std::cout << "AnalyzeOneFunction 错误: vectorProcess 队列中的节点没初始化" << std::endl;
                        exit(1);
                    }
                    break;
                }
            }

            std::cout << "AnalyzeOneFunction: vectorProcess = " << std::dec << indexProcess << std::endl;
            

            if(indexProcess>=0){ // 如果确实获得了一个没阻塞的节点

                std::cout << "AnalyzeOneFunction: 获得一个未阻塞节点 index = " << vectorProcess[indexProcess].index << ", type = " << vectorProcess[indexProcess].type << std::endl;

                // iterProcess = iterProcess + indexProcess;
                // 初始化 入口/出口 操作/状态
                pNodeBase->initSYNCInfo(indexVar); // 初始化, 说明当前循环正在处理 indexVar 指向的变量
                STATE_CONSTR CoExitStConstr(ST_REQ_UNINIT); // 表示协同出口状态满足的同步状态约束
                //std::vector<NODE_INDEX>* pvectorChildren;
                //std::vector<NODE_INDEX>::iterator iterChildren;
                std::vector<NODE_INDEX> pvectorParents;
                //std::vector<NODE_INDEX>::iterator iterParents;

                if(pNodeBase->isProcessed==true || indexNode==FuncCurrent.MapExit){
                    std::cout << "AnalyzeOneFunction: 遇到一个已经处理的节点 或 MapExit, index = " << std::dec << vectorProcess[indexProcess].index << ", type = " << vectorProcess[indexProcess].type << std::endl;

                    vectorProcess.erase(vectorProcess.begin()+indexProcess); // 出队当前节点
                    EnqueueChildren(vectorProcess, indexNode, indexVar, FuncCurrent); // 入队子节点
                    continue;
                }

                if(indexNode!=indexDefNode){
                    // 判断是否需要阻塞, 需要则存入阻塞信息, 阻塞的话 CoExitState==0; 若返回 SYNC_STATE::MASK(0xFFFFFFFF), 则说明是变量定义所在的节点
                    CoExitStConstr = CheckBlockGroup(vectorBlock, indexNode, indexVar, FuncCurrent);
                }else{
                    CoExitStConstr = StReqDiverg;
                }
                std::cout << "AnalyzeOneFunction: CoExitState: ZERO = 0x" << std::hex << CoExitStConstr.ZERO 
                    << ", ONE = 0x" << std::hex << CoExitStConstr.ONE << std::endl;
                if(CoExitStConstr==StReqUninit){
                    continue; // 如果需要阻塞, 直接进入下一次循环, 处理下一个没阻塞的节点
                }

                std::cout << "AnalyzeOneFunction: indexNode.type = " << std::dec << indexNode.type << ", indexNode.index = " << std::dec << indexNode.index << std::endl;

                // 先判断当前节点中是否用到了当前处理的变量
                bool VarUsed = false;
                std::vector<VAR_REF_LIST>::iterator iterVarRef;
                iterVarRef = find(pNodeBase->vectorVarRef.begin(), pNodeBase->vectorVarRef.end(), indexVar);
                if(iterVarRef!=pNodeBase->vectorVarRef.end()) { 
                    VarUsed=true;
                    pNodeBase->NeedInsertStTrans = iterVarRef->NeedInsertStTrans;
                    std::cout << "AnalyzeOneFunction: 入口同步状态: ZERO = 0x" << std::hex << iterVarRef->StReq.ZERO 
                        << ", ONE = 0x" << std::hex << iterVarRef->StReq.ONE << std::endl;
                }else{
                    std::cout << "AnalyzeOneFunction: 入口同步状态: ZERO = 0x" << std::hex << pNodeBase->EntryStConstr.ZERO 
                        << ", ONE = 0x" << std::hex << pNodeBase->EntryStConstr.ONE << std::endl;
                }

                // 1. 确定 实际的 入口状态约束 以及 入口处是否插入 运行时函数 OAODataTrans()
                if(indexNode==indexDefNode){ // SYNC_STATE::MASK(0xFFFFFFFF), 则说明是变量定义所在的节点
                    // pNodeBase->EntryStTrans = StTransNone;
                    if(indexNode.type==NODE_TYPE::SEQUENTIAL){
                        if(isParm==true){
                            pNodeBase->EntryStConstr = StReq;
                            pNodeBase->EntryStTrans = StTransNone;
                        }else if(vectorVar[indexVar].isGlobal==true){
                            if(VarUsed==true){
                                pNodeBase->EntryStConstr = iterVarRef->StReq;
                                pNodeBase->EntryStTrans = iterVarRef->StReq;
                            }else{
                                pNodeBase->EntryStConstr = StReq;
                                pNodeBase->EntryStTrans = StTransNone;
                            }
                        }else{
                            pNodeBase->EntryStConstr = StReqHostOnly;
                            pNodeBase->EntryStTrans = StTransNone;
                        }
                    }else{
                        std::cout << "AnalyzeOneFunction 错误: 遇到并行域中的变量定义" << std::endl;
                        exit(1);
                    }
                }else if( pNodeBase->TerminatorStmt!=NULL
                          && ( isa<ForStmt>(pNodeBase->TerminatorStmt) || isa<WhileStmt>(pNodeBase->TerminatorStmt)) ){
                    // wfr 20191209 修改判断条件, 判断节点是循环头节点
                    STATE_CONSTR UnusedStConstr; // wfr 20190719 这里将同步操作后的同步状态不再赋值给 pNodeBase->EntryState, 而是赋值给一个无用的变量 UnusedStConstr, 从而没有改入口状态
                    InferEnteryTrans(pNodeBase->EntryStTrans, UnusedStConstr, CoExitStConstr, pNodeBase->EntryStConstr);
                }else if(VarUsed==false){ // 如果当前节点没有用到当前变量 且 不是定义节点
                    // wfr 20191207 把没引用当前变量的情况提前到循环头节点情况(下一种情况)之前
                    pNodeBase->EntryStTrans = StTransNone;
                    pNodeBase->EntryStConstr = CoExitStConstr;
                }else{
                    std::cout << "AnalyzeOneFunction: a" << std::endl;
                    InferEnteryTrans(pNodeBase->EntryStTrans, pNodeBase->EntryStConstr, CoExitStConstr, iterVarRef->StReq);
                }

                // 至此处理完了 节点 入口同步操作 和 实际入口同步状态


                std::cout << "AnalyzeOneFunction: 入口同步操作 ZERO = 0x" << std::hex << pNodeBase->EntryStTrans.ZERO 
                        << ", ONE = 0x" << std::hex << pNodeBase->EntryStTrans.ONE << std::endl;
                std::cout << "AnalyzeOneFunction: 实际入口同步状态约束 ZERO = 0x" << std::hex << pNodeBase->EntryStConstr.ZERO 
                        << ", ONE = 0x" << std::hex << pNodeBase->EntryStConstr.ONE << std::endl;


                // 2. 下边 处理节点 出口同步操作 和 实际出口同步状态约束
                if(pNodeBase->ExitStTrans==StTransUninit){
                    if(VarUsed==false){ // 如果当前节点没有用到当前变量
                        pNodeBase->ExitStTrans = StTransNone;
                        pNodeBase->ExitStConstr = pNodeBase->EntryStConstr;
                        pNodeBase->NeedInsertStTrans = false;
                    }else{
                        std::cout << "AnalyzeOneFunction: a" << std::endl;
                        InferExitTrans(pNodeBase->ExitStTrans, pNodeBase->ExitStConstr, pNodeBase->EntryStConstr, iterVarRef->StTransFunc);
                        pNodeBase->NeedInsertStTrans = iterVarRef->NeedInsertStTrans;
                    }
                }else{ // wfr 20190727 如果 状态转移函数已经确定了, 应该是一个循环头节点
                    pNodeBase->NeedInsertStTrans = true; // wfr 20191119 这里设置为 true, 不能再使用 iterVarRef->NeedInsertStTrans 判断, 因为循环头节点可能没有引用当前处理的变量
                    pNodeBase->ExitStConstr = ExeStTrans(pNodeBase->EntryStConstr, pNodeBase->ExitStTrans);
                }

                // 至此处理完 节点 出口同步操作 和 实际出口同步状态

                std::cout << "AnalyzeOneFunction: 出口同步操作 ZERO = 0x" << std::hex << pNodeBase->ExitStTrans.ZERO 
                        << ", ONE = 0x" << std::hex << pNodeBase->ExitStTrans.ONE << std::endl;
                std::cout << "AnalyzeOneFunction: 实际出口同步状态约束 ZERO = 0x" << std::hex << pNodeBase->ExitStConstr.ZERO 
                        << ", ONE = 0x" << std::hex << pNodeBase->ExitStConstr.ONE << std::endl;

                pNodeBase->isProcessed=true;

                std::cout << "AnalyzeOneFunction: 处理完一个未阻塞节点 index = " << std::dec << vectorProcess[indexProcess].index << ", type = " << vectorProcess[indexProcess].type << std::endl;

                vectorProcess.erase(vectorProcess.begin()+indexProcess); // 出队当前节点
                EnqueueChildren(vectorProcess, indexNode, indexVar, FuncCurrent); // 入队子节点
                // 至此处理完一个不阻塞的节点

            // 如果 vectorProcess 队列中所有的节点都没阻塞了 / 队列中没有节点了
            // wfr 20190720 处理阻塞:
            // 1. 检查所有父节点的 ExitStConstr, 对于还是 StReqUninit 的, 根据该节点内部的信息 推导 ExitStConstr
            // 2. 处理子节点中的循环入口节点
            }else{ // 处理第一个阻塞
                
                if(vectorBlock.empty()){ // 执行到这里说明 vectorProcess 中已经没有可以处理的节点, 所以如果没找到阻塞, 则认为遍历完成
                    break;
                }
                std::cout << "AnalyzeOneFunction: 处理阻塞" << std::endl;

                // 否则处理第一个阻塞
                BLOCK_INFO& Block = vectorBlock[0];
                
                // wfr 20191209 检查阻塞组的 父节点/子节点 vector 是否为空, 为空则报错退出
                if(Block.Parents.empty()==true){
                    std::cout << "AnalyzeOneFunction 警告: 阻塞组 父节点列表为空" << std::endl;
                    //exit(1);
                }
                if(Block.Children.empty()==true){
                    std::cout << "AnalyzeOneFunction 警告: 阻塞组 子节点列表为空" << std::endl;
                    //exit(1);
                }

                // 1. 检查所有父节点的 ExitStConstr, 对于还是 StReqUninit 的, 根据该节点内部的信息 推导 ExitStConstr
                for(size_t i = 0; i < Block.Parents.size(); ++i){
                    
                    SEQ_PAR_NODE_BASE* pNodeBase;
                    pNodeBase = getNodeBasePtr(FuncCurrent, Block.Parents[i]);
                    if(pNodeBase->isNegotiated){
                        std::cout << "AnalyzeOneFunction 错误: 节点出口状态已经过协商, 不能再次协商" << std::endl;
                        exit(1);
                    }
                    
                    pNodeBase->isNegotiated = true; // 设置协商标识

                    if(pNodeBase->ExitStConstr!=StReqUninit){
                        continue;
                    }

                    // 判断当前节点中是否用到了当前处理的变量
                    bool VarUsed = false;
                    std::vector<VAR_REF_LIST>::iterator iterVarRef;
                    iterVarRef = find(pNodeBase->vectorVarRef.begin(), pNodeBase->vectorVarRef.end(), indexVar);
                    if(iterVarRef!=pNodeBase->vectorVarRef.end()) { 
                        VarUsed=true;
                    }
                    if(VarUsed==false){
                        pNodeBase->ExitStConstr = StReqDiverg;
                    }else{
                        STATE_CONSTR UnusedStTrans;
                        InferExitTrans(UnusedStTrans, pNodeBase->ExitStConstr, iterVarRef->StReq, iterVarRef->StTransFunc);
                    }
                }

                // 2. 如果子节点中有循环入口节点 且 变量入口状态没确定, 则先确定 当前处理的变量 在循环入口处 的 入口状态
                for(unsigned int i=0; i<Block.Children.size(); ++i){
                    SEQ_PAR_NODE_BASE* pBase;
                    pBase = getNodeBasePtr(FuncCurrent, Block.Children[i]);
                    pNodeBase->isBlocked = false; // 解除阻塞状态标识

                    if( pBase->TerminatorStmt!=NULL 
                        && ( isa<ForStmt>(pBase->TerminatorStmt) || isa<WhileStmt>(pBase->TerminatorStmt) )
                    ){
                        if(pBase->EntryStConstr==StReqUninit){
                            // 确定 入口状态
                            bool Unused0=0; bool Unused1=0; bool Unused2=0; bool Unused3=0;
                            InferLoopInterfaceState(pBase->EntryStConstr, pBase->ExitStTrans, pBase->LoopNeedAnalysis, 
                                Block.Children[i], indexVar, FuncCurrent, Unused0, Unused1, Unused2, Unused3);
                        }
                    }
                }

                // 至此 处理完 第一个阻塞

                // 出队 第一个阻塞
                vectorBlock.erase(vectorBlock.begin());
            }
            
        } // while 循环 处理 一个不阻塞节点 / 一个阻塞

        // 20190508 这里集中向 vectorTrans 中写入 同步操作
        // 先写 SEQ 节点的 vectorTrans
        for(unsigned int i=0; i<vectorSEQ.size(); ++i){
            SEQ_REGION& SEQ = vectorSEQ[i];
            if(SEQ.indexVar!=indexVar || SEQ.isProcessed==false){
                continue;
            }

            // wfr 20191223 插入 数据传输 API
            if( SEQ.EntryStTrans!=StTransUninit && SEQ.EntryStTrans!=StTransDiverg && 
                SEQ.EntryStTrans!=StTransNone )
            {
                SourceLocation EmptyLoc;
                if(SEQ.StmtRange.BeginLoc != EmptyLoc){ // wfr 20191223 如果是函数调用
                    SEQ.vectorTrans.emplace_back(indexVar, TRANS_TYPE::DATA_TRANS, SEQ.EntryStTrans, SEQ.StmtRange.BeginLoc, Rewrite);
                }else{
                    SEQ.vectorTrans.emplace_back(indexVar, TRANS_TYPE::DATA_TRANS, SEQ.EntryStTrans, SEQ.SEQRange.BeginLoc, Rewrite);
                }
            }

            // wfr 20191223 插入 状态更新 API
            if( SEQ.NeedInsertStTrans==true && SEQ.ExitStTrans!=StTransUninit 
                && SEQ.ExitStTrans!=StTransDiverg && SEQ.ExitStTrans!=StTransNone){
                SourceLocation TmpLoc;
                SourceLocation EmptyLoc;
                if( SEQ.TerminatorStmt!= NULL
                    && ( isa<ForStmt>(SEQ.TerminatorStmt) || isa<WhileStmt>(SEQ.TerminatorStmt) )
                ){
                    TmpLoc = SEQ.LoopBodyRange.EndLoc.getLocWithOffset(1);
                }else if( SEQ.TerminatorStmt!= NULL && isa<IfStmt>(SEQ.TerminatorStmt) ){
                    if(SEQ.ThenRange.EndLoc!=TmpLoc && SEQ.ElseRange.EndLoc!=TmpLoc){
                        std::cout << "AnalyzeOneFunction 错误: if 节点的 then 和 else 都存在, 不应该有出口同步操作; type = " << NODE_TYPE::SEQUENTIAL << ", index = " << i << std::endl;
                        exit(0);
                    }else{
                        TmpLoc = SEQ.ThenRange.EndLoc.getLocWithOffset(1);
                    }
                }else if(SEQ.StmtRange.EndLoc != EmptyLoc){ // wfr 20191223 如果是函数调用
                    TmpLoc = SEQ.StmtRange.EndLoc.getLocWithOffset(1);
                }else{
                    TmpLoc = SEQ.SEQRange.EndLoc.getLocWithOffset(1);
                }
                SEQ.vectorTrans.emplace_back(indexVar, TRANS_TYPE::STATE_TRANS, SEQ.ExitStTrans, TmpLoc, Rewrite);
            }

            // 这里处理下 return 语句, 如果当前节点中有 return 语句, 就使用 delete 从 data environment 中删除变量
            // 只处理 数组/指针 且 在 device 中
            // 不处理函数参数
            // wfr 20190815 只有 不是全局变量 或者 是main函数 的情况, 才在 return 前释放 device 内存
            if(vectorVar[indexVar].isGlobal==false || indexFunc==0){
                SourceLocation TmpLoc;
                if( SEQ.ReturnRange.EndLoc!=TmpLoc && isParm == false && SEQ.ExitStConstr!=StReqHostOnly && 
                    (vectorVar[indexVar].TypeName.back()=='*' || vectorVar[indexVar].TypeName.back()==']') )
                {
                    SEQ.vectorTrans.emplace_back(indexVar, TRANS_TYPE::DATA_TRANS, StReqHostOnly, SEQ.ReturnRange.BeginLoc, Rewrite);
                }
            }
        }
        // 再写 OMP 节点的 vectorTrans
        for(unsigned int i=0; i<vectorOMP.size(); ++i){
            OMP_REGION& OMP = vectorOMP[i];
            if(OMP.indexVar!=indexVar || OMP.isProcessed==false){
                continue;
            }

            NODE_INDEX indexEntry, indexExit;
            indexEntry.type = NODE_TYPE::PARALLEL;
            indexExit.type = NODE_TYPE::PARALLEL;
            indexEntry.index = OMP.indexEntry;
            indexExit.index = vectorOMP[indexEntry.index].indexExit;
            if( OMP.EntryStTrans!=StTransUninit && OMP.EntryStTrans!=StTransDiverg && 
                OMP.EntryStTrans!=StTransNone )
            {
                // 将插入位置设置为 OMP 指令 上一行的最后一个字符
                SourceLocation TmpLoc = vectorOMP[indexEntry.index].DirectorRange.BeginLoc.getLocWithOffset(-1*(OMP.DirectorRange.BeginCol));
                OMP.vectorTrans.emplace_back(indexVar, TRANS_TYPE::DATA_TRANS, OMP.EntryStTrans, TmpLoc, Rewrite);
            }
            NODE_INDEX  TmpIndexNode(NODE_TYPE::PARALLEL, i);
            if( OMP.NeedInsertStTrans==true && OMP.ExitStTrans!=StTransUninit 
                && OMP.ExitStTrans!=StTransDiverg && OMP.ExitStTrans!=StTransNone )
            {
                if(indexExit == TmpIndexNode){ // 说明是 并行域的 出口节点
                    SourceLocation TmpLoc = OMP.OMPRange.EndLoc.getLocWithOffset(1);
                    if( OMP.TerminatorStmt!= NULL
                        && ( isa<ForStmt>(OMP.TerminatorStmt) || isa<WhileStmt>(OMP.TerminatorStmt) )
                    ){
                        TmpLoc = OMP.LoopBodyRange.EndLoc.getLocWithOffset(1);
                    }else if( OMP.TerminatorStmt!= NULL && isa<IfStmt>(OMP.TerminatorStmt) ){
                        if(OMP.ThenRange.EndLoc!=TmpLoc && OMP.ElseRange.EndLoc!=TmpLoc){
                            std::cout << "AnalyzeOneFunction 错误: if 节点的 then 和 else 都存在, 不应该有出口同步操作; type = " << NODE_TYPE::PARALLEL << ", index = " << i << std::endl;
                            exit(0);
                        }else{
                            TmpLoc = OMP.ThenRange.EndLoc.getLocWithOffset(1);
                        }
                    }else{
                        TmpLoc = OMP.OMPRange.EndLoc.getLocWithOffset(1);
                    }
                    OMP.vectorTrans.emplace_back(indexVar, TRANS_TYPE::STATE_TRANS, OMP.ExitStTrans, TmpLoc, Rewrite);
                }else{ // 不是 并行域的 出口节点
                    std::cout << "AnalyzeOneFunction 错误: 并行域中不应该进行变量传输" << std::endl;
                    exit(1);
                }
            }
        }

        // wfr 20190815 只有不是全局变量的时候才在作用域结束之前释放 device 上的内存
        int indexRoot;
        if(FuncCurrent.vectorVar[indexVar].indexRoot >= 0){
            indexRoot = FuncCurrent.vectorVar[indexVar].indexRoot;

            std::vector<FUNC_PARM_VAR_INFO>::iterator iterParm;
            iterParm = find(vectorParm.begin(), vectorParm.end(), indexRoot);
            if(iterParm!=vectorParm.end()){ // 找到了一个函数参数
                isParm = true; // 设置标志变量, 说明在处理函数参数
            }
        }else{
            indexRoot = indexVar;
        }

        if(vectorVar[indexRoot].isGlobal==false){
            NODE_INDEX indexScopeExit;
            ScopeBeginEndNode(indexScopeExit, indexFunc, vectorVar[indexRoot].Scope, SCOPE_BEGIN_END::END);  // wfr 20190409 判断一个 Scope 在哪个 SEQ/OMP 结束, 根据源代码文本顺序
            std::cout << "AnalyzeOneFunction: 变量作用域出口所在的节点 index = " << indexScopeExit.index << ", type = " << indexScopeExit.type << std::endl;

            if(indexScopeExit.type==NODE_TYPE::NODE_UNINIT){
                std::cout << "AnalyzeOneFunction 错误: 变量作用域出口所在的节点没初始化" << std::endl;
                std::cout << "AnalyzeOneFunction 错误: vectorVar[indexRoot].Scope.EndOffset = " << vectorVar[indexRoot].Scope.EndOffset << std::endl;
                std::cout << "AnalyzeOneFunction 错误: vectorSEQ.size() = " << vectorSEQ.size() << std::endl;
                NODE_INDEX indexMapExit = FuncCurrent.MapExit;
                NODE_INDEX indexReturn = vectorSEQ[indexMapExit.index].vectorParents[0]; // 找到最后一个节点的 index
                std::cout << "AnalyzeOneFunction 错误: return节点的结束偏移 = " << vectorSEQ[indexReturn.index].SEQRange.EndOffset << std::endl;
                exit(1);
            }else if(indexScopeExit.type==NODE_TYPE::PARALLEL){
                std::cout << "AnalyzeOneFunction 错误: 变量作用域出口所在的节点是 OMP" << std::endl;
                exit(1);
            }
            SEQ_PAR_NODE_BASE* pScopeExitBase = getNodeBasePtr(FuncCurrent, indexScopeExit);

            // 这里考察下 变量作用域出口所在的节点的出口状态, 在作用域结束之前, 插入 delete 操作
            // 只处理 数组/指针 且 在 device 中
            // 不处理函数参数
            // 且 当前节点 没有 return 语句
            SourceLocation TmpLoc;
            
            if( pScopeExitBase->ReturnRange.EndLoc==TmpLoc && isParm == false && pScopeExitBase->ExitStConstr!=StReqHostOnly &&
                (vectorVar[indexRoot].TypeName.back()=='*' || vectorVar[indexRoot].TypeName.back()==']') )
            {
                if(pScopeExitBase->ExitState!=SYNC_STATE::HOST_ONLY){ // 如果不是 HOSY_ONLY, 就在作用域结束之前, 插入 delete 操作
                    std::cout << "AnalyzeOneFunction: d" << std::endl;
                    // 可以在这里将 变量 同步/传输操作信息 存入 函数中
                    pScopeExitBase->vectorTrans.emplace_back(indexRoot, TRANS_TYPE::DATA_TRANS, StReqHostOnly, vectorVar[indexRoot].Scope.EndLoc, Rewrite);
                }else{
                    std::cout << "AnalyzeOneFunction: 变量 " << vectorVar[indexRoot].Name << " 在离开作用范围之前已经是 HOST_ONLY !!!" << std::endl;
                }
            }
        }

        // 至此 针对 当前变量 完成了对串并行图的遍历

    } // for 循环 处理每一个外部变量

    return true;
}

// 进入这个函数之前, 函数中所有 SEQ/OMP 节点中的 所有变量(需要考虑的) 声明/读写类型信息/同步状态信息 都应该已经获得了
// 当然可能存在调用未处理完成函数的情况, 这种情况需要之后的链接处理
// 该函数需要完成的工作: 
// 遍历并处理每一个 SEQ/OMP 节点
// 在每一个节点中, 遍历处理每个需要考虑的变量
// 对每一个变量, 遍历引用信息, 根据读写类型信息, 推导出该节点入口出口处的 同步状态信息
// 目前的思路是将所有 函数调用 都拆分出 单独的 函数调用节点, 这样所有的普通 SEQ/OMP 应该都可以处理, 即变量读写信息都是全的
// 对于函数调用节点, 如果被调函数已经处理完了则直接使用, 否则等待之后链接处理
// 一个 SEQ/OMP 节点处理完, 则设置 isComplete = true
// 如果所有 SEQ/OMP 节点都处理完全, 即所有同步状态信息都推导出了, 
// 之后遍历串并行图, 在原码中写入 变量传输, 同步语句 / Offloading语句
// 该函数处理完则设置标识 isComplete = true
bool OAOASTVisitor::VisitFunctionDecl(FunctionDecl *S){

    int indexFunc = -1;
    if(BeginFlag==true && FuncStack.empty()==false){
        indexFunc = FuncStack.back().indexFunc;
        if(indexFunc<0){
            std::cout << "VisitFunctionDecl 错误: 当前函数的 indexFunc<0" << std::endl;
            exit(1);
        }
        if(vectorFunc[indexFunc].ptrDefine==NULL){
            return true; // 还没找到函数定义, 暂不继续处理
        }
    }else{
        return true; // 如果不是在函数中就退出
    }

    std::cout << "VisitFunctionDecl: 处理 " << vectorFunc[indexFunc].Name << std::endl;

    // 分析当前函数
    // AnalyzeOneFunction(indexFunc);

    std::cout << "VisitFunctionDecl: 退出" << std::endl;

	return true;
}

// wfr 20190724 对 函数的 UsedInOMP 进行传播, 标记出所有运行在 device 上的函数, 这些函数要保守分析
int OAOASTVisitor::SpreadFuncUsedInOMP(FUNC_INFO& Func){
    if(Func.UsedInOMP==false){
        return 0;
    }
    std::vector<int>& vectorCallee = Func.vectorCallee;
    for(unsigned long i=0; i<vectorCallee.size(); ++i){
        if(vectorCallee[i]<0 || vectorCallee[i]>=vectorFunc.size()){
            std::cout << "SpreadUsedInOMP 错误: vectorCallee 的 index 非法" << std::endl;
            exit(1);
        }
        vectorFunc[vectorCallee[i]].UsedInOMP = true;
        SpreadFuncUsedInOMP(vectorFunc[vectorCallee[i]]);
    }
    return 0;
}

// wfr 20200114 对 变量的 UsedInOMP 进行传播, 父函数中变量在 device 上被使用, 则在被调用函数中 UsedInOMP=true, 表示该变量也需要分析
int OAOASTVisitor::SpreadVarUsedInOMP(FUNC_INFO& Func){
    std::vector<VARIABLE_INFO>& vectorVar = Func.vectorVar;
    std::vector<SEQ_REGION>& vectorSEQ = Func.vectorSEQ;
    for(unsigned long i=0; i<vectorSEQ.size(); ++i){
        if (vectorSEQ[i].indexFunc<0) continue;
        bool NeedSpread = false;
        std::vector<VAR_REF_LIST>& vectorVarRef = vectorSEQ[i].vectorVarRef;
        std::vector<FUNC_PARM_VAR_INFO>& vectorCalleeParm = vectorFunc[vectorSEQ[i].indexFunc].vectorParm;
        std::vector<VARIABLE_INFO>& vectorCalleeVar = vectorFunc[vectorSEQ[i].indexFunc].vectorVar;

        for(unsigned long j=0; j<vectorVarRef.size(); ++j){
            if(j >= vectorCalleeParm.size()){
                break; // wfr 20200307 所有实参都处理完了, 跳出 for
            }
            if( vectorCalleeParm[j].indexVar < 0 || 
                vectorCalleeParm[j].indexVar > vectorCalleeVar.size() ){
                std::cout << "SpreadVarUsedInOMP 警告: 没找到实参对应的形参" << std::endl;
                continue;
            }
            VARIABLE_INFO& CalleeVar = vectorCalleeVar[vectorCalleeParm[j].indexVar];
            std::string CalleeVarTypeName = CalleeVar.TypeName;
            bool flag = (CalleeVarTypeName.back()=='&' || CalleeVarTypeName.back()=='*' || CalleeVarTypeName.back()==']' || CalleeVar.isClass==true);
            bool CallerVarUsedInOMP = vectorVar[vectorVarRef[j].index].UsedInOMP;
            bool CalleeVarUsedInOMP = CalleeVar.UsedInOMP;
            if(flag==true && CallerVarUsedInOMP==true && CalleeVarUsedInOMP==false){
                NeedSpread = true;
                CalleeVar.UsedInOMP = true;
            }
        }
        if(NeedSpread==true){
            SpreadVarUsedInOMP(vectorFunc[vectorSEQ[i].indexFunc]);
        }
    }
    return 0;
}

// wfr 20190725 找到 循环体的第一个节点 和 循环之后的第一个节点
int OAOASTVisitor::MarkLoopBranch(FUNC_INFO& Func){
    std::vector<SEQ_REGION>& vectorSEQ = Func.vectorSEQ;
    std::vector<OMP_REGION>& vectorOMP = Func.vectorOMP;

    for(unsigned long i=0; i<vectorSEQ.size(); ++i){
        if( vectorSEQ[i].TerminatorStmt!= NULL
            && ( isa<ForStmt>(vectorSEQ[i].TerminatorStmt) || isa<WhileStmt>(vectorSEQ[i].TerminatorStmt) )
        ){
            if(vectorSEQ[i].vectorChildren.size()!=2){
                std::cout << "MarkLoopBranch 错误: 循环头节点的子节点数量 != 2" << std::endl;
                exit(1);
            }

            for(unsigned long j=0; j<2; ++j){
                NODE_INDEX indexChild = vectorSEQ[i].vectorChildren[j];
                SEQ_PAR_NODE_BASE* pChildBase = getNodeBasePtr(Func, indexChild);

                if (indexChild.type == NODE_TYPE::SEQUENTIAL) {
                    SEQ_REGION *pSEQChild = (SEQ_REGION *)pChildBase;
                    if (pSEQChild->SEQRange.EndOffset < vectorSEQ[i].LoopBodyRange.BeginOffset || 
                        vectorSEQ[i].LoopBodyRange.EndOffset < pSEQChild->SEQRange.BeginOffset)
                    {
                        vectorSEQ[i].LoopExit = indexChild;
                    }else{
                        vectorSEQ[i].LoopBody = indexChild;
                    }
                }else if (indexChild.type == NODE_TYPE::PARALLEL) {
                    OMP_REGION *pOMPChild = (OMP_REGION *)pChildBase;
                    if (pOMPChild->OMPRange.EndOffset < vectorSEQ[i].LoopBodyRange.BeginOffset || 
                        vectorSEQ[i].LoopBodyRange.EndOffset < pOMPChild->DirectorRange.BeginOffset)
                    {
                        vectorSEQ[i].LoopExit = indexChild;
                    }else{
                        vectorSEQ[i].LoopBody = indexChild;
                    }
                }
            }

            if(vectorSEQ[i].LoopBody.index<0 || vectorSEQ[i].LoopExit.index<0){
                std::cout << "MarkLoopBranch 错误: 循环分支分析出错" << std::endl;
                exit(1);
            }
        }
    }

    //wfr 20190802 由于 CFG 中暂时没有并行域中的分支的信息, 即暂时没有并行域中的分支信息
    for(unsigned long i=0; i<vectorOMP.size(); ++i){
        if( vectorOMP[i].TerminatorStmt!= NULL
            && ( isa<ForStmt>(vectorOMP[i].TerminatorStmt) || isa<WhileStmt>(vectorOMP[i].TerminatorStmt) )
        ){
            if(vectorOMP[i].vectorChildren.size()!=2){
                std::cout << "MarkLoopBranch 错误: 循环头节点的子节点数量 != 2" << std::endl;
                exit(1);
            }

            for(unsigned long j=0; j<2; ++j){
                NODE_INDEX indexChild = vectorOMP[i].vectorChildren[j];
                SEQ_PAR_NODE_BASE* pChildBase = getNodeBasePtr(Func, indexChild);

                if (indexChild.type == NODE_TYPE::SEQUENTIAL) {
                    SEQ_REGION *pSEQChild = (SEQ_REGION *)pChildBase;
                    if (pSEQChild->SEQRange.EndOffset < vectorOMP[i].LoopBodyRange.BeginOffset || 
                        vectorOMP[i].LoopBodyRange.EndOffset < pSEQChild->SEQRange.BeginOffset)
                    {
                        vectorOMP[i].LoopExit = indexChild;
                    }else{
                        vectorOMP[i].LoopBody = indexChild;
                    }
                }else if (indexChild.type == NODE_TYPE::PARALLEL) {
                    OMP_REGION *pOMPChild = (OMP_REGION *)pChildBase;
                    if (pOMPChild->OMPRange.EndOffset < vectorOMP[i].LoopBodyRange.BeginOffset || 
                        vectorOMP[i].LoopBodyRange.EndOffset < pOMPChild->DirectorRange.BeginOffset)
                    {
                        vectorOMP[i].LoopExit = indexChild;
                    }else{
                        vectorOMP[i].LoopBody = indexChild;
                    }
                }
            }

            if(vectorOMP[i].LoopBody.index<0 || vectorOMP[i].LoopExit.index<0){
                std::cout << "MarkLoopBranch 错误: 循环分支分析出错" << std::endl;
                exit(1);
            }
        }
    }

    return 0;
}

// 在 AST遍历 结束后调用该函数, 分析所有函数
int OAOASTVisitor::AnalyzeAllFunctions(){
    std::vector<int> vectorProcess; // 队列中的函数等待分析

    // 这里对 UsedInOMP 进行传播, 标记出所有运行在 device 上的函数, 这些函数要保守分析
    for(unsigned long i=0; i<vectorFunc.size(); ++i){
        SpreadFuncUsedInOMP(vectorFunc[i]);
        SpreadVarUsedInOMP(vectorFunc[i]);
        MarkLoopBranch(vectorFunc[i]);
    }

    // 以下开始分析
    for(unsigned long i=0; i<vectorFunc.size(); ++i){
        if(vectorFunc[i].isComplete==false){
            vectorProcess.push_back(i); // 将所有需要分析的函数入队
        }
    }

    unsigned long OldSize = vectorProcess.size()+1;
    while(!vectorProcess.empty()){ // 待分析队列非空就进入循环
        if(OldSize == vectorProcess.size()){ // 说明上次循环没有分析任何函数, 应该是出错了
            std::cout << "AnalyzeAllFunctions 错误: 找不到可以分析的函数, 可能是因为没有分析子函数" << std::endl;
            exit(1);
        }else{
            OldSize = vectorProcess.size();
        }
        for(unsigned long i=0; i<vectorProcess.size(); ++i){
            if(AnalyzeOneFunction(vectorProcess[i])){ // 如果成功分析当前函数
                vectorProcess.erase(vectorProcess.begin()+i); // 将当前函数从待分析队列移除
                --i; // 游标回退一个, 以抵消下次进入循环时的 ++
            }
        }
    }

    return 0;
}

bool OAOASTConsumer::HandleTopLevelDecl(DeclGroupRef D)
{
	typedef DeclGroupRef::iterator iter;
    std::string tmp = "MY_DOMAIN";

	for (iter i = D.begin(), e = D.end(); i != e; ++i)
	{
		ASTVisitor.TraverseDecl(*i);
	}
	
	return true; // keep going
}
