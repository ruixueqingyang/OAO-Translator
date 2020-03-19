/*******************************************************************************
Copyright(C), 2010-2019, 瑞雪轻飏
     FileName: ASTStackNode.h
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190108
  Description: 在 AST 先深搜索中, 用来保存搜索顺序栈中的一个元素, 
               使遍历函数知道当前是在遍历 哪种操作符 的 第几个操作数, 以判断对变量的操作是 读/写
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

#ifndef AST_STACK_NODE_H
#define AST_STACK_NODE_H

#include "Clang_LLVM_Inc.h"

using namespace clang;

// wfr 20190307 CompoundStmt 信息节点类, 目的是保存当前所在 "{}" 的范围, 也就是变量的作用域
/*class CompoundStackNode{
public:
    Stmt* ptrDecl;
    MY_SOURCE_RANGE CompoundRange;

    CompoundStackNode(){ptrDecl=NULL;}
    CompoundStackNode(Stmt* ptr, Rewriter& Rewrite){
        std::cout << "进入 CompoundStackNode::CompoundStackNode" << std::endl;
        init(ptr, Rewrite);
        std::cout << "离开 CompoundStackNode::CompoundStackNode" << std::endl;
    }

    void init(Stmt* ptr, Rewriter& Rewrite){
        ptrDecl = ptr;
        CompoundRange.init(ptrDecl->getBeginLoc(), ptrDecl->getEndLoc(), Rewrite);
    }
};*/

// 函数栈元素节点
// 函数节点在 vectorFunc 中的 index
// 
class FuncStackNode{
public:
    Decl* ptrDecl; // 函数定义的地址
    int   indexFunc; // 函数在 vectorFunc 中的 index

    FuncStackNode(int in_indexFunc = -1, Decl* in_ptrDecl = NULL){
        indexFunc = in_indexFunc;
        ptrDecl = in_ptrDecl;
    }
    void init(int in_indexFunc = -1, Decl* in_ptrDecl = NULL){
        indexFunc = in_indexFunc;
        ptrDecl = in_ptrDecl;
    }
};

class ASTStackNode{
public:
    enum PTR_ARY{ PTR_UNINIT, PTR, ARRAY }; // 是 引用指针得到array的值, 或者 指针之间相互赋值

    Stmt* pOperator; // 运算符的地址 / 函数调用的地址
    VarDecl* pDecl; // 变量定义的地址,  注意: 这两个地址只能有一个被使用
    std::string Name; // AST 语句/节点的名称/类型
    int OperandID; // 表示正在遍历第几个操作数, 从 0 开始

    std::vector<bool>   OperandState; // 表示某个操作数是否处理完成
    std::vector<int>    Operand; // 操作数 在 vectorVar 中的 index
    std::vector<int>    NextOperandID; // 遍历 AST 时, 下一个要处理的 操作数的 ID
    std::vector<Stmt*>  SubStmts; // 当前 运算符 的 直接 子节点列表
    std::vector<PTR_ARY> AccessType; // 变量 是 指针 / 数组
    std::vector<std::string>  vectorInfo; // 保存运算操作数的信息, 例如被引用的函数的名称等, 例如 "malloc"

    Stmt* pStmt; // 如果需要用堆栈传递指针, 可以使用这个变量
    NODE_INDEX indexNode; // 表示当前要处理的是哪个串并行节点, 这个好像暂时没有利用起来
    int indexFunc; // 如果处理函数调用, indexFunc 表示被调用函数的 index
    int indexVarRef; // 如果是 OMP 中的函数引用, 这个用来标识 保存 函数实参信息的 VAR_REF_LIST 的 index

    void init(Stmt* in_pOperator, std::string in_Name = "NULL", int in_OperandID = -1, int in_OperandNum=0){
        pOperator = in_pOperator;
        pDecl = NULL;
        Name = in_Name;
        OperandID = in_OperandID;
        pStmt = NULL;
        indexFunc = -1;
        indexVarRef = -1;
        OperandState.resize(in_OperandNum);
        Operand.resize(in_OperandNum);
        NextOperandID.resize(in_OperandNum);
        SubStmts.resize(in_OperandNum);
        AccessType.resize(in_OperandNum);
        vectorInfo.resize(in_OperandNum);
        for(int i=0; i<in_OperandNum; i++){
            OperandState[i] = false;
            Operand[i] = -1;
            NextOperandID[i] = -1;
            SubStmts[i] = nullptr;
            AccessType[i] = PTR_ARY::PTR_UNINIT;
            vectorInfo[i] = "NULL";
        }
    }
    void init(VarDecl* in_pDecl, std::string in_Name = "NULL", int in_OperandID = -1, int in_OperandNum=0){
        pOperator = NULL;
        pDecl = in_pDecl;
        Name = in_Name;
        OperandID = in_OperandID;
        pStmt = NULL;
        indexFunc = -1;
        indexVarRef = -1;
        OperandState.resize(in_OperandNum);
        Operand.resize(in_OperandNum);
        NextOperandID.resize(in_OperandNum);
        SubStmts.resize(in_OperandNum);
        AccessType.resize(in_OperandNum);
        vectorInfo.resize(in_OperandNum);
        for(int i=0; i<in_OperandNum; i++){
            OperandState[i] = false;
            Operand[i] = -1;
            NextOperandID[i] = -1;
            SubStmts[i] = NULL;
            AccessType[i] = PTR_ARY::PTR_UNINIT;
            vectorInfo[i] = "NULL";
        }
    }
    ASTStackNode(Stmt* in_pOperator, std::string in_Name = "NULL", int in_OperandID = -1, int in_OperandNum = 0 ){
        init(in_pOperator, in_Name, in_OperandID, in_OperandNum);
    }
    ASTStackNode(VarDecl* in_pDecl, std::string in_Name = "NULL", int in_OperandID = -1, int in_OperandNum = 0 ){
        init(in_pDecl, in_Name, in_OperandID, in_OperandNum);
    }
    ASTStackNode() { pOperator = NULL; Name = "NULL"; OperandID = -1; pStmt = NULL; indexFunc = -1; indexVarRef = -1;}
};


#endif // AST_STACK_NODE_H