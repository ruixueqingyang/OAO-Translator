/***************************************************************************
Copyright(C), 2010-2018, 瑞雪轻飏
     FileName: VarInfo.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20181216
  Description: 变量信息类
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

#ifndef VAR_INFO_H
#define VAR_INFO_H

#include "Clang_LLVM_Inc.h"
#include "SrcRange.h"
#include "SeqParNodeBase.h"

using namespace clang;

/*
|-FunctionDecl 0x1a486861b38 <line:49:1, line:51:1> line:49:7 used AddFunc 'float (float &, float &, const float *, const float *const, MY_DOMAIN &, MY_DOMAIN *)'
| |-ParmVarDecl 0x1a486861740 <col:15, col:22> col:22 used a 'float &' 改变 REF
| |-ParmVarDecl 0x1a4868617b8 <col:25, col:32> col:32 used b 'float &' 改变 REF
| |-ParmVarDecl 0x1a486861830 <col:35, col:48> col:48 pa 'const float *' 不变 PTR_CONST
| |-ParmVarDecl 0x1a4868618a8 <col:52, col:71> col:71 pb 'const float *const' 不变 PTR_CONST
| |-ParmVarDecl 0x1a486861948 <col:75, col:86> col:86 domain 'MY_DOMAIN &' 改变 REF
| |-ParmVarDecl 0x1a4868619b8 <col:94, col:105> col:105 pdomain 'MY_DOMAIN *' 改变 PTR
*/

// 这里应该加上变量类型信息
// 1. 传入拷贝(且不是指针)(不改变)
// 2. 简单指针(可以有const修饰)(可能改变)
// 3. const TYPE* [const] VAR(认为不改变)
// 4. 简单引用/类的引用(可能改变)
// 5. 类的指针(可以有const修饰)(可能改变)
// 6. const CLASS_TYPE* [const] VAR(认为不改变)
// 7. 类(认为可能改变, 因为类的域可能是指针变量)
enum VAR_TYPE {
    VAR_UNINIT, // 没初始化
    COPY, // 传入拷贝(且不是指针)(不改变)
    PTR, // TYPE* [const] VAR 简单指针(可以有const修饰)(可能改变)
    PTR_CONST, // const TYPE* [const] VAR(认为不改变), 指向的数据区只读, 指针变量可以被改变, 方便函数参数分析, 此种类型传入参数则认为数据只读
    CLASS_PTR, // CLASS_TYPE* [const] VAR 类的指针(可以有const修饰)(可能改变)
    CLASS_PTR_CONST, // const CLASS_TYPE* [const] VAR(认为不改变)
    REF, // TYPE& VAR 简单引用
    CLASS_REF, // CLASS_TYPE& VAR 类的引用
    CLASS //类(认为可能改变, 因为类的域可能是指针变量)
};

class VARIABLE_INFO : public MY_SOURCE_RANGE{
public:
    std::string    Name; // 变量名/域的名字
    Decl*          ptrDecl; // 变量的定义的地址
    //DeclStmt*      pDeclStmt; // 知道变量定义所在的 DeclStmt, 才能知道变量定义语句的结尾的 位置
    MY_SOURCE_RANGE DeclRange;
    std::string    TypeName; // 变量的类型：int、int*、float*等等
    VAR_TYPE       Type;
    bool           isClass; // 1表示是类, 0表示不是, 默认是0
    bool           isMember; // 1表示变量是类的域, 0表示不是, 默认是0
    bool           isArrow; // 这个在变量是类的域时才有意义, 1表示使用“->”来引用类的域, 0表示使用“.”来引用类的域
    Decl*          ptrClass; // 如果是类的成员, 这里是类实例定义的地址
    int            indexClass; // 如果是类的成员, 这里是类的 VARIABLE_INFO 在列表中的 index
    std::string    RootName; // 逆着变量实参形参的传递方向, 找到变量最初的定义的变量名
    int            indexRoot; // 当前变量是指针时, 其值可能是从别的指针变量赋值而来, indexRoot 表示逆赋值链最初的那个指针变量
    Decl*          ptrRoot; // 变量最初定义的地址
    bool           UsedInOMP; // 标识该变量是否在 OMP 中被调用
    bool           isGlobal; // wfr 20190806 标识该变量是否是全局变量
    NODE_INDEX     indexDefNode; // 表示变量定义在哪个 node 中
    NODE_INDEX     indexLastRefNode; // 表示变量的最后一次引用在哪个 node 中
    MY_SOURCE_RANGE Scope; // 变量的作用域的范围

    std::string    Rename; // 对 结构体/类 的域 进行重命名
    std::string    FullName; // 结构体/类 的域 的全名, 即从 类./->域
    std::string    ArrayLen; // 数组长度
    //std::string    ArrayLenCalculate; // 数组长度
    //bool           isMapedWhileRewriting; // 表示该变量是否 在修改源代码的过程中 已经被 map 到 device
    
    // std::vector<int>    vectorIndexMember; // 这个暂时没用 如果是类的话, 表示类的域的 index
    std::vector<CODE_INSERT> vectorInsert; // 表示需要在源代码中插入的信息
    std::vector<CODE_REPLACE> vectorReplace; // 表示需要在源代码中替换的信息

    // ~VARIABLE_INFO(){
    //     //std::cout << "~VARIABLE_INFO: 析构 变量" << Name << " 的信息" << std::endl;
    //     // std::cout << "~VARIABLE_INFO: vectorIndexMember.size() = " << vectorIndexMember.size() << std::endl;
    //     if(Name=="flux_contribution_i_density_energy"){
    //         std::cout << "~VARIABLE_INFO: 到了出问题的地方!" << std::endl;
    //     }
    // }

    VARIABLE_INFO(Decl* ptr, size_t offset, Rewriter& Rewrite) : MY_SOURCE_RANGE(ptr->getBeginLoc(), ptr->getEndLoc(), offset, Rewrite){
        init();
    }
    VARIABLE_INFO(Decl* ptr, Rewriter& Rewrite) : MY_SOURCE_RANGE(ptr->getBeginLoc(), ptr->getEndLoc(), Rewrite){
        init();
    }

    const VARIABLE_INFO& operator=(const VARIABLE_INFO& in_Var){
        init(in_Var);
        return in_Var;
    }

    VARIABLE_INFO(const VARIABLE_INFO& in_Var){
        init(in_Var);
    }

    VARIABLE_INFO() : MY_SOURCE_RANGE(){
        init();
    }

    void init(const VARIABLE_INFO& in_Var){
        Name = in_Var.Name;
        ptrDecl = in_Var.ptrDecl;
        //pDeclStmt = in_Var.pDeclStmt;
        DeclRange = in_Var.DeclRange;
        TypeName = in_Var.TypeName;
        Type = in_Var.Type;
        isClass = in_Var.isClass;
        isMember = in_Var.isMember;
        isArrow = in_Var.isArrow;
        ptrClass = in_Var.ptrClass;
        indexClass = in_Var.indexClass;
        RootName = in_Var.RootName;
        indexRoot = in_Var.indexRoot;
        ptrRoot = in_Var.ptrRoot;
        UsedInOMP = in_Var.UsedInOMP;
        isGlobal = in_Var.isGlobal;
        indexDefNode = in_Var.indexDefNode;
        indexLastRefNode = in_Var.indexLastRefNode;
        Scope = in_Var.Scope;

        Rename = in_Var.Rename;
        FullName = in_Var.FullName;
        ArrayLen = in_Var.ArrayLen;
    
        // vectorIndexMember = in_Var.vectorIndexMember;
        vectorInsert = in_Var.vectorInsert;
        vectorReplace = in_Var.vectorReplace;


        BeginLoc = in_Var.BeginLoc;
        EndLoc = in_Var.EndLoc;
        BeginLine = in_Var.BeginLine;
        BeginCol = in_Var.BeginCol;
        EndLine = in_Var.EndLine;
        EndCol = in_Var.EndCol;
        BeginOffset = in_Var.BeginOffset;
        EndOffset = in_Var.EndOffset;
    }

    void init(){
        Name = "NULL";
        ptrDecl = NULL;
        TypeName = "NULL";
        Type = VAR_TYPE::VAR_UNINIT;
        isClass = false;
        isMember = false;
        isArrow = 0;
        ptrClass = NULL;
        indexClass = -1;
        RootName = "NULL";
        indexRoot = -1;
        ptrRoot = NULL;
        UsedInOMP = false;
        isGlobal = false;
        Rename = "NULL";
        FullName = "NULL";
        ArrayLen = "NULL";
        //ArrayLenCalculate = "NULL";
        //isMapedWhileRewriting = false;
    }

    // wfr 20190329 获得多维数组的元素的个数, 即数组长度
    void getArrayLen(){

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

        // if(TypeName.back()!=']'){
        //     std::cout << "getArrayLen 错误：不是数组类型" << std::endl;
        //     exit(1);
        // }

        // ArrayLen = "sizeof( ";
        // ArrayLen += FullName;
        // ArrayLen += " ) / sizeof( ";
        // ArrayLen += FullName;
        // ArrayLen += "[0] )";
    }
};

#endif // VAR_INFO_H
