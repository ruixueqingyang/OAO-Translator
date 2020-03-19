/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: OAO.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190105
  Description: OpenMP Automated Offloading 主函数文件
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

#include "OAO.h"

/*******************************************************************************
     Function: main
  Description: OpenMP Automated Offloading 主函数
        Input: // 输入参数说明, 包括每个参数的作用、取值说明及参数间关系。
       Output: 使用命令格式 OAO <options> <file>.c, eg: OAO -fopenmp source.cpp
       Return: // 函数返回值的说明
Data Accessed: // 被访问的表（此项仅对于牵扯到数据库操作的程序）
Data  Updated: // 被修改的表（此项仅对于牵扯到数据库操作的程序）
        Calls: // 被本函数调用的函数清单
    Called By: 无
       Others: // 其它说明
       Author: 瑞雪轻飏
Creation Date: 20190105
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/
int main(int argc, char **argv) {
    struct stat sb;

    // if (argc < 2)
    // {
    // 	// llvm::errs() << "Usage: CIrewriter <options> <filename>\n";
    // 	// return 1;
    // }

    // wfr 20191213 修改默认输入文件
    int NumArgs = 2; // 参数个数
    int MyArgc = NumArgs +1;
    char *MyArgv[MyArgc];
    char MyArg1[] = "-fopenmp";
    char FileDir[] = "/home/wfr/work/Coding/Rodinia_3.1/openmp/particlefilter/ex_particle_OPENMP_seq.c";
    
    MyArgv[0] = argv[0];
    MyArgv[1] = MyArg1;
    MyArgv[2] = FileDir;

    // Get filename
    std::string fileName; //(argv[argc - 1]);

    if (argc >= 2) {
        fileName = argv[argc - 1];
    } else {
        fileName = MyArgv[MyArgc -1];
    }

    //ghn 20191108 如果源文件是.c,需要修改为c++,语法树存在差异，按照c++标准处理
    int len = fileName.length();
    bool flag = false;
    std::string last = fileName.substr(len-2,len);
    if(last.compare(".c") == 0){
        flag = true;
        //复制一份文件到相同目录下，修改后缀为cpp
        std::string new_fileName = fileName + "pp";
        std::ifstream in;
        std::ofstream out;
    
        in.open(fileName, std::ios::binary);//打开源文件
        if (in.fail()){//打开源文件失败
            std::cout << "打开源文件失败" << std::endl;
            in.close();
            out.close();
            return 0;
        }
        out.open(new_fileName, std::ios::binary);//创建目标文件 
        if (out.fail()){//创建文件失败
            std::cout << "创建文件失败" << std::endl;
            out.close();
            in.close();
            return 0;
        }
        else{//复制文件
            out << in.rdbuf();
            out.close();
            in.close();
            std::cout << "源文件为c文件，从c文件转换为c++文件,创建成功" << std::endl;
        }
        fileName = new_fileName;
        char *p = (char*)fileName.c_str();
        if (argc >= 2) {
            argv[argc - 1] = p;
        } else {
            MyArgv[MyArgc -1] = p;
        }
    }

    std::cout << "fileName: " << fileName << std::endl;

    // Make sure it exists
    if (stat(fileName.c_str(), &sb) == -1) {
        perror(fileName.c_str());
        exit(EXIT_FAILURE);
    }

    CompilerInstance compiler;
    DiagnosticOptions diagnosticOptions;
    compiler.createDiagnostics();
    // compiler.createDiagnostics(argc, argv);

    // Create an invocation that passes any flags to preprocessor
    CompilerInvocation *Invocation = new CompilerInvocation;
    if (argc < 2) {
        CompilerInvocation::CreateFromArgs(*Invocation, MyArgv + 1, MyArgv + MyArgc, compiler.getDiagnostics());
    } else {
        CompilerInvocation::CreateFromArgs(*Invocation, argv + 1, argv + argc, compiler.getDiagnostics());
    }
    compiler.setInvocation((std::shared_ptr<CompilerInvocation>)Invocation);

    // Set default target triple
    std::shared_ptr<clang::TargetOptions> pto = std::make_shared<clang::TargetOptions>();
    pto->Triple = llvm::sys::getDefaultTargetTriple();
    TargetInfo *pti = TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), pto);
    compiler.setTarget(pti);

    compiler.createFileManager();
    compiler.createSourceManager(compiler.getFileManager());

    HeaderSearchOptions &headerSearchOptions = compiler.getHeaderSearchOpts();

    // <Warning!!> -- Platform Specific Code lives here
    // This depends on A) that you're running linux and
    // B) that you have the same GCC LIBs installed that
    // I do.
    // Search through Clang itself for something like this,
    // go on, you won't find it. The reason why is Clang
    // has its own versions of std* which are installed under
    // /usr/local/lib/clang/<version>/include/
    // See somewhere around Driver.cpp:77 to see Clang adding
    // its version of the headers to its include path.
    // To see what include paths need to be here, try
    // clang -v -c test.c
    // or clang++ for C++ paths as used below:
    
    // headerSearchOptions.AddPath("/usr/include/c++/9.1.0", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/usr/include/c++/9.1.0/x86_64-pc-linux-gnu", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/usr/include/c++/9.1.0/backward", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/usr/lib/gcc/x86_64-pc-linux-gnu/9.1.0/include", clang::frontend::Angled, false, false);
    

    headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/include/c++/8.3.0", clang::frontend::Angled, false, false);
    headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/include/c++/8.3.0/x86_64-pc-linux-gnu", clang::frontend::Angled, false, false);
    headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/include/c++/8.3.0/backward", clang::frontend::Angled, false, false);
    headerSearchOptions.AddPath("/usr/local/include", clang::frontend::Angled, false, false);
    headerSearchOptions.AddPath("/home/wfr/install/LLVM-9/install-9/lib/clang/9.0.1/include", clang::frontend::Angled, false, false);
    headerSearchOptions.AddPath("/usr/include", clang::frontend::Angled, false, false);


    // headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/lib/gcc/x86_64-pc-linux-gnu/8.3.0/include-fixed", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/include/c++/8.3.0/tr1", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/home/wfr/install/GCC-8/install/lib/gcc/x86_64-pc-linux-gnu/8.3.0/include", clang::frontend::Angled, false, false);
      
    // headerSearchOptions.AddPath("/usr/local/include", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/usr/include/x86_64-linux-gnu", clang::frontend::Angled, false, false);
    // headerSearchOptions.AddPath("/usr/include", clang::frontend::Angled, false, false);
    // </Warning!!> -- End of Platform Specific Code

    // Allow C++ code to get rewritten
    LangOptions langOpts;
    langOpts.GNUMode = 1;
    langOpts.CXXExceptions = 1;
    langOpts.RTTI = 1;
    langOpts.Bool = 1;
    langOpts.CPlusPlus = 1;
    //clang::InputKind MyInputKind = InputKind(clang::InputKind::CXX);
    clang::InputKind MyInputKind(clang::InputKind::Language::CXX);
    llvm::Triple MyTriple;
    clang::PreprocessorOptions PPOpts;
    PPOpts.resetNonModularOptions();
    clang::LangStandard::Kind MyKind = clang::LangStandard::lang_cxx11;
    // clang::LangStandard::Kind MyKind = LANGSTANDARD(cxx11, "c++11", CXX, "ISO C++ 2011 with amendments", LineComment
    // | CPlusPlus | CPlusPlus11 | Digraphs);
    Invocation->setLangDefaults(langOpts,
                                MyInputKind, // wfr 20181025
                                MyTriple,    // wfr 20181025
                                PPOpts,      // wfr 20181025
                                MyKind);     // wfr 20181025

    compiler.createPreprocessor(clang::TU_Complete);
    compiler.getPreprocessorOpts().UsePredefines = false;

    compiler.createASTContext();

    // Initialize rewriter
    Rewriter Rewrite;
    Rewrite.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

    llvm::ErrorOr< const FileEntry * > pFile = compiler.getFileManager().getFile(fileName);
    auto File = *pFile;
    compiler.getSourceManager().setMainFileID(
      compiler.getSourceManager().createFileID(File, clang::SourceLocation(), clang::SrcMgr::C_User));
    compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &compiler.getPreprocessor());

    // 创建 ASTDumper 方便调试
    // ASTDumper MyDumper(std::cout(), nullptr, &(compiler.getSourceManager()), true);
    OAOASTConsumer astConsumer(compiler.getASTContext(), Rewrite);

    // Convert <file>.c to <file_out>.c
    // ghn 20191119 如果是c文件，c转换为c++处理后，保存为c文件
    std::string reName = fileName;
    if(flag == true){
        //unlink(fileName);
        reName = fileName.substr(0,len);
        std::cout << "fileName_reName: " << reName << std::endl;
        //给fileName重新赋值
    }
    
    std::string outName(reName);
    size_t ext = outName.rfind(".");
    if (ext == std::string::npos)
        ext = outName.length();
    outName.insert(ext, "_out");

    llvm::errs() << "Output to: " << outName << "\n";
    std::error_code OutErrorInfo;
    std::error_code ok;
    llvm::raw_fd_ostream outFile(llvm::StringRef(outName), OutErrorInfo, llvm::sys::fs::F_None);

    if (OutErrorInfo == ok) {
        // Parse the AST 应该是在这里遍历 AST
        ParseAST(compiler.getPreprocessor(), &astConsumer, compiler.getASTContext());
        astConsumer.ASTVisitor.AnalyzeAllFunctions();
        std::cout << "解析源文件完成" << std::endl;
        astConsumer.ASTVisitor.translation();
        compiler.getDiagnosticClient().EndSourceFile();
        // Now output rewritten source code
        const RewriteBuffer *RewriteBuf = Rewrite.getRewriteBufferFor(compiler.getSourceManager().getMainFileID());
        outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());

        // wfr 20190730 这里将运行时源文件拷贝到目标目录
        std::string OAODir = argv[0];
        size_t TmpPos;
        std::string destDir, srcDir;

        TmpPos = OAODir.rfind('/');
        srcDir = OAODir.substr(0, TmpPos+1); srcDir += "RunTime.cpp";
        TmpPos = reName.rfind('/');
        destDir = reName.substr(0, TmpPos+1); destDir += "RunTime.cpp";
        FileCopy(destDir, srcDir);

        TmpPos = OAODir.rfind('/');
        srcDir = OAODir.substr(0, TmpPos+1); srcDir += "RunTime.h";
        TmpPos = reName.rfind('/');
        destDir = reName.substr(0, TmpPos+1); destDir += "RunTime.h";
        FileCopy(destDir, srcDir);
        
        std::cout << "转换完成, 输出文件: " << outName << std::endl;
    } else {
        llvm::errs() << "Cannot open " << outName << " for writing\n";
    }

    outFile.close();

    // DebugPrint();
    // ghn 20191119 如果是c文件，c转换为c++处理后，删除c++文件
    if(flag == true){
        char *m = (char*)fileName.c_str();
        remove(m);
    }


    return 0;
}

int FileCopy(std::string dest, std::string src){

    std::ifstream srcStream(src, std::ifstream::in|std::ifstream::binary);
    if(!srcStream.is_open()){
        srcStream.close();
        std::cout << "FileCopy 错误: 打开源文件失败" << std::endl;
        exit(1);
    }

    std::ofstream destStream(dest, std::ifstream::out|std::ifstream::binary);
    if(!destStream.is_open()){
        destStream.close();
        std::cout << "FileCopy 错误: 打开目标文件失败" << std::endl;
        exit(1);
    }

    char tmp;
    while(!srcStream.eof()){
        srcStream.get(tmp);
        destStream.put(tmp);
    }

    srcStream.close();
    destStream.close();

    return 0;
}