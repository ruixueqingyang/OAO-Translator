# ./My2dconv.sh 20
# ./My2mm.sh 20
# ./My3dconv.sh 20
# ./My3mm.sh 20
# ./Myatax.sh 20
# ./Mybicg.sh 20
# ./Mycorr.sh 20
# ./Mycovar.sh 20

# ./Mygemm.sh 20
# ./Mygesummv.sh 20
# ./Mymvt.sh 20
# ./Mysyrk.sh 20

# 这三个时间很长
# ./Myfdtd2d.sh 20
# ./Mysyr2k.sh 20
# ./Myfdtd2d_function.sh 20



# ./2dconv_native.sh 10
# ./2mm_native.sh 10
./3dconv_native.sh 10
./3mm_native.sh 10
./atax_native.sh 10
./bicg_native.sh 10
./corr_native.sh 10
./covar_native.sh 10
./gemm_native.sh 10
./gesummv_native.sh 10
./mvt_native.sh 10
./syrk_native.sh 10

# 这三个时间很长
./fdtd2d_native.sh 10
./syr2k_native.sh 10
./fdtd2d_function_native.sh 10