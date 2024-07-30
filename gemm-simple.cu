#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
// __global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k)
void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k)
{

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
  printf("\n A :"), print(A); // gmem_ptr[16b](0x7f400e000000) o (81920,256):(256,_1)
  printf("\n B :"), print(B); // gmem_ptr[16b](0x7f4010800000) o (256,256):(256,_1)
  printf("\n C :"), print(C); // gmem_ptr[16b](0x7f4012000000) o (81920,256):(256,_1)
  // int ix = blockIdx.x;
  // int iy = blockIdx.y;
  int ix = 1;
  int iy = 1;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  printf("\n gA :"), print(gA); // gmem_ptr[16b](0x7f400e010000) o (_128,_32,8):(256,_1,_32)
  //  gB(kTileN, kTileK, num_tile_k)
  printf("\n gB :"), print(gB); // gmem_ptr[16b](0x7f4010810000) o (_128,_32,8):(256,_1,_32)
  //  gC(kTileM, kTileN)
  printf("\n gC :"), print(gC); // gmem_ptr[16b](0x7f4012010100) o (_128,_128):(256,_1) coshape layout: (_2,_2,_1):(_1,_2,_0)

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(96);
  // auto thr_mma = tiled_mma.get_slice(threadIdx.x);

#if 0
  template <class ATensor>
  CUTE_HOST_DEVICE constexpr auto
  partition_A(ATensor && atensor) const
  {
    auto thr_tensor = make_tensor(std::forward<ATensor>(atensor).data(), this->thrfrg_A(atensor.layout()));
    printf("\n atensor :"), print(atensor);                                                   // gmem_ptr[16b](0x7f400e010000) o (_128,_32,8):(256,_1,_32)
    printf("\n this->thrfrg_A(atensor.layout()) :"), print(this->thrfrg_A(atensor.layout())); //(((_4,(_2,_4)),(_2,_1)),((_2,_2,_2),(_4,_2,8))):(((_2,(256,512)),(4096,_0)),((_1,2048,_8),(8192,_16,_32)))
    printf("\n thr_tensor :"), print(thr_tensor);                                             // gmem_ptr[16b](0x7f400e010000) o (((_4,(_2,_4)),(_2,_1)),((_2,_2,_2),(_4,_2,8))):(((_2,(256,512)),(4096,_0)),((_1,2048,_8),(8192,_16,_32)))

    auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));

    printf("\n thr_vmnk_ :"), print(thr_vmnk_); //(0,1,1,_0)
    printf("\n thr_vmk :"), print(thr_vmk);     //(0,(1,_0))

    printf("\n make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)) :");
    print(make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_))); //(_,(_,_,_))

    printf("\n thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_))) :");
    print(thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)))); // gmem_ptr[16b](0x7f400e012000) o ((_2,_2,_2),_4,_2,8):((_1,2048,_8),8192,_16,_32)

    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  }

  // Tile a tensor or a layout from shape
  //   (M,K,...)
  // to shape
  //   ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
  // where
  //   ThrV: The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
  //   ThrM: The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
  //   ThrK: The threads tiled in K.      layout<3>(ThrLayoutVMNK): ThrK -> thread_idx
  //   FrgV:  The values local to an MMA.
  //   RestM: The values tiled in M.
  //   RestK: The values tiled in K.
  template <class ATensor>
  CUTE_HOST_DEVICE constexpr auto
  thrfrg_A(ATensor && atensor) const
  {
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<2>{});
    // CUTE_STATIC_ASSERT_V(size<0>(atensor) % size<0>(TiledShape_MNK{}) == Int<0>{});
    // CUTE_STATIC_ASSERT_V(size<1>(atensor) % size<2>(TiledShape_MNK{}) == Int<0>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}),
                            get<2>(PermutationMNK{}));
    printf("\n PermutationMNK{} :"), print(PermutationMNK{}); //((_1,_2,_1):(_0,_1,_0),_,_)
    printf("\n t_tile :"), print(t_tile);                     //((_1,_2,_1):(_0,_1,_0),_)
    auto t_tensor = logical_divide(atensor, t_tile);          // (PermM,PermK)
    printf("\n t_tensor :"), print(t_tensor);                 //(((_1,_2,_1),_64),_32,8):(((_0,256,_0),512),_1,_32)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    printf("\n AtomShape_MNK{} :"), print(AtomShape_MNK{}); //(_16,_8,_16)
    printf("\n a_tile :"), print(a_tile);                   //(_16:_1,_16:_1)
    auto a_tensor = zipped_divide(t_tensor, a_tile);        // ((AtomM,AtomK),(RestM,RestK))
    printf("\n a_tensor :"), print(a_tensor);               //(((_2,_8),_16),(_8,_2,8)):(((256,512),_1),(4096,_16,_32))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutA_TV{}, _); // ((ThrV,FrgV),(RestM,RestK))
    printf("\n tv_tensor :"), print(tv_tensor);             //(((_4,(_2,_4)),(_2,_2,_2)),(_8,_2,8)):(((_2,(256,512)),(_1,2048,_8)),(4096,_16,_32))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                        make_layout(size<3>(thr_layout_vmnk_))));

    printf("\n thr_layout_vmnk_ :"), print(thr_layout_vmnk_); //(_32,_2,_2,_1):(_1,_32,_64,_0)
    printf("\n thr_tile :"), print(thr_tile);                 //(_,(_2:_1,_1:_0))

    auto thr_tensor = zipped_divide(tv_tensor, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
    printf("\n thr_tensor :"), print(thr_tensor);         //(((_4,(_2,_4)),(_2,_1)),((_2,_2,_2),(_4,_2,8))):(((_2,(256,512)),(4096,_0)),((_1,2048,_8),(8192,_16,_32)))

    return thr_tensor;
  }

#endif
  auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  printf("\n tAgA :"), print(tAgA);    //
  printf("\n tBgB :"), print(tBgB);    //
  printf("\n tCgC :"), print(tCgC);    //
  printf("\n tCgC :"), print(tCgC);    //

#if 0

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr auto
  partition_fragment_A(ATensor && atensor) const
  {

    printf("\n partition_A(atensor) :"), print(partition_A(atensor)); // gmem_ptr[16b](0x7f400e012000) o ((_2,_2,_2),_4,_2,8):((_1,2048,_8),8192,_16,_32)
    return TiledMMA::make_fragment_A(partition_A(atensor));
  }

  template <class ATensor>
  CUTE_HOST_DEVICE static constexpr auto
  make_fragment_A(ATensor && atensor)
  {
    // Check that this tensor is likely already partitioned
    CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<3>{}); // VMK
    CUTE_STATIC_ASSERT_V(size<0>(atensor) == size<1>(LayoutA_TV{}));
    printf("\n LayoutA_TV{} :"), print(LayoutA_TV{}); //((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))

    if constexpr (has_dereference<FrgTypeA>::value)
    {
      // If the intended FrgTypeA is a view (of the current tensor), forward the whole
      static_assert(is_same<ValTypeA, typename remove_cvref_t<ATensor>::value_type>::value, "Expecting ValTypeA type");
      auto res = make_tensor<FrgTypeA>(std::forward<ATensor>(atensor));
      printf("\n res :"), print(res); // ptr[16b](0x7ffc05a490b0) o ((_2,_2,_2),_4,_2):((_1,_2,_4),_8,_32) ?
      return res;
    }
    else
    {
      // Else, the intended FrgTypeA is a value type, construct a new tensor with a fragment layout
      auto res = make_fragment_like<FrgTypeA>(atensor);
      printf("\n res :"), print(res); //
      return res;
    }

    CUTE_GCC_UNREACHABLE;
  }
#endif
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)
  printf("\n tArA :"), print(tArA);                      // ptr[16b](0x7ffc05a48fb0) o ((_2,_2,_2),_4,_2):((_1,_2,_4),_8,_32)
  printf("\n tBrB :"), print(tBrB);                      // ptr[16b](0x7ffc05a49030) o ((_2,_2),_8,_2):((_1,_2),_4,_32)
  printf("\n tCrC :"), print(tCrC);                      // ptr[16b](0x7ffc05a490b0) o ((_2,_2),_4,_8):((_1,_2),_4,_16)
  clear(tCrC);

  int num_tile_k = size<2>(gA);

  printf("\n tAgA(_, _, _, itile) :"), print(tAgA(_, _, _, 0)); // gmem_ptr[16b](0x7f400e012000) o ((_2,_2,_2),_4,_2):((_1,2048,_8),8192,_16)
  printf("\n tBgB(_, _, _, itile) :"), print(tBgB(_, _, _, 0)); // gmem_ptr[16b](0x7f4010811000) o ((_2,_2),_8,_2):((_1,_8),4096,_16)
  printf("\n tBgB(_, _, _, itile) :"), print(tBgB(_, _, _, 0)); //

#pragma unroll 1
  for (int itile = 0; itile < num_tile_k; ++itile)
  {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }
  printf("\n ");
  cute::copy(tCrC, tCgC);
}

int main()
{
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T *)malloc(sizeof(T) * m * k);
  Bptr_host = (T *)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{},
                                      make_layout(Shape<_2, _2, _1>{}),
                                      make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 32;

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i)
  {
    // gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
    gemm_simple<T, kTileM, kTileN, kTileK, MMA>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i)
  {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     n, m, k,
                                     &alpha,
                                     (half *)Bptr, k,
                                     (half *)Aptr, k,
                                     &beta,
                                     (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T *)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T *)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i)
  {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold)
    {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n)
{
  for (int i = 0; i < n; ++i)
  {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
