// Function: sub_1D3D250
// Address: 0x1d3d250
//
__int64 *__fastcall sub_1D3D250(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  unsigned __int8 *v10; // rax
  unsigned int v11; // ebx
  __int128 v12; // rax
  const void **v14; // [rsp+10h] [rbp-38h]

  v10 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v14 = (const void **)*((_QWORD *)v10 + 1);
  *(_QWORD *)&v12 = sub_1D38BB0((__int64)a1, a4, a5, *v10, v14, 0, a6, a7, a8, 0);
  return sub_1D332F0(a1, 52, a5, v11, v14, 0, *(double *)a6.m128i_i64, a7, a8, a2, a3, v12);
}
