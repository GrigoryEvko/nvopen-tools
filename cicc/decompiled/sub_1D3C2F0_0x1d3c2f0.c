// Function: sub_1D3C2F0
// Address: 0x1d3c2f0
//
__int64 *__fastcall sub_1D3C2F0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        const void **a6,
        double a7,
        __m128i a8,
        __m128i a9)
{
  __int128 v12; // rax
  __int128 v14; // [rsp-10h] [rbp-50h]
  unsigned int v15; // [rsp+8h] [rbp-38h]

  v15 = a5;
  *((_QWORD *)&v14 + 1) = a6;
  *(_QWORD *)&v14 = a5;
  *(_QWORD *)&v12 = sub_1D395A0((__int64)a1, 1, a2, (unsigned int)a5, a6, (__int64)a6, a7, a8, a9, v14);
  return sub_1D332F0(a1, 120, a2, v15, a6, 0, a7, *(double *)a8.m128i_i64, a9, a3, a4, v12);
}
