// Function: sub_387DD00
// Address: 0x387dd00
//
_QWORD *__fastcall sub_387DD00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  int v11; // eax

  v11 = *(_DWORD *)(a2 + 32);
  if ( v11 == 1 )
    return sub_3876850(a1, a2, a3, (__m128)a4, *(double *)a5.m128i_i64, a6, a7, a8, a9, a10, a11);
  if ( v11 == 2 )
    return (_QWORD *)sub_387DB30(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  return (_QWORD *)sub_387DD30(
                     a1,
                     a2,
                     a3,
                     *(double *)a4.m128i_i64,
                     *(double *)a5.m128i_i64,
                     a6,
                     a7,
                     a8,
                     a9,
                     a10,
                     *(double *)a11.m128_u64);
}
