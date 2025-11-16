// Function: sub_1BE0150
// Address: 0x1be0150
//
__int64 __fastcall sub_1BE0150(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __m128i a5,
        __m128i a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 result; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8
  int v17; // ebx
  double v18; // xmm4_8
  double v19; // xmm5_8

  result = sub_1BDC4B0(
             a1,
             *(__int64 ***)(a2 - 48),
             *(__int64 ***)(a2 - 24),
             (__int64)a4,
             a5,
             *(double *)a6.m128i_i64,
             *(double *)a7.m128i_i64,
             *(double *)a8.m128i_i64,
             a9,
             a10,
             a11,
             a12);
  if ( !(_BYTE)result )
  {
    v17 = sub_1BE00E0(a1, 0, *(_QWORD *)(a2 - 48), a3, a4, *(__int64 **)(a1 + 8), a5, a6, a7, a8, v15, v16, a11, a12);
    return v17
         | (unsigned int)sub_1BE00E0(
                           a1,
                           0,
                           *(_QWORD *)(a2 - 24),
                           a3,
                           a4,
                           *(__int64 **)(a1 + 8),
                           a5,
                           a6,
                           a7,
                           a8,
                           v18,
                           v19,
                           a11,
                           a12);
  }
  return result;
}
