// Function: sub_2892AE0
// Address: 0x2892ae0
//
_DWORD *__fastcall sub_2892AE0(_DWORD *a1, int a2, __int64 a3, int a4, __m128d a5)
{
  __m128d v6; // xmm1
  __m128d v9; // xmm4
  __m128d v10; // xmm5

  if ( a2 == 91 )
  {
    *a1 = qword_5003A28;
    a1[1] = qword_5003B08;
    return a1;
  }
  else
  {
    a5.m128d_f64[0] = sub_C41B00((__int64 *)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 24LL));
    v6.m128d_f64[1] = a5.m128d_f64[1];
    v6.m128d_f64[0] = a5.m128d_f64[0] * 2147483646.0 + 1.0;
    if ( fabs(v6.m128d_f64[0]) < 4.503599627370496e15 )
    {
      v10.m128d_f64[0] = (double)(int)v6.m128d_f64[0];
      *(_QWORD *)&v6.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                      v10.m128d_f64[0]
                                    + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v6, v10) & 0x3FF0000000000000LL))
                                  | *(_QWORD *)&v6.m128d_f64[0] & 0x8000000000000000LL;
    }
    a5.m128d_f64[1] = 0.0;
    a5.m128d_f64[0] = (1.0 - a5.m128d_f64[0]) / (double)(a4 - 1) * 2147483646.0 + 1.0;
    if ( fabs(a5.m128d_f64[0]) < 4.503599627370496e15 )
    {
      v9.m128d_f64[0] = (double)(int)a5.m128d_f64[0];
      *(_QWORD *)&a5.m128d_f64[0] = COERCE_UNSIGNED_INT64(
                                      v9.m128d_f64[0]
                                    + COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(a5, v9) & 0x3FF0000000000000LL))
                                  | *(_QWORD *)&a5.m128d_f64[0] & 0x8000000000000000LL;
    }
    *a1 = (int)a5.m128d_f64[0];
    a1[1] = (int)v6.m128d_f64[0];
    return a1;
  }
}
