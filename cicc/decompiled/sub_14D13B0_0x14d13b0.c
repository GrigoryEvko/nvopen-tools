// Function: sub_14D13B0
// Address: 0x14d13b0
//
__m128d __fastcall sub_14D13B0(__m128d result)
{
  __m128d v1; // xmm1

  if ( fabs(result.m128d_f64[0]) < 4.503599627370496e15 )
  {
    v1 = 0;
    v1.m128d_f64[0] = (double)(int)result.m128d_f64[0];
    v1.m128d_f64[0] = v1.m128d_f64[0] - COERCE_DOUBLE(*(_OWORD *)&_mm_cmpgt_sd(v1, result) & 0x3FF0000000000000LL);
    return _mm_or_pd(v1, _mm_andn_pd((__m128d)0x7FFFFFFFFFFFFFFFuLL, result));
  }
  return result;
}
