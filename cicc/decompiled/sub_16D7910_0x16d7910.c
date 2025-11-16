// Function: sub_16D7910
// Address: 0x16d7910
//
double *__fastcall sub_16D7910(__m128i *a1)
{
  double *result; // rax
  __m128i v2; // xmm1
  __m128i v3; // [rsp+0h] [rbp-30h] BYREF
  __m128i v4; // [rsp+10h] [rbp-20h] BYREF

  a1[8].m128i_i16[0] = 257;
  result = sub_16D7810((double *)v3.m128i_i64, 1);
  v2 = _mm_loadu_si128(&v4);
  a1[2] = _mm_loadu_si128(&v3);
  a1[3] = v2;
  return result;
}
