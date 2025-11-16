// Function: sub_33772F0
// Address: 0x33772f0
//
__m128i *__fastcall sub_33772F0(unsigned __int64 *a1, __m128i *a2, const __m128i *a3)
{
  __m128i *result; // rax
  __m128i *v5; // rdi
  __int8 *v6; // r13
  __m128i v7; // xmm7
  __m128i v8; // xmm6
  __m128i v9; // xmm4
  __m128i v10; // xmm5
  __m128i v11; // [rsp+8h] [rbp-58h] BYREF
  __m128i v12; // [rsp+18h] [rbp-48h] BYREF
  __m128i v13[3]; // [rsp+28h] [rbp-38h] BYREF

  result = a2;
  v5 = (__m128i *)a1[1];
  v6 = &a2->m128i_i8[-*a1];
  if ( v5 == (__m128i *)a1[2] )
  {
    sub_332CDC0(a1, a2, a3);
    return (__m128i *)&v6[*a1];
  }
  else if ( v5 == a2 )
  {
    if ( v5 )
    {
      *v5 = _mm_loadu_si128(a3);
      v5[1] = _mm_loadu_si128(a3 + 1);
      v5[2] = _mm_loadu_si128(a3 + 2);
      v5 = (__m128i *)a1[1];
      result = (__m128i *)&v6[*a1];
    }
    a1[1] = (unsigned __int64)&v5[3];
  }
  else
  {
    v11 = _mm_loadu_si128(a3);
    v12 = _mm_loadu_si128(a3 + 1);
    v13[0] = _mm_loadu_si128(a3 + 2);
    if ( v5 )
    {
      v7 = _mm_loadu_si128(v5 - 2);
      *v5 = _mm_loadu_si128(v5 - 3);
      v8 = _mm_loadu_si128(v5 - 1);
      v5[1] = v7;
      v5[2] = v8;
      v5 = (__m128i *)a1[1];
    }
    a1[1] = (unsigned __int64)&v5[3];
    if ( a2 != &v5[-3] )
      memmove(&a2[3], a2, (char *)&v5[-3] - (char *)a2);
    v9 = _mm_loadu_si128(&v12);
    v10 = _mm_loadu_si128(v13);
    *a2 = _mm_loadu_si128(&v11);
    a2[1] = v9;
    a2[2] = v10;
    return (__m128i *)&v6[*a1];
  }
  return result;
}
