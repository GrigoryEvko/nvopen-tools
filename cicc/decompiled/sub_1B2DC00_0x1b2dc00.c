// Function: sub_1B2DC00
// Address: 0x1b2dc00
//
char *__fastcall sub_1B2DC00(
        const __m128i *src,
        const __m128i *a2,
        const __m128i *a3,
        const __m128i *a4,
        __m128i *a5,
        __int64 a6)
{
  const __m128i *v6; // r13
  const __m128i *v7; // r12
  __m128i v9; // xmm0
  __m128i v10; // xmm3
  __int8 *v11; // rbx
  __int64 v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = src;
  v7 = a3;
  v14[0] = a6;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      if ( sub_1B2B020(v14, (__int64)v7, (__int64)v6) )
      {
        v9 = _mm_loadu_si128(v7);
        a5 += 3;
        v7 += 3;
        a5[-3] = v9;
        a5[-2] = _mm_loadu_si128(v7 - 2);
        a5[-1] = _mm_loadu_si128(v7 - 1);
        if ( v6 == a2 )
          break;
      }
      else
      {
        v10 = _mm_loadu_si128(v6);
        v6 += 3;
        a5 += 3;
        a5[-3] = v10;
        a5[-2] = _mm_loadu_si128(v6 - 2);
        a5[-1] = _mm_loadu_si128(v6 - 1);
        if ( v6 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  if ( a2 != v6 )
    memmove(a5, v6, (char *)a2 - (char *)v6);
  v11 = &a5->m128i_i8[(char *)a2 - (char *)v6];
  if ( a4 != v7 )
    memmove(v11, v7, (char *)a4 - (char *)v7);
  return &v11[(char *)a4 - (char *)v7];
}
