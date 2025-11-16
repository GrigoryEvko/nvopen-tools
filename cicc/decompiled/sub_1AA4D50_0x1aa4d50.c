// Function: sub_1AA4D50
// Address: 0x1aa4d50
//
char *__fastcall sub_1AA4D50(
        const __m128i *src,
        const __m128i *a2,
        const __m128i *a3,
        const __m128i *a4,
        _QWORD *a5,
        unsigned __int8 (__fastcall *a6)(const __m128i *, const __m128i *))
{
  const __m128i *v6; // r14
  const __m128i *v7; // r12
  __m128i v10; // xmm0
  __m128i v11; // xmm3
  char *v12; // rbx

  v6 = a3;
  v7 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      if ( a6(v6, v7) )
      {
        v10 = _mm_loadu_si128(v6);
        a5 += 7;
        v6 = (const __m128i *)((char *)v6 + 56);
        *(__m128i *)(a5 - 7) = v10;
        *(__m128i *)(a5 - 5) = _mm_loadu_si128((const __m128i *)((char *)v6 - 40));
        *(__m128i *)(a5 - 3) = _mm_loadu_si128((const __m128i *)((char *)v6 - 24));
        *(a5 - 1) = v6[-1].m128i_i64[1];
        if ( v7 == a2 )
          break;
      }
      else
      {
        v11 = _mm_loadu_si128(v7);
        v7 = (const __m128i *)((char *)v7 + 56);
        a5 += 7;
        *(__m128i *)(a5 - 7) = v11;
        *(__m128i *)(a5 - 5) = _mm_loadu_si128((const __m128i *)((char *)v7 - 40));
        *(__m128i *)(a5 - 3) = _mm_loadu_si128((const __m128i *)((char *)v7 - 24));
        *(a5 - 1) = v7[-1].m128i_i64[1];
        if ( v7 == a2 )
          break;
      }
    }
    while ( v6 != a4 );
  }
  if ( a2 != v7 )
    memmove(a5, v7, (char *)a2 - (char *)v7);
  v12 = (char *)a5 + (char *)a2 - (char *)v7;
  if ( a4 != v6 )
    memmove(v12, v6, (char *)a4 - (char *)v6);
  return &v12[(char *)a4 - (char *)v6];
}
