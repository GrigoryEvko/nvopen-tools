// Function: sub_396B8C0
// Address: 0x396b8c0
//
char *__fastcall sub_396B8C0(const __m128i *src, const __m128i *a2, const __m128i *a3, const __m128i *a4, _QWORD *a5)
{
  const __m128i *v5; // r12
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  size_t v9; // r13
  char *v10; // r8

  v5 = a3;
  if ( a4 != a3 && a2 != src )
  {
    do
    {
      if ( v5->m128i_i32[0] < src->m128i_i32[0] )
      {
        v7 = _mm_loadu_si128(v5);
        a5 += 3;
        v5 = (const __m128i *)((char *)v5 + 24);
        *(__m128i *)(a5 - 3) = v7;
        *(a5 - 1) = v5[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
      else
      {
        v8 = _mm_loadu_si128(src);
        src = (const __m128i *)((char *)src + 24);
        a5 += 3;
        *(__m128i *)(a5 - 3) = v8;
        *(a5 - 1) = src[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
    }
    while ( a4 != v5 );
  }
  v9 = (char *)a2 - (char *)src;
  if ( a2 != src )
    a5 = memmove(a5, src, v9);
  v10 = (char *)a5 + v9;
  if ( a4 != v5 )
    v10 = (char *)memmove(v10, v5, (char *)a4 - (char *)v5);
  return &v10[(char *)a4 - (char *)v5];
}
