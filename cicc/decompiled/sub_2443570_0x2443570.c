// Function: sub_2443570
// Address: 0x2443570
//
char *__fastcall sub_2443570(const __m128i *src, const __m128i *a2, const __m128i *a3, const __m128i *a4, __m128i *a5)
{
  const __m128i *v5; // r12
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  size_t v9; // r13
  __int8 *v10; // r8

  v5 = a3;
  if ( a4 != a3 && a2 != src )
  {
    do
    {
      if ( v5->m128i_i64[1] > (unsigned __int64)src->m128i_i64[1] )
      {
        v7 = _mm_loadu_si128(v5);
        ++a5;
        ++v5;
        a5[-1] = v7;
        if ( a2 == src )
          break;
      }
      else
      {
        v8 = _mm_loadu_si128(src++);
        ++a5;
        a5[-1] = v8;
        if ( a2 == src )
          break;
      }
    }
    while ( a4 != v5 );
  }
  v9 = (char *)a2 - (char *)src;
  if ( a2 != src )
    a5 = (__m128i *)memmove(a5, src, v9);
  v10 = &a5->m128i_i8[v9];
  if ( a4 != v5 )
    v10 = (__int8 *)memmove(v10, v5, (char *)a4 - (char *)v5);
  return &v10[(char *)a4 - (char *)v5];
}
