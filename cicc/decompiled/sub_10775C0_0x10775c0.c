// Function: sub_10775C0
// Address: 0x10775c0
//
char *__fastcall sub_10775C0(const __m128i *src, const __m128i *a2, const __m128i *a3, const __m128i *a4, _QWORD *a5)
{
  __m128i v7; // xmm0
  __m128i v8; // xmm2
  size_t v9; // r13
  char *v10; // r8

  if ( a2 != src )
  {
    while ( a4 != a3 )
    {
      if ( *(_QWORD *)(a3[2].m128i_i64[0] + 160) + a3->m128i_i64[0] < (unsigned __int64)(*(_QWORD *)(src[2].m128i_i64[0] + 160)
                                                                                       + src->m128i_i64[0]) )
      {
        v7 = _mm_loadu_si128(a3);
        a5 += 5;
        a3 = (const __m128i *)((char *)a3 + 40);
        *(__m128i *)(a5 - 5) = v7;
        *(__m128i *)(a5 - 3) = _mm_loadu_si128((const __m128i *)((char *)a3 - 24));
        *(a5 - 1) = a3[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
      else
      {
        v8 = _mm_loadu_si128(src);
        src = (const __m128i *)((char *)src + 40);
        a5 += 5;
        *(__m128i *)(a5 - 5) = v8;
        *(__m128i *)(a5 - 3) = _mm_loadu_si128((const __m128i *)((char *)src - 24));
        *(a5 - 1) = src[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
    }
  }
  v9 = (char *)a2 - (char *)src;
  if ( a2 != src )
    a5 = memmove(a5, src, v9);
  v10 = (char *)a5 + v9;
  if ( a4 != a3 )
    v10 = (char *)memmove(v10, a3, (char *)a4 - (char *)a3);
  return &v10[(char *)a4 - (char *)a3];
}
