// Function: sub_1076E90
// Address: 0x1076e90
//
char *__fastcall sub_1076E90(const __m128i *src, const __m128i *a2, const __m128i *a3, const __m128i *a4, char *a5)
{
  __m128i v7; // xmm0
  __m128i v8; // xmm2
  signed __int64 v9; // rbx

  if ( src == a2 )
  {
LABEL_7:
    v9 = (char *)a4 - (char *)a3;
    if ( a4 != a3 )
      return (char *)memmove(a5, a3, (char *)a4 - (char *)a3) + v9;
  }
  else
  {
    while ( a4 != a3 )
    {
      if ( *(_QWORD *)(a3[2].m128i_i64[0] + 160) + a3->m128i_i64[0] < (unsigned __int64)(*(_QWORD *)(src[2].m128i_i64[0] + 160)
                                                                                       + src->m128i_i64[0]) )
      {
        v7 = _mm_loadu_si128(a3);
        a5 += 40;
        a3 = (const __m128i *)((char *)a3 + 40);
        *(__m128i *)(a5 - 40) = v7;
        *(__m128i *)(a5 - 24) = _mm_loadu_si128((const __m128i *)((char *)a3 - 24));
        *((_QWORD *)a5 - 1) = a3[-1].m128i_i64[1];
        if ( src == a2 )
          goto LABEL_7;
      }
      else
      {
        v8 = _mm_loadu_si128(src);
        src = (const __m128i *)((char *)src + 40);
        a5 += 40;
        *(__m128i *)(a5 - 40) = v8;
        *(__m128i *)(a5 - 24) = _mm_loadu_si128((const __m128i *)((char *)src - 24));
        *((_QWORD *)a5 - 1) = src[-1].m128i_i64[1];
        if ( src == a2 )
          goto LABEL_7;
      }
    }
    a5 = (char *)memmove(a5, src, (char *)a2 - (char *)src) + (char *)a2 - (char *)src;
    v9 = 0;
  }
  return &a5[v9];
}
