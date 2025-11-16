// Function: sub_2912FB0
// Address: 0x2912fb0
//
char *__fastcall sub_2912FB0(const __m128i *src, const __m128i *a2, const __m128i *a3, const __m128i *a4, _QWORD *a5)
{
  __int64 v7; // rax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  size_t v10; // r13
  char *v11; // r8

  if ( a2 != src )
  {
    while ( a4 != a3 )
    {
      if ( a3->m128i_i64[0] < (unsigned __int64)src->m128i_i64[0] )
        goto LABEL_5;
      if ( a3->m128i_i64[0] > (unsigned __int64)src->m128i_i64[0] )
        goto LABEL_9;
      v7 = (a3[1].m128i_i64[0] >> 2) & 1;
      if ( (_BYTE)v7 == ((src[1].m128i_i64[0] >> 2) & 1) )
      {
        if ( a3->m128i_i64[1] <= (unsigned __int64)src->m128i_i64[1] )
          goto LABEL_9;
LABEL_5:
        v8 = _mm_loadu_si128(a3);
        a5 += 3;
        a3 = (const __m128i *)((char *)a3 + 24);
        *(__m128i *)(a5 - 3) = v8;
        *(a5 - 1) = a3[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
      else
      {
        if ( !(_BYTE)v7 )
          goto LABEL_5;
LABEL_9:
        v9 = _mm_loadu_si128(src);
        src = (const __m128i *)((char *)src + 24);
        a5 += 3;
        *(__m128i *)(a5 - 3) = v9;
        *(a5 - 1) = src[-1].m128i_i64[1];
        if ( a2 == src )
          break;
      }
    }
  }
  v10 = (char *)a2 - (char *)src;
  if ( a2 != src )
    a5 = memmove(a5, src, v10);
  v11 = (char *)a5 + v10;
  if ( a4 != a3 )
    v11 = (char *)memmove(v11, a3, (char *)a4 - (char *)a3);
  return &v11[(char *)a4 - (char *)a3];
}
