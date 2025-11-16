// Function: sub_1029CF0
// Address: 0x1029cf0
//
void __fastcall sub_1029CF0(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  __m128i v3; // xmm0
  unsigned __int64 v4; // rdx
  __m128i *v5; // rcx
  const __m128i *j; // rax
  __m128i v7; // xmm1
  __m128i v8; // [rsp-38h] [rbp-38h] BYREF

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v3 )
    {
      while ( 1 )
      {
        v4 = i->m128i_i64[0];
        v5 = (__m128i *)i;
        v3 = _mm_loadu_si128(i);
        if ( i->m128i_i64[0] < (unsigned __int64)src->m128i_i64[0] )
          break;
        for ( j = i - 1; v4 < j->m128i_i64[0]; j[2] = v7 )
        {
          v7 = _mm_loadu_si128(j);
          v5 = (__m128i *)j--;
        }
        ++i;
        *v5 = v3;
        if ( a2 == i )
          return;
      }
      if ( src != i )
      {
        v8 = v3;
        memmove(&src[1], src, (char *)i - (char *)src);
        v3 = _mm_load_si128(&v8);
      }
      ++i;
    }
  }
}
