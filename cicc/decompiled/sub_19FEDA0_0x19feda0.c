// Function: sub_19FEDA0
// Address: 0x19feda0
//
void __fastcall sub_19FEDA0(__m128i *src, __m128i *a2)
{
  __m128i *i; // rbx
  unsigned __int32 v3; // r12d
  __int64 v4; // r15
  __m128i *v5; // rdx
  const __m128i *j; // rax
  __m128i v7; // xmm0

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; src->m128i_i64[1] = v4 )
    {
      while ( 1 )
      {
        v3 = i->m128i_i32[0];
        v4 = i->m128i_i64[1];
        v5 = i;
        if ( i->m128i_i32[0] > (unsigned __int32)src->m128i_i32[0] )
          break;
        for ( j = i - 1; v3 > j->m128i_i32[0]; j[2] = v7 )
        {
          v7 = _mm_loadu_si128(j);
          v5 = (__m128i *)j--;
        }
        ++i;
        v5->m128i_i32[0] = v3;
        v5->m128i_i64[1] = v4;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(&src[1], src, (char *)i - (char *)src);
      ++i;
      src->m128i_i32[0] = v3;
    }
  }
}
