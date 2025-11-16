// Function: sub_26BA340
// Address: 0x26ba340
//
void __fastcall sub_26BA340(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // r12
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r14
  const __m128i *j; // rax
  bool v6; // dl
  __int64 *v7; // rcx

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; src->m128i_i64[1] = v3 )
    {
      while ( 1 )
      {
        v3 = i->m128i_u64[1];
        v4 = i->m128i_i64[0];
        if ( *(_OWORD *)src < *(_OWORD *)i )
          break;
        for ( j = i; ; j[1] = _mm_loadu_si128(j) )
        {
          v7 = (__int64 *)j;
          v6 = v3 == j[-1].m128i_i64[1] ? j[-1].m128i_i64[0] < v4 : v3 > j[-1].m128i_i64[1];
          --j;
          if ( !v6 )
            break;
        }
        ++i;
        *v7 = v4;
        v7[1] = v3;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(&src[1], src, (char *)i - (char *)src);
      ++i;
      src->m128i_i64[0] = v4;
    }
  }
}
