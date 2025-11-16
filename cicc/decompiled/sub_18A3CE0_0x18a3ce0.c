// Function: sub_18A3CE0
// Address: 0x18a3ce0
//
void __fastcall sub_18A3CE0(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r15
  const __m128i *j; // rax
  unsigned __int64 v7; // rdx
  __int64 *v8; // rsi
  bool v9; // cl

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; src->m128i_i64[1] = v4 )
    {
      while ( 1 )
      {
        v4 = i->m128i_u64[1];
        v5 = i->m128i_i64[0];
        if ( *(_OWORD *)src < *(_OWORD *)i )
          break;
        for ( j = i; ; j[1] = _mm_loadu_si128(j) )
        {
          v7 = j[-1].m128i_u64[1];
          v8 = (__int64 *)j;
          v9 = v4 > v7;
          if ( v4 == v7 )
            v9 = j[-1].m128i_i64[0] < v5;
          --j;
          if ( !v9 )
            break;
        }
        ++i;
        *v8 = v5;
        v8[1] = v4;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(&src[1], src, (char *)i - (char *)src);
      ++i;
      src->m128i_i64[0] = v5;
    }
  }
}
