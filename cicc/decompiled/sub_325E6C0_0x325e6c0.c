// Function: sub_325E6C0
// Address: 0x325e6c0
//
void __fastcall sub_325E6C0(__int64 *src, __int64 *a2)
{
  __int64 *i; // rbx
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 *v5; // rdx
  const __m128i *v6; // rax
  __m128i v7; // xmm0

  if ( src != a2 )
  {
    for ( i = src + 2; a2 != i; src[1] = v3 )
    {
      while ( 1 )
      {
        v3 = i[1];
        v4 = *i;
        v5 = i;
        if ( v3 < src[1] )
          break;
        v6 = (const __m128i *)(i - 2);
        if ( v3 < *(i - 1) )
        {
          do
          {
            v7 = _mm_loadu_si128(v6);
            v5 = (__int64 *)v6--;
            v6[2] = v7;
          }
          while ( v3 < v6->m128i_i64[1] );
        }
        i += 2;
        *v5 = v4;
        v5[1] = v3;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 2, src, (char *)i - (char *)src);
      i += 2;
      *src = v4;
    }
  }
}
