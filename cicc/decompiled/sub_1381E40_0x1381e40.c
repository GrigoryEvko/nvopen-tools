// Function: sub_1381E40
// Address: 0x1381e40
//
void __fastcall sub_1381E40(char *src, char *a2)
{
  char *v2; // rdx
  __m128i *i; // rax
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // r13
  __int64 v6; // r14
  char *v7; // r12
  __m128i v8; // xmm0

  if ( src != a2 )
  {
    v2 = src + 16;
    if ( a2 != src + 16 )
    {
      do
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)v2;
          v6 = *((_QWORD *)v2 + 1);
          if ( *(_QWORD *)src > *(_QWORD *)v2 || v6 < *((_QWORD *)src + 1) && v5 == *(_QWORD *)src )
            break;
          for ( i = (__m128i *)v2; ; i[1] = v8 )
          {
            v4 = i[-1].m128i_u64[0];
            if ( v4 <= v5 && (v6 >= i[-1].m128i_i64[1] || v5 != v4) )
              break;
            v8 = _mm_loadu_si128(--i);
          }
          i->m128i_i64[0] = v5;
          i->m128i_i64[1] = v6;
          v2 += 16;
          if ( a2 == v2 )
            return;
        }
        v7 = v2 + 16;
        if ( src != v2 )
          memmove(src + 16, src, v2 - src);
        *(_QWORD *)src = v5;
        v2 = v7;
        *((_QWORD *)src + 1) = v6;
      }
      while ( a2 != v7 );
    }
  }
}
