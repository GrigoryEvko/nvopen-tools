// Function: sub_3510CA0
// Address: 0x3510ca0
//
void __fastcall sub_3510CA0(unsigned __int64 *src, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // rbx
  char *v3; // r13
  __int64 v4; // rcx
  unsigned __int64 v5; // r12
  __int64 *v6; // rdx
  __int64 v7; // r15
  const __m128i *v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v2 = src + 3;
    if ( a2 != src + 3 )
    {
      v3 = (char *)src + (((char *)(a2 - 6) - (char *)src) & 0xFFFFFFFFFFFFFFF8LL) + 48;
      do
      {
        while ( 1 )
        {
          v5 = *v2;
          v4 = v2[1];
          v6 = (__int64 *)v2;
          v7 = v2[2];
          if ( *v2 > *src )
            break;
          v8 = (const __m128i *)(v2 - 3);
          if ( v5 > *(v2 - 3) )
          {
            do
            {
              v9 = _mm_loadu_si128(v8);
              v8[2].m128i_i64[1] = v8[1].m128i_i64[0];
              v6 = (__int64 *)v8;
              v8 = (const __m128i *)((char *)v8 - 24);
              v8[3] = v9;
            }
            while ( v5 > v8->m128i_i64[0] );
          }
          v2 += 3;
          *v6 = v5;
          v6[1] = v4;
          v6[2] = v7;
          if ( v2 == (unsigned __int64 *)v3 )
            return;
        }
        if ( src != v2 )
        {
          v10 = v2[1];
          memmove(src + 3, src, (char *)v2 - (char *)src);
          v4 = v10;
        }
        v2 += 3;
        *src = v5;
        src[1] = v4;
        src[2] = v7;
      }
      while ( v2 != (unsigned __int64 *)v3 );
    }
  }
}
