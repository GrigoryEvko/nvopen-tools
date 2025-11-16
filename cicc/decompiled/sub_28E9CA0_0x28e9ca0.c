// Function: sub_28E9CA0
// Address: 0x28e9ca0
//
void __fastcall sub_28E9CA0(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  __m128i v3; // xmm0
  unsigned __int32 v4; // ecx
  __m128i *v5; // rdx
  const __m128i *v6; // rax
  __m128i v7; // [rsp-58h] [rbp-58h] BYREF
  __m128i v8; // [rsp-48h] [rbp-48h]
  __m128i v9; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = src + 1; i != a2; src->m128i_i32[2] = v9.m128i_i32[2] )
    {
      while ( 1 )
      {
        v4 = i->m128i_u32[2];
        v5 = (__m128i *)i;
        v3 = _mm_loadu_si128(i);
        if ( v4 > src->m128i_i32[2] )
          break;
        v6 = i - 1;
        if ( v4 > i[-1].m128i_i32[2] )
        {
          do
          {
            v6[1].m128i_i64[0] = v6->m128i_i64[0];
            v6[1].m128i_i32[2] = v6->m128i_i32[2];
            v5 = (__m128i *)v6--;
          }
          while ( v4 > v6->m128i_i32[2] );
        }
        v8 = v3;
        ++i;
        v5->m128i_i64[0] = v3.m128i_i64[0];
        v5->m128i_i32[2] = v8.m128i_i32[2];
        if ( i == a2 )
          return;
      }
      if ( src != i )
      {
        v7 = v3;
        memmove(&src[1], src, (char *)i - (char *)src);
        v3 = _mm_load_si128(&v7);
      }
      v9 = v3;
      ++i;
      src->m128i_i64[0] = v3.m128i_i64[0];
    }
  }
}
