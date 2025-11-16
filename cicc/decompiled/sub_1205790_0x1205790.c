// Function: sub_1205790
// Address: 0x1205790
//
__int64 __fastcall sub_1205790(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  __int64 v4; // rax
  __m128i v5; // xmm1
  __int64 result; // rax
  __m128i *v7; // rdi
  __m128i v8; // [rsp-48h] [rbp-48h] BYREF
  __int64 v9; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = (__m128i *)((char *)src + 24); a2 != i; *src = v5 )
    {
      while ( (i->m128i_i32[0] & 6u) >= (src->m128i_i32[0] & 6u) )
      {
        v7 = (__m128i *)i;
        i = (const __m128i *)((char *)i + 24);
        result = sub_1205720(v7);
        if ( a2 == i )
          return result;
      }
      v4 = i[1].m128i_i64[0];
      v8 = _mm_loadu_si128(i);
      v9 = v4;
      if ( src != i )
        memmove(&src[1].m128i_u64[1], src, (char *)i - (char *)src);
      v5 = _mm_loadu_si128(&v8);
      result = v9;
      i = (const __m128i *)((char *)i + 24);
      src[1].m128i_i64[0] = v9;
    }
  }
  return result;
}
