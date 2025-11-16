// Function: sub_35D7A00
// Address: 0x35d7a00
//
void __fastcall sub_35D7A00(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  const __m128i *v4; // rdi
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i v7; // xmm3
  __int32 v8; // eax
  __m128i v9; // [rsp-58h] [rbp-58h] BYREF
  __m128i v10; // [rsp-48h] [rbp-48h] BYREF
  __int64 v11; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = (__m128i *)((char *)src + 40); a2 != i; src[1] = v7 )
    {
      while ( (int)sub_C4C880(i->m128i_i64[1] + 24, src->m128i_i64[1] + 24) >= 0 )
      {
        v4 = i;
        i = (const __m128i *)((char *)i + 40);
        sub_35D7970(v4);
        if ( a2 == i )
          return;
      }
      v5 = _mm_loadu_si128(i + 1);
      v6 = i[2].m128i_i64[0];
      v9 = _mm_loadu_si128(i);
      v11 = v6;
      v10 = v5;
      if ( src != i )
        memmove(&src[2].m128i_u64[1], src, (char *)i - (char *)src);
      v7 = _mm_loadu_si128(&v10);
      i = (const __m128i *)((char *)i + 40);
      v8 = v11;
      *src = _mm_loadu_si128(&v9);
      src[2].m128i_i32[0] = v8;
    }
  }
}
