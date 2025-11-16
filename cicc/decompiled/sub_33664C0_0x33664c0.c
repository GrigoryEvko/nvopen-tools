// Function: sub_33664C0
// Address: 0x33664c0
//
void __fastcall sub_33664C0(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  __m128i v4; // xmm1
  __int64 v5; // rax
  __m128i v6; // xmm3
  __int32 v7; // eax
  unsigned int v8; // eax
  const __m128i *v9; // rdi
  __m128i v10; // [rsp-58h] [rbp-58h] BYREF
  __m128i v11; // [rsp-48h] [rbp-48h] BYREF
  __int64 v12; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = (__m128i *)((char *)src + 40); a2 != i; src[1] = v6 )
    {
      while ( 1 )
      {
        v8 = i[2].m128i_u32[0];
        if ( src[2].m128i_i32[0] == v8 )
          v8 = (unsigned int)sub_C4C880(i->m128i_i64[1] + 24, src->m128i_i64[1] + 24) >> 31;
        else
          LOBYTE(v8) = src[2].m128i_i32[0] < v8;
        if ( (_BYTE)v8 )
          break;
        v9 = i;
        i = (const __m128i *)((char *)i + 40);
        sub_3365BE0(v9);
        if ( a2 == i )
          return;
      }
      v4 = _mm_loadu_si128(i + 1);
      v5 = i[2].m128i_i64[0];
      v10 = _mm_loadu_si128(i);
      v12 = v5;
      v11 = v4;
      if ( src != i )
        memmove(&src[2].m128i_u64[1], src, (char *)i - (char *)src);
      v6 = _mm_loadu_si128(&v11);
      i = (const __m128i *)((char *)i + 40);
      v7 = v12;
      *src = _mm_loadu_si128(&v10);
      src[2].m128i_i32[0] = v7;
    }
  }
}
