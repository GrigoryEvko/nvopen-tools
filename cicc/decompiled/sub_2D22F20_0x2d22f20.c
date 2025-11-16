// Function: sub_2D22F20
// Address: 0x2d22f20
//
void __fastcall sub_2D22F20(__m128i *src, const __m128i *a2)
{
  const __m128i *v3; // rbx
  unsigned __int64 v4; // rax
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i v7; // xmm3
  __int64 v8; // rax
  __int64 v9; // rdi
  __m128i v10; // [rsp-58h] [rbp-58h] BYREF
  __m128i v11; // [rsp-48h] [rbp-48h] BYREF
  __int64 v12; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    v3 = (__m128i *)((char *)src + 40);
    while ( a2 != v3 )
    {
      while ( src[1].m128i_i8[8] )
      {
        v4 = src->m128i_u64[1];
        if ( !v3[1].m128i_i8[8] )
          goto LABEL_11;
LABEL_5:
        if ( v3->m128i_i64[1] < v4 )
          goto LABEL_6;
LABEL_12:
        v9 = (__int64)v3;
        v3 = (const __m128i *)((char *)v3 + 40);
        sub_2D22E80(v9);
        if ( a2 == v3 )
          return;
      }
      v4 = qword_4F81350[0];
      if ( v3[1].m128i_i8[8] )
        goto LABEL_5;
LABEL_11:
      if ( qword_4F81350[0] >= v4 )
        goto LABEL_12;
LABEL_6:
      v5 = _mm_loadu_si128(v3 + 1);
      v6 = v3[2].m128i_i64[0];
      v10 = _mm_loadu_si128(v3);
      v12 = v6;
      v11 = v5;
      if ( src != v3 )
        memmove(&src[2].m128i_u64[1], src, (char *)v3 - (char *)src);
      v7 = _mm_loadu_si128(&v11);
      v3 = (const __m128i *)((char *)v3 + 40);
      v8 = v12;
      *src = _mm_loadu_si128(&v10);
      src[2].m128i_i64[0] = v8;
      src[1] = v7;
    }
  }
}
