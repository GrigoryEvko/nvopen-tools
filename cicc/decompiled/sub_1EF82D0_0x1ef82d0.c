// Function: sub_1EF82D0
// Address: 0x1ef82d0
//
__m128i *__fastcall sub_1EF82D0(
        const __m128i *a1,
        const __m128i *a2,
        const __m128i *a3,
        const __m128i *a4,
        __m128i *a5)
{
  const __m128i *v5; // r15
  const __m128i *v6; // rbx
  __m128i *v7; // r14
  __m128i *v8; // r13
  unsigned __int64 v9; // r14
  const __m128i *v10; // rbx
  __m128i *v11; // r12
  __int64 v12; // rsi
  __int64 v13; // r14
  unsigned __int64 v14; // rbx
  const __m128i *v15; // r15
  __m128i *v16; // r12
  __int64 v18; // [rsp+0h] [rbp-40h]

  v5 = a3;
  v6 = a1;
  if ( a2 != a1 && a4 != a3 )
  {
    v7 = a5 + 1;
    while ( 1 )
    {
      if ( v5->m128i_i32[2] > (unsigned __int32)v6->m128i_i32[2] )
      {
        v7[-1].m128i_i64[0] = v5->m128i_i64[0];
        v7[-1].m128i_i32[2] = v5->m128i_i32[2];
        v7[-1].m128i_i32[3] = v5->m128i_i32[3];
        if ( v7 != &v5[1] )
        {
          _libc_free(v7->m128i_i64[0]);
          *v7 = _mm_loadu_si128(v5 + 1);
          v7[1].m128i_i32[0] = v5[2].m128i_i32[0];
          v5[1].m128i_i64[0] = 0;
          v5[1].m128i_i64[1] = 0;
          v5[2].m128i_i32[0] = 0;
        }
        v8 = (__m128i *)((char *)v7 + 24);
        v5 = (const __m128i *)((char *)v5 + 40);
        v7 = (__m128i *)((char *)v7 + 40);
        if ( a2 == v6 )
          goto LABEL_12;
      }
      else
      {
        v7[-1].m128i_i64[0] = v6->m128i_i64[0];
        v7[-1].m128i_i32[2] = v6->m128i_i32[2];
        v7[-1].m128i_i32[3] = v6->m128i_i32[3];
        if ( v7 != &v6[1] )
        {
          _libc_free(v7->m128i_i64[0]);
          *v7 = _mm_loadu_si128(v6 + 1);
          v7[1].m128i_i32[0] = v6[2].m128i_i32[0];
          v6[1].m128i_i64[0] = 0;
          v6[1].m128i_i64[1] = 0;
          v6[2].m128i_i32[0] = 0;
        }
        v6 = (const __m128i *)((char *)v6 + 40);
        v8 = (__m128i *)((char *)v7 + 24);
        v7 = (__m128i *)((char *)v7 + 40);
        if ( a2 == v6 )
          goto LABEL_12;
      }
      if ( a4 == v5 )
        goto LABEL_12;
    }
  }
  v8 = a5;
LABEL_12:
  v18 = (char *)a2 - (char *)v6;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)v6) >> 3);
  if ( (char *)a2 - (char *)v6 > 0 )
  {
    v10 = v6 + 1;
    v11 = v8 + 1;
    do
    {
      v11[-1].m128i_i64[0] = v10[-1].m128i_i64[0];
      v11[-1].m128i_i32[2] = v10[-1].m128i_i32[2];
      v11[-1].m128i_i32[3] = v10[-1].m128i_i32[3];
      if ( v10 != v11 )
      {
        _libc_free(v11->m128i_i64[0]);
        *v11 = _mm_loadu_si128(v10);
        v11[1].m128i_i32[0] = v10[1].m128i_i32[0];
        v10->m128i_i64[0] = 0;
        v10->m128i_i64[1] = 0;
        v10[1].m128i_i32[0] = 0;
      }
      v10 = (const __m128i *)((char *)v10 + 40);
      v11 = (__m128i *)((char *)v11 + 40);
      --v9;
    }
    while ( v9 );
    v12 = v18;
    if ( v18 <= 0 )
      v12 = 40;
    v8 = (__m128i *)((char *)v8 + v12);
  }
  v13 = (char *)a4 - (char *)v5;
  v14 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a4 - (char *)v5) >> 3);
  if ( (char *)a4 - (char *)v5 > 0 )
  {
    v15 = v5 + 1;
    v16 = v8 + 1;
    do
    {
      v16[-1].m128i_i64[0] = v15[-1].m128i_i64[0];
      v16[-1].m128i_i32[2] = v15[-1].m128i_i32[2];
      v16[-1].m128i_i32[3] = v15[-1].m128i_i32[3];
      if ( v15 != v16 )
      {
        _libc_free(v16->m128i_i64[0]);
        *v16 = _mm_loadu_si128(v15);
        v16[1].m128i_i32[0] = v15[1].m128i_i32[0];
        v15->m128i_i64[0] = 0;
        v15->m128i_i64[1] = 0;
        v15[1].m128i_i32[0] = 0;
      }
      v15 = (const __m128i *)((char *)v15 + 40);
      v16 = (__m128i *)((char *)v16 + 40);
      --v14;
    }
    while ( v14 );
    if ( v13 <= 0 )
      v13 = 40;
    return (__m128i *)((char *)v8 + v13);
  }
  return v8;
}
