// Function: sub_29133D0
// Address: 0x29133d0
//
unsigned __int64 __fastcall sub_29133D0(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rbp
  const __m128i *v5; // r11
  const __m128i *v6; // r10
  __int64 v7; // r11
  signed __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r9
  const __m128i *v11; // rcx
  const __m128i *v12; // rsi
  __int64 v13; // rdx
  __m128i v14; // xmm1
  __m128i v15; // xmm0
  __int64 v16; // r8
  __int64 v17; // rbx
  __m128i *v18; // rcx
  __m128i *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r8
  __m128i v22; // xmm0
  __m128i v23; // xmm2
  __int64 v24; // rbx
  const __m128i *v26; // rdx
  const __m128i *v27; // rax
  __m128i v28; // xmm3
  __m128i v29; // xmm0
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 v32; // [rsp-10h] [rbp-10h]
  __int64 v33; // [rsp-8h] [rbp-8h]

  v5 = a3;
  if ( a1 == a2 )
    return (unsigned __int64)v5;
  v6 = a1;
  if ( a2 == a3 )
    return (unsigned __int64)a1;
  v33 = v4;
  v7 = (__int64)a1->m128i_i64 + (char *)a3 - (char *)a2;
  v8 = 0xAAAAAAAAAAAAAAABLL * (((char *)a3 - (char *)a1) >> 3);
  v32 = v3;
  v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  if ( v9 == v8 - v9 )
  {
    v26 = a2;
    v27 = v6;
    do
    {
      v28 = _mm_loadu_si128(v26);
      v29 = _mm_loadu_si128(v27);
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (const __m128i *)((char *)v26 + 24);
      v30 = v27[-1].m128i_i64[1];
      *(__m128i *)((char *)v27 - 24) = v28;
      v31 = v26[-1].m128i_i64[1];
      *(&v33 - 4) = v30;
      v27[-1].m128i_i64[1] = v31;
      *((__m128i *)&v33 - 3) = v29;
      *(__m128i *)((char *)v26 - 24) = v29;
      v26[-1].m128i_i64[1] = v30;
    }
    while ( a2 != v27 );
    return (unsigned __int64)&v6[1].m128i_u64[((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v6) >> 3) + 1];
  }
  else
  {
    v10 = v8 - v9;
    if ( v9 >= v8 - v9 )
      goto LABEL_12;
    while ( 1 )
    {
      v11 = (const __m128i *)((char *)v6 + 24 * v9);
      if ( v10 > 0 )
      {
        v12 = v6;
        v13 = 0;
        do
        {
          v14 = _mm_loadu_si128(v11);
          v15 = _mm_loadu_si128(v12);
          ++v13;
          v12 = (const __m128i *)((char *)v12 + 24);
          v16 = v12[-1].m128i_i64[1];
          v11 = (const __m128i *)((char *)v11 + 24);
          *(__m128i *)((char *)v12 - 24) = v14;
          v17 = v11[-1].m128i_i64[1];
          *(&v33 - 4) = v16;
          v12[-1].m128i_i64[1] = v17;
          *((__m128i *)&v33 - 3) = v15;
          *(__m128i *)((char *)v11 - 24) = v15;
          v11[-1].m128i_i64[1] = v16;
        }
        while ( v10 != v13 );
        v6 = (const __m128i *)((char *)v6 + 24 * v10);
      }
      if ( !(v8 % v9) )
        break;
      v10 = v9;
      v9 -= v8 % v9;
      while ( 1 )
      {
        v8 = v10;
        v10 -= v9;
        if ( v9 < v10 )
          break;
LABEL_12:
        v18 = (__m128i *)((char *)v6 + 24 * v8);
        v6 = (__m128i *)((char *)v18 - 24 * v10);
        if ( v9 > 0 )
        {
          v19 = (__m128i *)((char *)v18 - 24 * v10);
          v20 = 0;
          do
          {
            v21 = v19[-1].m128i_i64[1];
            ++v20;
            v19 = (__m128i *)((char *)v19 - 24);
            v22 = _mm_loadu_si128(v19);
            v23 = _mm_loadu_si128((__m128i *)((char *)v18 - 24));
            v18 = (__m128i *)((char *)v18 - 24);
            *(&v33 - 4) = v21;
            *v19 = v23;
            v24 = v18[1].m128i_i64[0];
            *((__m128i *)&v33 - 3) = v22;
            v19[1].m128i_i64[0] = v24;
            v18[1].m128i_i64[0] = v21;
            *v18 = v22;
          }
          while ( v9 != v20 );
          v6 = (const __m128i *)((char *)v6 - 24 * v9);
        }
        v9 = v8 % v10;
        if ( !(v8 % v10) )
          return v7;
      }
    }
    return v7;
  }
}
