// Function: sub_3984680
// Address: 0x3984680
//
const __m128i *__fastcall sub_3984680(__m128i *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r11
  __m128i *v4; // r10
  __int8 *v5; // r11
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  const __m128i *v9; // rcx
  __m128i *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdx
  __m128i v14; // xmm0
  __m128i *v15; // rcx
  __m128i *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rdx
  __m128i v20; // xmm1
  const __m128i *v22; // rax
  __m128i v23; // xmm2
  __int64 v24; // rcx
  __int64 v25; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 == a3 )
    return a1;
  v5 = &a1->m128i_i8[(char *)a3 - (char *)a2];
  v6 = a3 - a1;
  v7 = a2 - a1;
  if ( v7 == v6 - v7 )
  {
    v22 = a2;
    do
    {
      v23 = _mm_loadu_si128(v22);
      v24 = v4->m128i_i64[0];
      ++v4;
      ++v22;
      v25 = v4[-1].m128i_i64[1];
      v4[-1] = v23;
      v22[-1].m128i_i64[0] = v24;
      v22[-1].m128i_i64[1] = v25;
    }
    while ( a2 != v4 );
    return a2;
  }
  else
  {
    v8 = v6 - v7;
    if ( v7 >= v6 - v7 )
      goto LABEL_12;
    while ( 1 )
    {
      v9 = &v4[v7];
      if ( v8 > 0 )
      {
        v10 = v4;
        v11 = 0;
        do
        {
          v12 = v10->m128i_i64[0];
          v13 = v10->m128i_i64[1];
          ++v11;
          ++v10;
          v14 = _mm_loadu_si128(v9++);
          v10[-1] = v14;
          v9[-1].m128i_i64[0] = v12;
          v9[-1].m128i_i64[1] = v13;
        }
        while ( v8 != v11 );
        v4 += v8;
      }
      if ( !(v6 % v7) )
        break;
      v8 = v7;
      v7 -= v6 % v7;
      while ( 1 )
      {
        v6 = v8;
        v8 -= v7;
        if ( v7 < v8 )
          break;
LABEL_12:
        v15 = &v4[v6];
        v4 = &v15[-v8];
        if ( v7 > 0 )
        {
          v16 = &v15[-v8];
          v17 = 0;
          do
          {
            v18 = v16[-1].m128i_i64[0];
            v19 = v16[-1].m128i_i64[1];
            ++v17;
            --v16;
            v20 = _mm_loadu_si128(--v15);
            *v16 = v20;
            v15->m128i_i64[0] = v18;
            v15->m128i_i64[1] = v19;
          }
          while ( v7 != v17 );
          v4 -= v7;
        }
        v7 = v6 % v8;
        if ( !(v6 % v8) )
          return (const __m128i *)v5;
      }
    }
    return (const __m128i *)v5;
  }
}
