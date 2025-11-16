// Function: sub_19FE520
// Address: 0x19fe520
//
const __m128i *__fastcall sub_19FE520(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r11
  const __m128i *v4; // r10
  __int8 *v5; // r11
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  const __m128i *v9; // rcx
  const __m128i *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __m128i *v14; // rcx
  __m128i *v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __m128i v18; // xmm0
  __int64 *v20; // rax
  __int64 v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // [rsp-8h] [rbp-8h]

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
    v20 = (__int64 *)a2;
    do
    {
      v21 = *v20;
      v22 = _mm_loadu_si128(v4++);
      v20 += 2;
      v4[-1].m128i_i64[0] = v21;
      v4[-1].m128i_i32[2] = *((_DWORD *)v20 - 2);
      *((__m128i *)&v23 - 1) = v22;
      *(v20 - 2) = v22.m128i_i64[0];
      *((_DWORD *)v20 - 2) = *((_DWORD *)&v23 - 2);
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
          v12 = v9->m128i_i64[0];
          v13 = _mm_loadu_si128(v10);
          ++v11;
          ++v10;
          ++v9;
          v10[-1].m128i_i64[0] = v12;
          v10[-1].m128i_i32[2] = v9[-1].m128i_i32[2];
          *((__m128i *)&v23 - 2) = v13;
          v9[-1].m128i_i64[0] = v13.m128i_i64[0];
          v9[-1].m128i_i32[2] = *((_DWORD *)&v23 - 6);
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
        v14 = (__m128i *)&v4[v6];
        v4 = &v14[-v8];
        if ( v7 > 0 )
        {
          v15 = &v14[-v8];
          v16 = 0;
          do
          {
            v17 = v14[-1].m128i_i64[0];
            ++v16;
            --v14;
            v18 = _mm_loadu_si128(--v15);
            v15->m128i_i64[0] = v17;
            v15->m128i_i32[2] = v14->m128i_i32[2];
            *((__m128i *)&v23 - 3) = v18;
            v14->m128i_i64[0] = v18.m128i_i64[0];
            v14->m128i_i32[2] = *((_DWORD *)&v23 - 10);
          }
          while ( v7 != v16 );
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
