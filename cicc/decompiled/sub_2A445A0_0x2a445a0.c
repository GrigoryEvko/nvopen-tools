// Function: sub_2A445A0
// Address: 0x2a445a0
//
__m128i *__fastcall sub_2A445A0(__m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __m128i *v3; // r11
  signed __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r8
  const __m128i *v7; // rcx
  __m128i *v8; // rsi
  __int64 v9; // rdx
  __m128i v10; // xmm0
  __int32 v11; // r15d
  __int32 v12; // r14d
  __int32 v13; // r13d
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r10
  __int8 v17; // r9
  __m128i *v18; // rcx
  __m128i *v19; // rsi
  __int64 v20; // rdx
  __int32 v21; // r15d
  __int32 v22; // r14d
  __m128i v23; // xmm3
  __int32 v24; // r13d
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r10
  __int8 v28; // r9
  const __m128i *v30; // rdx
  __m128i *v31; // rax
  __m128i v32; // xmm6
  __int32 v33; // r12d
  __int32 v34; // ebx
  __int32 v35; // r10d
  __int64 v36; // r9
  __int64 v37; // r8
  __int64 v38; // rdi
  __int8 v39; // cl
  __int64 v40; // [rsp+0h] [rbp-38h]

  if ( a1 == a2 )
    return (__m128i *)a3;
  v3 = a1;
  if ( a2 == a3 )
    return a1;
  v40 = (__int64)a1->m128i_i64 + (char *)a3 - (char *)a2;
  v4 = 0xAAAAAAAAAAAAAAABLL * (a3 - a1);
  v5 = 0xAAAAAAAAAAAAAAABLL * (a2 - a1);
  if ( v5 == v4 - v5 )
  {
    v30 = a2;
    v31 = v3;
    do
    {
      v32 = _mm_loadu_si128(v30);
      v33 = v31->m128i_i32[0];
      v31 += 3;
      v30 += 3;
      v34 = v31[-3].m128i_i32[1];
      v35 = v31[-3].m128i_i32[2];
      v31[-3] = v32;
      v36 = v31[-2].m128i_i64[0];
      v37 = v31[-2].m128i_i64[1];
      v38 = v31[-1].m128i_i64[0];
      v31[-2] = _mm_loadu_si128(v30 - 2);
      v39 = v31[-1].m128i_i8[8];
      v31[-1] = _mm_loadu_si128(v30 - 1);
      v30[-3].m128i_i32[0] = v33;
      v30[-3].m128i_i32[1] = v34;
      v30[-3].m128i_i32[2] = v35;
      v30[-2].m128i_i64[0] = v36;
      v30[-2].m128i_i64[1] = v37;
      v30[-1].m128i_i64[0] = v38;
      v30[-1].m128i_i8[8] = v39;
    }
    while ( a2 != v31 );
    return &v3[3
             * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&a2[-3] - (char *)v3) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
             + 3];
  }
  else
  {
    v6 = v4 - v5;
    if ( v5 >= v4 - v5 )
      goto LABEL_12;
    while ( 1 )
    {
      v7 = &v3[3 * v5];
      if ( v6 > 0 )
      {
        v8 = v3;
        v9 = 0;
        do
        {
          v10 = _mm_loadu_si128(v7);
          v11 = v8->m128i_i32[0];
          ++v9;
          v8 += 3;
          v12 = v8[-3].m128i_i32[1];
          v13 = v8[-3].m128i_i32[2];
          v7 += 3;
          v8[-3] = v10;
          v14 = v8[-2].m128i_i64[0];
          v15 = v8[-2].m128i_i64[1];
          v16 = v8[-1].m128i_i64[0];
          v17 = v8[-1].m128i_i8[8];
          v8[-2] = _mm_loadu_si128(v7 - 2);
          v8[-1] = _mm_loadu_si128(v7 - 1);
          v7[-3].m128i_i32[0] = v11;
          v7[-3].m128i_i32[1] = v12;
          v7[-3].m128i_i32[2] = v13;
          v7[-2].m128i_i64[0] = v14;
          v7[-2].m128i_i64[1] = v15;
          v7[-1].m128i_i64[0] = v16;
          v7[-1].m128i_i8[8] = v17;
        }
        while ( v6 != v9 );
        v3 += 3 * v6;
      }
      if ( !(v4 % v5) )
        break;
      v6 = v5;
      v5 -= v4 % v5;
      while ( 1 )
      {
        v4 = v6;
        v6 -= v5;
        if ( v5 < v6 )
          break;
LABEL_12:
        v18 = &v3[3 * v4];
        v3 = &v18[-3 * v6];
        if ( v5 > 0 )
        {
          v19 = &v18[-3 * v6];
          v20 = 0;
          do
          {
            v21 = v19[-3].m128i_i32[0];
            v22 = v19[-3].m128i_i32[1];
            ++v20;
            v19 -= 3;
            v23 = _mm_loadu_si128(v18 - 3);
            v24 = v19->m128i_i32[2];
            v18 -= 3;
            v25 = v19[1].m128i_i64[0];
            v26 = v19[1].m128i_i64[1];
            *v19 = v23;
            v27 = v19[2].m128i_i64[0];
            v28 = v19[2].m128i_i8[8];
            v19[1] = _mm_loadu_si128(v18 + 1);
            v19[2] = _mm_loadu_si128(v18 + 2);
            v18->m128i_i32[0] = v21;
            v18->m128i_i32[1] = v22;
            v18->m128i_i32[2] = v24;
            v18[1].m128i_i64[0] = v25;
            v18[1].m128i_i64[1] = v26;
            v18[2].m128i_i64[0] = v27;
            v18[2].m128i_i8[8] = v28;
          }
          while ( v5 != v20 );
          v3 -= 3 * v5;
        }
        v5 = v4 % v6;
        if ( !(v4 % v6) )
          return (__m128i *)v40;
      }
    }
    return (__m128i *)v40;
  }
}
