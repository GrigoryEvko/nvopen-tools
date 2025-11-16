// Function: sub_27396F0
// Address: 0x27396f0
//
const __m128i *__fastcall sub_27396F0(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rbp
  const __m128i *v5; // r11
  const __m128i *v6; // r10
  __int8 *v7; // r11
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  const __m128i *v11; // rcx
  const __m128i *v12; // rsi
  __int64 v13; // rdi
  __m128i v14; // xmm4
  __m128i v15; // xmm2
  __m128i v16; // xmm1
  __m128i v17; // xmm0
  __m128i v18; // xmm5
  __m128i v19; // xmm3
  __int64 v20; // rdx
  __m128i v21; // xmm6
  __int64 v22; // rbx
  __m128i *v23; // rcx
  __m128i *v24; // rsi
  __int64 v25; // rdi
  __m128i v26; // xmm3
  __m128i v27; // xmm2
  __m128i v28; // xmm1
  __m128i v29; // xmm0
  __m128i v30; // xmm4
  __m128i v31; // xmm7
  __int64 v32; // rdx
  __m128i v33; // xmm5
  __int64 v34; // rbx
  const __m128i *v36; // rax
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm2
  __m128i v40; // xmm1
  __m128i v41; // xmm0
  __int64 v42; // rdx
  __m128i v43; // xmm6
  __m128i v44; // xmm7
  __int64 v45; // rcx
  __int64 v46; // [rsp-10h] [rbp-10h]
  __int64 v47; // [rsp-8h] [rbp-8h]

  v5 = a3;
  if ( a1 == a2 )
    return v5;
  v6 = a1;
  if ( a2 == a3 )
    return a1;
  v47 = v4;
  v7 = &a1->m128i_i8[(char *)a3 - (char *)a2];
  v8 = ((char *)a3 - (char *)a1) >> 6;
  v9 = ((char *)a2 - (char *)a1) >> 6;
  v46 = v3;
  if ( v9 == v8 - v9 )
  {
    v36 = a2;
    do
    {
      v37 = _mm_loadu_si128(v6 + 3);
      v38 = _mm_loadu_si128(v36);
      v6 += 4;
      v36 += 4;
      v39 = _mm_loadu_si128(v6 - 4);
      v40 = _mm_loadu_si128(v6 - 3);
      v6[-4] = v38;
      v41 = _mm_loadu_si128(v6 - 2);
      v42 = v6[-1].m128i_i64[0];
      *((__m128i *)&v47 - 2) = v37;
      v43 = _mm_loadu_si128(v36 - 3);
      *((__m128i *)&v47 - 5) = v39;
      v6[-3] = v43;
      v44 = _mm_loadu_si128(v36 - 2);
      *((__m128i *)&v47 - 4) = v40;
      v6[-2] = v44;
      v45 = v36[-1].m128i_i64[0];
      *((__m128i *)&v47 - 3) = v41;
      v6[-1].m128i_i64[0] = v45;
      v6[-1].m128i_i32[2] = v36[-1].m128i_i32[2];
      v36[-1].m128i_i64[0] = v42;
      LODWORD(v42) = *((_DWORD *)&v47 - 6);
      v36[-4] = v39;
      v36[-3] = v40;
      v36[-2] = v41;
      v36[-1].m128i_i32[2] = v42;
    }
    while ( a2 != v6 );
    return a2;
  }
  else
  {
    v10 = v8 - v9;
    if ( v9 >= v8 - v9 )
      goto LABEL_12;
    while ( 1 )
    {
      v11 = &v6[4 * v9];
      if ( v10 > 0 )
      {
        v12 = v6;
        v13 = 0;
        do
        {
          v14 = _mm_loadu_si128(v11);
          v15 = _mm_loadu_si128(v12);
          ++v13;
          v12 += 4;
          v16 = _mm_loadu_si128(v12 - 3);
          v17 = _mm_loadu_si128(v12 - 2);
          v11 += 4;
          v12[-4] = v14;
          v18 = _mm_loadu_si128(v11 - 3);
          v19 = _mm_loadu_si128(v12 - 1);
          v20 = v12[-1].m128i_i64[0];
          *((__m128i *)&v47 - 5) = v15;
          v12[-3] = v18;
          v21 = _mm_loadu_si128(v11 - 2);
          *((__m128i *)&v47 - 2) = v19;
          v12[-2] = v21;
          v22 = v11[-1].m128i_i64[0];
          *((__m128i *)&v47 - 4) = v16;
          v12[-1].m128i_i64[0] = v22;
          LODWORD(v22) = v11[-1].m128i_i32[2];
          *((__m128i *)&v47 - 3) = v17;
          v12[-1].m128i_i32[2] = v22;
          v11[-1].m128i_i64[0] = v20;
          LODWORD(v20) = *((_DWORD *)&v47 - 6);
          v11[-4] = v15;
          v11[-3] = v16;
          v11[-2] = v17;
          v11[-1].m128i_i32[2] = v20;
        }
        while ( v10 != v13 );
        v6 += 4 * v10;
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
        v23 = (__m128i *)&v6[4 * v8];
        v6 = &v23[-4 * v10];
        if ( v9 > 0 )
        {
          v24 = &v23[-4 * v10];
          v25 = 0;
          do
          {
            v26 = _mm_loadu_si128(v23 - 4);
            v27 = _mm_loadu_si128(v24 - 4);
            ++v25;
            v24 -= 4;
            v28 = _mm_loadu_si128(v24 + 1);
            v29 = _mm_loadu_si128(v24 + 2);
            v23 -= 4;
            *v24 = v26;
            v30 = _mm_loadu_si128(v23 + 1);
            v31 = _mm_loadu_si128(v24 + 3);
            v32 = v24[3].m128i_i64[0];
            *((__m128i *)&v47 - 5) = v27;
            v24[1] = v30;
            v33 = _mm_loadu_si128(v23 + 2);
            *((__m128i *)&v47 - 2) = v31;
            v24[2] = v33;
            v34 = v23[3].m128i_i64[0];
            *((__m128i *)&v47 - 4) = v28;
            v24[3].m128i_i64[0] = v34;
            LODWORD(v34) = v23[3].m128i_i32[2];
            *((__m128i *)&v47 - 3) = v29;
            v24[3].m128i_i32[2] = v34;
            v23[3].m128i_i64[0] = v32;
            LODWORD(v32) = *((_DWORD *)&v47 - 6);
            *v23 = v27;
            v23[3].m128i_i32[2] = v32;
            v23[1] = v28;
            v23[2] = v29;
          }
          while ( v9 != v25 );
          v6 -= 4 * v9;
        }
        v9 = v8 % v10;
        if ( !(v8 % v10) )
          return (const __m128i *)v7;
      }
    }
    return (const __m128i *)v7;
  }
}
