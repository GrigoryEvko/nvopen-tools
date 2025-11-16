// Function: sub_1077D10
// Address: 0x1077d10
//
const __m128i *__fastcall sub_1077D10(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 *v4; // r9
  __int8 *v5; // r10
  signed __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  const __m128i *v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r12
  __int32 v16; // ebx
  __int64 v17; // r11
  __int64 *v18; // rcx
  __int64 *v19; // rsi
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __m128i v22; // xmm2
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // r12
  int v26; // ebx
  __int64 v27; // r11
  const __m128i *v29; // rax
  const __m128i *v30; // rdx
  __m128i v31; // xmm4
  __int64 v32; // r10
  __int64 v33; // r9
  __int64 v34; // r8
  __int32 v35; // edi
  __int64 v36; // rcx

  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v4 = (__int64 *)a1;
  v5 = &a1->m128i_i8[(char *)a3 - (char *)a2];
  v6 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a3 - (char *)a1) >> 3);
  v7 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3);
  if ( v7 == v6 - v7 )
  {
    v29 = a1;
    v30 = a2;
    do
    {
      v31 = _mm_loadu_si128(v30);
      v32 = v29->m128i_i64[0];
      v29 = (const __m128i *)((char *)v29 + 40);
      v30 = (const __m128i *)((char *)v30 + 40);
      v33 = v29[-2].m128i_i64[0];
      v34 = v29[-2].m128i_i64[1];
      *(__m128i *)((char *)v29 - 40) = v31;
      v35 = v29[-1].m128i_i32[0];
      v36 = v29[-1].m128i_i64[1];
      *(const __m128i *)((char *)&v29[-2] + 8) = _mm_loadu_si128((const __m128i *)((char *)v30 - 24));
      v29[-1].m128i_i64[1] = v30[-1].m128i_i64[1];
      v30[-3].m128i_i64[1] = v32;
      v30[-2].m128i_i64[0] = v33;
      v30[-2].m128i_i64[1] = v34;
      v30[-1].m128i_i32[0] = v35;
      v30[-1].m128i_i64[1] = v36;
    }
    while ( a2 != v29 );
    return a2;
  }
  else
  {
    v8 = v6 - v7;
    if ( v7 >= v6 - v7 )
      goto LABEL_12;
    while ( 1 )
    {
      v9 = (const __m128i *)&v4[5 * v7];
      if ( v8 > 0 )
      {
        v10 = v4;
        v11 = 0;
        do
        {
          v12 = _mm_loadu_si128(v9);
          v13 = *v10;
          ++v11;
          v10 += 5;
          v14 = *(v10 - 4);
          v15 = *(v10 - 3);
          v9 = (const __m128i *)((char *)v9 + 40);
          *(__m128i *)(v10 - 5) = v12;
          v16 = *((_DWORD *)v10 - 4);
          v17 = *(v10 - 1);
          *(__m128i *)(v10 - 3) = _mm_loadu_si128((const __m128i *)((char *)v9 - 24));
          *(v10 - 1) = v9[-1].m128i_i64[1];
          v9[-3].m128i_i64[1] = v13;
          v9[-2].m128i_i64[0] = v14;
          v9[-2].m128i_i64[1] = v15;
          v9[-1].m128i_i32[0] = v16;
          v9[-1].m128i_i64[1] = v17;
        }
        while ( v8 != v11 );
        v4 += 5 * v8;
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
        v18 = &v4[5 * v6];
        v4 = &v18[-5 * v8];
        if ( v7 > 0 )
        {
          v19 = v4 - 5;
          v20 = v18 - 5;
          v21 = 0;
          do
          {
            v22 = _mm_loadu_si128((const __m128i *)v20);
            v23 = *v19;
            ++v21;
            v19 -= 5;
            v24 = v19[6];
            v25 = v19[7];
            v20 -= 5;
            *(__m128i *)(v19 + 5) = v22;
            v26 = *((_DWORD *)v19 + 16);
            v27 = v19[9];
            *(__m128i *)(v19 + 7) = _mm_loadu_si128((const __m128i *)(v20 + 7));
            v19[9] = v20[9];
            v20[5] = v23;
            v20[6] = v24;
            v20[7] = v25;
            *((_DWORD *)v20 + 16) = v26;
            v20[9] = v27;
          }
          while ( v7 != v21 );
          v4 -= 5 * v7;
        }
        v7 = v6 % v8;
        if ( !(v6 % v8) )
          return (const __m128i *)v5;
      }
    }
    return (const __m128i *)v5;
  }
}
