// Function: sub_376D3B0
// Address: 0x376d3b0
//
const __m128i *__fastcall sub_376D3B0(const __m128i *a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r12
  char v5; // al
  unsigned __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r13
  bool v9; // zf
  const __m128i *v10; // r11
  const __m128i *v11; // rax
  __int64 v12; // rdx
  const __m128i *i; // rdx
  const __m128i *result; // rax
  unsigned __int64 v15; // rsi
  const __m128i *v16; // r9
  int v17; // edi
  __int32 v18; // r10d
  int v19; // r15d
  __m128i *v20; // r14
  unsigned int k; // ecx
  __m128i *v22; // rdx
  unsigned int v23; // ecx
  const __m128i *v24; // r13
  const __m128i *v25; // rdx
  const __m128i *v26; // rax
  __m128i *v27; // r12
  __m128i v28; // xmm0
  __int64 v29; // rax
  const __m128i *v30; // rax
  __int64 v31; // rdx
  const __m128i *m; // rdx
  unsigned __int64 v33; // rsi
  const __m128i *v34; // r8
  int v35; // edi
  __int32 v36; // r9d
  int v37; // r14d
  __m128i *v38; // r11
  unsigned int n; // r10d
  __m128i *v40; // rdx
  unsigned int v41; // edx
  __int32 v42; // edi
  __int32 v43; // edi
  __m128i v44; // xmm3
  __int32 v45; // ecx
  __int64 v46; // rax
  __int32 v47; // r8d
  __int32 v48; // ecx
  const __m128i *j; // [rsp+8h] [rbp-838h]
  _BYTE v50[2096]; // [rsp+10h] [rbp-830h] BYREF

  v2 = a2;
  v4 = a1[1].m128i_i64[0];
  v5 = a1->m128i_i8[8] & 1;
  if ( a2 <= 0x40 )
  {
    v24 = a1 + 1;
    v25 = a1 + 129;
    if ( !v5 )
      goto LABEL_4;
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      if ( !v5 )
      {
        v7 = a1[1].m128i_u32[2];
        v46 = sub_C7D670(32LL * (unsigned int)v6, 8);
        a1[1].m128i_i32[2] = v6;
        a1[1].m128i_i64[0] = v46;
        goto LABEL_5;
      }
      v24 = a1 + 1;
      v25 = a1 + 129;
    }
    else
    {
      if ( !v5 )
      {
LABEL_4:
        a1->m128i_i8[8] |= 1u;
        v7 = a1[1].m128i_u32[2];
LABEL_5:
        v8 = 32 * v7;
        v9 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        v10 = (const __m128i *)(v4 + v8);
        if ( v9 )
        {
          v11 = (const __m128i *)a1[1].m128i_i64[0];
          v12 = 2LL * a1[1].m128i_u32[2];
        }
        else
        {
          v11 = a1 + 1;
          v12 = 128;
        }
        for ( i = &v11[v12]; i != v11; v11 += 2 )
        {
          if ( v11 )
          {
            v11->m128i_i64[0] = 0;
            v11->m128i_i32[2] = -1;
          }
        }
        result = (const __m128i *)v4;
        for ( j = a1 + 1; v10 != result; result += 2 )
        {
          v15 = result->m128i_i64[0];
          if ( result->m128i_i64[0] || result->m128i_i32[2] <= 0xFFFFFFFD )
          {
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v16 = j;
              v17 = 63;
            }
            else
            {
              v42 = a1[1].m128i_i32[2];
              v16 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v42 )
                goto LABEL_79;
              v17 = v42 - 1;
            }
            v18 = result->m128i_i32[2];
            v19 = 1;
            v20 = 0;
            for ( k = v17 & (v18 + ((v15 >> 9) ^ (v15 >> 4))); ; k = v17 & v23 )
            {
              v22 = (__m128i *)&v16[2 * k];
              if ( v15 == v22->m128i_i64[0] && v18 == v22->m128i_i32[2] )
                break;
              if ( !v22->m128i_i64[0] )
              {
                v47 = v22->m128i_i32[2];
                if ( v47 == -1 )
                {
                  if ( v20 )
                    v22 = v20;
                  break;
                }
                if ( v47 == -2 && !v20 )
                  v20 = (__m128i *)&v16[2 * k];
              }
              v23 = v19 + k;
              ++v19;
            }
            v22->m128i_i64[0] = result->m128i_i64[0];
            v22->m128i_i32[2] = result->m128i_i32[2];
            v22[1] = _mm_loadu_si128(result + 1);
            a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          }
        }
        return (const __m128i *)sub_C7D6A0(v4, v8, 8);
      }
      v24 = a1 + 1;
      v25 = a1 + 129;
      v2 = 64;
    }
  }
  v26 = v24;
  v27 = (__m128i *)v50;
  do
  {
    while ( !v26->m128i_i64[0] && v26->m128i_i32[2] > 0xFFFFFFFD )
    {
      v26 += 2;
      if ( v26 == v25 )
        goto LABEL_27;
    }
    if ( v27 )
      *v27 = _mm_loadu_si128(v26);
    v28 = _mm_loadu_si128(v26 + 1);
    v26 += 2;
    v27 += 2;
    v27[-1] = v28;
  }
  while ( v26 != v25 );
LABEL_27:
  if ( v2 > 0x40 )
  {
    a1->m128i_i8[8] &= ~1u;
    v29 = sub_C7D670(32LL * v2, 8);
    a1[1].m128i_i32[2] = v2;
    a1[1].m128i_i64[0] = v29;
  }
  v9 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  if ( v9 )
  {
    v30 = (const __m128i *)a1[1].m128i_i64[0];
    v31 = 2LL * a1[1].m128i_u32[2];
  }
  else
  {
    v30 = v24;
    v31 = 128;
  }
  for ( m = &v30[v31]; m != v30; v30 += 2 )
  {
    if ( v30 )
    {
      v30->m128i_i64[0] = 0;
      v30->m128i_i32[2] = -1;
    }
  }
  result = (const __m128i *)v50;
  if ( v27 != (__m128i *)v50 )
  {
    do
    {
      v33 = result->m128i_i64[0];
      if ( result->m128i_i64[0] || result->m128i_i32[2] <= 0xFFFFFFFD )
      {
        if ( (a1->m128i_i8[8] & 1) != 0 )
        {
          v34 = v24;
          v35 = 63;
        }
        else
        {
          v43 = a1[1].m128i_i32[2];
          v34 = (const __m128i *)a1[1].m128i_i64[0];
          if ( !v43 )
          {
LABEL_79:
            MEMORY[0] = result->m128i_i64[0];
            MEMORY[8] = result->m128i_i32[2];
            BUG();
          }
          v35 = v43 - 1;
        }
        v36 = result->m128i_i32[2];
        v37 = 1;
        v38 = 0;
        for ( n = v35 & (v36 + ((v33 >> 9) ^ (v33 >> 4))); ; n = v35 & v41 )
        {
          v40 = (__m128i *)&v34[2 * n];
          if ( v33 == v40->m128i_i64[0] && v36 == v40->m128i_i32[2] )
            break;
          if ( !v40->m128i_i64[0] )
          {
            v48 = v40->m128i_i32[2];
            if ( v48 == -1 )
            {
              if ( v38 )
                v40 = v38;
              break;
            }
            if ( v48 == -2 && !v38 )
              v38 = (__m128i *)&v34[2 * n];
          }
          v41 = n + v37++;
        }
        v44 = _mm_loadu_si128(result + 1);
        v40->m128i_i64[0] = result->m128i_i64[0];
        v45 = result->m128i_i32[2];
        v40[1] = v44;
        v40->m128i_i32[2] = v45;
        a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
      }
      result += 2;
    }
    while ( v27 != result );
  }
  return result;
}
