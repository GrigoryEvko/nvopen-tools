// Function: sub_116D540
// Address: 0x116d540
//
__m128i *__fastcall sub_116D540(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r10d
  __m128i *v7; // rdx
  unsigned int i; // eax
  __m128i *v9; // r9
  __m128i *v10; // r13
  unsigned int v11; // eax
  __int64 v13; // r11
  int v14; // eax
  int v15; // ecx
  __m128i *v16; // r9
  __int64 v17; // rdi
  int v18; // r10d
  unsigned int k; // eax
  unsigned int v20; // eax
  int v21; // ecx
  __m128i v22; // xmm0
  int v23; // eax
  int v24; // eax
  int v25; // edi
  __int64 v26; // rsi
  int v27; // r9d
  unsigned int j; // eax
  unsigned int v29; // eax
  __int64 v30; // r8
  __int64 v31; // r8

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  for ( i = (v4 - 1)
          & (((unsigned int)a2->m128i_i64[0] >> 9)
           ^ ((unsigned int)a2->m128i_i64[0] >> 4)
           ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; i = (v4 - 1) & v11 )
  {
    v9 = (__m128i *)(v5 + 24LL * i);
    v10 = (__m128i *)v9->m128i_i64[0];
    if ( a2->m128i_i64[0] == v9->m128i_i64[0] && a2->m128i_i64[1] == v9->m128i_i64[1] )
      return v9 + 1;
    if ( !v10 )
      break;
LABEL_5:
    v11 = v6 + i;
    ++v6;
  }
  v13 = v9->m128i_i64[1];
  if ( v13 )
  {
    if ( v13 == 1 && !v7 )
      v7 = (__m128i *)(v5 + 24LL * i);
    goto LABEL_5;
  }
  v23 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v9;
  ++*(_QWORD *)a1;
  v21 = v23 + 1;
  if ( 4 * (v23 + 1) < 3 * v4 )
  {
    if ( v4 - *(_DWORD *)(a1 + 20) - v21 <= v4 >> 3 )
    {
      sub_116D1D0(a1, v4);
      v24 = *(_DWORD *)(a1 + 24);
      if ( v24 )
      {
        v25 = v24 - 1;
        v27 = 1;
        for ( j = (v24 - 1)
                & (((unsigned int)a2->m128i_i64[0] >> 9)
                 ^ ((unsigned int)a2->m128i_i64[0] >> 4)
                 ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; j = v25 & v29 )
        {
          v26 = *(_QWORD *)(a1 + 8);
          v7 = (__m128i *)(v26 + 24LL * j);
          if ( a2->m128i_i64[0] == v7->m128i_i64[0] && a2->m128i_i64[1] == v7->m128i_i64[1] )
            break;
          if ( !v7->m128i_i64[0] )
          {
            v31 = v7->m128i_i64[1];
            if ( !v31 )
            {
              if ( v10 )
                v7 = v10;
              v21 = *(_DWORD *)(a1 + 16) + 1;
              goto LABEL_21;
            }
            if ( v31 == 1 && !v10 )
              v10 = (__m128i *)(v26 + 24LL * j);
          }
          v29 = v27 + j;
          ++v27;
        }
        goto LABEL_20;
      }
LABEL_53:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    goto LABEL_21;
  }
LABEL_14:
  sub_116D1D0(a1, 2 * v4);
  v14 = *(_DWORD *)(a1 + 24);
  if ( !v14 )
    goto LABEL_53;
  v15 = v14 - 1;
  v16 = 0;
  v18 = 1;
  for ( k = (v14 - 1)
          & (((unsigned int)a2->m128i_i64[0] >> 9)
           ^ ((unsigned int)a2->m128i_i64[0] >> 4)
           ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; k = v15 & v20 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    v7 = (__m128i *)(v17 + 24LL * k);
    if ( a2->m128i_i64[0] == v7->m128i_i64[0] && a2->m128i_i64[1] == v7->m128i_i64[1] )
      break;
    if ( !v7->m128i_i64[0] )
    {
      v30 = v7->m128i_i64[1];
      if ( !v30 )
      {
        if ( v16 )
          v7 = v16;
        v21 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_21;
      }
      if ( v30 == 1 && !v16 )
        v16 = (__m128i *)(v17 + 24LL * k);
    }
    v20 = v18 + k;
    ++v18;
  }
LABEL_20:
  v21 = *(_DWORD *)(a1 + 16) + 1;
LABEL_21:
  *(_DWORD *)(a1 + 16) = v21;
  if ( v7->m128i_i64[0] || v7->m128i_i64[1] )
    --*(_DWORD *)(a1 + 20);
  v22 = _mm_loadu_si128(a2);
  v7[1].m128i_i64[0] = 0;
  *v7 = v22;
  return v7 + 1;
}
