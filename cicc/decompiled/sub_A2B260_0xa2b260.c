// Function: sub_A2B260
// Address: 0xa2b260
//
_QWORD *__fastcall sub_A2B260(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  const __m128i *v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r10
  const __m128i *v9; // r11
  _QWORD *i; // rdx
  const __m128i *v11; // rbx
  const __m128i *v12; // r13
  __int64 v13; // rdi
  int v14; // r14d
  int v15; // eax
  size_t v16; // rdx
  __int64 v17; // r9
  char *v18; // rdi
  int v19; // r11d
  unsigned int j; // r8d
  __int64 v21; // rcx
  bool v22; // al
  const void *v23; // r14
  unsigned int v24; // r8d
  __m128i v25; // xmm1
  int v26; // eax
  _QWORD *k; // rdx
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  size_t v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  int v32; // [rsp+28h] [rbp-48h]
  int v33; // [rsp+2Ch] [rbp-44h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  __int64 v35; // [rsp+38h] [rbp-38h]
  unsigned int v36; // [rsp+38h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(const __m128i **)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 24 * v4;
    v9 = (const __m128i *)((char *)v5 + 24 * v4);
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v9 == v5 )
      return (_QWORD *)sub_C7D6A0(v5, v8, 8);
    v11 = v5;
    v12 = v9;
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = v11->m128i_i64[0];
        if ( v11->m128i_i64[0] != -1 && v13 != -2 )
          break;
        v11 = (const __m128i *)((char *)v11 + 24);
        if ( v12 == v11 )
          return (_QWORD *)sub_C7D6A0(v5, v8, 8);
      }
      v14 = *(_DWORD *)(a1 + 24);
      v35 = v8;
      if ( !v14 )
      {
        MEMORY[0] = _mm_loadu_si128(v11);
        BUG();
      }
      v34 = *(_QWORD *)(a1 + 8);
      v15 = sub_C94890(v13, v11->m128i_i64[1]);
      v33 = v14 - 1;
      v16 = v11->m128i_u64[1];
      v17 = 0;
      v18 = (char *)v11->m128i_i64[0];
      v8 = v35;
      v19 = 1;
      for ( j = (v14 - 1) & v15; ; j = v33 & v24 )
      {
        v21 = v34 + 24LL * j;
        v22 = v18 + 1 == 0;
        v23 = *(const void **)v21;
        if ( *(_QWORD *)v21 == -1 )
          break;
        v22 = v18 + 2 == 0;
        if ( v23 == (const void *)-2LL )
          break;
        if ( v16 == *(_QWORD *)(v21 + 8) )
        {
          v32 = v19;
          v31 = v17;
          v36 = j;
          if ( !v16 )
            goto LABEL_23;
          v28 = v34 + 24LL * j;
          v29 = v8;
          v30 = v16;
          v26 = memcmp(v18, v23, v16);
          v16 = v30;
          v8 = v29;
          v21 = v28;
          j = v36;
          v17 = v31;
          v19 = v32;
          if ( !v26 )
            goto LABEL_23;
        }
LABEL_18:
        if ( !v17 && v23 == (const void *)-2LL )
          v17 = v21;
        v24 = v19 + j;
        ++v19;
      }
      if ( v22 )
        goto LABEL_23;
      if ( v23 != (const void *)-1LL )
        goto LABEL_18;
      if ( v17 )
        v21 = v17;
LABEL_23:
      v25 = _mm_loadu_si128(v11);
      v11 = (const __m128i *)((char *)v11 + 24);
      *(__m128i *)v21 = v25;
      *(_QWORD *)(v21 + 16) = v11[-1].m128i_i64[1];
      ++*(_DWORD *)(a1 + 16);
      if ( v12 == v11 )
        return (_QWORD *)sub_C7D6A0(v5, v8, 8);
    }
  }
  *(_QWORD *)(a1 + 16) = 0;
  for ( k = &result[3 * *(unsigned int *)(a1 + 24)]; k != result; result += 3 )
  {
    if ( result )
    {
      *result = -1;
      result[1] = 0;
    }
  }
  return result;
}
