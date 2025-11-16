// Function: sub_3141930
// Address: 0x3141930
//
_QWORD *__fastcall sub_3141930(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r10
  __int64 v9; // r11
  _QWORD *i; // rdx
  const __m128i *v11; // rbx
  const __m128i *v12; // r14
  _QWORD *v13; // rdi
  int v14; // r13d
  int v15; // eax
  char *v16; // rdi
  size_t v17; // rdx
  int v18; // r8d
  __int64 v19; // r10
  int v20; // r11d
  unsigned int j; // r9d
  __int64 v22; // r13
  const void *v23; // rcx
  bool v24; // al
  unsigned int v25; // r9d
  __m128i v26; // xmm1
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // eax
  _QWORD *k; // rdx
  size_t v31; // [rsp+0h] [rbp-70h]
  const void *v32; // [rsp+8h] [rbp-68h]
  int v33; // [rsp+1Ch] [rbp-54h]
  __int64 v34; // [rsp+20h] [rbp-50h]
  unsigned int v35; // [rsp+28h] [rbp-48h]
  int v36; // [rsp+2Ch] [rbp-44h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  __int64 v38; // [rsp+38h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 48 * v4;
    v9 = v5 + 48 * v4;
    for ( i = &result[6 * *(unsigned int *)(a1 + 24)]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v9 != v5 )
    {
      v38 = 48 * v4;
      v11 = (const __m128i *)v5;
      v12 = (const __m128i *)v9;
      while ( 1 )
      {
        while ( 1 )
        {
          v13 = (_QWORD *)v11->m128i_i64[0];
          if ( v11->m128i_i64[0] != -1 && v13 != (_QWORD *)-2LL )
            break;
          v11 += 3;
          if ( v12 == v11 )
            goto LABEL_24;
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = _mm_loadu_si128(v11);
          BUG();
        }
        v37 = *(_QWORD *)(a1 + 8);
        v15 = sub_C94890(v13, v11->m128i_i64[1]);
        v16 = (char *)v11->m128i_i64[0];
        v17 = v11->m128i_u64[1];
        v18 = 1;
        v19 = 0;
        v20 = v14 - 1;
        for ( j = (v14 - 1) & v15; ; j = v20 & v25 )
        {
          v22 = v37 + 48LL * j;
          v23 = *(const void **)v22;
          v24 = v16 + 1 == 0;
          if ( *(_QWORD *)v22 == -1 )
            break;
          v24 = v16 + 2 == 0;
          if ( v23 == (const void *)-2LL )
            break;
          if ( v17 == *(_QWORD *)(v22 + 8) )
          {
            v33 = v18;
            v34 = v19;
            v35 = j;
            v36 = v20;
            if ( !v17 )
              goto LABEL_23;
            v31 = v17;
            v32 = *(const void **)v22;
            v29 = memcmp(v16, v23, v17);
            v23 = v32;
            v17 = v31;
            v20 = v36;
            j = v35;
            v19 = v34;
            v18 = v33;
            if ( !v29 )
              goto LABEL_23;
          }
LABEL_18:
          if ( !v19 && v23 == (const void *)-2LL )
            v19 = v22;
          v25 = v18 + j;
          ++v18;
        }
        if ( v24 )
          goto LABEL_23;
        if ( v23 != (const void *)-1LL )
          goto LABEL_18;
        if ( v19 )
          v22 = v19;
LABEL_23:
        v26 = _mm_loadu_si128(v11);
        *(_QWORD *)(v22 + 32) = 0;
        *(_QWORD *)(v22 + 24) = 0;
        *(_DWORD *)(v22 + 40) = 0;
        *(_QWORD *)(v22 + 16) = 1;
        *(__m128i *)v22 = v26;
        v27 = v11[1].m128i_i64[1];
        ++v11[1].m128i_i64[0];
        v28 = *(_QWORD *)(v22 + 24);
        v11 += 3;
        *(_QWORD *)(v22 + 24) = v27;
        LODWORD(v27) = v11[-1].m128i_i32[0];
        v11[-2].m128i_i64[1] = v28;
        LODWORD(v28) = *(_DWORD *)(v22 + 32);
        *(_DWORD *)(v22 + 32) = v27;
        LODWORD(v27) = v11[-1].m128i_i32[1];
        v11[-1].m128i_i32[0] = v28;
        LODWORD(v28) = *(_DWORD *)(v22 + 36);
        *(_DWORD *)(v22 + 36) = v27;
        LODWORD(v27) = v11[-1].m128i_i32[2];
        v11[-1].m128i_i32[1] = v28;
        LODWORD(v28) = *(_DWORD *)(v22 + 40);
        *(_DWORD *)(v22 + 40) = v27;
        v11[-1].m128i_i32[2] = v28;
        ++*(_DWORD *)(a1 + 16);
        sub_C7D6A0(v11[-2].m128i_i64[1], 32LL * v11[-1].m128i_u32[2], 8);
        if ( v12 == v11 )
        {
LABEL_24:
          v8 = v38;
          return (_QWORD *)sub_C7D6A0(v5, v8, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * *(unsigned int *)(a1 + 24)]; k != result; result += 6 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
  }
  return result;
}
