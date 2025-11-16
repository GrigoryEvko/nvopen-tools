// Function: sub_BA8070
// Address: 0xba8070
//
_QWORD *__fastcall sub_BA8070(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  const __m128i *v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  const __m128i *v10; // r13
  _QWORD *i; // rdx
  __int64 v12; // rax
  const __m128i *v13; // rbx
  const __m128i *v14; // r12
  __int64 v15; // r14
  __int64 v16; // rdi
  int v17; // eax
  size_t v18; // rdx
  __int64 v19; // r10
  char *v20; // rdi
  int v21; // r11d
  int v22; // r8d
  unsigned int j; // r9d
  __int64 v24; // rcx
  const void *v25; // rsi
  bool v26; // al
  int v27; // eax
  __m128i v28; // xmm1
  _QWORD *k; // rdx
  unsigned int v30; // r9d
  __int64 v31; // [rsp+0h] [rbp-70h]
  size_t v32; // [rsp+8h] [rbp-68h]
  int v33; // [rsp+24h] [rbp-4Ch]
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]
  unsigned int v36; // [rsp+38h] [rbp-38h]
  int v37; // [rsp+3Ch] [rbp-34h]
  int v38; // [rsp+3Ch] [rbp-34h]

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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16 * v4;
    v10 = &v5[v4];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v10 != v5 )
    {
      v12 = a1;
      v13 = v5;
      v14 = v5;
      v15 = v12;
      while ( 1 )
      {
        while ( 1 )
        {
          v16 = v13->m128i_i64[0];
          if ( v13->m128i_i64[0] != -1 && v16 != -2 )
            break;
          if ( v10 == ++v13 )
            goto LABEL_22;
        }
        v37 = *(_DWORD *)(v15 + 24);
        if ( !v37 )
        {
          MEMORY[0] = _mm_loadu_si128(v13);
          BUG();
        }
        v35 = *(_QWORD *)(v15 + 8);
        v17 = sub_C94890(v16, v13->m128i_i64[1]);
        v18 = v13->m128i_u64[1];
        v19 = 0;
        v20 = (char *)v13->m128i_i64[0];
        v21 = 1;
        v22 = v37 - 1;
        for ( j = (v37 - 1) & v17; ; j = v22 & v30 )
        {
          v24 = v35 + 16LL * j;
          v25 = *(const void **)v24;
          if ( *(_QWORD *)v24 == -1 )
            break;
          v26 = v20 + 2 == 0;
          if ( v25 != (const void *)-2LL )
          {
            if ( v18 != *(_QWORD *)(v24 + 8) )
              goto LABEL_36;
            v33 = v21;
            v34 = v19;
            v36 = j;
            v38 = v22;
            if ( !v18 )
              goto LABEL_21;
            v31 = v35 + 16LL * j;
            v32 = v18;
            v27 = memcmp(v20, v25, v18);
            v18 = v32;
            v24 = v31;
            v22 = v38;
            j = v36;
            v26 = v27 == 0;
            v19 = v34;
            v21 = v33;
          }
          if ( v26 )
            goto LABEL_21;
          if ( !v19 && v25 == (const void *)-2LL )
            v19 = v24;
LABEL_36:
          v30 = v21 + j;
          ++v21;
        }
        if ( v20 != (char *)-1LL && v19 )
          v24 = v19;
LABEL_21:
        v28 = _mm_loadu_si128(v13++);
        *(__m128i *)v24 = v28;
        ++*(_DWORD *)(v15 + 16);
        if ( v10 == v13 )
        {
LABEL_22:
          v5 = v14;
          return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * *(unsigned int *)(a1 + 24)]; k != result; result += 2 )
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
