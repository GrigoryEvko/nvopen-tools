// Function: sub_E41C70
// Address: 0xe41c70
//
_QWORD *__fastcall sub_E41C70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r11
  _QWORD *i; // rdx
  const __m128i *v9; // r13
  const __m128i *v10; // rbx
  const __m128i *v11; // r12
  __int64 v13; // rax
  int v14; // eax
  size_t v15; // rdx
  __int64 v16; // r8
  char *v17; // rdi
  int v18; // r9d
  int v19; // r10d
  unsigned int j; // r11d
  __int64 v21; // rcx
  const void *v22; // rsi
  bool v23; // al
  unsigned int v24; // r11d
  __m128i v25; // xmm1
  const __m128i *v26; // rax
  const __m128i *v27; // rdi
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rdx
  _QWORD *k; // rdx
  __int64 v32; // [rsp+8h] [rbp-78h]
  size_t v33; // [rsp+10h] [rbp-70h]
  int v34; // [rsp+2Ch] [rbp-54h]
  __int64 v35; // [rsp+30h] [rbp-50h]
  __int64 v36; // [rsp+38h] [rbp-48h]
  __int64 v37; // [rsp+40h] [rbp-40h]
  unsigned int v38; // [rsp+48h] [rbp-38h]
  int v39; // [rsp+4Ch] [rbp-34h]
  int v40; // [rsp+4Ch] [rbp-34h]

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(48LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 48 * v3;
    for ( i = &result[6 * *(unsigned int *)(a1 + 24)]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v4 + 48 * v3 != v4 )
    {
      v37 = 48 * v3;
      v9 = (const __m128i *)(v4 + 48 * v3);
      v10 = (const __m128i *)(v4 + 32);
      v11 = (const __m128i *)v4;
      while ( 1 )
      {
        v13 = v10[-2].m128i_i64[0];
        if ( v13 == -1 || v13 == -2 )
          goto LABEL_10;
        v39 = *(_DWORD *)(a1 + 24);
        if ( !v39 )
        {
          MEMORY[0] = _mm_loadu_si128(v11);
          BUG();
        }
        v36 = *(_QWORD *)(a1 + 8);
        v14 = sub_C94890(v11->m128i_i64[0], v11->m128i_i64[1]);
        v15 = v10[-2].m128i_u64[1];
        v16 = 0;
        v17 = (char *)v10[-2].m128i_i64[0];
        v18 = 1;
        v19 = v39 - 1;
        for ( j = (v39 - 1) & v14; ; j = v19 & v24 )
        {
          v21 = v36 + 48LL * j;
          v22 = *(const void **)v21;
          v23 = v17 + 1 == 0;
          if ( *(_QWORD *)v21 == -1 )
            break;
          v23 = v17 + 2 == 0;
          if ( v22 == (const void *)-2LL )
            break;
          if ( v15 == *(_QWORD *)(v21 + 8) )
          {
            v34 = v18;
            v35 = v16;
            v38 = j;
            v40 = v19;
            if ( !v15 )
              goto LABEL_23;
            v32 = v36 + 48LL * j;
            v33 = v15;
            v29 = memcmp(v17, v22, v15);
            v15 = v33;
            v21 = v32;
            v19 = v40;
            j = v38;
            v16 = v35;
            v18 = v34;
            if ( !v29 )
              goto LABEL_23;
          }
LABEL_18:
          if ( !v16 && v22 == (const void *)-2LL )
            v16 = v21;
          v24 = v18 + j;
          ++v18;
        }
        if ( v23 )
          goto LABEL_23;
        if ( v22 != (const void *)-1LL )
          goto LABEL_18;
        if ( v16 )
          v21 = v16;
LABEL_23:
        v25 = _mm_loadu_si128(v10 - 2);
        *(_QWORD *)(v21 + 16) = v21 + 32;
        *(__m128i *)v21 = v25;
        v26 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v26 == v10 )
        {
          *(__m128i *)(v21 + 32) = _mm_loadu_si128(v10);
        }
        else
        {
          *(_QWORD *)(v21 + 16) = v26;
          *(_QWORD *)(v21 + 32) = v10->m128i_i64[0];
        }
        *(_QWORD *)(v21 + 24) = v10[-1].m128i_i64[1];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        ++*(_DWORD *)(a1 + 16);
        v27 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v27 == v10 )
        {
LABEL_10:
          v11 += 3;
          v10 += 3;
          if ( v9 == v11 )
            goto LABEL_27;
        }
        else
        {
          v28 = v10->m128i_i64[0];
          v11 += 3;
          v10 += 3;
          j_j___libc_free_0(v27, v28 + 1);
          if ( v9 == v11 )
          {
LABEL_27:
            v7 = v37;
            return (_QWORD *)sub_C7D6A0(v4, v7, 8);
          }
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, v7, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v30]; k != result; result += 6 )
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
