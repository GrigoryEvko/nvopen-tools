// Function: sub_12E0A40
// Address: 0x12e0a40
//
_QWORD *__fastcall sub_12E0A40(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r14
  _QWORD *i; // rdx
  __int64 v10; // rax
  const __m128i *v11; // rbx
  const __m128i *v12; // r12
  __int64 v13; // r13
  int v14; // r15d
  int v15; // eax
  int v16; // ecx
  const void *v17; // rdi
  size_t v18; // rdx
  int v19; // r11d
  __int64 v20; // r9
  unsigned int j; // r8d
  __int64 v22; // r15
  const void *v23; // rsi
  unsigned int v24; // r8d
  _QWORD *k; // rdx
  size_t v26; // [rsp+0h] [rbp-60h]
  int v27; // [rsp+14h] [rbp-4Ch]
  __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned int v29; // [rsp+20h] [rbp-40h]
  int v30; // [rsp+24h] [rbp-3Ch]
  __int64 v31; // [rsp+28h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(const __m128i **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v8 != v4 )
    {
      v10 = a1;
      v11 = v4;
      v12 = v4;
      v13 = v10;
      do
      {
        if ( v11->m128i_i64[0] != -1 && v11->m128i_i64[0] != -2 )
        {
          v14 = *(_DWORD *)(v13 + 24);
          if ( !v14 )
          {
            MEMORY[0] = _mm_loadu_si128(v11);
            BUG();
          }
          v31 = *(_QWORD *)(v13 + 8);
          v15 = sub_16D3930(v11->m128i_i64[0], v11->m128i_i64[1]);
          v16 = v14 - 1;
          v17 = (const void *)v11->m128i_i64[0];
          v18 = v11->m128i_u64[1];
          v19 = 1;
          v20 = 0;
          for ( j = (v14 - 1) & v15; ; j = v16 & v24 )
          {
            v22 = v31 + 16LL * j;
            v23 = *(const void **)v22;
            if ( *(_QWORD *)v22 == -1 )
              break;
            if ( v23 == (const void *)-2LL )
            {
              if ( v17 == (const void *)-2LL )
                goto LABEL_22;
              if ( !v20 )
                v20 = v31 + 16LL * j;
            }
            else if ( *(_QWORD *)(v22 + 8) == v18 )
            {
              v27 = v19;
              v28 = v20;
              v29 = j;
              v30 = v16;
              if ( !v18 )
                goto LABEL_22;
              v26 = v18;
              if ( !memcmp(v17, v23, v18) )
                goto LABEL_22;
              v18 = v26;
              v16 = v30;
              j = v29;
              v20 = v28;
              v19 = v27;
            }
            v24 = v19 + j;
            ++v19;
          }
          if ( v17 != (const void *)-1LL && v20 )
            v22 = v20;
LABEL_22:
          *(__m128i *)v22 = _mm_loadu_si128(v11);
          ++*(_DWORD *)(v13 + 16);
        }
        ++v11;
      }
      while ( v8 != v11 );
      v4 = v12;
    }
    return (_QWORD *)j___libc_free_0(v4);
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
