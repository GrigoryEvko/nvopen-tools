// Function: sub_29A07B0
// Address: 0x29a07b0
//
__int64 __fastcall sub_29A07B0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r11
  __int64 i; // rdx
  __int64 v11; // rax
  const __m128i *v12; // rbx
  __int64 v13; // r12
  const __m128i *v14; // r14
  __int64 v15; // r15
  __int64 v16; // r13
  unsigned __int64 v17; // rax
  size_t v18; // rdx
  __int64 v19; // r10
  char *v20; // rdi
  int v21; // r11d
  int v22; // r8d
  unsigned int j; // r13d
  __int64 v24; // rcx
  const void *v25; // rsi
  bool v26; // al
  int v27; // eax
  unsigned int v28; // r13d
  __m128i v29; // xmm1
  __int64 k; // rdx
  __int64 v31; // [rsp+8h] [rbp-68h]
  int v32; // [rsp+10h] [rbp-60h]
  int v33; // [rsp+14h] [rbp-5Ch]
  size_t v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  int v36; // [rsp+30h] [rbp-40h]
  __int64 v37; // [rsp+38h] [rbp-38h]

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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v37 = 32 * v4;
    v9 = v5 + 32 * v4;
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v9 != v5 )
    {
      v11 = a1;
      v12 = (const __m128i *)v5;
      v13 = v5;
      v14 = (const __m128i *)v9;
      v15 = v11;
      while ( 1 )
      {
        while ( v12->m128i_i64[0] != -1 )
        {
          if ( v12->m128i_i64[0] != -2 || v12[1].m128i_i32[0] != -2 )
            goto LABEL_12;
LABEL_26:
          v12 += 2;
          if ( v14 == v12 )
            goto LABEL_27;
        }
        if ( v12[1].m128i_i32[0] == -1 )
          goto LABEL_26;
LABEL_12:
        v36 = *(_DWORD *)(v15 + 24);
        if ( !v36 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          BUG();
        }
        v35 = *(_QWORD *)(v15 + 8);
        v16 = (unsigned int)(37 * v12[1].m128i_i32[0]);
        v17 = sub_C94890(v12->m128i_i64[0], v12->m128i_i64[1]);
        v18 = v12->m128i_u64[1];
        v19 = 0;
        v20 = (char *)v12->m128i_i64[0];
        v21 = 1;
        v22 = v36 - 1;
        for ( j = (v36 - 1) & (((0xBF58476D1CE4E5B9LL * ((v17 << 32) | v16)) >> 31) ^ (484763065 * v16)); ; j = v22 & v28 )
        {
          v24 = v35 + 32LL * j;
          v25 = *(const void **)v24;
          if ( *(_QWORD *)v24 == -1 )
            break;
          v26 = v20 + 2 == 0;
          if ( v25 == (const void *)-2LL )
            goto LABEL_19;
          if ( *(_QWORD *)(v24 + 8) != v18 )
            goto LABEL_23;
          if ( v18 )
          {
            v32 = v21;
            v31 = v19;
            v33 = v22;
            v34 = v18;
            v27 = memcmp(v20, v25, v18);
            v18 = v34;
            v22 = v33;
            v19 = v31;
            v21 = v32;
            v26 = v27 == 0;
            v24 = v35 + 32LL * j;
LABEL_19:
            if ( !v26 )
              goto LABEL_21;
          }
LABEL_20:
          if ( v12[1].m128i_i32[0] == *(_DWORD *)(v24 + 16) )
            goto LABEL_31;
LABEL_21:
          if ( v25 == (const void *)-1LL )
            goto LABEL_22;
          if ( v25 == (const void *)-2LL && *(_DWORD *)(v24 + 16) == -2 && !v19 )
            v19 = v24;
LABEL_23:
          v28 = v21 + j;
          ++v21;
        }
        if ( v20 == (char *)-1LL )
          goto LABEL_20;
LABEL_22:
        if ( *(_DWORD *)(v24 + 16) != -1 )
          goto LABEL_23;
        if ( v19 )
          v24 = v19;
LABEL_31:
        v29 = _mm_loadu_si128(v12);
        v12 += 2;
        *(__m128i *)v24 = v29;
        *(_DWORD *)(v24 + 16) = v12[-1].m128i_i32[0];
        *(_DWORD *)(v24 + 24) = v12[-1].m128i_i32[2];
        ++*(_DWORD *)(v15 + 16);
        if ( v14 == v12 )
        {
LABEL_27:
          v5 = v13;
          return sub_C7D6A0(v5, v37, 8);
        }
      }
    }
    return sub_C7D6A0(v5, v37, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32LL * *(unsigned int *)(a1 + 24); k != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
  }
  return result;
}
