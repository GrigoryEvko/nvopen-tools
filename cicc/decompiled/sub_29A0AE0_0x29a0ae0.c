// Function: sub_29A0AE0
// Address: 0x29a0ae0
//
__int64 __fastcall sub_29A0AE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // r10
  __int64 v8; // r11
  __int64 i; // rdx
  const __m128i *v10; // rbx
  __int64 v11; // r12
  const __m128i *v12; // r14
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  size_t v16; // rdx
  __int64 v17; // r11
  char *v18; // rdi
  int v19; // r9d
  unsigned int j; // r13d
  __int64 v21; // r8
  const void *v22; // rcx
  bool v23; // al
  int v24; // eax
  unsigned int v25; // r13d
  __m128i v26; // xmm1
  __int64 k; // rdx
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  size_t v30; // [rsp+18h] [rbp-58h]
  const void *v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]
  int v34; // [rsp+38h] [rbp-38h]
  int v35; // [rsp+3Ch] [rbp-34h]
  int v36; // [rsp+3Ch] [rbp-34h]

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
  result = sub_C7D670(24LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 24 * v3;
    v8 = v4 + 24 * v3;
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = (const __m128i *)v4;
      v11 = v4;
      v12 = (const __m128i *)v8;
      while ( 1 )
      {
        while ( v10->m128i_i64[0] != -1 )
        {
          if ( v10->m128i_i64[0] != -2 || v10[1].m128i_i32[0] != -2 )
            goto LABEL_12;
LABEL_26:
          v10 = (const __m128i *)((char *)v10 + 24);
          if ( v12 == v10 )
            goto LABEL_27;
        }
        if ( v10[1].m128i_i32[0] == -1 )
          goto LABEL_26;
LABEL_12:
        v33 = v7;
        v35 = *(_DWORD *)(a1 + 24);
        if ( !v35 )
        {
          MEMORY[0] = _mm_loadu_si128(v10);
          BUG();
        }
        v32 = *(_QWORD *)(a1 + 8);
        v14 = (unsigned int)(37 * v10[1].m128i_i32[0]);
        v15 = sub_C94890(v10->m128i_i64[0], v10->m128i_i64[1]);
        v16 = v10->m128i_u64[1];
        v17 = 0;
        v18 = (char *)v10->m128i_i64[0];
        v7 = v33;
        v19 = 1;
        v36 = v35 - 1;
        for ( j = v36 & (((0xBF58476D1CE4E5B9LL * ((v15 << 32) | v14)) >> 31) ^ (484763065 * v14)); ; j = v36 & v25 )
        {
          v21 = v32 + 24LL * j;
          v22 = *(const void **)v21;
          if ( *(_QWORD *)v21 == -1 )
            break;
          v23 = v18 + 2 == 0;
          if ( v22 != (const void *)-2LL )
          {
            if ( *(_QWORD *)(v21 + 8) != v16 )
              goto LABEL_23;
            if ( !v16 )
            {
              if ( v10[1].m128i_i32[0] == *(_DWORD *)(v21 + 16) )
                goto LABEL_31;
              goto LABEL_41;
            }
            v28 = v7;
            v34 = v19;
            v29 = v17;
            v30 = v16;
            v31 = *(const void **)v21;
            v24 = memcmp(v18, *(const void **)v21, v16);
            v22 = v31;
            v16 = v30;
            v17 = v29;
            v19 = v34;
            v7 = v28;
            v23 = v24 == 0;
            v21 = v32 + 24LL * j;
          }
          if ( v23 )
            goto LABEL_20;
LABEL_21:
          if ( v22 == (const void *)-1LL )
            goto LABEL_22;
LABEL_41:
          if ( v22 == (const void *)-2LL && *(_DWORD *)(v21 + 16) == -2 && !v17 )
            v17 = v21;
LABEL_23:
          v25 = v19 + j;
          ++v19;
        }
        if ( v18 == (char *)-1LL )
        {
LABEL_20:
          if ( v10[1].m128i_i32[0] == *(_DWORD *)(v21 + 16) )
            goto LABEL_31;
          goto LABEL_21;
        }
LABEL_22:
        if ( *(_DWORD *)(v21 + 16) != -1 )
          goto LABEL_23;
        if ( v17 )
          v21 = v17;
LABEL_31:
        v26 = _mm_loadu_si128(v10);
        v10 = (const __m128i *)((char *)v10 + 24);
        *(__m128i *)v21 = v26;
        *(_DWORD *)(v21 + 16) = v10[-1].m128i_i32[2];
        ++*(_DWORD *)(a1 + 16);
        if ( v12 == v10 )
        {
LABEL_27:
          v4 = v11;
          return sub_C7D6A0(v4, v7, 8);
        }
      }
    }
    return sub_C7D6A0(v4, v7, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24LL * *(unsigned int *)(a1 + 24); k != result; result += 24 )
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
