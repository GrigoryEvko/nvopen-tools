// Function: sub_29A03D0
// Address: 0x29a03d0
//
__int64 __fastcall sub_29A03D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // r10
  __int64 v10; // r11
  __int64 i; // rdx
  const __m128i *v12; // rbx
  const __m128i *v13; // r14
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  char *v16; // rdi
  __m128i *v17; // r10
  size_t v18; // rdx
  int v19; // r9d
  int v20; // r11d
  unsigned int j; // ecx
  __int64 v22; // r8
  const void *v23; // r13
  bool v24; // al
  int v25; // eax
  unsigned int v26; // ecx
  __m128i *v27; // r13
  __int32 v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 k; // rdx
  __int64 v32; // [rsp+0h] [rbp-70h]
  int v33; // [rsp+Ch] [rbp-64h]
  __m128i *v34; // [rsp+10h] [rbp-60h]
  unsigned int v35; // [rsp+18h] [rbp-58h]
  int v36; // [rsp+1Ch] [rbp-54h]
  size_t v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39; // [rsp+30h] [rbp-40h]
  int v40; // [rsp+38h] [rbp-38h]

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
  result = sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 56 * v4;
    v10 = v5 + 56 * v4;
    for ( i = result + 56 * v8; i != result; result += 56 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v10 != v5 )
    {
      v39 = 56 * v4;
      v12 = (const __m128i *)v5;
      v13 = (const __m128i *)v10;
      while ( 1 )
      {
        while ( v12->m128i_i64[0] != -1 )
        {
          if ( v12->m128i_i64[0] != -2 || v12[1].m128i_i32[0] != -2 )
            goto LABEL_12;
LABEL_26:
          v12 = (const __m128i *)((char *)v12 + 56);
          if ( v13 == v12 )
            goto LABEL_27;
        }
        if ( v12[1].m128i_i32[0] == -1 )
          goto LABEL_26;
LABEL_12:
        v40 = *(_DWORD *)(a1 + 24);
        if ( !v40 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          BUG();
        }
        v38 = *(_QWORD *)(a1 + 8);
        v14 = (unsigned int)(37 * v12[1].m128i_i32[0]);
        v15 = sub_C94890(v12->m128i_i64[0], v12->m128i_i64[1]);
        v16 = (char *)v12->m128i_i64[0];
        v17 = 0;
        v18 = v12->m128i_u64[1];
        v19 = 1;
        v20 = v40 - 1;
        for ( j = (v40 - 1) & (((0xBF58476D1CE4E5B9LL * (v14 | (v15 << 32))) >> 31) ^ (484763065 * v14)); ; j = v20 & v26 )
        {
          v22 = v38 + 56LL * j;
          v23 = *(const void **)v22;
          if ( *(_QWORD *)v22 == -1 )
            break;
          v24 = v16 + 2 == 0;
          if ( v23 == (const void *)-2LL )
            goto LABEL_19;
          if ( *(_QWORD *)(v22 + 8) != v18 )
            goto LABEL_23;
          if ( v18 )
          {
            v32 = v38 + 56LL * j;
            v33 = v19;
            v34 = v17;
            v35 = j;
            v36 = v20;
            v37 = v18;
            v25 = memcmp(v16, *(const void **)v22, v18);
            v18 = v37;
            v20 = v36;
            j = v35;
            v17 = v34;
            v19 = v33;
            v24 = v25 == 0;
            v22 = v32;
LABEL_19:
            if ( !v24 )
              goto LABEL_21;
          }
LABEL_20:
          if ( v12[1].m128i_i32[0] == *(_DWORD *)(v22 + 16) )
          {
            v27 = (__m128i *)v22;
            goto LABEL_32;
          }
LABEL_21:
          if ( v23 == (const void *)-1LL )
            goto LABEL_22;
          if ( v23 == (const void *)-2LL && *(_DWORD *)(v22 + 16) == -2 && !v17 )
            v17 = (__m128i *)v22;
LABEL_23:
          v26 = v19 + j;
          ++v19;
        }
        if ( v16 == (char *)-1LL )
          goto LABEL_20;
LABEL_22:
        if ( *(_DWORD *)(v22 + 16) != -1 )
          goto LABEL_23;
        v27 = (__m128i *)v22;
        if ( v17 )
          v27 = v17;
LABEL_32:
        *v27 = _mm_loadu_si128(v12);
        v28 = v12[1].m128i_i32[0];
        v27[2].m128i_i64[1] = 0;
        v27[2].m128i_i64[0] = 0;
        v27[3].m128i_i32[0] = 0;
        v27[1].m128i_i32[0] = v28;
        v27[1].m128i_i64[1] = 1;
        v29 = v12[2].m128i_i64[0];
        ++v12[1].m128i_i64[1];
        v30 = v27[2].m128i_i64[0];
        v12 = (const __m128i *)((char *)v12 + 56);
        v27[2].m128i_i64[0] = v29;
        LODWORD(v29) = v12[-1].m128i_i32[0];
        v12[-2].m128i_i64[1] = v30;
        LODWORD(v30) = v27[2].m128i_i32[2];
        v27[2].m128i_i32[2] = v29;
        LODWORD(v29) = v12[-1].m128i_i32[1];
        v12[-1].m128i_i32[0] = v30;
        LODWORD(v30) = v27[2].m128i_i32[3];
        v27[2].m128i_i32[3] = v29;
        LODWORD(v29) = v12[-1].m128i_i32[2];
        v12[-1].m128i_i32[1] = v30;
        LODWORD(v30) = v27[3].m128i_i32[0];
        v27[3].m128i_i32[0] = v29;
        v12[-1].m128i_i32[2] = v30;
        ++*(_DWORD *)(a1 + 16);
        sub_C7D6A0(v12[-2].m128i_i64[1], 8LL * v12[-1].m128i_u32[2], 8);
        if ( v13 == v12 )
        {
LABEL_27:
          v9 = v39;
          return sub_C7D6A0(v5, v9, 8);
        }
      }
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 56LL * *(unsigned int *)(a1 + 24); k != result; result += 56 )
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
