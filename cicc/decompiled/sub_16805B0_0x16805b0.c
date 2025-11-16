// Function: sub_16805B0
// Address: 0x16805b0
//
__int64 __fastcall sub_16805B0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __int64 result; // rax
  const __m128i *v6; // r11
  const __m128i *v7; // r10
  __int64 i; // rdx
  const __m128i *v9; // rbx
  __int32 v10; // r12d
  int v11; // r14d
  int v12; // r14d
  __int64 v13; // r9
  __m128i *v14; // r11
  unsigned int j; // r8d
  __m128i *v16; // rcx
  __int32 v17; // r15d
  unsigned int v18; // r8d
  char *v19; // rdi
  const void *v20; // rsi
  bool v21; // al
  size_t v22; // rdx
  __m128i *v23; // r15
  int v24; // eax
  __int64 v25; // rdx
  __int64 k; // rdx
  __m128i *v27; // [rsp+8h] [rbp-68h]
  const __m128i *v28; // [rsp+10h] [rbp-60h]
  __m128i *v29; // [rsp+18h] [rbp-58h]
  unsigned int v30; // [rsp+24h] [rbp-4Ch]
  __int64 v31; // [rsp+28h] [rbp-48h]
  const __m128i *v32; // [rsp+30h] [rbp-40h]
  const __m128i *v33; // [rsp+38h] [rbp-38h]
  int v34; // [rsp+38h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v33 = *(const __m128i **)(a1 + 8);
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_22077B0(24LL * (unsigned int)v4);
  v6 = v33;
  *(_QWORD *)(a1 + 8) = result;
  if ( v33 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)((char *)v33 + 24 * v3);
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
    if ( v7 != v33 )
    {
      v32 = v33;
      v9 = v33;
      while ( 1 )
      {
        v10 = v9->m128i_i32[3];
        if ( v10 )
        {
          if ( v10 == 1 && v9->m128i_i64[0] == -2 )
            goto LABEL_22;
        }
        else if ( v9->m128i_i64[0] == -1 )
        {
          goto LABEL_22;
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = _mm_loadu_si128(v9);
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 0;
        v34 = 1;
        for ( j = v12 & v10; ; j = v12 & v18 )
        {
          v16 = (__m128i *)(v13 + 24LL * j);
          v17 = v16->m128i_i32[3];
          if ( v10 != v17 )
            goto LABEL_15;
          v19 = (char *)v9->m128i_i64[0];
          v20 = (const void *)v16->m128i_i64[0];
          v21 = v9->m128i_i64[0] == -1;
          if ( v16->m128i_i64[0] == -1 )
            break;
          v21 = v19 + 2 == 0;
          if ( v20 == (const void *)-2LL )
            break;
          v22 = v16->m128i_u32[2];
          if ( v22 == v9->m128i_i32[2] )
          {
            v29 = v14;
            v30 = j;
            v31 = v13;
            if ( !v16->m128i_i32[2] )
              goto LABEL_33;
            v27 = (__m128i *)(v13 + 24LL * j);
            v28 = v7;
            v24 = memcmp(v19, v20, v22);
            v7 = v28;
            v16 = v27;
            v13 = v31;
            j = v30;
            v14 = v29;
            if ( !v24 )
            {
LABEL_33:
              v23 = v16;
              goto LABEL_34;
            }
          }
          if ( v17 )
            goto LABEL_16;
LABEL_20:
          v18 = v34 + j;
          ++v34;
        }
        if ( v21 )
          goto LABEL_33;
LABEL_15:
        if ( v17 )
        {
LABEL_16:
          if ( v17 == 1 && v16->m128i_i64[0] == -2 && !v14 )
            v14 = v16;
          goto LABEL_20;
        }
        if ( v16->m128i_i64[0] != -1 )
          goto LABEL_20;
        v23 = (__m128i *)(v13 + 24LL * j);
        if ( v14 )
          v23 = v14;
LABEL_34:
        *v23 = _mm_loadu_si128(v9);
        v23[1].m128i_i64[0] = v9[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 16);
LABEL_22:
        v9 = (const __m128i *)((char *)v9 + 24);
        if ( v7 == v9 )
        {
          v6 = v32;
          return j___libc_free_0(v6);
        }
      }
    }
    return j___libc_free_0(v6);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v25; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
  }
  return result;
}
