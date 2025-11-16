// Function: sub_1928450
// Address: 0x1928450
//
_QWORD *__fastcall sub_1928450(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  const __m128i *v7; // r14
  _QWORD *i; // rdx
  const __m128i *v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // edx
  __int64 *v17; // r15
  __int64 v18; // rdi
  __m128i *v19; // rax
  unsigned int v20; // r8d
  unsigned __int64 v21; // rdi
  const __m128i *v22; // rdx
  __int64 v23; // rsi
  const __m128i *v24; // rdx
  const __m128i *j; // rdi
  _QWORD *k; // rdx
  unsigned int v27; // [rsp+Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)(v4 + 72 * v3);
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
    v9 = (const __m128i *)(v4 + 24);
    if ( v7 != (const __m128i *)v4 )
    {
      while ( 1 )
      {
        v10 = v9[-2].m128i_i64[1];
        if ( v10 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = v9[-2].m128i_i64[1];
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 72LL * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 72LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_15;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_15:
          *v17 = v10;
          v19 = (__m128i *)(v17 + 3);
          v17[1] = (__int64)(v17 + 3);
          v17[2] = 0x200000000LL;
          v20 = v9[-1].m128i_u32[2];
          if ( v20 && v17 + 1 != (__int64 *)&v9[-1] )
          {
            v22 = (const __m128i *)v9[-1].m128i_i64[0];
            if ( v22 == v9 )
            {
              v23 = v20;
              v24 = v9;
              if ( v20 > 2 )
              {
                v27 = v9[-1].m128i_u32[2];
                sub_1923080((__int64)(v17 + 1), v20);
                v19 = (__m128i *)v17[1];
                v24 = (const __m128i *)v9[-1].m128i_i64[0];
                v23 = v9[-1].m128i_u32[2];
                v20 = v27;
              }
              for ( j = (const __m128i *)((char *)v24 + 24 * v23); j != v24; v19 = (__m128i *)((char *)v19 + 24) )
              {
                if ( v19 )
                {
                  *v19 = _mm_loadu_si128(v24);
                  v19[1].m128i_i64[0] = v24[1].m128i_i64[0];
                }
                v24 = (const __m128i *)((char *)v24 + 24);
              }
              *((_DWORD *)v17 + 4) = v20;
              v9[-1].m128i_i32[2] = 0;
            }
            else
            {
              v17[1] = (__int64)v22;
              *((_DWORD *)v17 + 4) = v9[-1].m128i_i32[2];
              *((_DWORD *)v17 + 5) = v9[-1].m128i_i32[3];
              v9[-1].m128i_i64[0] = (__int64)v9;
              v9[-1].m128i_i32[3] = 0;
              v9[-1].m128i_i32[2] = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v21 = v9[-1].m128i_u64[0];
          if ( (const __m128i *)v21 != v9 )
            _libc_free(v21);
        }
        if ( v7 == &v9[3] )
          break;
        v9 = (const __m128i *)((char *)v9 + 72);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[9 * *(unsigned int *)(a1 + 24)]; k != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
