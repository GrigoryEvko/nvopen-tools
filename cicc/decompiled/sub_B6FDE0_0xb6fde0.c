// Function: sub_B6FDE0
// Address: 0xb6fde0
//
_QWORD *__fastcall sub_B6FDE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  const __m128i *v10; // r15
  _QWORD *i; // rdx
  const __m128i *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  __int64 *v18; // r9
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r8
  const __m128i *v22; // rax
  const __m128i *v23; // rdi
  __int64 v24; // rdx
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 40 * v4;
    v10 = (const __m128i *)(v5 + 40 * v4);
    for ( i = &result[5 * v8]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    v12 = (const __m128i *)(v5 + 24);
    if ( v10 != (const __m128i *)v5 )
    {
      while ( 1 )
      {
        v13 = v12[-2].m128i_i64[1];
        if ( v13 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = v12[-2].m128i_i64[1];
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (__int64 *)(v16 + 40LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__int64 *)(v16 + 40LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_15;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_15:
          *v20 = v13;
          v20[1] = (__int64)(v20 + 3);
          v22 = (const __m128i *)v12[-1].m128i_i64[0];
          if ( v12 == v22 )
          {
            *(__m128i *)(v20 + 3) = _mm_loadu_si128(v12);
          }
          else
          {
            v20[1] = (__int64)v22;
            v20[3] = v12->m128i_i64[0];
          }
          v20[2] = v12[-1].m128i_i64[1];
          v12[-1].m128i_i64[0] = (__int64)v12;
          v12[-1].m128i_i64[1] = 0;
          v12->m128i_i8[0] = 0;
          ++*(_DWORD *)(a1 + 16);
          v23 = (const __m128i *)v12[-1].m128i_i64[0];
          if ( v12 != v23 )
            j_j___libc_free_0(v23, v12->m128i_i64[0] + 1);
        }
        if ( v10 == &v12[1] )
          break;
        v12 = (const __m128i *)((char *)v12 + 40);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v24]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
