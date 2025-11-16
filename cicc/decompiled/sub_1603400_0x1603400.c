// Function: sub_1603400
// Address: 0x1603400
//
_QWORD *__fastcall sub_1603400(__int64 a1, int a2)
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
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rdi
  const __m128i *v19; // rax
  const __m128i *v20; // rdi
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)(v4 + 40 * v3);
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
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
          v17 = (__int64 *)(v13 + 40LL * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 40LL * v16);
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
          v17[1] = (__int64)(v17 + 3);
          v19 = (const __m128i *)v9[-1].m128i_i64[0];
          if ( v9 == v19 )
          {
            *(__m128i *)(v17 + 3) = _mm_loadu_si128(v9);
          }
          else
          {
            v17[1] = (__int64)v19;
            v17[3] = v9->m128i_i64[0];
          }
          v17[2] = v9[-1].m128i_i64[1];
          v9[-1].m128i_i64[0] = (__int64)v9;
          v9[-1].m128i_i64[1] = 0;
          v9->m128i_i8[0] = 0;
          ++*(_DWORD *)(a1 + 16);
          v20 = (const __m128i *)v9[-1].m128i_i64[0];
          if ( v9 != v20 )
            j_j___libc_free_0(v20, v9->m128i_i64[0] + 1);
        }
        if ( v7 == &v9[1] )
          break;
        v9 = (const __m128i *)((char *)v9 + 40);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
