// Function: sub_1418D60
// Address: 0x1418d60
//
_QWORD *__fastcall sub_1418D60(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  const __m128i *v7; // r14
  _QWORD *i; // rdx
  const __m128i *v9; // rbx
  unsigned __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  __m128i *v14; // r9
  int v15; // r10d
  unsigned int v16; // ecx
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)((char *)v4 + 72 * v3);
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -4;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = v9->m128i_i64[0];
        if ( v9->m128i_i64[0] != -16 && v10 != -4 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = v9->m128i_i64[0];
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 0;
          v15 = 1;
          v16 = (v11 - 1) & (v10 ^ (v10 >> 9));
          v17 = (__m128i *)(v13 + 72LL * v16);
          v18 = v17->m128i_i64[0];
          if ( v10 != v17->m128i_i64[0] )
          {
            while ( v18 != -4 )
            {
              if ( v18 == -16 && !v14 )
                v14 = v17;
              v16 = v12 & (v15 + v16);
              v17 = (__m128i *)(v13 + 72LL * v16);
              v18 = v17->m128i_i64[0];
              if ( v10 == v17->m128i_i64[0] )
                goto LABEL_14;
              ++v15;
            }
            if ( v14 )
              v17 = v14;
          }
LABEL_14:
          v17->m128i_i64[0] = v9->m128i_i64[0];
          v17->m128i_i64[1] = v9->m128i_i64[1];
          v17[1].m128i_i64[0] = v9[1].m128i_i64[0];
          v17[1].m128i_i64[1] = v9[1].m128i_i64[1];
          v17[2].m128i_i64[0] = v9[2].m128i_i64[0];
          v19 = v9[2].m128i_i64[1];
          v9[2].m128i_i64[0] = 0;
          v9[1].m128i_i64[0] = 0;
          v9[1].m128i_i64[1] = 0;
          v17[2].m128i_i64[1] = v19;
          v17[3] = _mm_loadu_si128(v9 + 3);
          v17[4].m128i_i64[0] = v9[4].m128i_i64[0];
          ++*(_DWORD *)(a1 + 16);
          v20 = v9[1].m128i_i64[0];
          if ( v20 )
            j_j___libc_free_0(v20, v9[2].m128i_i64[0] - v20);
        }
        v9 = (const __m128i *)((char *)v9 + 72);
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * *(unsigned int *)(a1 + 24)]; j != result; result += 9 )
    {
      if ( result )
        *result = -4;
    }
  }
  return result;
}
