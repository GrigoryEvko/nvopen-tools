// Function: sub_14C9BF0
// Address: 0x14c9bf0
//
_DWORD *__fastcall sub_14C9BF0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  const __m128i *v7; // r14
  _DWORD *i; // rdx
  const __m128i *v9; // rbx
  const __m128i *v10; // rax
  unsigned int v11; // eax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // r8
  unsigned int *v15; // r9
  int v16; // r10d
  unsigned int v17; // esi
  unsigned int *v18; // rdx
  unsigned int v19; // edi
  const __m128i *v20; // rax
  const __m128i *v21; // rdi
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)(v4 + 72 * v3);
    for ( i = &result[18 * *(unsigned int *)(a1 + 24)]; i != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
    v9 = (const __m128i *)(v4 + 56);
    if ( v7 != (const __m128i *)v4 )
    {
      while ( 1 )
      {
        v11 = v9[-4].m128i_u32[2];
        if ( v11 > 0xFFFFFFFD )
          goto LABEL_10;
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v16 = 1;
        v17 = (v12 - 1) & (37 * v11);
        v18 = (unsigned int *)(v14 + 72LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v15 && v19 == -2 )
              v15 = v18;
            v17 = v13 & (v16 + v17);
            v18 = (unsigned int *)(v14 + 72LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_15;
            ++v16;
          }
          if ( v15 )
            v18 = v15;
        }
LABEL_15:
        *v18 = v11;
        *((_QWORD *)v18 + 1) = v9[-3].m128i_i64[0];
        *((_QWORD *)v18 + 2) = v9[-3].m128i_i64[1];
        *((_QWORD *)v18 + 3) = v9[-2].m128i_i64[0];
        *((_BYTE *)v18 + 32) = v9[-2].m128i_i8[8];
        *((_QWORD *)v18 + 5) = v18 + 14;
        v20 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v20 == v9 )
        {
          *(__m128i *)(v18 + 14) = _mm_loadu_si128(v9);
        }
        else
        {
          *((_QWORD *)v18 + 5) = v20;
          *((_QWORD *)v18 + 7) = v9->m128i_i64[0];
        }
        *((_QWORD *)v18 + 6) = v9[-1].m128i_i64[1];
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        ++*(_DWORD *)(a1 + 16);
        v21 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v21 == v9 )
        {
LABEL_10:
          v10 = (const __m128i *)((char *)v9 + 72);
          if ( v7 == &v9[1] )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
          j_j___libc_free_0(v21, v9->m128i_i64[0] + 1);
          v10 = (const __m128i *)((char *)v9 + 72);
          if ( v7 == &v9[1] )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v9 = v10;
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[18 * *(unsigned int *)(a1 + 24)]; j != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
