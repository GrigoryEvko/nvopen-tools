// Function: sub_210EDA0
// Address: 0x210eda0
//
_QWORD *__fastcall sub_210EDA0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // r14
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  int v12; // ecx
  __int64 v13; // rdi
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // edx
  __int64 *v17; // r8
  __int64 v18; // rsi
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __int64 v22; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(104LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[13 * v3];
    for ( i = &result[13 * *(unsigned int *)(a1 + 24)]; i != result; result += 13 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *v9;
        if ( *v9 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 104LL * v16);
          v18 = *v17;
          if ( *v17 != v10 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 104LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_14:
          *v17 = v10;
          *(__m128i *)(v17 + 1) = _mm_loadu_si128((const __m128i *)(v9 + 1));
          *((_DWORD *)v17 + 6) = *((_DWORD *)v9 + 6);
          v19 = _mm_loadu_si128((const __m128i *)v9 + 2);
          v9[1] = 0;
          v9[2] = 0;
          *((_DWORD *)v9 + 6) = 0;
          *((__m128i *)v17 + 2) = v19;
          *((_DWORD *)v17 + 12) = *((_DWORD *)v9 + 12);
          v20 = _mm_loadu_si128((const __m128i *)(v9 + 7));
          v9[4] = 0;
          v9[5] = 0;
          *((_DWORD *)v9 + 12) = 0;
          *(__m128i *)(v17 + 7) = v20;
          *((_DWORD *)v17 + 18) = *((_DWORD *)v9 + 18);
          v21 = _mm_loadu_si128((const __m128i *)v9 + 5);
          v9[7] = 0;
          v9[8] = 0;
          *((_DWORD *)v9 + 18) = 0;
          *((__m128i *)v17 + 5) = v21;
          *((_DWORD *)v17 + 24) = *((_DWORD *)v9 + 24);
          v9[10] = 0;
          v9[11] = 0;
          *((_DWORD *)v9 + 24) = 0;
          ++*(_DWORD *)(a1 + 16);
          _libc_free(v9[10]);
          _libc_free(v9[7]);
          _libc_free(v9[4]);
          _libc_free(v9[1]);
        }
        v9 += 13;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[13 * v22]; j != result; result += 13 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
