// Function: sub_1874300
// Address: 0x1874300
//
_QWORD *__fastcall sub_1874300(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r14
  _QWORD *i; // rdx
  const __m128i *v10; // rbx
  __int64 v11; // rsi
  __m128i *v12; // rax
  _QWORD *j; // rdx
  __m128i *v14; // [rsp+8h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( v10->m128i_i64[0] == -1 || v10->m128i_i64[0] == -2 )
        {
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        v11 = (__int64)v10;
        v10 += 2;
        sub_1872D70(a1, v11, &v14);
        v12 = v14;
        *v14 = _mm_loadu_si128(v10 - 2);
        v12[1] = _mm_loadu_si128(v10 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * *(unsigned int *)(a1 + 24)]; j != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
      }
    }
  }
  return result;
}
