// Function: sub_18FFA60
// Address: 0x18ffa60
//
_QWORD *__fastcall sub_18FFA60(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __m128i *v3; // r13
  unsigned __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __m128i *v7; // r14
  _QWORD *i; // rdx
  __m128i *v9; // rbx
  __m128i *v10; // rax
  _QWORD *j; // rdx
  __m128i *v12; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(__m128i **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(48LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v3[3 * v2];
    for ( i = &result[6 * v6]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = 0;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
      }
    }
    if ( v7 != v3 )
    {
      v9 = v3;
      do
      {
        if ( v9->m128i_i64[0] != -8 && v9->m128i_i64[0] != -16
          || v9->m128i_i64[1]
          || v9[1].m128i_i64[0]
          || v9[1].m128i_i64[1]
          || v9[2].m128i_i64[0] )
        {
          sub_18FEB70(a1, v9->m128i_i64, (__int64 **)&v12);
          v10 = v12;
          *v12 = _mm_loadu_si128(v9);
          v10[1] = _mm_loadu_si128(v9 + 1);
          v10[2].m128i_i64[0] = v9[2].m128i_i64[0];
          v10[2].m128i_i64[1] = v9[2].m128i_i64[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v9 += 3;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[6 * *(unsigned int *)(a1 + 24)]; j != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = 0;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
      }
    }
  }
  return result;
}
