// Function: sub_1B7A1D0
// Address: 0x1b7a1d0
//
__int64 __fastcall sub_1B7A1D0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  const __m128i *v3; // r13
  unsigned __int64 v4; // rax
  __int64 result; // rax
  const __m128i *v6; // r14
  __int64 i; // rdx
  const __m128i *j; // rbx
  __m128i *v9; // rax
  __int64 k; // rdx
  __m128i *v11; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(const __m128i **)(a1 + 8);
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
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = (const __m128i *)((char *)v3 + 24 * v2);
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v6 != v3 )
    {
      for ( j = v3; v6 != j; j = (const __m128i *)((char *)j + 24) )
      {
        while ( j->m128i_i64[0] == -1 )
        {
          if ( j[1].m128i_i32[0] != -1 )
            goto LABEL_11;
LABEL_12:
          j = (const __m128i *)((char *)j + 24);
          if ( v6 == j )
            return j___libc_free_0(v3);
        }
        if ( j->m128i_i64[0] != -2 || j[1].m128i_i32[0] != -2 )
        {
LABEL_11:
          sub_1B79C30(a1, (__int64)j, &v11);
          v9 = v11;
          *v11 = _mm_loadu_si128(j);
          v9[1].m128i_i32[0] = j[1].m128i_i32[0];
          ++*(_DWORD *)(a1 + 16);
          goto LABEL_12;
        }
      }
    }
    return j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24LL * *(unsigned int *)(a1 + 24); k != result; result += 24 )
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
