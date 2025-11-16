// Function: sub_1B7A020
// Address: 0x1b7a020
//
__int64 __fastcall sub_1B7A020(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r14
  __int64 i; // rdx
  const __m128i *j; // rbx
  __m128i *v11; // rax
  __int64 k; // rdx
  __m128i *v13; // [rsp+8h] [rbp-38h] BYREF

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
  result = sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = result + 32 * v7; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v8 != v4 )
    {
      for ( j = v4; v8 != j; j += 2 )
      {
        while ( j->m128i_i64[0] == -1 )
        {
          if ( j[1].m128i_i32[0] != -1 )
            goto LABEL_11;
LABEL_12:
          j += 2;
          if ( v8 == j )
            return j___libc_free_0(v4);
        }
        if ( j->m128i_i64[0] != -2 || j[1].m128i_i32[0] != -2 )
        {
LABEL_11:
          sub_1B79A80(a1, (__int64)j, &v13);
          v11 = v13;
          *v13 = _mm_loadu_si128(j);
          v11[1].m128i_i32[0] = j[1].m128i_i32[0];
          v11[1].m128i_i32[2] = j[1].m128i_i32[2];
          ++*(_DWORD *)(a1 + 16);
          goto LABEL_12;
        }
      }
    }
    return j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32LL * *(unsigned int *)(a1 + 24); k != result; result += 32 )
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
