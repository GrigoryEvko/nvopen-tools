// Function: sub_1B79DE0
// Address: 0x1b79de0
//
__int64 __fastcall sub_1B79DE0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  const __m128i *v3; // r13
  unsigned __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rcx
  const __m128i *v7; // r14
  __int64 i; // rdx
  const __m128i *j; // rbx
  __m128i *v10; // rax
  __int32 v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 k; // rdx
  __m128i *v15; // [rsp+8h] [rbp-38h] BYREF

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
  result = sub_22077B0(56LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)((char *)v3 + 56 * v2);
    for ( i = result + 56 * v6; i != result; result += 56 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = -1;
      }
    }
    if ( v7 != v3 )
    {
      for ( j = v3; v7 != j; j = (const __m128i *)((char *)j + 56) )
      {
        while ( j->m128i_i64[0] == -1 )
        {
          if ( j[1].m128i_i32[0] != -1 )
            goto LABEL_11;
LABEL_12:
          j = (const __m128i *)((char *)j + 56);
          if ( v7 == j )
            return j___libc_free_0(v3);
        }
        if ( j->m128i_i64[0] != -2 || j[1].m128i_i32[0] != -2 )
        {
LABEL_11:
          sub_1B798C0(a1, (__int64)j, &v15);
          v10 = v15;
          *v15 = _mm_loadu_si128(j);
          v11 = j[1].m128i_i32[0];
          v10[2].m128i_i64[1] = 0;
          v10[2].m128i_i64[0] = 0;
          v10[3].m128i_i32[0] = 0;
          v10[1].m128i_i32[0] = v11;
          v10[1].m128i_i64[1] = 1;
          v12 = j[2].m128i_i64[0];
          ++j[1].m128i_i64[1];
          v13 = v10[2].m128i_i64[0];
          v10[2].m128i_i64[0] = v12;
          LODWORD(v12) = j[2].m128i_i32[2];
          j[2].m128i_i64[0] = v13;
          LODWORD(v13) = v10[2].m128i_i32[2];
          v10[2].m128i_i32[2] = v12;
          LODWORD(v12) = j[2].m128i_i32[3];
          j[2].m128i_i32[2] = v13;
          LODWORD(v13) = v10[2].m128i_i32[3];
          v10[2].m128i_i32[3] = v12;
          LODWORD(v12) = j[3].m128i_i32[0];
          j[2].m128i_i32[3] = v13;
          LODWORD(v13) = v10[3].m128i_i32[0];
          v10[3].m128i_i32[0] = v12;
          j[3].m128i_i32[0] = v13;
          ++*(_DWORD *)(a1 + 16);
          j___libc_free_0(j[2].m128i_i64[0]);
          goto LABEL_12;
        }
      }
    }
    return j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 56LL * *(unsigned int *)(a1 + 24); k != result; result += 56 )
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
