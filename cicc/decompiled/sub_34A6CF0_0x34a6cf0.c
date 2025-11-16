// Function: sub_34A6CF0
// Address: 0x34a6cf0
//
_QWORD *__fastcall sub_34A6CF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10; // r12
  _QWORD *i; // rdx
  __int64 j; // rbx
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 *v19; // rdi
  __int64 v20; // rcx
  _QWORD *k; // rdx
  __int64 *v22; // [rsp+18h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 56 * v4;
    v10 = v5 + 56 * v4;
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
        result[2] = -1;
      }
    }
    if ( v10 != v5 )
    {
      for ( j = v5; v10 != j; j += 56 )
      {
        while ( *(_QWORD *)j != -4096 )
        {
          if ( *(_QWORD *)j != -8192 || *(_QWORD *)(j + 8) != -2 || *(_QWORD *)(j + 16) != -2 )
            goto LABEL_15;
LABEL_12:
          j += 56;
          if ( v10 == j )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        if ( *(_QWORD *)(j + 8) != -1 || *(_QWORD *)(j + 16) != -1 )
        {
LABEL_15:
          sub_34A25D0(a1, (__int64 *)j, &v22);
          v17 = *(_QWORD *)j;
          v18 = v22;
          *v22 = *(_QWORD *)j;
          v19 = v22;
          *(__m128i *)(v18 + 1) = _mm_loadu_si128((const __m128i *)(j + 8));
          v19[4] = 0x100000000LL;
          v19[3] = (__int64)(v19 + 5);
          if ( *(_DWORD *)(j + 32) )
            sub_349D880((__int64)(v19 + 3), (char **)(j + 24), v17, v14, v15, v16);
          ++*(_DWORD *)(a1 + 16);
          v13 = *(_QWORD *)(j + 24);
          if ( v13 != j + 40 )
            _libc_free(v13);
          goto LABEL_12;
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[7 * v20]; k != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
        result[2] = -1;
      }
    }
  }
  return result;
}
