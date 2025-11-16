// Function: sub_B03490
// Address: 0xb03490
//
_QWORD *__fastcall sub_B03490(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // rdx
  __int64 *i; // r13
  _QWORD *j; // rdx
  __int64 *v13; // [rsp+8h] [rbp-48h]
  __int64 *v14; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v13 = v4;
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &result[v8];
    for ( i = &v4[v9]; v10 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( ; i != v4; ++v4 )
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        sub_AFC700(a1, v4, &v14);
        *v14 = *v4;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v13, v9 * 8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
