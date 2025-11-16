// Function: sub_256AAA0
// Address: 0x256aaa0
//
_QWORD *__fastcall sub_256AAA0(__int64 a1, int a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned int v4; // eax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 *v7; // r15
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 v10; // rdx
  _QWORD *j; // rdx
  __int64 *v12; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_C7D670(8LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (__int64 *)(v3 + 8 * v2);
    for ( i = &result[v6]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    if ( v7 != (__int64 *)v3 )
    {
      v9 = (__int64 *)v3;
      do
      {
        if ( *v9 != -8192 && *v9 != -4096 )
        {
          sub_255F790(a1, v9, &v12);
          *v12 = *v9;
          ++*(_DWORD *)(a1 + 16);
        }
        ++v9;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)sub_C7D6A0(v3, 8 * v2, 8);
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v10]; j != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
