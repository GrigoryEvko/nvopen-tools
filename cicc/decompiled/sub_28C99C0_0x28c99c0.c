// Function: sub_28C99C0
// Address: 0x28c99c0
//
_QWORD *__fastcall sub_28C99C0(__int64 a1, int a2)
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
  result = (_QWORD *)sub_C7D670(16LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (__int64 *)(v3 + 16 * v2);
    for ( i = &result[2 * v6]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != (__int64 *)v3 )
    {
      v9 = (__int64 *)v3;
      do
      {
        if ( *v9 != 0x7FFFFFFF0LL && *v9 != -8 )
        {
          sub_28C8F20(a1, v9, &v12);
          *v12 = *v9;
          v12[1] = v9[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v9 += 2;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)sub_C7D6A0(v3, 16 * v2, 8);
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v10]; j != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
