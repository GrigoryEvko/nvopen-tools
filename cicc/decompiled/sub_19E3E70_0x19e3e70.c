// Function: sub_19E3E70
// Address: 0x19e3e70
//
_QWORD *__fastcall sub_19E3E70(__int64 a1, int a2)
{
  __int64 v2; // r14
  __int64 *v3; // r12
  unsigned int v4; // eax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 *v7; // r14
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rdx
  _QWORD *j; // rdx
  __int64 *v13; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(__int64 **)(a1 + 8);
  v4 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(16LL * v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v3[2 * v2];
    for ( i = &result[2 * v6]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v3 )
    {
      v9 = v3;
      do
      {
        if ( *v9 != 0x7FFFFFFF0LL && *v9 != -8 )
        {
          sub_19E3400(a1, v9, &v13);
          v10 = v13;
          *v13 = *v9;
          v10[1] = v9[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v9 += 2;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v3);
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v11]; j != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
