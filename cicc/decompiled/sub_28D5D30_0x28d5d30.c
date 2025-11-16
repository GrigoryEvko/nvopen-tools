// Function: sub_28D5D30
// Address: 0x28d5d30
//
_QWORD *__fastcall sub_28D5D30(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // eax
  _QWORD *result; // rax
  __int64 v6; // r15
  _QWORD *i; // rdx
  __int64 v8; // rbx
  __int64 v9; // rcx
  _QWORD *j; // rdx
  __int64 v11; // [rsp+8h] [rbp-48h]
  _QWORD *v12; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_C7D670(56LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v11 = 56 * v2;
    v6 = v3 + 56 * v2;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
    if ( v6 != v3 )
    {
      v8 = v3;
      do
      {
        if ( *(_QWORD *)v8 != 0x7FFFFFFF0LL && *(_QWORD *)v8 != -8 )
        {
          sub_28CCF80(a1, (__int64 *)v8, &v12);
          *v12 = *(_QWORD *)v8;
          sub_C8CF70((__int64)(v12 + 1), v12 + 5, 2, v8 + 40, v8 + 8);
          ++*(_DWORD *)(a1 + 16);
          if ( !*(_BYTE *)(v8 + 36) )
            _libc_free(*(_QWORD *)(v8 + 16));
        }
        v8 += 56;
      }
      while ( v6 != v8 );
    }
    return (_QWORD *)sub_C7D6A0(v3, v11, 8);
  }
  else
  {
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v9]; j != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
