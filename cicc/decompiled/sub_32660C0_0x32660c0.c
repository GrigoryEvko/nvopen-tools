// Function: sub_32660C0
// Address: 0x32660c0
//
__int64 __fastcall sub_32660C0(_QWORD **a1, __int64 a2)
{
  _BYTE *v2; // r13
  __int64 v3; // r14
  __int64 result; // rax
  unsigned int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // [rsp+0h] [rbp-40h]
  __int64 v9; // [rsp+0h] [rbp-40h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v2 = *a1;
  v3 = (*a1)[1];
  if ( !(unsigned __int8)sub_33CB110(*(unsigned int *)(a2 + 24)) )
  {
    if ( *(_DWORD *)(a2 + 24) != 98 )
      return 0;
    goto LABEL_3;
  }
  v10 = sub_33CB280(*(unsigned int *)(a2 + 24), ((unsigned __int8)(*(_DWORD *)(a2 + 28) >> 12) ^ 1) & 1);
  if ( !BYTE4(v10) || (_DWORD)v10 != 98 )
    return 0;
  v5 = *(_DWORD *)(a2 + 24);
  v8 = sub_33CB160(v5);
  if ( !BYTE4(v8)
    || (v6 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v8, *(_QWORD *)v6 == *(_QWORD *)(v3 + 16))
    && *(_DWORD *)(v6 + 8) == *(_DWORD *)(v3 + 24)
    || (result = sub_33D1720(*(_QWORD *)v6, 0), (_BYTE)result) )
  {
    v9 = sub_33CB1F0(v5);
    if ( BYTE4(v9) )
    {
      v7 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v9;
      if ( *(_QWORD *)(v3 + 32) != *(_QWORD *)v7 )
        return 0;
      if ( *(_DWORD *)(v3 + 40) != *(_DWORD *)(v7 + 8) )
        return 0;
    }
LABEL_3:
    if ( *v2 || (*(_BYTE *)(a2 + 29) & 2) != 0 )
    {
      result = 1;
      if ( (*(_BYTE *)(*a1[1] + 8LL) & 1) == 0 )
        return (*(_DWORD *)(a2 + 28) >> 11) & 1;
      return result;
    }
    return 0;
  }
  return result;
}
