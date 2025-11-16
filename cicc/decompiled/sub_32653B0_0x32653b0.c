// Function: sub_32653B0
// Address: 0x32653b0
//
__int64 __fastcall sub_32653B0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // [rsp+0h] [rbp-30h]
  __int64 v8; // [rsp+0h] [rbp-30h]
  __int64 v9; // [rsp+8h] [rbp-28h]

  v2 = *((_QWORD *)a1 + 1);
  result = sub_33CB110(*(unsigned int *)(a2 + 24));
  if ( !(_BYTE)result )
  {
    if ( *(_DWORD *)(a2 + 24) != 98 )
      return result;
    goto LABEL_3;
  }
  v9 = sub_33CB280(*(unsigned int *)(a2 + 24), ((unsigned __int8)(*(_DWORD *)(a2 + 28) >> 12) ^ 1) & 1);
  if ( !BYTE4(v9) || (_DWORD)v9 != 98 )
    return 0;
  v4 = *(_DWORD *)(a2 + 24);
  v7 = sub_33CB160(v4);
  if ( !BYTE4(v7)
    || (v5 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v7, *(_QWORD *)v5 == *(_QWORD *)(v2 + 16))
    && *(_DWORD *)(v5 + 8) == *(_DWORD *)(v2 + 24)
    || (result = sub_33D1720(*(_QWORD *)v5, 0), (_BYTE)result) )
  {
    v8 = sub_33CB1F0(v4);
    if ( !BYTE4(v8)
      || (v6 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v8, *(_QWORD *)(v2 + 32) == *(_QWORD *)v6)
      && *(_DWORD *)(v2 + 40) == *(_DWORD *)(v6 + 8) )
    {
LABEL_3:
      result = *a1;
      if ( !(_BYTE)result )
        return (*(_DWORD *)(a2 + 28) >> 9) & 1;
      return result;
    }
    return 0;
  }
  return result;
}
