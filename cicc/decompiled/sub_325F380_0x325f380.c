// Function: sub_325F380
// Address: 0x325f380
//
__int64 __fastcall sub_325F380(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 v3; // r13
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // [rsp+0h] [rbp-30h]
  __int64 v10; // [rsp+8h] [rbp-28h]
  __int64 v11; // [rsp+8h] [rbp-28h]

  if ( !(unsigned __int8)sub_33CB110(*(unsigned int *)(a2 + 24)) )
  {
    LOBYTE(v3) = *(_DWORD *)(a2 + 24) == a3;
    return (unsigned int)v3;
  }
  v9 = sub_33CB280(*(unsigned int *)(a2 + 24), ((unsigned __int8)(*(_DWORD *)(a2 + 28) >> 12) ^ 1) & 1);
  v3 = HIDWORD(v9);
  if ( !BYTE4(v9) )
    return (unsigned int)v3;
  if ( a3 == (_DWORD)v9 )
  {
    if ( (v6 = *(_DWORD *)(a2 + 24), v10 = sub_33CB160(v6), !BYTE4(v10))
      || (v7 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v10, *(_QWORD *)v7 == *(_QWORD *)(a1 + 16))
      && *(_DWORD *)(v7 + 8) == *(_DWORD *)(a1 + 24)
      || (unsigned __int8)sub_33D1720(*(_QWORD *)v7, 0) )
    {
      v11 = sub_33CB1F0(v6);
      if ( !BYTE4(v11) )
        return (unsigned int)v3;
      v8 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v11;
      if ( *(_QWORD *)(a1 + 32) == *(_QWORD *)v8 && *(_DWORD *)(a1 + 40) == *(_DWORD *)(v8 + 8) )
        return (unsigned int)v3;
    }
  }
  return 0;
}
