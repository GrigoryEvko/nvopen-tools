// Function: sub_326A1D0
// Address: 0x326a1d0
//
__int64 __fastcall sub_326A1D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v3; // r15d
  unsigned __int64 v4; // r13
  unsigned __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+0h] [rbp-40h]
  __int64 v16; // [rsp+0h] [rbp-40h]
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v2 = *a1;
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(a2 + 24)) )
  {
    v6 = sub_33CB280(*(unsigned int *)(a2 + 24), ((unsigned __int8)(*(_DWORD *)(a2 + 28) >> 12) ^ 1) & 1);
    v3 = *(_DWORD *)(a2 + 24);
    v4 = HIDWORD(v6);
    if ( BYTE4(v6) && (_DWORD)v6 == 150 )
    {
      v13 = sub_33CB160(v3);
      if ( !BYTE4(v13)
        || (v8 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v13, *(_QWORD *)v8 == *(_QWORD *)(v2 + 16))
        && *(_DWORD *)(v8 + 8) == *(_DWORD *)(v2 + 24)
        || (unsigned __int8)sub_33D1720(*(_QWORD *)v8, 0) )
      {
        v14 = sub_33CB1F0(v3);
        if ( !BYTE4(v14) )
          return (unsigned int)v4;
        v9 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v14;
        if ( *(_QWORD *)(v2 + 32) == *(_QWORD *)v9 && *(_DWORD *)(v2 + 40) == *(_DWORD *)(v9 + 8) )
          return (unsigned int)v4;
      }
      v3 = *(_DWORD *)(a2 + 24);
    }
  }
  else
  {
    v3 = *(_DWORD *)(a2 + 24);
    LODWORD(v4) = 1;
    if ( v3 == 150 )
      return (unsigned int)v4;
  }
  v7 = *a1;
  if ( (unsigned __int8)sub_33CB110(v3) )
  {
    v17 = sub_33CB280(*(unsigned int *)(a2 + 24), ((unsigned __int8)(*(_DWORD *)(a2 + 28) >> 12) ^ 1) & 1);
    v4 = HIDWORD(v17);
    if ( !BYTE4(v17)
      || (_DWORD)v17 != 151
      || (v10 = *(_DWORD *)(a2 + 24), v15 = sub_33CB160(v10), BYTE4(v15))
      && ((v11 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v15, *(_QWORD *)v11 != *(_QWORD *)(v7 + 16))
       || *(_DWORD *)(v11 + 8) != *(_DWORD *)(v7 + 24))
      && !(unsigned __int8)sub_33D1720(*(_QWORD *)v11, 0)
      || (v16 = sub_33CB1F0(v10), BYTE4(v16))
      && ((v12 = *(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v16, *(_QWORD *)(v7 + 32) != *(_QWORD *)v12)
       || *(_DWORD *)(v7 + 40) != *(_DWORD *)(v12 + 8)) )
    {
      LODWORD(v4) = 0;
    }
  }
  else
  {
    LOBYTE(v4) = *(_DWORD *)(a2 + 24) == 151;
  }
  return (unsigned int)v4;
}
