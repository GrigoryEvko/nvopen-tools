// Function: sub_1FDCE70
// Address: 0x1fdce70
//
__int64 __fastcall sub_1FDCE70(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v6; // rax
  __int64 v7; // r15
  int v8; // r14d
  __int64 v9; // rdi
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 (*v16)(); // rax

  v6 = sub_1648700(*(_QWORD *)(a2 + 8));
  if ( a3 != v6 )
  {
    v7 = a3[5];
    v8 = 6;
    while ( v6[5] == v7 )
    {
      if ( !--v8 )
        break;
      v9 = v6[1];
      if ( !v9 || *(_QWORD *)(v9 + 8) )
        break;
      v6 = sub_1648700(v9);
      if ( v6 == a3 )
        goto LABEL_8;
    }
    return 0;
  }
LABEL_8:
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    return 0;
  v11 = sub_1FD8F60(a1, a2);
  if ( !v11 )
    return 0;
  v12 = a1[7];
  v13 = v11 < 0
      ? *(_QWORD *)(*(_QWORD *)(v12 + 24) + 16LL * (v11 & 0x7FFFFFFF) + 8)
      : *(_QWORD *)(*(_QWORD *)(v12 + 272) + 8LL * (unsigned int)v11);
  if ( !v13 )
    return 0;
  v14 = v13;
  if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
    goto LABEL_17;
  v14 = *(_QWORD *)(v13 + 32);
  if ( !v14 )
    return 0;
  while ( (*(_BYTE *)(v14 + 3) & 0x10) != 0 )
  {
    v14 = *(_QWORD *)(v14 + 32);
    if ( !v14 )
      return 0;
  }
LABEL_17:
  while ( 1 )
  {
    v14 = *(_QWORD *)(v14 + 32);
    if ( !v14 )
      break;
    if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
      return 0;
  }
  v15 = *(_QWORD *)(v13 + 16);
  *(_QWORD *)(a1[5] + 792LL) = v15;
  *(_QWORD *)(a1[5] + 784LL) = *(_QWORD *)(v15 + 24);
  v16 = *(__int64 (**)())(*a1 + 16LL);
  if ( v16 == sub_1FD3450 )
    return 0;
  return ((__int64 (__fastcall *)(_QWORD *, __int64, unsigned __int64, __int64))v16)(
           a1,
           v15,
           0xCCCCCCCCCCCCCCCDLL * ((v13 - *(_QWORD *)(*(_QWORD *)(v13 + 16) + 32LL)) >> 3),
           a2);
}
