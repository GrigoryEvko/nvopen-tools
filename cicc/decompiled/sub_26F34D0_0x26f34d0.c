// Function: sub_26F34D0
// Address: 0x26f34d0
//
char __fastcall sub_26F34D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v20[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 24);
  v19 = a2;
  v5 = **(_QWORD **)(v4 + 16);
  if ( *(_BYTE *)(v5 + 8) != 12 || *(_DWORD *)(v5 + 8) > 0x40FFu || !*(_QWORD *)(a2 + 104) )
    return v5;
  if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
  {
    v5 = *(_QWORD *)(a2 + 96);
    if ( *(_QWORD *)(v5 + 16) )
      return v5;
    v9 = a2;
    goto LABEL_12;
  }
  sub_B2C6D0(a2, a2, a3, a4);
  v5 = *(_QWORD *)(a2 + 96);
  if ( *(_QWORD *)(v5 + 16) )
    return v5;
  v9 = v19;
  if ( (*(_BYTE *)(v19 + 2) & 1) == 0 )
  {
LABEL_12:
    v12 = *(_QWORD *)(v9 + 96);
    v13 = v12 + 40LL * *(_QWORD *)(v9 + 104);
    goto LABEL_13;
  }
  sub_B2C6D0(v19, a2, v7, v8);
  v12 = *(_QWORD *)(v19 + 96);
  v13 = v12 + 40LL * *(_QWORD *)(v19 + 104);
  if ( (*(_BYTE *)(v19 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v19, a2, v10, v11);
    v12 = *(_QWORD *)(v19 + 96);
  }
LABEL_13:
  v5 = sub_26F15C0(v12, v13, 1);
  v15 = v14;
  if ( v5 == v14 )
  {
LABEL_17:
    LOBYTE(v5) = sub_B2FC80(v19);
    if ( !(_BYTE)v5 )
    {
      v17 = (*(__int64 (__fastcall **)(_QWORD, __int64))*a1)(*(_QWORD *)(*a1 + 8), v19);
      LODWORD(v5) = sub_25BFB50(v19, v17);
      if ( !(_DWORD)v5 )
        LOBYTE(v5) = sub_2686B10((__int64)v20, a1[1], &v19);
    }
  }
  else
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v5 + 8);
      if ( *(_BYTE *)(v16 + 8) != 12 || *(_DWORD *)(v16 + 8) > 0x40FFu )
        break;
      v5 += 40;
      if ( v15 == v5 )
        goto LABEL_17;
    }
  }
  return v5;
}
