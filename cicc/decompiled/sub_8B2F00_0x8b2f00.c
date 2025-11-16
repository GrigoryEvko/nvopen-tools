// Function: sub_8B2F00
// Address: 0x8b2f00
//
__int64 __fastcall sub_8B2F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 *a6)
{
  _QWORD *v6; // r15
  int v7; // r12d
  __int64 v9; // r12
  __int64 v10; // rax
  _BOOL4 v11; // r13d
  _BOOL4 v12; // r8d
  __int64 v13; // rsi
  _QWORD *v16; // rax
  __int64 v17; // rax
  int v20; // [rsp+2Ch] [rbp-94h] BYREF
  _QWORD v21[18]; // [rsp+30h] [rbp-90h] BYREF

  v6 = *(_QWORD **)(a2 + 104);
  v7 = unk_4D04854;
  if ( (*(_BYTE *)(a2 + 266) & 4) == 0 )
  {
    if ( unk_4D04854 )
      return sub_8B2D80(*(__int64 **)(a1 + 32), v6);
    v9 = **(_QWORD **)(sub_8794A0(*(_QWORD **)(a2 + 104)) + 32);
    v10 = sub_8794A0(*(_QWORD **)(a1 + 32));
    if ( (*(_BYTE *)(v10 + 160) & 2) != 0 )
    {
      v11 = 1;
      v12 = 1;
      goto LABEL_6;
    }
    v11 = 1;
    v13 = **(_QWORD **)(v10 + 32);
LABEL_10:
    v12 = sub_89B3C0(v9, v13, 0, 4u, 0, 8u) != 0;
    goto LABEL_6;
  }
  v20 = 0;
  sub_892150(v21);
  v16 = (_QWORD *)sub_8A4520((__int64)v6, a3, a4, a6, 0, &v20, (__int64)v21);
  *(_QWORD *)(a1 + 40) = v16;
  v6 = v16;
  v11 = v20 == 0;
  if ( v20 | v7 )
  {
LABEL_12:
    v12 = 0;
    goto LABEL_6;
  }
  v9 = **(_QWORD **)(sub_8794A0(v16) + 32);
  v17 = sub_8794A0(*(_QWORD **)(a1 + 32));
  v12 = 1;
  if ( (*(_BYTE *)(v17 + 160) & 2) != 0 )
    goto LABEL_6;
  v13 = **(_QWORD **)(v17 + 32);
  if ( (a5 & 1) == 0 )
    goto LABEL_10;
  for ( ; v13; v9 = *(_QWORD *)v9 )
  {
    if ( !v9 )
      break;
    if ( *(_BYTE *)(*(_QWORD *)(v9 + 8) + 80LL) != *(_BYTE *)(*(_QWORD *)(v13 + 8) + 80LL) )
      goto LABEL_12;
    v13 = *(_QWORD *)v13;
  }
  v12 = (v13 | v9) == 0;
LABEL_6:
  if ( unk_4D04854 )
  {
    v12 = 0;
    if ( v11 )
      return sub_8B2D80(*(__int64 **)(a1 + 32), v6);
  }
  return v12;
}
