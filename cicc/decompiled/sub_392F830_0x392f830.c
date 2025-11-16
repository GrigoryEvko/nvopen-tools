// Function: sub_392F830
// Address: 0x392f830
//
__int64 __fastcall sub_392F830(__int64 a1, unsigned int a2, __int64 a3, __int64 **a4)
{
  __int64 v7; // r14
  _BYTE *v8; // r13
  char v9; // al
  char v10; // dl
  char v11; // al
  __int64 v12; // rsi
  char v13; // r14
  unsigned int v14; // ecx
  unsigned __int64 v15; // rcx
  __int64 v16; // rdi
  char v17; // al
  unsigned __int64 v18; // r8
  char v20; // al
  __int64 v21; // [rsp-10h] [rbp-70h]
  bool v23; // [rsp+12h] [rbp-4Eh]
  char v24; // [rsp+13h] [rbp-4Dh]
  char v25; // [rsp+14h] [rbp-4Ch]
  char v26; // [rsp+14h] [rbp-4Ch]
  char v27; // [rsp+18h] [rbp-48h]
  char v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  _QWORD v31[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a3;
  v8 = sub_38CF550(a4, *(_QWORD *)a3);
  if ( !v8 )
  {
    v27 = sub_38E27C0(v7);
    v23 = 0;
    v24 = 1;
    v10 = sub_38E2700(v7);
    goto LABEL_8;
  }
  v24 = (*(_BYTE *)(v7 + 9) & 0xC) == 12;
  v27 = sub_38E27C0(v7);
  v25 = sub_38E2700(v7);
  v9 = sub_38E2700((__int64)v8);
  v10 = v25;
  if ( v25 == 6 )
  {
    v10 = 6;
    v23 = (unsigned __int8)v9 <= 2u || v9 == 10;
    if ( v23 )
      goto LABEL_8;
    goto LABEL_7;
  }
  if ( (unsigned __int8)v25 > 6u )
  {
    if ( v25 != 10 )
      goto LABEL_7;
    v10 = 10;
    v23 = v9 == 6 || (unsigned __int8)v9 <= 2u;
    if ( !v23 )
      goto LABEL_7;
  }
  else if ( v25 == 1 )
  {
    v23 = 1;
    if ( v9 )
      v10 = v9;
  }
  else if ( v25 != 2 || (v10 = 2, !(v23 = v9 == 6 || (unsigned __int8)v9 <= 1u)) )
  {
LABEL_7:
    v23 = 1;
    v10 = v9;
  }
LABEL_8:
  v26 = (16 * v27) | v10;
  v28 = sub_38E2740(v7);
  v11 = sub_38E2750(v7);
  v12 = *(_QWORD *)a3;
  v13 = v11 | v28;
  if ( (*(_BYTE *)(*(_QWORD *)a3 + 9LL) & 0xC) == 0xC && (*(_BYTE *)(v12 + 8) & 0x10) != 0 )
  {
    v14 = *(_DWORD *)(v12 + 8);
    if ( (v14 & 0x1F000) != 0 )
    {
      v15 = (unsigned int)(1 << (((v14 >> 12) & 0x1F) - 1));
      goto LABEL_12;
    }
    goto LABEL_11;
  }
  v30 = *(_QWORD *)a3;
  if ( !(unsigned __int8)sub_38D0480(a4, v12, v31) )
  {
    v12 = *(_QWORD *)a3;
LABEL_11:
    v15 = 0;
    goto LABEL_12;
  }
  v20 = sub_390AF00((__int64)*a4, v30);
  v15 = v31[0];
  if ( v20 )
    v15 = v31[0] | 1LL;
  v12 = *(_QWORD *)a3;
LABEL_12:
  v16 = *(_QWORD *)(v12 + 32);
  if ( v16 )
    goto LABEL_16;
  if ( v23 )
    v16 = *((_QWORD *)v8 + 4);
  if ( v16 )
  {
LABEL_16:
    v29 = v15;
    v17 = sub_38CF260(v16, v31, a4);
    v15 = v29;
    if ( !v17 )
      sub_16BD130("Size expression must be absolute.", 1u);
    v18 = v31[0];
  }
  else
  {
    v18 = 0;
  }
  sub_392F510(a1, a2, v26, v15, v18, v13, *(_DWORD *)(a3 + 8), v24);
  return v21;
}
