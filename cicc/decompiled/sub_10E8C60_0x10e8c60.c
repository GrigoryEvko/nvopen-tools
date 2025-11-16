// Function: sub_10E8C60
// Address: 0x10e8c60
//
unsigned __int64 __fastcall sub_10E8C60(__int64 a1, __int64 a2)
{
  __int64 *v3; // r13
  __int64 v4; // r15
  __int64 **v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r9
  _QWORD *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r15
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v14; // rdi
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = (__int64 *)sub_BD5C60(a2);
  v4 = sub_ACD6D0(v3);
  v5 = (__int64 **)sub_BCE3C0(v3, 0);
  v6 = sub_ACADE0(v5);
  v7 = sub_BD2C40(80, unk_3F10A10);
  v9 = v7;
  if ( v7 )
    sub_B4D3C0((__int64)v7, v4, v6, 0, 0, v8, 0, 0);
  v10 = *(_QWORD *)(a2 + 48);
  v16[0] = v10;
  if ( v10 )
  {
    v11 = (__int64)(v9 + 6);
    sub_B96E90((__int64)v16, v10, 1);
    v12 = v9[6];
    if ( !v12 )
      goto LABEL_6;
  }
  else
  {
    v12 = v9[6];
    v11 = (__int64)(v9 + 6);
    if ( !v12 )
      goto LABEL_8;
  }
  sub_B91220(v11, v12);
LABEL_6:
  v13 = (unsigned __int8 *)v16[0];
  v9[6] = v16[0];
  if ( v13 )
    sub_B976B0((__int64)v16, v13, v11);
LABEL_8:
  sub_B44220(v9, a2 + 24, 0);
  v14 = *(_QWORD *)(a1 + 40);
  v16[0] = (__int64)v9;
  return sub_10E8740(v14 + 2096, v16);
}
