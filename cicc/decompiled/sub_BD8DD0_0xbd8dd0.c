// Function: sub_BD8DD0
// Address: 0xbd8dd0
//
__int64 __fastcall sub_BD8DD0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  int v4; // r15d
  int *v6; // rbx
  __int64 v7; // rsi
  int v8; // r15d
  unsigned __int16 v10; // ax
  _QWORD v11[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = 85;
  v6 = (int *)&unk_3F64024;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  v11[0] = a4;
  *(_QWORD *)a1 = a2;
  while ( 1 )
  {
    v12[0] = sub_A744E0(v11, a3);
    v7 = sub_A734C0(v12, v4);
    if ( v7 )
      sub_A77670(a1, v7);
    if ( v6 == (int *)&unk_3F64048 )
      break;
    v4 = *v6++;
  }
  v8 = a3 + 1;
  if ( (unsigned __int8)sub_A74710(v11, a3 + 1, 86)
    && ((unsigned __int8)sub_A74710(v11, v8, 81) || (unsigned __int8)sub_A74710(v11, v8, 80)) )
  {
    v10 = sub_A74840(v11, a3);
    sub_A77B90((__int64 **)a1, v10);
  }
  return a1;
}
