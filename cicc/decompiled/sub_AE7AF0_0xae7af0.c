// Function: sub_AE7AF0
// Address: 0xae7af0
//
__int64 __fastcall sub_AE7AF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  int v4; // r15d
  int v5; // ebx
  int v6; // eax
  __int64 v7; // rax
  _QWORD v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 + 24);
  v9[0] = v3;
  if ( v3 )
    sub_B96E90(v9, v3, 1);
  v4 = sub_B10D00(v9);
  v5 = sub_B10D40(v9);
  v6 = sub_B141C0(a2);
  v7 = sub_B01860(v6, 0, 0, v4, v5, 0, 0, 1);
  sub_B10CB0(a1, v7);
  if ( v9[0] )
    sub_B91220(v9);
  return a1;
}
