// Function: sub_F51DC0
// Address: 0xf51dc0
//
void __fastcall sub_F51DC0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r14
  _QWORD *v4; // rax
  __int64 v5; // r15
  __int64 v7; // [rsp+8h] [rbp-48h]
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_B12000(a1 + 72);
  v4 = (_QWORD *)sub_B11F60(a1 + 80);
  v5 = sub_F4EF60(v4);
  v7 = *(_QWORD *)(a2 - 64);
  sub_AE7AF0((__int64)v8, a1);
  sub_F4EE60(a3, v7, v3, v5, (__int64)v8, v7, a2 + 24, 0);
  if ( v8[0] )
    sub_B91220((__int64)v8, v8[0]);
}
