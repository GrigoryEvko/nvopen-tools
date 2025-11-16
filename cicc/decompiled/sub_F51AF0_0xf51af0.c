// Function: sub_F51AF0
// Address: 0xf51af0
//
void __fastcall sub_F51AF0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rdx
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v7; // [rsp+8h] [rbp-48h]
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (1 - v3)) + 24LL);
  v5 = sub_F4EF60(*(_QWORD **)(*(_QWORD *)(a1 + 32 * (2 - v3)) + 24LL));
  v7 = *(_QWORD *)(a2 - 64);
  sub_AE7A80((__int64)v8, a1);
  sub_F4EE60(a3, v7, v4, v5, (__int64)v8, v7, a2 + 24, 0);
  if ( v8[0] )
    sub_B91220((__int64)v8, v8[0]);
}
