// Function: sub_23328D0
// Address: 0x23328d0
//
__int64 __fastcall sub_23328D0(__int64 a1, __int64 a2)
{
  __int64 v3[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v4; // [rsp+10h] [rbp-50h]
  __int64 v5; // [rsp+18h] [rbp-48h]
  __int64 v6; // [rsp+20h] [rbp-40h]
  __int64 v7; // [rsp+28h] [rbp-38h]
  __int64 v8; // [rsp+30h] [rbp-30h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v7 = 0x100000000LL;
  v8 = a1;
  v3[0] = (__int64)&unk_49DD210;
  v3[1] = 0;
  v4 = 0;
  v5 = 0;
  v6 = 0;
  sub_CB5980((__int64)v3, 0, 0, 0);
  sub_CB6840((__int64)v3, a2);
  if ( v6 != v4 )
    sub_CB5AE0(v3);
  v3[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v3);
  return a1;
}
