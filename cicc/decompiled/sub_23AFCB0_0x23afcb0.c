// Function: sub_23AFCB0
// Address: 0x23afcb0
//
void __fastcall sub_23AFCB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD v4[2]; // [rsp+0h] [rbp-100h] BYREF
  unsigned __int8 *v5; // [rsp+10h] [rbp-F0h] BYREF
  size_t v6; // [rsp+18h] [rbp-E8h]
  __int64 v7; // [rsp+20h] [rbp-E0h]
  _BYTE v8[24]; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v9[8]; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD v10[4]; // [rsp+80h] [rbp-80h] BYREF
  char v11; // [rsp+A0h] [rbp-60h]
  _QWORD v12[2]; // [rsp+A8h] [rbp-58h] BYREF
  _QWORD v13[2]; // [rsp+B8h] [rbp-48h] BYREF
  _QWORD v14[7]; // [rsp+C8h] [rbp-38h] BYREF

  v10[0] = "*** IR Dump After {0} on {1} filtered out ***\n";
  v10[2] = v14;
  v4[1] = a3;
  v12[1] = a4;
  v12[0] = &unk_49E6618;
  v4[0] = a2;
  v9[5] = 0x100000000LL;
  v13[0] = &unk_49DB108;
  v13[1] = v4;
  v14[0] = v13;
  v14[1] = v12;
  v9[0] = &unk_49DD288;
  v9[6] = &v5;
  v10[1] = 46;
  v10[3] = 2;
  v11 = 1;
  v5 = v8;
  v6 = 0;
  v7 = 20;
  v9[1] = 2;
  memset(&v9[2], 0, 24);
  sub_CB5980((__int64)v9, 0, 0, 0);
  sub_CB6840((__int64)v9, (__int64)v10);
  v9[0] = &unk_49DD388;
  sub_CB5840((__int64)v9);
  sub_CB6200(*(_QWORD *)(a1 + 40), v5, v6);
  if ( v5 != v8 )
    _libc_free((unsigned __int64)v5);
}
