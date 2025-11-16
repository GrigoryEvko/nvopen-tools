// Function: sub_1087CB0
// Address: 0x1087cb0
//
__int64 __fastcall sub_1087CB0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned __int8 *v4; // rsi
  unsigned int v5; // r13d
  unsigned int v7; // [rsp+Ch] [rbp-114h] BYREF
  _QWORD v8[8]; // [rsp+10h] [rbp-110h] BYREF
  unsigned __int8 *v9; // [rsp+50h] [rbp-D0h] BYREF
  size_t v10; // [rsp+58h] [rbp-C8h]
  __int64 v11; // [rsp+60h] [rbp-C0h]
  _BYTE v12[184]; // [rsp+68h] [rbp-B8h] BYREF

  v8[5] = 0x100000000LL;
  v8[6] = &v9;
  v9 = v12;
  v8[0] = &unk_49DD288;
  v10 = 0;
  v11 = 128;
  v8[1] = 2;
  memset(&v8[2], 0, 24);
  sub_CB5980((__int64)v8, 0, 0, 0);
  sub_E5CCC0(a2, v8, a3);
  sub_CB6200(*(_QWORD *)(a1 + 8), v9, v10);
  v4 = v9;
  v7 = 0;
  sub_1098F90(&v7, v9, v10);
  v5 = v7;
  v8[0] = &unk_49DD388;
  sub_CB5840((__int64)v8);
  if ( v9 != v12 )
    _libc_free(v9, v4);
  return v5;
}
