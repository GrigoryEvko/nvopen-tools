// Function: sub_B16D50
// Address: 0xb16d50
//
__int64 __fastcall sub_B16D50(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD v6[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v7[10]; // [rsp+10h] [rbp-50h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  v6[1] = a5;
  v6[0] = a4;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  v7[6] = a1 + 32;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v7[5] = 0x100000000LL;
  v7[0] = &unk_49DD210;
  memset(&v7[1], 0, 32);
  sub_CB5980(v7, 0, 0, 0);
  sub_C68B50(v6, v7);
  v7[0] = &unk_49DD210;
  return sub_CB5840(v7);
}
