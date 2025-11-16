// Function: sub_B16360
// Address: 0xb16360
//
__int64 __fastcall sub_B16360(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _QWORD v6[12]; // [rsp+0h] [rbp-60h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  v6[6] = a1 + 32;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v6[5] = 0x100000000LL;
  v6[0] = &unk_49DD210;
  memset(&v6[1], 0, 32);
  sub_CB5980(v6, 0, 0, 0);
  sub_A587F0(a4, (__int64)v6, 0, 0);
  v6[0] = &unk_49DD210;
  return sub_CB5840(v6);
}
