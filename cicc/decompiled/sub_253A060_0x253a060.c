// Function: sub_253A060
// Address: 0x253a060
//
__int64 __fastcall sub_253A060(__int64 a1, __int64 a2)
{
  _QWORD v3[12]; // [rsp+0h] [rbp-60h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  v3[5] = 0x100000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v3[6] = a1;
  v3[0] = &unk_49DD210;
  memset(&v3[1], 0, 32);
  sub_CB5980((__int64)v3, 0, 0, 0);
  sub_250FB90((__int64)v3, a2 + 88);
  v3[0] = &unk_49DD210;
  sub_CB5840((__int64)v3);
  return a1;
}
