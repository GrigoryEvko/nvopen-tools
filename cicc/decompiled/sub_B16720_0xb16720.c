// Function: sub_B16720
// Address: 0xb16720
//
__int64 __fastcall sub_B16720(__int64 a1, _BYTE *a2, __int64 a3, float a4)
{
  __int64 result; // rax
  _QWORD v5[12]; // [rsp+10h] [rbp-60h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  v5[5] = 0x100000000LL;
  v5[0] = &unk_49DD210;
  v5[6] = a1 + 32;
  memset(&v5[1], 0, 32);
  sub_CB5980(v5, 0, 0, 0);
  sub_CB5AB0(v5, a4);
  v5[0] = &unk_49DD210;
  result = sub_CB5840(v5);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  return result;
}
