// Function: sub_2CF92F0
// Address: 0x2cf92f0
//
__int64 __fastcall sub_2CF92F0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE v4[9]; // [rsp+Fh] [rbp-11h] BYREF

  sub_2CF8860((__int64)v4, a3);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
