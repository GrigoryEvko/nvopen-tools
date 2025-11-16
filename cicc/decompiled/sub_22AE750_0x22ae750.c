// Function: sub_22AE750
// Address: 0x22ae750
//
__int64 __fastcall sub_22AE750(__int64 a1, __int64 a2)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_BYTE *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  *(_QWORD *)(a1 + 160) = a2;
  return 0x400000000LL;
}
