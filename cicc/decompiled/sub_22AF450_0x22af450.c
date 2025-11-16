// Function: sub_22AF450
// Address: 0x22af450
//
__int64 __fastcall sub_22AF450(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  *(_BYTE *)(a1 + 72) = a3;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  *(_QWORD *)(a1 + 160) = a4;
  return sub_22AF1D0(a1, a2, a1 + 144, a4, a5, a6);
}
