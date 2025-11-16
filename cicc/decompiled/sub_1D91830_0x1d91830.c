// Function: sub_1D91830
// Address: 0x1d91830
//
__int64 __fastcall sub_1D91830(__int64 a1)
{
  *(_QWORD *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_QWORD *)a1 = &unk_49F9CF0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 44) = 0x100000000000000LL;
  *(_BYTE *)(a1 + 52) = 0;
  return 0x100000000000000LL;
}
