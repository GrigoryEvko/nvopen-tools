// Function: sub_DF5F00
// Address: 0xdf5f00
//
__int64 __fastcall sub_DF5F00(__int64 a1)
{
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  *(_DWORD *)(a1 + 56) = 1;
  *(_BYTE *)(a1 + 60) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0x400000000LL;
  return a1;
}
