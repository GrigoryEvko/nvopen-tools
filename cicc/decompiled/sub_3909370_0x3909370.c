// Function: sub_3909370
// Address: 0x3909370
//
__int64 __fastcall sub_3909370(__int64 a1)
{
  *(_WORD *)(a1 + 16) &= 0xFEu;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 18) = 0;
  *(_QWORD *)a1 = &unk_4A3EAE0;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x100000000LL;
  return 0x100000000LL;
}
