// Function: sub_727530
// Address: 0x727530
//
void __fastcall sub_727530(__int64 a1)
{
  *(_WORD *)(a1 + 32) &= 0xC000u;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = -1;
}
