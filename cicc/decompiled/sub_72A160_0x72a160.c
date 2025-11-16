// Function: sub_72A160
// Address: 0x72a160
//
void __fastcall sub_72A160(__int64 a1)
{
  *(_WORD *)(a1 + 88) &= 0x83FCu;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  sub_72A140(a1);
}
