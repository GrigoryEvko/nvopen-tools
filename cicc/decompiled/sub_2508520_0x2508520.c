// Function: sub_2508520
// Address: 0x2508520
//
__int64 __fastcall sub_2508520(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 8;
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAIsDead");
  return a1;
}
