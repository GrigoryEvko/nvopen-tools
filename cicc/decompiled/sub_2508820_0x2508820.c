// Function: sub_2508820
// Address: 0x2508820
//
__int64 __fastcall sub_2508820(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAPointerInfo");
  *(_QWORD *)(a1 + 8) = 13;
  return a1;
}
