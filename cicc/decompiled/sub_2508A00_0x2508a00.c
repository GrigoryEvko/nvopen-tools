// Function: sub_2508A00
// Address: 0x2508a00
//
__int64 __fastcall sub_2508A00(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAInstanceInfo");
  *(_QWORD *)(a1 + 8) = 14;
  return a1;
}
