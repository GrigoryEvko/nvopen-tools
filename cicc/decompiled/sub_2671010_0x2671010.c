// Function: sub_2671010
// Address: 0x2671010
//
__int64 __fastcall sub_2671010(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAHeapToShared");
  *(_QWORD *)(a1 + 8) = 14;
  return a1;
}
