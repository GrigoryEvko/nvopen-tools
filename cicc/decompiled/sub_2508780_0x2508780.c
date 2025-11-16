// Function: sub_2508780
// Address: 0x2508780
//
__int64 __fastcall sub_2508780(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 10;
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoUnwind");
  return a1;
}
