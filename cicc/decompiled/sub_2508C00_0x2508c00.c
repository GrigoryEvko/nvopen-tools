// Function: sub_2508C00
// Address: 0x2508c00
//
__int64 __fastcall sub_2508C00(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 8;
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoFree");
  return a1;
}
