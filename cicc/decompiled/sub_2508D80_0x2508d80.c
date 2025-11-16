// Function: sub_2508D80
// Address: 0x2508d80
//
__int64 __fastcall sub_2508D80(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AACallEdges");
  *(_QWORD *)(a1 + 8) = 11;
  return a1;
}
