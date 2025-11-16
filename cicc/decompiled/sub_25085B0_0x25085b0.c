// Function: sub_25085B0
// Address: 0x25085b0
//
__int64 __fastcall sub_25085B0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAHeapToStack");
  *(_QWORD *)(a1 + 8) = 13;
  return a1;
}
