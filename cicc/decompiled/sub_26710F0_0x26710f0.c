// Function: sub_26710F0
// Address: 0x26710f0
//
__int64 __fastcall sub_26710F0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAKernelInfo");
  *(_QWORD *)(a1 + 8) = 12;
  return a1;
}
