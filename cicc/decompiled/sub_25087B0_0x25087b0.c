// Function: sub_25087B0
// Address: 0x25087b0
//
__int64 __fastcall sub_25087B0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAAddressSpace");
  *(_QWORD *)(a1 + 8) = 14;
  return a1;
}
