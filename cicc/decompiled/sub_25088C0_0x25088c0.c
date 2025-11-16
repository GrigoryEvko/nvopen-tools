// Function: sub_25088C0
// Address: 0x25088c0
//
__int64 __fastcall sub_25088C0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANonConvergent");
  *(_QWORD *)(a1 + 8) = 15;
  return a1;
}
