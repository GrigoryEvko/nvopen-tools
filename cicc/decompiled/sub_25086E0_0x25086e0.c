// Function: sub_25086E0
// Address: 0x25086e0
//
__int64 __fastcall sub_25086E0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANonNull");
  *(_QWORD *)(a1 + 8) = 9;
  return a1;
}
