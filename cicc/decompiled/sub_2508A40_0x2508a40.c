// Function: sub_2508A40
// Address: 0x2508a40
//
__int64 __fastcall sub_2508A40(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoRecurse");
  *(_QWORD *)(a1 + 8) = 11;
  return a1;
}
