// Function: sub_2508CD0
// Address: 0x2508cd0
//
__int64 __fastcall sub_2508CD0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoFPClass");
  *(_QWORD *)(a1 + 8) = 11;
  return a1;
}
