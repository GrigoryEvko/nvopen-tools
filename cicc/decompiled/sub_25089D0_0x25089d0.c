// Function: sub_25089D0
// Address: 0x25089d0
//
__int64 __fastcall sub_25089D0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 10;
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoReturn");
  return a1;
}
