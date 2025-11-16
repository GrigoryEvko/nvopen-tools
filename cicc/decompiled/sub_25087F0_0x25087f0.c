// Function: sub_25087F0
// Address: 0x25087f0
//
__int64 __fastcall sub_25087F0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 8;
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoSync");
  return a1;
}
