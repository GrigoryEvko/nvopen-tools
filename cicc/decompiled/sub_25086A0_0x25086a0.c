// Function: sub_25086A0
// Address: 0x25086a0
//
__int64 __fastcall sub_25086A0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoCapture");
  *(_QWORD *)(a1 + 8) = 11;
  return a1;
}
