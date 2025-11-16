// Function: sub_2508AB0
// Address: 0x2508ab0
//
__int64 __fastcall sub_2508AB0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AAWillReturn");
  *(_QWORD *)(a1 + 8) = 12;
  return a1;
}
