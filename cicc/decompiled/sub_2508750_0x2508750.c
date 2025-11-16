// Function: sub_2508750
// Address: 0x2508750
//
__int64 __fastcall sub_2508750(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoUndef");
  *(_QWORD *)(a1 + 8) = 9;
  return a1;
}
