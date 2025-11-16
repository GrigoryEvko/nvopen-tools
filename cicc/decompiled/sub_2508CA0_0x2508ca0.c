// Function: sub_2508CA0
// Address: 0x2508ca0
//
__int64 __fastcall sub_2508CA0(__int64 a1)
{
  *(_QWORD *)a1 = a1 + 16;
  strcpy((char *)(a1 + 16), "AANoAlias");
  *(_QWORD *)(a1 + 8) = 9;
  return a1;
}
