// Function: sub_2534CA0
// Address: 0x2534ca0
//
_QWORD *__fastcall sub_2534CA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4; // rdx

  v2 = *(unsigned int *)(a2 + 248);
  *a1 = a2 + 224;
  v4 = *(_QWORD *)(a2 + 232) + 96 * v2;
  a1[1] = *(_QWORD *)(a2 + 224);
  a1[2] = v4;
  a1[3] = v4;
  return a1;
}
