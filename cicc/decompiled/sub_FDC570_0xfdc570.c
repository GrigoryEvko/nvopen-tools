// Function: sub_FDC570
// Address: 0xfdc570
//
_QWORD *__fastcall sub_FDC570(_QWORD *a1, __int64 a2, __int64 a3)
{
  *a1 = a2;
  a1[2] = sub_FDB180;
  a1[1] = a3;
  a1[3] = sub_FDC4D0;
  return a1;
}
