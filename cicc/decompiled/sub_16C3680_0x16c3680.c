// Function: sub_16C3680
// Address: 0x16c3680
//
_QWORD *__fastcall sub_16C3680(_QWORD *a1, __int64 a2, __int64 a3)
{
  a1[2] = 0;
  a1[3] = 0;
  *a1 = a2;
  a1[1] = a3;
  a1[4] = a3;
  return a1;
}
