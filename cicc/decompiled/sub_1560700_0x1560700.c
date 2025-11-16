// Function: sub_1560700
// Address: 0x1560700
//
_QWORD *__fastcall sub_1560700(_QWORD *a1, int a2)
{
  _QWORD *result; // rax

  result = a1;
  *a1 &= ~(1LL << a2);
  switch ( a2 )
  {
    case 1:
      a1[7] = 0;
      break;
    case 48:
      a1[8] = 0;
      break;
    case 9:
      a1[9] = 0;
      break;
    case 10:
      a1[10] = 0;
      break;
    case 2:
      a1[11] = 0;
      break;
  }
  return result;
}
