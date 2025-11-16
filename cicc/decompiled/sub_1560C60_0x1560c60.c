// Function: sub_1560C60
// Address: 0x1560c60
//
_QWORD *__fastcall sub_1560C60(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax

  result = a1;
  if ( a2 )
  {
    *a1 |= 0x400uLL;
    a1[10] = a2;
  }
  return result;
}
