// Function: sub_1560C00
// Address: 0x1560c00
//
_QWORD *__fastcall sub_1560C00(_QWORD *a1, unsigned int a2)
{
  _QWORD *result; // rax

  result = a1;
  if ( a2 )
  {
    *a1 |= 2uLL;
    a1[7] = a2;
  }
  return result;
}
