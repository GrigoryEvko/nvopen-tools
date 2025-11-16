// Function: sub_1560C20
// Address: 0x1560c20
//
_QWORD *__fastcall sub_1560C20(_QWORD *a1, unsigned int a2)
{
  _QWORD *result; // rax

  result = a1;
  if ( a2 )
  {
    *a1 |= 0x1000000000000uLL;
    a1[8] = a2;
  }
  return result;
}
