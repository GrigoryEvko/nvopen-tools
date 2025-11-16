// Function: sub_1560C40
// Address: 0x1560c40
//
_QWORD *__fastcall sub_1560C40(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax

  result = a1;
  if ( a2 )
  {
    *a1 |= 0x200uLL;
    a1[9] = a2;
  }
  return result;
}
