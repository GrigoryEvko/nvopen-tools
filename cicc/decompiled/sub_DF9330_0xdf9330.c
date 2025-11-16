// Function: sub_DF9330
// Address: 0xdf9330
//
_QWORD *__fastcall sub_DF9330(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_22077B0(16);
  if ( result )
  {
    result[1] = a2;
    *result = off_4979D10;
  }
  *a1 = result;
  return result;
}
