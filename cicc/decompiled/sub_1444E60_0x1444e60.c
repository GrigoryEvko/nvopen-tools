// Function: sub_1444E60
// Address: 0x1444e60
//
_QWORD *__fastcall sub_1444E60(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *result; // rax

  result = sub_1443FA0(a1, a2);
  if ( !result )
    return (_QWORD *)sub_1444DB0(a1, a2);
  return result;
}
