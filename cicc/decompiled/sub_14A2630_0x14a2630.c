// Function: sub_14A2630
// Address: 0x14a2630
//
_QWORD *__fastcall sub_14A2630(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_22077B0(16);
  if ( result )
  {
    result[1] = a2;
    *result = off_4984830;
  }
  *a1 = result;
  return result;
}
