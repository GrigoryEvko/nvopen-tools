// Function: sub_7AECD0
// Address: 0x7aecd0
//
_QWORD *__fastcall sub_7AECD0(_QWORD *a1, _QWORD **a2, __int64 a3, __int64 a4)
{
  _QWORD *result; // rax

  result = qword_4F064A0;
  if ( qword_4F064A0 )
    qword_4F064A0 = (_QWORD *)*qword_4F064A0;
  else
    result = (_QWORD *)sub_823970(24);
  if ( *a2 )
    **a2 = result;
  else
    *a1 = result;
  *a2 = result;
  *result = 0;
  result[1] = a3;
  result[2] = a4;
  return result;
}
