// Function: sub_F8DB50
// Address: 0xf8db50
//
_QWORD *__fastcall sub_F8DB50(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_F894B0((__int64)a1, a2);
  if ( a3 )
  {
    if ( a3 != result[1] )
      return sub_F80EB0(a1, (unsigned __int64)result, a3);
  }
  return result;
}
