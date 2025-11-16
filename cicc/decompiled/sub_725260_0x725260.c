// Function: sub_725260
// Address: 0x725260
//
_QWORD *sub_725260()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = (_QWORD *)qword_4F07968;
  if ( qword_4F07968 )
  {
    v1 = *(_QWORD *)qword_4F07968;
    *(_QWORD *)(qword_4F07968 + 8) = 0;
    qword_4F07968 = v1;
    *result = 0;
  }
  else
  {
    result = sub_7247C0(16);
    *result = 0;
    result[1] = 0;
  }
  return result;
}
