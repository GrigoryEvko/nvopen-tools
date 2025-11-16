// Function: sub_7E7010
// Address: 0x7e7010
//
_QWORD *__fastcall sub_7E7010(_QWORD *a1)
{
  _QWORD *result; // rax

  if ( a1 )
  {
    sub_7E67B0(a1);
    result = qword_4D03F68;
    if ( qword_4D03F68 )
    {
      for ( result = (_QWORD *)qword_4D03F68[9]; result; result = (_QWORD *)*result )
        *((_BYTE *)result + 16) = 0;
    }
  }
  else
  {
    result = qword_4D03F68;
    if ( qword_4D03F68 )
    {
      for ( result = (_QWORD *)qword_4D03F68[9]; result; result = (_QWORD *)*result )
        *((_BYTE *)result + 16) = 0;
    }
  }
  return result;
}
