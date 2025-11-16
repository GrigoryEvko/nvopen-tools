// Function: sub_725090
// Address: 0x725090
//
_QWORD *__fastcall sub_725090(unsigned __int8 a1)
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F07970;
  if ( qword_4F07970 )
    qword_4F07970 = *(_QWORD *)qword_4F07970;
  else
    result = sub_7247C0(56);
  *((_WORD *)result + 12) &= 0xF000u;
  *result = 0;
  *((_BYTE *)result + 8) = a1;
  result[2] = 0;
  if ( a1 == 2 )
  {
    result[4] = 0;
    result[5] = 0;
    result[6] = 0;
  }
  else
  {
    if ( a1 > 2u )
    {
      if ( a1 != 3 )
        sub_721090();
    }
    else
    {
      result[4] = 0;
    }
    result[6] = 0;
  }
  return result;
}
