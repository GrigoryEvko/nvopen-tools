// Function: sub_727D20
// Address: 0x727d20
//
_QWORD *__fastcall sub_727D20(_QWORD *a1)
{
  char v1; // al
  __int64 v2; // rsi
  _QWORD *result; // rax

  if ( a1 )
  {
    v1 = *((_BYTE *)a1 - 8);
    if ( (v1 & 1) != 0 )
    {
      if ( (v1 & 2) != 0 )
      {
        if ( *a1 )
        {
          v2 = sub_72B7A0(a1);
        }
        else if ( unk_4D03FE8 )
        {
          v2 = *qword_4D03FD0;
        }
        else
        {
          v2 = unk_4D03FF0;
        }
        result = (_QWORD *)sub_727B10(16, v2);
      }
      else
      {
        result = (_QWORD *)sub_7279A0(16);
      }
    }
    else
    {
      result = sub_7246D0(16);
    }
  }
  else
  {
    result = sub_7247C0(16);
  }
  *result = 0;
  result[1] = 0;
  return result;
}
