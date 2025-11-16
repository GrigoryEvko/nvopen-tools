// Function: sub_222DCC0
// Address: 0x222dcc0
//
__int64 __fastcall sub_222DCC0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_2231340(a2) )
  {
    a1[30] = sub_222F790(a2);
    if ( (unsigned __int8)sub_22313E0(a2) )
      goto LABEL_3;
LABEL_6:
    a1[31] = 0;
    result = sub_2231430(a2);
    if ( (_BYTE)result )
      goto LABEL_4;
    goto LABEL_7;
  }
  a1[30] = 0;
  if ( !(unsigned __int8)sub_22313E0(a2) )
    goto LABEL_6;
LABEL_3:
  a1[31] = sub_22302C0(a2);
  result = sub_2231430(a2);
  if ( (_BYTE)result )
  {
LABEL_4:
    result = sub_2230310(a2);
    a1[32] = result;
    return result;
  }
LABEL_7:
  a1[32] = 0;
  return result;
}
