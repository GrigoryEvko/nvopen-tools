// Function: sub_222DE10
// Address: 0x222de10
//
__int64 __fastcall sub_222DE10(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_2244C40(a2) )
  {
    a1[30] = sub_2243120(a2);
    if ( (unsigned __int8)sub_2244C90(a2) )
      goto LABEL_3;
LABEL_6:
    a1[31] = 0;
    result = sub_2244CE0(a2);
    if ( (_BYTE)result )
      goto LABEL_4;
    goto LABEL_7;
  }
  a1[30] = 0;
  if ( !(unsigned __int8)sub_2244C90(a2) )
    goto LABEL_6;
LABEL_3:
  a1[31] = sub_2243B70(a2);
  result = sub_2244CE0(a2);
  if ( (_BYTE)result )
  {
LABEL_4:
    result = sub_2243BC0(a2);
    a1[32] = result;
    return result;
  }
LABEL_7:
  a1[32] = 0;
  return result;
}
