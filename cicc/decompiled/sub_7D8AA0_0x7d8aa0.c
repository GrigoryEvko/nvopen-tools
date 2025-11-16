// Function: sub_7D8AA0
// Address: 0x7d8aa0
//
_QWORD *__fastcall sub_7D8AA0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  _QWORD *result; // rax

  if ( *((_BYTE *)a1 + 56) == 20 )
    return (_QWORD *)sub_7F1170();
  v1 = *a1;
  v2 = *(_QWORD *)a1[9];
  sub_7D8A60(a1);
  if ( (unsigned int)sub_8D2AF0(v1) )
    return sub_7D8470(a1);
  result = (_QWORD *)sub_8D2AF0(v2);
  if ( (_DWORD)result )
    return sub_7D8470(a1);
  return result;
}
