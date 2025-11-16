// Function: sub_5EBAE0
// Address: 0x5ebae0
//
_QWORD *__fastcall sub_5EBAE0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx

  result = (_QWORD *)qword_4CF7FB0;
  if ( qword_4CF7FB0 )
  {
    v3 = *(_QWORD *)qword_4CF7FB0;
    *(_QWORD *)(qword_4CF7FB0 + 16) = a1;
    qword_4CF7FB0 = v3;
  }
  else
  {
    result = (_QWORD *)sub_724FB0();
    result[2] = a1;
  }
  *result = a2;
  return result;
}
