// Function: sub_7E17A0
// Address: 0x7e17a0
//
_QWORD *__fastcall sub_7E17A0(__int64 a1)
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F18A20;
  if ( qword_4F18A20 )
    qword_4F18A20 = *(_QWORD *)qword_4F18A20;
  else
    result = (_QWORD *)sub_823970(16);
  result[1] = a1;
  *result = qword_4D03F60;
  qword_4D03F60 = result;
  return result;
}
