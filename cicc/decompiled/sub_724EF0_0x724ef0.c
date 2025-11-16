// Function: sub_724EF0
// Address: 0x724ef0
//
_QWORD *__fastcall sub_724EF0(__int64 a1)
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F07978;
  if ( qword_4F07978 )
    qword_4F07978 = *(_QWORD *)qword_4F07978;
  else
    result = sub_7247C0(88);
  *result = 0;
  result[4] &= 0xFE000000uLL;
  result[1] = a1;
  result[2] = 0;
  result[3] = 0;
  result[5] = 0;
  result[6] = 0;
  result[7] = 0;
  result[8] = 0;
  result[9] = 0;
  result[10] = 0;
  return result;
}
