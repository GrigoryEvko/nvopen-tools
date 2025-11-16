// Function: sub_7B8160
// Address: 0x7b8160
//
_QWORD *sub_7B8160()
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F061C8;
  qword_4F061C8 = *(_QWORD *)qword_4F061C8;
  *result = qword_4F08530;
  qword_4F08530 = (__int64)result;
  return result;
}
