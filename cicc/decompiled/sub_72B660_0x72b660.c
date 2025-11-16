// Function: sub_72B660
// Address: 0x72b660
//
_QWORD *sub_72B660()
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F07B48;
  if ( !qword_4F07B48 )
  {
    result = sub_7259C0(17);
    qword_4F07B48 = (__int64)result;
  }
  return result;
}
