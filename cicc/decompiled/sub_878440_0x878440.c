// Function: sub_878440
// Address: 0x878440
//
_QWORD *sub_878440()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = (_QWORD *)qword_4F60000;
  if ( qword_4F60000 )
  {
    v1 = *(_QWORD *)qword_4F60000;
    *(_QWORD *)(qword_4F60000 + 8) = 0;
    qword_4F60000 = v1;
    *result = 0;
  }
  else
  {
    result = (_QWORD *)sub_823970(16);
    *result = 0;
    result[1] = 0;
  }
  return result;
}
