// Function: sub_8787C0
// Address: 0x8787c0
//
_QWORD *sub_8787C0()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = (_QWORD *)qword_4F5FFF0;
  if ( qword_4F5FFF0 )
  {
    v1 = *(_QWORD *)qword_4F5FFF0;
    *(_QWORD *)(qword_4F5FFF0 + 8) = 0;
    qword_4F5FFF0 = v1;
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
