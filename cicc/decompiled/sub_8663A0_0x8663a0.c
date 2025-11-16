// Function: sub_8663A0
// Address: 0x8663a0
//
_QWORD *sub_8663A0()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = (_QWORD *)qword_4F5FD38;
  if ( qword_4F5FD38 )
    qword_4F5FD38 = *(_QWORD *)qword_4F5FD38;
  else
    result = (_QWORD *)sub_823970(64);
  *result = 0;
  result[1] = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result[2] = 0;
  result[3] = 0;
  result[4] = v1;
  LODWORD(v1) = dword_4F5FCD8;
  result[5] = 0;
  result[6] = 0;
  result[7] = (unsigned int)v1;
  return result;
}
