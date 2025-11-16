// Function: sub_729510
// Address: 0x729510
//
_QWORD *__fastcall sub_729510(unsigned __int64 a1)
{
  _QWORD *v1; // r14
  __int64 v2; // rbx
  _QWORD *result; // rax

  v1 = qword_4F06C50;
  v2 = qword_4F06C48 + 2000LL;
  if ( qword_4F06C48 + 2000LL < a1 )
    v2 = a1;
  if ( unk_4F06C38 )
  {
    qword_4F06C50 = (_QWORD *)sub_823970(v2);
    memcpy(qword_4F06C50, v1, qword_4F06C48);
    result = (_QWORD *)sub_823A00(v1, qword_4F06C48);
    qword_4F06C48 = v2;
  }
  else
  {
    result = (_QWORD *)sub_822C60(qword_4F06C50, qword_4F06C48, v2);
    qword_4F06C48 = v2;
    qword_4F06C50 = result;
  }
  return result;
}
