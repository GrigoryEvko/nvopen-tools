// Function: sub_7ABA40
// Address: 0x7aba40
//
__int64 sub_7ABA40()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F08558;
  if ( qword_4F08558 )
    qword_4F08558 = *(_QWORD *)qword_4F08558;
  else
    result = sub_823970(112);
  *(_QWORD *)result = 0;
  *(_WORD *)(result + 24) = 0;
  v1 = qword_4F08560;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)result = v1;
  *(_QWORD *)(result + 28) = 0;
  *(_BYTE *)(result + 26) = 7;
  qword_4F08560 = result;
  return result;
}
