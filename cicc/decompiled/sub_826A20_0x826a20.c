// Function: sub_826A20
// Address: 0x826a20
//
_QWORD *__fastcall sub_826A20(unsigned int a1, int a2)
{
  _DWORD *v3; // rax
  __int64 v4; // rdi
  _DWORD *v5; // r12
  _QWORD *result; // rax

  v3 = (_DWORD *)sub_823970(8);
  v4 = qword_4F1F650;
  *v3 = a1;
  v5 = v3;
  v3[1] = a2;
  if ( !v4 )
  {
    qword_4F1F650 = sub_881A70(0, 512, 31, 52);
    v4 = qword_4F1F650;
  }
  result = (_QWORD *)sub_881B20(v4, a1, 1);
  *result = v5;
  return result;
}
