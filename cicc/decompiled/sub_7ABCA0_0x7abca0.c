// Function: sub_7ABCA0
// Address: 0x7abca0
//
__int64 sub_7ABCA0()
{
  __int64 v0; // rbx
  char *v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 result; // rax

  v0 = 2 * (qword_4F17F88 - (_QWORD)qword_4F17F90);
  v1 = (char *)sub_822C60(qword_4F17F90, qword_4F17F88 - (_QWORD)qword_4F17F90, v0);
  v2 = qword_4F17F80 - (_QWORD)qword_4F17F90;
  v3 = (__int64)&v1[v0];
  qword_4F17F90 = v1;
  result = (__int64)&v1[v2];
  qword_4F17F88 = v3;
  qword_4F17F80 = result;
  return result;
}
