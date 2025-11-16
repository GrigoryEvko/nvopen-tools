// Function: sub_AF4FD0
// Address: 0xaf4fd0
//
_QWORD *__fastcall sub_AF4FD0(_QWORD *a1, unsigned int a2, unsigned int a3, char a4)
{
  _QWORD *result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx

  result = a1;
  v5 = a3;
  *result = 4097;
  result[3] = 4097;
  v6 = a4 == 0 ? 7LL : 5LL;
  result[1] = a2;
  result[2] = v6;
  result[4] = v5;
  result[5] = v6;
  return result;
}
