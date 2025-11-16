// Function: sub_E63F00
// Address: 0xe63f00
//
void *__fastcall sub_E63F00(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rdi
  _QWORD *v4; // rbx
  void *result; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rsi

  v2 = a1[7];
  if ( v2 )
    j_j___libc_free_0(v2, a1[9] - v2);
  v3 = (_QWORD *)a1[2];
  v4 = a1 + 6;
  sub_E62D10(v3);
  result = memset((void *)*(v4 - 6), 0, 8LL * *(v4 - 5));
  v6 = (_QWORD *)*(v4 - 6);
  v7 = *(v4 - 5);
  *(v4 - 3) = 0;
  *(v4 - 4) = 0;
  if ( v6 != v4 )
    return (void *)j_j___libc_free_0(v6, 8 * v7);
  return result;
}
