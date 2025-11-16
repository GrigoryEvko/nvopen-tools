// Function: sub_E8BC70
// Address: 0xe8bc70
//
_QWORD *__fastcall sub_E8BC70(_QWORD *a1, const void *a2, size_t a3, __int64 a4, __int64 a5)
{
  _QWORD *result; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  _QWORD *v10; // r12

  sub_E7BC40(a1, *(unsigned int **)(a1[36] + 8LL), a3, a4, a5);
  result = (_QWORD *)sub_E8BB10(a1, 0);
  v9 = result[6];
  v10 = result;
  if ( a3 + v9 > result[7] )
  {
    result = (_QWORD *)sub_C8D290((__int64)(result + 5), result + 8, a3 + v9, 1u, v7, v8);
    v9 = v10[6];
  }
  if ( a3 )
  {
    result = memcpy((void *)(v10[5] + v9), a2, a3);
    v9 = v10[6];
  }
  v10[6] = v9 + a3;
  return result;
}
