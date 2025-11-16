// Function: sub_C3C790
// Address: 0xc3c790
//
_QWORD *__fastcall sub_C3C790(_QWORD *a1, _QWORD **a2)
{
  _QWORD *result; // rax
  bool v4; // zf
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  __int64 *v8; // rsi
  _QWORD *v9; // rdi
  void *v10; // r15
  _QWORD *v11; // rdi
  __int64 *v12; // rsi

  result = *a2;
  v4 = a2[1] == 0;
  *a1 = *a2;
  if ( v4 || (result = (_QWORD *)sub_2207820(56), (v6 = result) == 0) )
  {
    v7 = 0;
  }
  else
  {
    *result = 2;
    v7 = result + 1;
    v8 = a2[1];
    v9 = result + 1;
    v10 = sub_C33340();
    if ( (void *)*v8 == v10 )
      sub_C3C790(v9, v8);
    else
      sub_C33EB0(v9, v8);
    v11 = v6 + 4;
    v12 = a2[1] + 3;
    if ( (void *)*v12 == v10 )
      result = (_QWORD *)sub_C3C790(v11, v12);
    else
      result = (_QWORD *)sub_C33EB0(v11, v12);
  }
  a1[1] = v7;
  return result;
}
