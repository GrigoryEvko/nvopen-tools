// Function: sub_C3C930
// Address: 0xc3c930
//
_QWORD *__fastcall sub_C3C930(_QWORD *a1, __int64 a2, void **a3, _QWORD *a4)
{
  _QWORD *result; // rax
  _QWORD *v8; // r14
  _QWORD *v9; // rbx
  _QWORD *v10; // rdi
  _QWORD *v11; // rdi
  _QWORD *v12; // rsi
  void *v13; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  result = (_QWORD *)sub_2207820(56);
  if ( !result )
  {
    v8 = 0;
    goto LABEL_5;
  }
  *result = 2;
  v8 = result + 1;
  v9 = result;
  v10 = result + 1;
  v13 = sub_C33340();
  if ( *a3 == v13 )
  {
    sub_C3C840(v10, a3);
    v11 = v9 + 4;
    v12 = a4;
    if ( (void *)*a4 != v13 )
      goto LABEL_4;
LABEL_7:
    result = sub_C3C840(v11, v12);
    goto LABEL_5;
  }
  sub_C338E0((__int64)v10, (__int64)a3);
  v11 = v9 + 4;
  v12 = a4;
  if ( (void *)*a4 == v13 )
    goto LABEL_7;
LABEL_4:
  result = (_QWORD *)sub_C338E0((__int64)v11, (__int64)v12);
LABEL_5:
  a1[1] = v8;
  return result;
}
