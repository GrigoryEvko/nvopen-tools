// Function: sub_169C810
// Address: 0x169c810
//
_QWORD *__fastcall sub_169C810(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *result; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  _QWORD *v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rdi
  _QWORD *v13; // rsi
  void *v14; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  result = (_QWORD *)sub_2207820(72);
  if ( !result )
  {
    v9 = 0;
    goto LABEL_5;
  }
  *result = 2;
  v8 = result;
  v9 = result + 1;
  v10 = result + 2;
  v11 = (_QWORD *)(a3 + 8);
  v14 = sub_16982C0();
  if ( *(void **)(a3 + 8) == v14 )
  {
    sub_169C7E0(v10, v11);
    v12 = v8 + 6;
    v13 = (_QWORD *)(a4 + 8);
    if ( *(void **)(a4 + 8) != v14 )
      goto LABEL_4;
LABEL_7:
    result = sub_169C7E0(v12, v13);
    goto LABEL_5;
  }
  sub_1698450((__int64)v10, (__int64)v11);
  v12 = v8 + 6;
  v13 = (_QWORD *)(a4 + 8);
  if ( *(void **)(a4 + 8) == v14 )
    goto LABEL_7;
LABEL_4:
  result = (_QWORD *)sub_1698450((__int64)v12, (__int64)v13);
LABEL_5:
  a1[1] = v9;
  return result;
}
