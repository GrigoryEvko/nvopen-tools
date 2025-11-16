// Function: sub_169C6E0
// Address: 0x169c6e0
//
_QWORD *__fastcall sub_169C6E0(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  bool v4; // zf
  _QWORD *v6; // r12
  _QWORD *v7; // r14
  void *v8; // r15
  __int64 *v9; // rsi
  _QWORD *v10; // rdi
  __int64 *v11; // rsi
  __int64 v12; // [rsp+0h] [rbp-40h]
  _QWORD *v13; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)a2;
  v4 = *(_QWORD *)(a2 + 8) == 0;
  *a1 = *(_QWORD *)a2;
  if ( v4 || (result = (_QWORD *)sub_2207820(72), (v6 = result) == 0) )
  {
    v7 = 0;
  }
  else
  {
    *result = 2;
    v7 = result + 1;
    v13 = result + 2;
    v12 = *(_QWORD *)(a2 + 8);
    v8 = sub_16982C0();
    v9 = (__int64 *)(v12 + 8);
    if ( *(void **)(v12 + 8) == v8 )
      sub_169C6E0(v13, v9);
    else
      sub_16986C0(v13, v9);
    v10 = v6 + 6;
    v11 = (__int64 *)(*(_QWORD *)(a2 + 8) + 40LL);
    if ( v8 == (void *)*v11 )
      result = (_QWORD *)sub_169C6E0(v10, v11);
    else
      result = (_QWORD *)sub_16986C0(v10, v11);
  }
  a1[1] = v7;
  return result;
}
