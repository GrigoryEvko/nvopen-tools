// Function: sub_169C630
// Address: 0x169c630
//
_QWORD *__fastcall sub_169C630(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  _QWORD *v6; // r15
  _QWORD *v7; // r12
  void *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  result = (_QWORD *)sub_2207820(72);
  if ( result )
  {
    *result = 2;
    v6 = result + 1;
    v10 = (__int64)(result + 2);
    v7 = result + 6;
    v8 = sub_16982C0();
    v9 = v10;
    if ( v8 == &unk_42AE9D0 )
    {
      v11 = (__int64)v8;
      sub_169C630(v9, v8, a3);
      result = sub_169C4E0(v7, v11);
    }
    else
    {
      sub_1699170(v10, (__int64)&unk_42AE9D0, a3);
      result = (_QWORD *)sub_1698360((__int64)v7, (__int64)&unk_42AE9D0);
    }
  }
  else
  {
    v6 = 0;
  }
  a1[1] = v6;
  return result;
}
