// Function: sub_169C4E0
// Address: 0x169c4e0
//
_QWORD *__fastcall sub_169C4E0(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // r15
  __int64 v4; // r12
  void *v5; // rax
  void *v6; // r14
  __int64 v7; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  result = (_QWORD *)sub_2207820(72);
  if ( result )
  {
    *result = 2;
    v3 = result + 1;
    v7 = (__int64)(result + 2);
    v4 = (__int64)(result + 6);
    v5 = sub_16982C0();
    v6 = v5;
    if ( v5 == &unk_42AE9D0 )
    {
      sub_169C4E0(v7, v5);
      result = (_QWORD *)sub_169C4E0(v4, v6);
    }
    else
    {
      sub_1698360(v7, (__int64)&unk_42AE9D0);
      result = (_QWORD *)sub_1698360(v4, (__int64)&unk_42AE9D0);
    }
  }
  else
  {
    v3 = 0;
  }
  a1[1] = v3;
  return result;
}
