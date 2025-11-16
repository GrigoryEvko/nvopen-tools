// Function: sub_C3C5A0
// Address: 0xc3c5a0
//
_QWORD *__fastcall sub_C3C5A0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // r15
  _QWORD *v6; // r12
  _DWORD *v7; // rax
  __int64 v8; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  result = (_QWORD *)sub_2207820(56);
  if ( result )
  {
    *result = 2;
    v5 = (__int64)(result + 1);
    v6 = result + 4;
    v7 = sub_C33340();
    if ( v7 == dword_3F657A0 )
    {
      v8 = (__int64)v7;
      sub_C3C5A0(v5, v7, a3);
      result = sub_C3C460(v6, v8);
    }
    else
    {
      sub_C36740(v5, (__int64)dword_3F657A0, a3);
      result = (_QWORD *)sub_C37380(v6, (__int64)dword_3F657A0);
    }
  }
  else
  {
    v5 = 0;
  }
  a1[1] = v5;
  return result;
}
