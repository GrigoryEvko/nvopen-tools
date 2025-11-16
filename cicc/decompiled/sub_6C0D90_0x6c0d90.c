// Function: sub_6C0D90
// Address: 0x6c0d90
//
_QWORD *__fastcall sub_6C0D90(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rbp
  _QWORD *result; // rax
  __int64 v5; // [rsp-18h] [rbp-18h] BYREF
  _QWORD v6[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( a2 )
    return (_QWORD *)sub_6E6470(a3);
  if ( !a1 || (result = (_QWORD *)qword_4D03C50) != 0 && (result = *(_QWORD **)(qword_4D03C50 + 136LL)) != 0 && *result )
  {
    v6[1] = v3;
    sub_6C0910(0, 0, 1u, &v5, 1, 0, 1u, 0, a1, 0, 0, v6, 0, 0, 0);
    sub_6E6470(v6[0]);
    return (_QWORD *)sub_6E1990(v6[0]);
  }
  return result;
}
