// Function: sub_3215550
// Address: 0x3215550
//
__int64 __fastcall sub_3215550(_QWORD *a1, __int64 a2, __int16 a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // eax
  int v7; // [rsp+Ah] [rbp-26h] BYREF
  __int16 v8; // [rsp+Eh] [rbp-22h]

  v4 = sub_31DF6E0(a2);
  v7 = v4;
  v8 = WORD2(v4);
  v5 = sub_3215500((__int64)a1, (__int64)&v7, a3);
  return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, bool))(*(_QWORD *)a2 + 432LL))(
           a2,
           *a1,
           0,
           v5,
           a3 != 1);
}
