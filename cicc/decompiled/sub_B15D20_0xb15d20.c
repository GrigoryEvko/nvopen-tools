// Function: sub_B15D20
// Address: 0xb15d20
//
_QWORD *__fastcall sub_B15D20(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, _QWORD *); // r14
  __int64 v3; // rax
  __int64 v4; // rax
  _QWORD *result; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v7[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL);
  sub_B15A70((__int64)v6, a1);
  v3 = v2(a2, v6);
  v4 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v3 + 48LL))(v3, ": ");
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 128LL))(v4, *(_QWORD *)(a1 + 40));
  result = v7;
  if ( (_QWORD *)v6[0] != v7 )
    return (_QWORD *)j_j___libc_free_0(v6[0], v7[0] + 1LL);
  return result;
}
