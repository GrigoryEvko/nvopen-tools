// Function: sub_B15B90
// Address: 0xb15b90
//
_QWORD *__fastcall sub_B15B90(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, _QWORD *); // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *result; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL);
  sub_B15A70((__int64)v9, a1);
  v3 = v2(a2, v9);
  v4 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v3 + 48LL))(v3, ": ");
  v5 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 128LL))(v4, *(_QWORD *)(a1 + 40));
  v6 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v5 + 48LL))(v5, " in function '");
  v7 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v6 + 136LL))(v6, *(_QWORD *)(a1 + 16));
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 16LL))(v7, 39);
  result = v10;
  if ( (_QWORD *)v9[0] != v10 )
    return (_QWORD *)j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return result;
}
