// Function: sub_B15C40
// Address: 0xb15c40
//
_QWORD *__fastcall sub_B15C40(_QWORD *a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, _QWORD *); // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *result; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL);
  sub_B15A70((__int64)v13, (__int64)a1);
  v3 = v2(a2, v13);
  v4 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v3 + 48LL))(v3, ": ");
  v5 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 48LL))(v4, a1[6]);
  v6 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v5 + 48LL))(v5, " (");
  v7 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v6 + 64LL))(v6, a1[7]);
  v8 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v7 + 48LL))(v7, ") exceeds limit (");
  v9 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v8 + 64LL))(v8, a1[8]);
  v10 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v9 + 48LL))(v9, ") in function '");
  v11 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v10 + 136LL))(v10, a1[5]);
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 16LL))(v11, 39);
  result = v14;
  if ( (_QWORD *)v13[0] != v14 )
    return (_QWORD *)j_j___libc_free_0(v13[0], v14[0] + 1LL);
  return result;
}
