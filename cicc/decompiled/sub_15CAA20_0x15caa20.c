// Function: sub_15CAA20
// Address: 0x15caa20
//
_QWORD *__fastcall sub_15CAA20(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, _QWORD *); // r14
  __int64 v3; // rax
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, __int64 *); // r15
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *result; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL);
  sub_15C9210((__int64)v13, a1);
  v3 = v2(a2, v13);
  v4 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v3 + 48LL))(v3, ": ");
  v5 = *(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v4 + 56LL);
  sub_15CA8E0(v15, a1, v6, v7, v8, v9);
  v5(v4, v15);
  if ( (__int64 *)v15[0] != &v16 )
    j_j___libc_free_0(v15[0], v16 + 1);
  result = v14;
  if ( (_QWORD *)v13[0] != v14 )
    result = (_QWORD *)j_j___libc_free_0(v13[0], v14[0] + 1LL);
  if ( *(_BYTE *)(a1 + 80) )
  {
    v11 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a2 + 48LL))(a2, " (hotness: ");
    v12 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 64LL))(v11, *(_QWORD *)(a1 + 72));
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v12 + 48LL))(v12, ")");
  }
  return result;
}
