// Function: sub_B17C60
// Address: 0xb17c60
//
_QWORD *__fastcall sub_B17C60(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, _QWORD *); // r14
  __int64 v3; // rax
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, _QWORD *); // r15
  _QWORD *result; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v12; // [rsp+30h] [rbp-40h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL);
  sub_B15A70((__int64)v9, a1);
  v3 = v2(a2, v9);
  v4 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v3 + 48LL))(v3, ": ");
  v5 = *(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v4 + 56LL);
  sub_B17B60((__int64)v11, a1);
  v5(v4, v11);
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0], v12 + 1);
  result = v10;
  if ( (_QWORD *)v9[0] != v10 )
    result = (_QWORD *)j_j___libc_free_0(v9[0], v10[0] + 1LL);
  if ( *(_BYTE *)(a1 + 72) )
  {
    v7 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a2 + 48LL))(a2, " (hotness: ");
    v8 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v7 + 64LL))(v7, *(_QWORD *)(a1 + 64));
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)v8 + 48LL))(v8, ")");
  }
  return result;
}
