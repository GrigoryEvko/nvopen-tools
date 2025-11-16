// Function: sub_ED10E0
// Address: 0xed10e0
//
_QWORD *__fastcall sub_ED10E0(int *a1, __int64 a2)
{
  __int64 *(__fastcall *v2)(__int64 *, __int64); // rax
  _QWORD *result; // rax
  unsigned __int8 *v4[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v5[3]; // [rsp+10h] [rbp-20h] BYREF

  v2 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)a1 + 24LL);
  if ( v2 == sub_ED10B0 )
    sub_ED0C50((__int64 *)v4, a1[2], (__int64)(a1 + 4));
  else
    v2((__int64 *)v4, (__int64)a1);
  sub_CB6200(a2, v4[0], (size_t)v4[1]);
  result = v5;
  if ( (_QWORD *)v4[0] != v5 )
    return (_QWORD *)j_j___libc_free_0(v4[0], v5[0] + 1LL);
  return result;
}
