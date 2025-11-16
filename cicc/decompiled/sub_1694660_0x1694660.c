// Function: sub_1694660
// Address: 0x1694660
//
_QWORD *__fastcall sub_1694660(int *a1, __int64 a2)
{
  __int64 *(__fastcall *v2)(__int64 *, __int64); // rax
  _QWORD *result; // rax
  const char *v4[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v5[3]; // [rsp+10h] [rbp-20h] BYREF

  v2 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)a1 + 24LL);
  if ( v2 == sub_1694640 )
    sub_1693E90((__int64 *)v4, a1[2]);
  else
    v2((__int64 *)v4, (__int64)a1);
  sub_16E7EE0(a2, v4[0], v4[1]);
  result = v5;
  if ( (_QWORD *)v4[0] != v5 )
    return (_QWORD *)j_j___libc_free_0(v4[0], v5[0] + 1LL);
  return result;
}
