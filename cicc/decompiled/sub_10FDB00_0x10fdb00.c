// Function: sub_10FDB00
// Address: 0x10fdb00
//
_BOOL8 __fastcall sub_10FDB00(__int64 a1, _DWORD *a2)
{
  __int64 *v3; // rsi
  void *v4; // rbx
  _BOOL4 v5; // r12d
  _QWORD *i; // rbx
  bool v8; // [rsp+Fh] [rbp-41h] BYREF
  void *v9; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v10; // [rsp+18h] [rbp-38h]

  v3 = (__int64 *)(a1 + 24);
  v4 = sub_C33340();
  if ( *(void **)(a1 + 24) == v4 )
    sub_C3C790(&v9, (_QWORD **)v3);
  else
    sub_C33EB0(&v9, v3);
  sub_C41640((__int64 *)&v9, a2, 1, &v8);
  v5 = !v8;
  if ( v9 != v4 )
  {
    sub_C338F0((__int64)&v9);
    return v5;
  }
  if ( !v10 )
    return v5;
  for ( i = &v10[3 * *(v10 - 1)]; v10 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0(i - 1);
  return v5;
}
