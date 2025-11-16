// Function: sub_E4CB10
// Address: 0xe4cb10
//
__int64 __fastcall sub_E4CB10(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi

  v3 = (__int64)(a1 + 87);
  *(_QWORD *)(v3 - 696) = off_49E14D8;
  sub_CB58D0(v3);
  a1[80] = &unk_49DD388;
  sub_CB5840((__int64)(a1 + 80));
  v4 = (_QWORD *)a1[61];
  if ( v4 != a1 + 64 )
    _libc_free(v4, a2);
  v5 = (_QWORD *)a1[42];
  if ( v5 != a1 + 45 )
    _libc_free(v5, a2);
  v6 = a1[41];
  if ( v6 )
  {
    if ( !*(_BYTE *)(v6 + 108) )
      _libc_free(*(_QWORD *)(v6 + 88), a2);
    v7 = *(_QWORD *)(v6 + 56);
    if ( v7 != v6 + 72 )
      _libc_free(v7, a2);
    v8 = *(_QWORD *)(v6 + 40);
    if ( v6 + 56 != v8 )
      _libc_free(v8, a2);
    v9 = *(_QWORD *)(v6 + 24);
    if ( v9 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
    v10 = *(_QWORD *)(v6 + 16);
    if ( v10 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
    v11 = *(_QWORD *)(v6 + 8);
    if ( v11 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
    j_j___libc_free_0(v6, 376);
  }
  v12 = a1[40];
  if ( v12 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  v13 = a1[37];
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  return sub_E98B30(a1);
}
