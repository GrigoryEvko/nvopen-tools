// Function: sub_C9FD10
// Address: 0xc9fd10
//
void __fastcall sub_C9FD10(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  void (__fastcall *v5)(__int64, __int64, __int64); // rax
  __int64 v6; // rdi
  void (__fastcall *v7)(__int64, __int64, __int64); // rax
  __int64 v8; // rdi
  __int64 v9; // rdi

  if ( !a1 )
    return;
  if ( *(_BYTE *)(a1 + 864) )
  {
    *(_BYTE *)(a1 + 864) = 0;
    sub_C9FAB0(a1 + 840, a2);
  }
  sub_F04DF0(a1 + 824);
  sub_C9F930(a1 + 712);
  *(_QWORD *)(a1 + 464) = &unk_49DC090;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 648);
  if ( v3 )
  {
    a2 = a1 + 632;
    v3(a1 + 632, a1 + 632, 3);
  }
  if ( !*(_BYTE *)(a1 + 588) )
    _libc_free(*(_QWORD *)(a1 + 568), a2);
  v4 = *(_QWORD *)(a1 + 536);
  if ( v4 != a1 + 552 )
    _libc_free(v4, a2);
  *(_QWORD *)(a1 + 264) = &unk_49DC090;
  v5 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 448);
  if ( v5 )
  {
    a2 = a1 + 432;
    v5(a1 + 432, a1 + 432, 3);
  }
  if ( !*(_BYTE *)(a1 + 388) )
    _libc_free(*(_QWORD *)(a1 + 368), a2);
  v6 = *(_QWORD *)(a1 + 336);
  if ( v6 != a1 + 352 )
    _libc_free(v6, a2);
  *(_QWORD *)(a1 + 32) = &unk_49DCA98;
  v7 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 248);
  if ( v7 )
  {
    a2 = a1 + 232;
    v7(a1 + 232, a1 + 232, 3);
  }
  v8 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(a1 + 176) = &unk_49DACE8;
  if ( v8 != a1 + 200 )
  {
    a2 = *(_QWORD *)(a1 + 200) + 1LL;
    j_j___libc_free_0(v8, a2);
  }
  if ( !*(_BYTE *)(a1 + 156) )
  {
    _libc_free(*(_QWORD *)(a1 + 136), a2);
    v9 = *(_QWORD *)(a1 + 104);
    if ( v9 == a1 + 120 )
      goto LABEL_23;
    goto LABEL_22;
  }
  v9 = *(_QWORD *)(a1 + 104);
  if ( v9 != a1 + 120 )
LABEL_22:
    _libc_free(v9, a2);
LABEL_23:
  if ( *(_QWORD *)a1 != a1 + 16 )
    j_j___libc_free_0(*(_QWORD *)a1, *(_QWORD *)(a1 + 16) + 1LL);
  j_j___libc_free_0(a1, 872);
}
