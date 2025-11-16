// Function: sub_35B69E0
// Address: 0x35b69e0
//
__int64 __fastcall sub_35B69E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  void (__fastcall *v7)(__int64, __int64, __int64); // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // r13
  __int64 i; // rbx
  unsigned __int64 v15; // rdi

  *(_QWORD *)a1 = off_4A3A088;
  *(_QWORD *)(a1 + 200) = &unk_4A3A180;
  *(_QWORD *)(a1 + 960) = &unk_4A3A1D8;
  v2 = *(_QWORD *)(a1 + 1016);
  if ( v2 != a1 + 1032 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 984);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 976);
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  v5 = *(_QWORD *)(a1 + 928);
  *(_QWORD *)(a1 + 200) = &unk_4A3A030;
  sub_35B6810(v5);
  v6 = *(_QWORD *)(a1 + 888);
  if ( v6 != a1 + 904 )
    _libc_free(v6);
  if ( !*(_BYTE *)(a1 + 628) )
    _libc_free(*(_QWORD *)(a1 + 608));
  v7 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 584);
  if ( v7 )
    v7(a1 + 568, a1 + 568, 3);
  v8 = *(_QWORD *)(a1 + 544);
  if ( v8 )
    j_j___libc_free_0_0(v8);
  v9 = *(_QWORD *)(a1 + 472);
  if ( v9 != a1 + 488 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 400);
  if ( v10 != a1 + 416 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 336);
  if ( v11 != a1 + 360 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 280);
  if ( v12 != a1 + 304 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 248);
  if ( v13 )
  {
    for ( i = v13 + 24LL * *(_QWORD *)(v13 - 8); v13 != i; i -= 24 )
    {
      v15 = *(_QWORD *)(i - 8);
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    j_j_j___libc_free_0_0(v13 - 8);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
