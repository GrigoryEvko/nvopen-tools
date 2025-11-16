// Function: sub_1D021E0
// Address: 0x1d021e0
//
__int64 __fastcall sub_1D021E0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi

  *(_QWORD *)a1 = off_49F94A0;
  v2 = *(_QWORD *)(a1 + 704);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(_QWORD *)(a1 + 672);
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 920));
  _libc_free(*(_QWORD *)(a1 + 888));
  v4 = *(_QWORD *)(a1 + 864);
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 880) - v4);
  v5 = *(_QWORD *)(a1 + 840);
  if ( v5 )
    j_j___libc_free_0(v5, *(_QWORD *)(a1 + 856) - v5);
  v6 = *(unsigned int *)(a1 + 816);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 800);
    v8 = &v7[5 * v6];
    do
    {
      if ( *v7 != -16 && *v7 != -8 )
      {
        v9 = v7[1];
        if ( (_QWORD *)v9 != v7 + 3 )
          _libc_free(v9);
      }
      v7 += 5;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 800));
  v10 = *(_QWORD *)(a1 + 744);
  if ( v10 != a1 + 760 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 736);
  if ( v11 )
    j_j___libc_free_0_0(v11);
  v12 = *(_QWORD *)(a1 + 728);
  if ( v12 )
    j_j___libc_free_0_0(v12);
  v13 = *(_QWORD *)(a1 + 680);
  if ( v13 )
    j_j___libc_free_0(v13, *(_QWORD *)(a1 + 696) - v13);
  v14 = *(_QWORD *)(a1 + 640);
  *(_QWORD *)a1 = &unk_49F9818;
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 656) - v14);
  return sub_1F012F0(a1);
}
