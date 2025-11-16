// Function: sub_3352780
// Address: 0x3352780
//
void __fastcall sub_3352780(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi

  *(_QWORD *)a1 = off_4A36238;
  v2 = *(_QWORD *)(a1 + 672);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(_QWORD *)(a1 + 640);
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  if ( (*(_BYTE *)(a1 + 1216) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 1224), 16LL * *(unsigned int *)(a1 + 1232), 8);
  v4 = *(_QWORD *)(a1 + 1136);
  if ( v4 != a1 + 1152 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 1112);
  if ( v5 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 1088);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 816);
  if ( v7 != a1 + 832 )
    _libc_free(v7);
  v8 = *(unsigned int *)(a1 + 784);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 768);
    v10 = &v9[5 * v8];
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[1];
        if ( (_QWORD *)v11 != v9 + 3 )
          _libc_free(v11);
      }
      v9 += 5;
    }
    while ( v10 != v9 );
    v8 = *(unsigned int *)(a1 + 784);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 768), 40 * v8, 8);
  v12 = *(_QWORD *)(a1 + 712);
  if ( v12 != a1 + 728 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 704);
  if ( v13 )
    j_j___libc_free_0_0(v13);
  v14 = *(_QWORD *)(a1 + 696);
  if ( v14 )
    j_j___libc_free_0_0(v14);
  v15 = *(_QWORD *)(a1 + 648);
  if ( v15 )
    j_j___libc_free_0(v15);
  v16 = *(_QWORD *)(a1 + 608);
  *(_QWORD *)a1 = &unk_4A365B8;
  if ( v16 )
    j_j___libc_free_0(v16);
  sub_2F8EAD0((_QWORD *)a1);
}
