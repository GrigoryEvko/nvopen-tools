// Function: sub_2DB3530
// Address: 0x2db3530
//
void __fastcall sub_2DB3530(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 1264);
  if ( v2 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 1216);
  if ( v3 != a1 + 1232 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 1144);
  if ( v4 != a1 + 1160 )
    _libc_free(v4);
  if ( !*(_BYTE *)(a1 + 1076) )
    _libc_free(*(_QWORD *)(a1 + 1056));
  v5 = *(_QWORD *)(a1 + 872);
  if ( v5 != a1 + 888 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 600);
  if ( v6 != a1 + 616 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 424);
  if ( v7 != a1 + 440 )
    _libc_free(v7);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
