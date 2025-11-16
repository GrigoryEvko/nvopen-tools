// Function: sub_1BE4730
// Address: 0x1be4730
//
__int64 __fastcall sub_1BE4730(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rdi

  *a1 = &unk_49F7138;
  v2 = a1[14];
  if ( v2 )
    sub_1BE4260(v2);
  v3 = a1[10];
  *a1 = &unk_49F6D50;
  if ( (_QWORD *)v3 != a1 + 12 )
    _libc_free(v3);
  v4 = a1[7];
  if ( (_QWORD *)v4 != a1 + 9 )
    _libc_free(v4);
  v5 = (_QWORD *)a1[2];
  if ( v5 != a1 + 4 )
    j_j___libc_free_0(v5, a1[4] + 1LL);
  return j_j___libc_free_0(a1, 136);
}
