// Function: sub_160CDC0
// Address: 0x160cdc0
//
void __fastcall sub_160CDC0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = &unk_49ED740;
  v2 = (_QWORD *)a1[28];
  if ( v2 != a1 + 30 )
    _libc_free((unsigned __int64)v2);
  v3 = a1[23];
  if ( v3 )
    j_j___libc_free_0(v3, a1[25] - v3);
  v4 = a1[20];
  if ( v4 )
    j_j___libc_free_0(v4, a1[22] - v4);
  v5 = a1[12];
  if ( v5 != a1[11] )
    _libc_free(v5);
}
