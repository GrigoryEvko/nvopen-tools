// Function: sub_15A93E0
// Address: 0x15a93e0
//
void __fastcall sub_15A93E0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  sub_15A9210((__int64)a1);
  v2 = a1[51];
  if ( (_QWORD *)v2 != a1 + 53 )
    _libc_free(v2);
  v3 = a1[28];
  if ( (_QWORD *)v3 != a1 + 30 )
    _libc_free(v3);
  v4 = (_QWORD *)a1[24];
  if ( v4 != a1 + 26 )
    j_j___libc_free_0(v4, a1[26] + 1LL);
  v5 = a1[6];
  if ( (_QWORD *)v5 != a1 + 8 )
    _libc_free(v5);
  v6 = a1[3];
  if ( (_QWORD *)v6 != a1 + 5 )
    _libc_free(v6);
}
