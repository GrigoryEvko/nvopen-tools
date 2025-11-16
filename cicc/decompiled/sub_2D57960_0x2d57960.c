// Function: sub_2D57960
// Address: 0x2d57960
//
void __fastcall sub_2D57960(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = (_QWORD *)a1[16];
  *a1 = off_49D4180;
  if ( v1 )
  {
    v3 = v1[15];
    if ( (_QWORD *)v3 != v1 + 17 )
      _libc_free(v3);
    v4 = v1[12];
    if ( (_QWORD *)v4 != v1 + 14 )
      _libc_free(v4);
    v5 = v1[2];
    if ( (_QWORD *)v5 != v1 + 4 )
      _libc_free(v5);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  v6 = a1[10];
  if ( (_QWORD *)v6 != a1 + 12 )
    _libc_free(v6);
  j_j___libc_free_0((unsigned __int64)a1);
}
