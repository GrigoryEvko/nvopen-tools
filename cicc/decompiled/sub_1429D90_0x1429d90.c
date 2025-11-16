// Function: sub_1429D90
// Address: 0x1429d90
//
__int64 __fastcall sub_1429D90(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  *a1 = off_49EB3D0;
  v2 = a1[72];
  if ( v2 != a1[71] )
    _libc_free(v2);
  v3 = a1[60];
  if ( (_QWORD *)v3 != a1 + 62 )
    _libc_free(v3);
  v4 = a1[50];
  if ( (_QWORD *)v4 != a1 + 52 )
    _libc_free(v4);
  v5 = a1[40];
  if ( (_QWORD *)v5 != a1 + 42 )
    _libc_free(v5);
  v6 = a1[30];
  if ( (_QWORD *)v6 != a1 + 32 )
    _libc_free(v6);
  v7 = a1[20];
  if ( (_QWORD *)v7 != a1 + 22 )
    _libc_free(v7);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 856);
}
