// Function: sub_166FD70
// Address: 0x166fd70
//
__int64 __fastcall sub_166FD70(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = off_49EE4A0;
  v2 = a1[61];
  if ( v2 != a1[60] )
    _libc_free(v2);
  v3 = a1[41];
  if ( (_QWORD *)v3 != a1 + 43 )
    _libc_free(v3);
  v4 = a1[23];
  if ( (_QWORD *)v4 != a1 + 25 )
    _libc_free(v4);
  v5 = a1[5];
  if ( (_QWORD *)v5 != a1 + 7 )
    _libc_free(v5);
  return j___libc_free_0(a1[2]);
}
