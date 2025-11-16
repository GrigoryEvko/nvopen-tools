// Function: sub_1D47060
// Address: 0x1d47060
//
void __fastcall sub_1D47060(_QWORD *a1)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = (_QWORD *)a1[27];
  qword_4FC1B10[2] = 0;
  if ( v2 != a1 + 29 )
    _libc_free((unsigned __int64)v2);
  v3 = a1[12];
  if ( v3 != a1[11] )
    _libc_free(v3);
}
