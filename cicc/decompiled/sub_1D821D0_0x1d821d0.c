// Function: sub_1D821D0
// Address: 0x1d821d0
//
void *__fastcall sub_1D821D0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  _libc_free(a1[130]);
  v2 = a1[124];
  if ( (_QWORD *)v2 != a1 + 126 )
    _libc_free(v2);
  _libc_free(a1[121]);
  v3 = a1[110];
  if ( v3 != a1[109] )
    _libc_free(v3);
  v4 = a1[86];
  if ( (_QWORD *)v4 != a1 + 88 )
    _libc_free(v4);
  v5 = a1[52];
  if ( (_QWORD *)v5 != a1 + 54 )
    _libc_free(v5);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
