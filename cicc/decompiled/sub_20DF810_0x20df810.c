// Function: sub_20DF810
// Address: 0x20df810
//
void *__fastcall sub_20DF810(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  _QWORD *v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = off_4A007D8;
  _libc_free(a1[55]);
  v2 = a1[49];
  if ( (_QWORD *)v2 != a1 + 51 )
    _libc_free(v2);
  v3 = (_QWORD *)a1[47];
  if ( v3 )
  {
    _libc_free(v3[22]);
    _libc_free(v3[19]);
    _libc_free(v3[16]);
    _libc_free(v3[13]);
    v4 = v3[6];
    if ( (_QWORD *)v4 != v3 + 8 )
      _libc_free(v4);
    j_j___libc_free_0(v3, 200);
  }
  v5 = a1[29];
  if ( (_QWORD *)v5 != a1 + 31 )
    _libc_free(v5);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
