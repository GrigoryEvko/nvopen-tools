// Function: sub_1EA3960
// Address: 0x1ea3960
//
void *__fastcall sub_1EA3960(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_49FD228;
  j___libc_free_0(a1[46]);
  v2 = a1[38];
  if ( v2 != a1[37] )
    _libc_free(v2);
  j___libc_free_0(a1[33]);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
