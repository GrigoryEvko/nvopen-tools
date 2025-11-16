// Function: sub_1EA39F0
// Address: 0x1ea39f0
//
__int64 __fastcall sub_1EA39F0(_QWORD *a1)
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
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 392);
}
