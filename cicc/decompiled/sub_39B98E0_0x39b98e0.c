// Function: sub_39B98E0
// Address: 0x39b98e0
//
void *__fastcall sub_39B98E0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A40370;
  v2 = a1[30];
  if ( (_QWORD *)v2 != a1 + 32 )
    j_j___libc_free_0(v2);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
