// Function: sub_39B9960
// Address: 0x39b9960
//
void __fastcall sub_39B9960(_QWORD *a1)
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
  sub_16366C0(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
