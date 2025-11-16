// Function: sub_141FE30
// Address: 0x141fe30
//
__int64 __fastcall sub_141FE30(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_49EB390;
  j___libc_free_0(a1[265]);
  v2 = a1[6];
  if ( (_QWORD *)v2 != a1 + 8 )
    _libc_free(v2);
  return j_j___libc_free_0(a1, 2144);
}
