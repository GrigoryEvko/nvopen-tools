// Function: sub_1848570
// Address: 0x1848570
//
__int64 __fastcall sub_1848570(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_49F10C0;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    _libc_free(v2);
  nullsub_518();
  return j_j___libc_free_0(a1, 72);
}
