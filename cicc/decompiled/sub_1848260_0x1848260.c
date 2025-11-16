// Function: sub_1848260
// Address: 0x1848260
//
void __fastcall sub_1848260(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_49F10C0;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    _libc_free(v2);
  nullsub_518();
}
