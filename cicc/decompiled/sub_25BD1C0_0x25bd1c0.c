// Function: sub_25BD1C0
// Address: 0x25bd1c0
//
void __fastcall sub_25BD1C0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A1F2A8;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    _libc_free(v2);
  nullsub_185();
  j_j___libc_free_0((unsigned __int64)a1);
}
