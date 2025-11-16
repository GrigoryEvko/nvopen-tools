// Function: sub_16E3E40
// Address: 0x16e3e40
//
void __fastcall sub_16E3E40(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_49EF9A8;
  v2 = a1[4];
  if ( (_QWORD *)v2 != a1 + 6 )
    _libc_free(v2);
  nullsub_627();
}
