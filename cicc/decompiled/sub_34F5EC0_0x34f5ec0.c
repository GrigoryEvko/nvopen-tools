// Function: sub_34F5EC0
// Address: 0x34f5ec0
//
void __fastcall sub_34F5EC0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1 + 456;
  *(_QWORD *)(v2 - 456) = off_4A38700;
  sub_34F5380(v2);
  v3 = *(_QWORD *)(a1 + 376);
  if ( v3 != a1 + 392 )
    _libc_free(v3);
  if ( *(_BYTE *)(a1 + 308) )
  {
    if ( *(_BYTE *)(a1 + 212) )
      goto LABEL_5;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 288));
    if ( *(_BYTE *)(a1 + 212) )
      goto LABEL_5;
  }
  _libc_free(*(_QWORD *)(a1 + 192));
LABEL_5:
  v4 = *(_QWORD *)(a1 + 136);
  if ( v4 != a1 + 152 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 88);
  if ( v5 != a1 + 104 )
    _libc_free(v5);
  nullsub_1888();
}
