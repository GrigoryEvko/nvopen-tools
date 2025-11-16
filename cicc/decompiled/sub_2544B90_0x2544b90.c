// Function: sub_2544B90
// Address: 0x2544b90
//
__int64 __fastcall sub_2544B90(__int64 a1)
{
  unsigned __int64 v2; // rdi

  if ( *(_BYTE *)(a1 + 140) )
  {
    if ( *(_BYTE *)(a1 + 44) )
      goto LABEL_3;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 120));
    if ( *(_BYTE *)(a1 + 44) )
      goto LABEL_3;
  }
  _libc_free(*(_QWORD *)(a1 + 24));
LABEL_3:
  v2 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v2 != a1 - 32 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
