// Function: sub_23AF020
// Address: 0x23af020
//
void __fastcall sub_23AF020(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 84) )
  {
    if ( *(_BYTE *)(a1 + 36) )
      return;
LABEL_5:
    _libc_free(*(_QWORD *)(a1 + 16));
    return;
  }
  _libc_free(*(_QWORD *)(a1 + 64));
  if ( !*(_BYTE *)(a1 + 36) )
    goto LABEL_5;
}
