// Function: sub_26740A0
// Address: 0x26740a0
//
void __fastcall sub_26740A0(__int64 a1)
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
