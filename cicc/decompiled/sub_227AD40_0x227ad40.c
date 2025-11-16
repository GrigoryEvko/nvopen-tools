// Function: sub_227AD40
// Address: 0x227ad40
//
void __fastcall sub_227AD40(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 76) )
  {
    if ( *(_BYTE *)(a1 + 28) )
      return;
LABEL_5:
    _libc_free(*(_QWORD *)(a1 + 8));
    return;
  }
  _libc_free(*(_QWORD *)(a1 + 56));
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_5;
}
