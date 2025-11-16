// Function: sub_23AF060
// Address: 0x23af060
//
void __fastcall sub_23AF060(unsigned __int64 a1)
{
  if ( !*(_BYTE *)(a1 + 84) )
  {
    _libc_free(*(_QWORD *)(a1 + 64));
    if ( *(_BYTE *)(a1 + 36) )
      goto LABEL_3;
LABEL_5:
    _libc_free(*(_QWORD *)(a1 + 16));
    goto LABEL_3;
  }
  if ( !*(_BYTE *)(a1 + 36) )
    goto LABEL_5;
LABEL_3:
  j_j___libc_free_0(a1);
}
