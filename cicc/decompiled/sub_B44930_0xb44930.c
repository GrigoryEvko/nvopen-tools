// Function: sub_B44930
// Address: 0xb44930
//
__int64 __fastcall sub_B44930(__int64 a1)
{
  unsigned int v1; // r12d

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  v1 = 1;
  if ( !sub_B91C10(a1, 4)
    && ((*(_BYTE *)(a1 + 7) & 0x20) == 0
     || !sub_B91C10(a1, 11) && ((*(_BYTE *)(a1 + 7) & 0x20) == 0 || !sub_B91C10(a1, 17))) )
  {
    return 0;
  }
  return v1;
}
