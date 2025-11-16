// Function: sub_2217280
// Address: 0x2217280
//
bool __fastcall sub_2217280(__int64 a1, unsigned __int16 a2)
{
  __int64 v2; // rbx

  v2 = 0;
  if ( *(_WORD *)(a1 + 1190) == a2 )
    return (unsigned int)__iswctype_l() != 0;
  while ( 1 )
  {
    if ( (*(_WORD *)(a1 + 2 * v2 + 1180) & a2) == 0 )
      goto LABEL_3;
    if ( (unsigned int)__iswctype_l() )
      return 1;
    if ( *(_WORD *)(a1 + 2 * v2 + 1180) == a2 )
      return 0;
LABEL_3:
    if ( ++v2 == 12 )
      return 0;
  }
}
