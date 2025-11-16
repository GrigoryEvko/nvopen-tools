// Function: sub_7A6790
// Address: 0x7a6790
//
__int64 __fastcall sub_7A6790(__int64 a1)
{
  __int64 i; // r12

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !(unsigned int)sub_8D3410(i) )
    return i;
  while ( (*(_WORD *)(i + 168) & 0x180) == 0 && !(unsigned int)sub_8D23B0(i) && *(_QWORD *)(i + 176) != 1 )
  {
    do
      i = *(_QWORD *)(i + 160);
    while ( *(_BYTE *)(i + 140) == 12 );
    if ( !(unsigned int)sub_8D3410(i) )
      return i;
  }
  return a1;
}
