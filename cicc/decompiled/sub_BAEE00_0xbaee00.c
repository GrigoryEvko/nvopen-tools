// Function: sub_BAEE00
// Address: 0xbaee00
//
char __fastcall sub_BAEE00(_BYTE *a1)
{
  char result; // al

  result = a1[336];
  if ( a1[342] )
    result |= 2u;
  if ( a1[344] )
    result |= 8u;
  if ( a1[346] )
    result |= 0x10u;
  if ( a1[337] )
    result |= 0x20u;
  if ( a1[338] )
    result |= 0x40u;
  if ( a1[339] )
    return result | 0x80;
  return result;
}
