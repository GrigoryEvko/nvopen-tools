// Function: sub_BAEE70
// Address: 0xbaee70
//
void __fastcall sub_BAEE70(_BYTE *a1, __int16 a2)
{
  if ( (a2 & 1) != 0 )
    a1[336] = 1;
  if ( (a2 & 2) != 0 )
    a1[342] = 1;
  if ( (a2 & 8) != 0 )
    a1[344] = 1;
  if ( (a2 & 0x10) != 0 )
    a1[346] = 1;
  if ( (a2 & 0x20) != 0 )
    a1[337] = 1;
  if ( (a2 & 0x40) != 0 )
    a1[338] = 1;
  if ( (a2 & 0x80u) != 0 )
    a1[339] = 1;
  if ( (a2 & 0x100) != 0 )
    a1[341] = 1;
  if ( (a2 & 0x200) != 0 )
    a1[345] = 1;
}
