// Function: sub_CEA480
// Address: 0xcea480
//
bool __fastcall sub_CEA480(int a1)
{
  if ( a1 == 27 )
    return 0;
  if ( a1 > 27 )
    return (unsigned int)(a1 - 74) > 0xB;
  return (unsigned int)(a1 - 15) > 7;
}
