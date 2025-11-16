// Function: sub_10B8470
// Address: 0x10b8470
//
bool __fastcall sub_10B8470(unsigned __int8 a1)
{
  bool result; // al

  result = 1;
  if ( a1 > 3u && a1 != 5 )
    return (a1 & 0xFD) == 4;
  return result;
}
