// Function: sub_2D9F590
// Address: 0x2d9f590
//
bool __fastcall sub_2D9F590(char *s2, __int64 a2)
{
  __int64 v2; // rsi
  bool result; // al

  v2 = 4 * a2;
  result = 1;
  if ( v2 )
  {
    if ( v2 != 4 )
      return memcmp(s2 + 4, s2, v2 - 4) == 0;
  }
  return result;
}
