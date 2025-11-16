// Function: sub_325F060
// Address: 0x325f060
//
bool __fastcall sub_325F060(char *s2, __int64 a2)
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
