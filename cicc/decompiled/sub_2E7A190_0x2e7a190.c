// Function: sub_2E7A190
// Address: 0x2e7a190
//
char __fastcall sub_2E7A190(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned int v3; // eax
  unsigned int v4; // edx

  if ( sub_2E7A170(a1) )
    return 20;
  v3 = sub_2E7A0A0(a1, a2);
  v4 = v3;
  if ( v3 == 16 )
    return 10;
  if ( v3 > 0x10 )
  {
    result = 11;
    if ( v4 != 32 )
      return 4;
  }
  else
  {
    result = 8;
    if ( v4 != 4 )
    {
      if ( v4 == 8 )
        return 9;
      return 4;
    }
  }
  return result;
}
