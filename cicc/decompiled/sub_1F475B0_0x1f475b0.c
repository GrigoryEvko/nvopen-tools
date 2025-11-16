// Function: sub_1F475B0
// Address: 0x1f475b0
//
bool __fastcall sub_1F475B0(__int64 a1)
{
  bool result; // al

  result = 1;
  if ( dword_4FCD500 != 1 )
  {
    result = 0;
    if ( dword_4FCD500 != 2 )
      return (unsigned int)sub_1F45DD0(a1) != 0;
  }
  return result;
}
