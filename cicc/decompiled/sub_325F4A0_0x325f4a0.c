// Function: sub_325F4A0
// Address: 0x325f4a0
//
bool __fastcall sub_325F4A0(__int64 a1, char a2, unsigned __int16 a3, unsigned __int16 a4)
{
  bool result; // al

  result = 0;
  if ( a3 )
  {
    if ( a4 )
      return (((int)*(unsigned __int16 *)(a1 + 2 * (a4 + 274LL * a3 + 71704) + 6) >> (4 * a2)) & 0xB) == 0;
  }
  return result;
}
