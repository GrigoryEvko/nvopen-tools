// Function: sub_16BA250
// Address: 0x16ba250
//
bool __fastcall sub_16BA250(_BYTE *a1, unsigned __int64 a2)
{
  bool result; // al

  result = 0;
  if ( a2 > 1 )
  {
    if ( *a1 == 0xFF )
    {
      return a1[1] == 0xFE;
    }
    else if ( *a1 == 0xFE )
    {
      return a1[1] == 0xFF;
    }
  }
  return result;
}
