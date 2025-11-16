// Function: sub_72F880
// Address: 0x72f880
//
__int64 __fastcall sub_72F880(_BYTE *a1)
{
  __int64 result; // rax

  result = 0;
  if ( (a1[193] & 0x10) == 0 )
  {
    result = 1;
    if ( (a1[206] & 8) != 0 )
      return a1[203] & 1;
  }
  return result;
}
