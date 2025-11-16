// Function: sub_109CE80
// Address: 0x109ce80
//
__int64 __fastcall sub_109CE80(_BYTE *a1)
{
  __int64 result; // rax

  result = 1;
  if ( *a1 <= 0x15u )
  {
    result = 0;
    if ( *a1 != 5 )
      return (unsigned int)sub_AD6CA0((__int64)a1) ^ 1;
  }
  return result;
}
