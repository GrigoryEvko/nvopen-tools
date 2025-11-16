// Function: sub_FFEE20
// Address: 0xffee20
//
__int64 __fastcall sub_FFEE20(_BYTE *a1)
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
