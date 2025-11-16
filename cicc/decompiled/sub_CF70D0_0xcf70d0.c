// Function: sub_CF70D0
// Address: 0xcf70d0
//
__int64 __fastcall sub_CF70D0(unsigned __int8 *a1)
{
  __int64 result; // rax

  if ( *a1 == 60 )
    return 1;
  result = sub_CF6FD0(a1);
  if ( (_BYTE)result )
    return 1;
  if ( *a1 == 22 )
  {
    if ( !(unsigned __int8)sub_B2D700((__int64)a1) )
      return sub_B2D680((__int64)a1);
    return 1;
  }
  return result;
}
