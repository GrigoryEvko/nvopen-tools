// Function: sub_DB6630
// Address: 0xdb6630
//
__int16 __fastcall sub_DB6630(__int64 a1, __int64 a2)
{
  __int16 result; // ax

  result = sub_D4A480(a2);
  if ( !(_BYTE)result )
  {
    result = sub_D4A4C0(a2);
    if ( (_BYTE)result )
      return (unsigned __int16)sub_DB5FD0(a1, a2) >> 8;
  }
  return result;
}
