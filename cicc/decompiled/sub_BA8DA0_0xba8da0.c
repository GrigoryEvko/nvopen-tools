// Function: sub_BA8DA0
// Address: 0xba8da0
//
_BYTE *__fastcall sub_BA8DA0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  _BYTE *result; // rax

  result = (_BYTE *)sub_BA8B30(a1, a2, a3);
  if ( result )
  {
    if ( *result != 1 )
      return 0;
  }
  return result;
}
