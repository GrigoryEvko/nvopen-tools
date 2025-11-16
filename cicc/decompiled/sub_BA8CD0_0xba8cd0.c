// Function: sub_BA8CD0
// Address: 0xba8cd0
//
_BYTE *__fastcall sub_BA8CD0(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  _BYTE *result; // rax

  result = (_BYTE *)sub_BA8B30(a1, a2, a3);
  if ( result )
  {
    if ( *result == 3 )
    {
      if ( !a4 && (result[32] & 0xFu) - 7 <= 1 )
        return 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
