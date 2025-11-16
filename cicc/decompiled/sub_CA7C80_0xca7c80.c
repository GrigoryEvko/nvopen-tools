// Function: sub_CA7C80
// Address: 0xca7c80
//
_BYTE *__fastcall sub_CA7C80(__int64 a1, _BYTE *a2)
{
  _BYTE *v2; // rdx
  _BYTE *result; // rax

  v2 = *(_BYTE **)(a1 + 48);
  result = a2;
  if ( v2 != a2 )
  {
    if ( *a2 == 13 )
    {
      if ( v2 == a2 + 1 )
      {
        return *(_BYTE **)(a1 + 48);
      }
      else if ( a2[1] == 10 )
      {
        return a2 + 2;
      }
      else
      {
        return a2 + 1;
      }
    }
    else
    {
      return &a2[*a2 == 10];
    }
  }
  return result;
}
