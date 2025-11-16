// Function: sub_72DB00
// Address: 0x72db00
//
__int64 *__fastcall sub_72DB00(__int64 a1, char a2, __int64 a3)
{
  __int64 *result; // rax

  if ( !a3 )
    return 0;
  result = *(__int64 **)(a3 + 216);
  if ( result )
  {
    while ( result[4] != a1 || *((_BYTE *)result + 16) != a2 )
    {
      result = (__int64 *)*result;
      if ( !result )
        return result;
    }
    return (__int64 *)result[1];
  }
  return result;
}
