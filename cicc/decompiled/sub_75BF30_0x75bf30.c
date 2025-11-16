// Function: sub_75BF30
// Address: 0x75bf30
//
__int64 *__fastcall sub_75BF30(__int64 a1, char a2)
{
  __int64 *result; // rax
  __int64 v4; // rdi

  result = (__int64 *)sub_72A270(a1, a2);
  if ( dword_4F07588 )
  {
    if ( result )
    {
      result = (__int64 *)result[4];
      if ( result )
      {
        v4 = *result;
        if ( a1 != *result && (*(_BYTE *)(v4 - 8) & 2) != 0 )
          return (__int64 *)sub_75B260(v4, a2);
      }
    }
  }
  return result;
}
