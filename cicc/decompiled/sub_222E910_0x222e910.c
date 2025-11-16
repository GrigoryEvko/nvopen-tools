// Function: sub_222E910
// Address: 0x222e910
//
__int64 __fastcall sub_222E910(__int64 a1, char *a2, char *a3)
{
  __int64 result; // rax
  __int64 v5; // rdx

  result = 0;
  if ( a2 < a3 )
  {
    do
    {
      v5 = *a2++;
      result = v5 + __ROL8__(result, 7);
    }
    while ( a3 != a2 );
  }
  return result;
}
