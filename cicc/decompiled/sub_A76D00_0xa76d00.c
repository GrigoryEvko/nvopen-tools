// Function: sub_A76D00
// Address: 0xa76d00
//
__int64 *__fastcall sub_A76D00(__int64 *a1, __int64 *a2, char a3)
{
  __int64 v3; // rsi

  v3 = *a2;
  if ( v3 )
  {
    sub_A76BD0(a1, v3, a3);
  }
  else
  {
    *a1 = (__int64)(a1 + 2);
    sub_A6E150(a1, byte_3F871B3, (__int64)byte_3F871B3);
  }
  return a1;
}
