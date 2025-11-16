// Function: sub_16BC7D0
// Address: 0x16bc7d0
//
__int64 *__fastcall sub_16BC7D0(__int64 *a1, __int64 a2, int a3)
{
  *a1 = (__int64)(a1 + 2);
  if ( a3 == 1 )
    sub_16BC640(a1, "Multiple errors", (__int64)"");
  else
    sub_16BC640(
      a1,
      "Inconvertible error value. An error has occurred that could not be converted to a known std::error_code. Please file a bug.",
      (__int64)"");
  return a1;
}
