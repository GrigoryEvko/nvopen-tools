// Function: sub_AD8340
// Address: 0xad8340
//
unsigned __int8 *__fastcall sub_AD8340(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  if ( *a1 == 17 )
    return a1 + 24;
  if ( *a1 == 5 )
    return sub_AD7630((__int64)a1, 0, a3) + 24;
  return (unsigned __int8 *)(sub_AD69F0(a1, 0) + 24);
}
