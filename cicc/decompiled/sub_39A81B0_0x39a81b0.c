// Function: sub_39A81B0
// Address: 0x39a81b0
//
unsigned __int8 *__fastcall sub_39A81B0(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al

  if ( !a2 )
    return (unsigned __int8 *)(a1 + 8);
  v2 = *a2;
  if ( *a2 == 15 )
    return (unsigned __int8 *)(a1 + 8);
  if ( v2 <= 0xEu )
  {
    if ( v2 <= 0xAu )
      return sub_39A23D0(a1, a2);
    return (unsigned __int8 *)sub_39A64F0((__int64 *)a1, (__int64)a2);
  }
  if ( (unsigned __int8)(v2 - 32) <= 1u )
    return (unsigned __int8 *)sub_39A64F0((__int64 *)a1, (__int64)a2);
  if ( v2 == 20 )
    return (unsigned __int8 *)sub_39A82F0(a1);
  if ( v2 == 17 )
    return (unsigned __int8 *)sub_39A8220(a1, a2, 0);
  if ( v2 != 21 )
    return sub_39A23D0(a1, a2);
  return (unsigned __int8 *)sub_39A8430(a1);
}
