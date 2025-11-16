// Function: sub_C33E70
// Address: 0xc33e70
//
__int64 *__fastcall sub_C33E70(__int64 *a1, __int64 *a2)
{
  if ( a1 != a2 )
  {
    if ( *a1 != *a2 )
    {
      sub_C33830((__int64)a1);
      sub_C337F0(a1, *a2);
    }
    sub_C33E20((__int64)a1, (__int64)a2);
  }
  return a1;
}
