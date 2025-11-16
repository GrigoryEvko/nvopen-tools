// Function: sub_2670E90
// Address: 0x2670e90
//
void __fastcall sub_2670E90(__int64 *a1, unsigned __int8 **a2)
{
  unsigned __int8 *v3; // rsi

  if ( *a1 )
    sub_B91220((__int64)a1, *a1);
  v3 = *a2;
  *a1 = (__int64)*a2;
  if ( v3 )
  {
    sub_B976B0((__int64)a2, v3, (__int64)a1);
    *a2 = 0;
  }
}
