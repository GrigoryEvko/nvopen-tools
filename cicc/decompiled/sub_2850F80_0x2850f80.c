// Function: sub_2850F80
// Address: 0x2850f80
//
void __fastcall sub_2850F80(__int64 *a1, unsigned __int8 **a2)
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
