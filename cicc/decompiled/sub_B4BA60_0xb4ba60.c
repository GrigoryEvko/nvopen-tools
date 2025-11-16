// Function: sub_B4BA60
// Address: 0xb4ba60
//
__int64 __fastcall sub_B4BA60(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  int v5; // r9d

  v5 = *a1;
  if ( v5 == 40 )
    return sub_B4B4A0(a1, a2, a3, a4, a5);
  if ( v5 == 85 )
    return sub_B4A5D0(a1, a2, a3, a4, a5);
  if ( v5 != 34 )
    BUG();
  return sub_B4AD00(a1, a2, a3, a4, a5);
}
