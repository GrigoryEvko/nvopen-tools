// Function: sub_BA6610
// Address: 0xba6610
//
void __fastcall sub_BA6610(__m128i *a1, unsigned int a2, unsigned __int8 *a3)
{
  unsigned __int8 v3; // al
  unsigned __int8 **v4; // r8

  v3 = a1[-1].m128i_u8[0];
  if ( (v3 & 2) != 0 )
  {
    v4 = (unsigned __int8 **)(a1[-2].m128i_i64[0] + 8LL * a2);
    if ( a3 == *v4 )
      return;
  }
  else
  {
    v4 = (unsigned __int8 **)&a1[-1] + a2 - ((v3 >> 2) & 0xF);
    if ( a3 == *v4 )
      return;
  }
  if ( (a1->m128i_i8[1] & 0x7F) != 0 )
    sub_B97110((__int64)a1, a2, (__int64)a3);
  else
    sub_BA56C0(a1, (__int64)v4, a3);
}
