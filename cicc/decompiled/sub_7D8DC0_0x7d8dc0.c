// Function: sub_7D8DC0
// Address: 0x7d8dc0
//
void __fastcall sub_7D8DC0(unsigned __int8 a1, __m128i **a2)
{
  if ( a1 == 1 )
  {
    sub_802E80(*a2, 0, 0, 0);
    sub_7D8CF0(*a2);
  }
  else if ( a1 > 1u && ((a1 - 2) & 0xFD) != 0 )
  {
    sub_721090();
  }
}
