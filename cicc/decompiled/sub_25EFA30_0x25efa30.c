// Function: sub_25EFA30
// Address: 0x25efa30
//
__int64 __fastcall sub_25EFA30(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d

  v2 = 0;
  if ( !(unsigned __int8)sub_B2D610(a1, 5) )
  {
    v2 = 1;
    sub_B2CD30(a1, 5);
  }
  if ( !(unsigned __int8)sub_B2D610(a1, 18) )
  {
    v2 = 1;
    sub_B2CD30(a1, 18);
  }
  if ( !(_BYTE)a2 )
    return v2;
  sub_B2F560(a1, 0, 0, 0);
  return a2;
}
