// Function: sub_F58C20
// Address: 0xf58c20
//
__int64 __fastcall sub_F58C20(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned int v5; // r13d
  unsigned int v6; // eax

  if ( (unsigned __int8)sub_B2D610(a1, 39)
    || (LOBYTE(v3) = sub_B2DCC0(a1), v1 = v3, !(_BYTE)v3)
    || (unsigned __int8)sub_B2D610(a1, 6) )
  {
    v1 = 0;
  }
  else
  {
    sub_B2CD30(a1, 39);
  }
  if ( !(unsigned __int8)sub_B2D610(a1, 29) )
  {
    v6 = sub_B2DCE0(a1);
    if ( (_BYTE)v6 )
    {
      v1 = v6;
      sub_B2CD30(a1, 29);
    }
  }
  if ( (unsigned __int8)sub_B2D610(a1, 19) )
    return v1;
  v4 = sub_B2D610(a1, 76);
  if ( !(_BYTE)v4 )
    return v1;
  v5 = v4;
  sub_B2CD30(a1, 19);
  return v5;
}
