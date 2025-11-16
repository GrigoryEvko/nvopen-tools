// Function: sub_B2DB90
// Address: 0xb2db90
//
__int64 __fastcall sub_B2DB90(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  unsigned int v4; // ecx

  if ( a2 != sub_C33310(a1, a2) )
    return sub_B2D9D0(a1);
  v3 = sub_B2DAA0(a1);
  if ( (_BYTE)v3 == 0xFF || HIBYTE(v3) == 0xFF )
    return sub_B2D9D0(a1);
  v4 = (unsigned __int8)v3;
  BYTE1(v4) = HIBYTE(v3);
  return v4;
}
