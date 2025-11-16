// Function: sub_12F9680
// Address: 0x12f9680
//
__int64 __fastcall sub_12F9680(unsigned __int16 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rcx

  v5 = ~(unsigned __int64)a1;
  BYTE1(v5) &= 0x7Cu;
  if ( BYTE1(v5) || (a1 & 0x3FF) == 0 )
  {
    v5 = ~(unsigned __int64)(unsigned __int16)a2;
    BYTE1(v5) = ((unsigned __int16)~(_WORD)a2 >> 8) & 0x7C;
    if ( BYTE1(v5) || (a2 & 0x3FF) == 0 )
    {
      LODWORD(a5) = 1;
      if ( a1 != (unsigned __int64)(unsigned __int16)a2 )
        LOBYTE(a5) = 2 * ((unsigned __int16)a2 | a1) == 0;
      return (unsigned int)a5;
    }
  }
  if ( (a1 & 0x7E00) != 0x7C00 || (a1 & 0x1FF) == 0 )
  {
    a5 = 0;
    if ( (a2 & 0x7E00) != 0x7C00 || (a2 & 0x1FF) == 0 )
      return (unsigned int)a5;
  }
  sub_12F9B70(16, a2, (unsigned __int16)a2, v5, a5);
  return 0;
}
