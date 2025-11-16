// Function: sub_9C2500
// Address: 0x9c2500
//
unsigned __int64 __fastcall sub_9C2500(unsigned __int64 a1, unsigned __int64 a2)
{
  __int16 v2; // r9

  if ( (a1 & 0x10) != 0 )
  {
    if ( ((a1 >> 4) & 2) == 0 )
    {
      v2 = 1;
      LOBYTE(a2) = a2 <= 2;
      return (((a1 >> 10) & 1) << 10)
           | (((a1 >> 7) & 1) << 9)
           | (((unsigned __int16)(a1 >> 6) & 1u) << 8)
           | (unsigned __int16)(((unsigned __int8)a2 << 7) | (v2 << 6) | a1 & 0xF | (16 * (BYTE1(a1) & 3)));
    }
  }
  else if ( a2 > 2 )
  {
    v2 = 0;
    a2 = (a1 >> 5) & 1;
    return (((a1 >> 10) & 1) << 10)
         | (((a1 >> 7) & 1) << 9)
         | (((unsigned __int16)(a1 >> 6) & 1u) << 8)
         | (unsigned __int16)(((unsigned __int8)a2 << 7) | (v2 << 6) | a1 & 0xF | (16 * (BYTE1(a1) & 3)));
  }
  v2 = 1;
  LOBYTE(a2) = 1;
  return (((a1 >> 10) & 1) << 10)
       | (((a1 >> 7) & 1) << 9)
       | (((unsigned __int16)(a1 >> 6) & 1u) << 8)
       | (unsigned __int16)(((unsigned __int8)a2 << 7) | (v2 << 6) | a1 & 0xF | (16 * (BYTE1(a1) & 3)));
}
