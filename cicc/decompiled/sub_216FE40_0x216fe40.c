// Function: sub_216FE40
// Address: 0x216fe40
//
__int64 __fastcall sub_216FE40(unsigned __int64 a1, unsigned __int64 a2, int *a3)
{
  __int64 result; // rax

  result = 0;
  if ( (((unsigned __int8)a2 | (unsigned __int8)a1) & 7) == 0 )
  {
    if ( a1 + a2 - 1 <= 0x1F && a2 )
    {
      *a3 = ((((unsigned __int16)(1 << (a2 >> 1)) - 1) & 0x3210) << (a1 >> 1))
          | ((unsigned __int16)((1 << (a1 >> 1)) - 1) | (unsigned __int16)(-1 << ((a1 + a2) >> 1))) & 0x7654;
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
