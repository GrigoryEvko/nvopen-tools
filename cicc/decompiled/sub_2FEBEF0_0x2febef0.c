// Function: sub_2FEBEF0
// Address: 0x2febef0
//
__int64 __fastcall sub_2FEBEF0(__int64 a1, int a2)
{
  __int64 v2; // rsi

  v2 = (unsigned int)(a2 - 1);
  if ( (unsigned int)v2 > 0x42 )
    BUG();
  return (unsigned int)word_44561C0[v2];
}
