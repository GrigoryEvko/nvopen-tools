// Function: sub_143A9A0
// Address: 0x143a9a0
//
__int64 __fastcall sub_143A9A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rsi

  v3 = *(__int64 **)(a2 + 8);
  if ( v3 )
    sub_1368C40(a1, v3, a3);
  else
    *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
