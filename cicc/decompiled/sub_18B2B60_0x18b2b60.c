// Function: sub_18B2B60
// Address: 0x18b2b60
//
__int64 __fastcall sub_18B2B60(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r13d

  v2 = 0;
  if ( (unsigned __int8)sub_1636800(a1, a2) )
    return v2;
  v2 = sub_15ACB40(a2);
  if ( *(_BYTE *)(a1 + 153) )
    return v2;
  else
    return (unsigned int)sub_18B24F0(a2, 0) | v2;
}
