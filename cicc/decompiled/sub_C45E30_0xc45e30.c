// Function: sub_C45E30
// Address: 0xc45e30
//
__int64 __fastcall sub_C45E30(__int64 a1, int a2)
{
  __int64 v2; // rsi
  unsigned __int64 *v3; // rax
  unsigned __int64 v4; // rdx

  v2 = (unsigned int)(a2 - 1);
  v3 = (unsigned __int64 *)(a1 + 8 * v2);
  while ( 1 )
  {
    if ( *v3 )
    {
      _BitScanReverse64(&v4, *v3);
      return ((_DWORD)v2 << 6) - ((unsigned int)v4 ^ 0x3F) + 63;
    }
    --v3;
    if ( !(_DWORD)v2 )
      break;
    LODWORD(v2) = v2 - 1;
  }
  return 0xFFFFFFFFLL;
}
