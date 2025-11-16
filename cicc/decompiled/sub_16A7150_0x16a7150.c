// Function: sub_16A7150
// Address: 0x16a7150
//
__int64 __fastcall sub_16A7150(__int64 a1, int a2)
{
  __int64 v2; // rax
  unsigned __int64 *v3; // rdx
  unsigned __int64 v4; // rcx

  v2 = (unsigned int)(a2 - 1);
  v3 = (unsigned __int64 *)(a1 + 8 * v2);
  while ( 1 )
  {
    if ( *v3 )
    {
      _BitScanReverse64(&v4, *v3);
      return (unsigned int)(v4 + ((_DWORD)v2 << 6));
    }
    --v3;
    if ( !(_DWORD)v2 )
      break;
    LODWORD(v2) = v2 - 1;
  }
  return 0xFFFFFFFFLL;
}
