// Function: sub_138ECE0
// Address: 0x138ece0
//
__int64 __fastcall sub_138ECE0(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // r8d

  v1 = *(_BYTE *)(a1 + 16);
  v2 = 0;
  if ( v1 > 0x10u )
    return 0;
  if ( v1 == 5 || v1 <= 3u )
    return 0;
  LOBYTE(v2) = (unsigned __int8)(v1 - 6) > 2u;
  return v2;
}
