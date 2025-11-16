// Function: sub_1F3D9C0
// Address: 0x1f3d9c0
//
__int64 __fastcall sub_1F3D9C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int8 v3; // dl
  unsigned __int8 v5; // dl

  v2 = 8 * sub_15A9520(a2, 0);
  if ( v2 == 32 )
    return 5;
  if ( v2 <= 0x20 )
  {
    v3 = 3;
    if ( v2 != 8 )
      return (unsigned __int8)(4 * (v2 == 16));
    return v3;
  }
  v3 = 6;
  if ( v2 == 64 )
    return v3;
  v5 = 0;
  if ( v2 == 128 )
    return 7;
  return v5;
}
