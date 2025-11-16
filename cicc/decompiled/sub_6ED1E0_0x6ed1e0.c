// Function: sub_6ED1E0
// Address: 0x6ed1e0
//
__int64 __fastcall sub_6ED1E0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  _BOOL4 v4; // eax

  v1 = *(_BYTE *)(a1 + 17);
  if ( v1 == 1 )
  {
    v4 = sub_6ED0A0(a1);
    v2 = 1;
    if ( !v4 )
      return v2;
    v1 = *(_BYTE *)(a1 + 17);
  }
  v2 = 0;
  if ( v1 == 3 )
    return (unsigned __int8)(*(_BYTE *)(a1 + 16) - 3) > 1u;
  return v2;
}
