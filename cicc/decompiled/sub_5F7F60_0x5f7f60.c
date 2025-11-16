// Function: sub_5F7F60
// Address: 0x5f7f60
//
__int64 __fastcall sub_5F7F60(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v5; // rax

  *a2 = 0;
  v2 = sub_7D3790(2, a1);
  if ( v2 )
  {
    v3 = v2;
    if ( (*(_BYTE *)(v2 + 82) & 4) != 0 )
    {
      *a2 = 1;
      return v3;
    }
  }
  else
  {
    v3 = sub_7D3810(2);
  }
  if ( *a2 )
    return v3;
  v5 = sub_87C270(v3, a1, a2);
  if ( !*a2 )
    return v5;
  return v3;
}
