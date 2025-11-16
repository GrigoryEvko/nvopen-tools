// Function: sub_3211420
// Address: 0x3211420
//
__int64 __fastcall sub_3211420(__int64 a1, __int64 a2)
{
  signed __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax

  v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
  if ( v2 >> 2 > 0 )
  {
    v3 = a1 + 160 * (v2 >> 2);
    do
    {
      if ( !*(_BYTE *)a1 && *(_DWORD *)(a1 + 8) )
        return a1;
      if ( !*(_BYTE *)(a1 + 40) && *(_DWORD *)(a1 + 48) )
        return a1 + 40;
      if ( !*(_BYTE *)(a1 + 80) && *(_DWORD *)(a1 + 88) )
        return a1 + 80;
      if ( !*(_BYTE *)(a1 + 120) && *(_DWORD *)(a1 + 128) )
        return a1 + 120;
      a1 += 160;
    }
    while ( a1 != v3 );
    v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
  }
  if ( v2 == 2 )
  {
LABEL_25:
    if ( !*(_BYTE *)a1 )
    {
      result = a1;
      if ( *(_DWORD *)(a1 + 8) )
        return result;
    }
    a1 += 40;
    goto LABEL_28;
  }
  if ( v2 == 3 )
  {
    if ( !*(_BYTE *)a1 )
    {
      result = a1;
      if ( *(_DWORD *)(a1 + 8) )
        return result;
    }
    a1 += 40;
    goto LABEL_25;
  }
  if ( v2 != 1 )
    return a2;
LABEL_28:
  result = a2;
  if ( !*(_BYTE *)a1 )
  {
    if ( *(_DWORD *)(a1 + 8) )
      return a1;
  }
  return result;
}
