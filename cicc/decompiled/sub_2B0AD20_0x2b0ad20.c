// Function: sub_2B0AD20
// Address: 0x2b0ad20
//
__int64 __fastcall sub_2B0AD20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // rax
  __int64 v6; // rdx

  v2 = (a2 - a1) >> 5;
  v3 = (a2 - a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_13:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        v4 = a2;
        if ( v3 != 1 )
          return v4;
LABEL_16:
        v4 = a1;
        if ( *(_BYTE *)(a1 + 4) )
        {
          if ( *(_DWORD *)a1 == 7 )
            return a2;
        }
        return v4;
      }
      v4 = a1;
      if ( !*(_BYTE *)(a1 + 4) || *(_DWORD *)a1 != 7 )
        return v4;
      a1 += 8;
    }
    v4 = a1;
    if ( !*(_BYTE *)(a1 + 4) || *(_DWORD *)a1 != 7 )
      return v4;
    a1 += 8;
    goto LABEL_16;
  }
  v4 = a1;
  v5 = a1 + 32 * v2;
  while ( *(_BYTE *)(v4 + 4) && *(_DWORD *)v4 == 7 )
  {
    v6 = v4 + 8;
    if ( !*(_BYTE *)(v4 + 12) )
      return v6;
    if ( *(_DWORD *)(v4 + 8) != 7 )
      return v6;
    v6 = v4 + 16;
    if ( !*(_BYTE *)(v4 + 20) )
      return v6;
    if ( *(_DWORD *)(v4 + 16) != 7 )
      return v6;
    v6 = v4 + 24;
    if ( !*(_BYTE *)(v4 + 28) || *(_DWORD *)(v4 + 24) != 7 )
      return v6;
    v4 += 32;
    if ( v4 == v5 )
    {
      a1 = v4;
      v3 = (a2 - v4) >> 3;
      goto LABEL_13;
    }
  }
  return v4;
}
