// Function: sub_22A71D0
// Address: 0x22a71d0
//
__int64 __fastcall sub_22A71D0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r13d
  unsigned int v5; // edx
  unsigned int v6; // eax
  unsigned int v7; // ecx
  unsigned int v8; // ecx
  unsigned int v9; // eax
  unsigned int v10; // eax

  v2 = *(_BYTE *)(a2 + 10);
  if ( *(_BYTE *)(a1 + 10) >= v2 )
  {
    v3 = 0;
    if ( *(_BYTE *)(a1 + 10) != v2 )
      return v3;
    v5 = *(_DWORD *)(a1 + 16);
    v6 = *(_DWORD *)(a2 + 16);
    if ( v5 >= v6 )
    {
      if ( v5 != v6
        || (v7 = *(_DWORD *)(a2 + 20), *(_DWORD *)(a1 + 20) >= v7)
        && (*(_DWORD *)(a1 + 20) != v7
         || (v8 = *(_DWORD *)(a2 + 24), *(_DWORD *)(a1 + 24) >= v8)
         && (*(_DWORD *)(a1 + 24) != v8 || *(_DWORD *)(a1 + 28) >= *(_DWORD *)(a2 + 28))) )
      {
        if ( v5 > v6 )
          return 0;
        v9 = *(_DWORD *)(a1 + 20);
        if ( *(_DWORD *)(a2 + 20) < v9 )
          return 0;
        if ( *(_DWORD *)(a2 + 20) == v9 )
        {
          v10 = *(_DWORD *)(a1 + 24);
          if ( *(_DWORD *)(a2 + 24) < v10 || *(_DWORD *)(a2 + 24) == v10 && *(_DWORD *)(a2 + 28) < *(_DWORD *)(a1 + 28) )
            return 0;
        }
        v3 = sub_22A6F20(a1, (__int64 *)a2);
        if ( !(_BYTE)v3 )
        {
          sub_22A6F20(a2, (__int64 *)a1);
          return v3;
        }
      }
    }
  }
  return 1;
}
