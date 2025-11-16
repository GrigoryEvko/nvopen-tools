// Function: sub_2AA8390
// Address: 0x2aa8390
//
__int64 __fastcall sub_2AA8390(__int64 a1, __int64 a2, int *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 result; // rax
  int v9; // edx

  v4 = (a2 - a1) >> 5;
  v5 = (a2 - a1) >> 3;
  if ( v4 <= 0 )
  {
LABEL_9:
    switch ( v5 )
    {
      case 2LL:
        v9 = *a3;
        break;
      case 3LL:
        v9 = *a3;
        if ( *(_DWORD *)a1 == *a3 )
        {
          result = a1;
          if ( *(_BYTE *)(a1 + 4) == *((_BYTE *)a3 + 4) )
            return result;
        }
        a1 += 8;
        break;
      case 1LL:
        v9 = *a3;
LABEL_23:
        result = a2;
        if ( *(_DWORD *)a1 == v9 && *(_BYTE *)(a1 + 4) == *((_BYTE *)a3 + 4) )
          return a1;
        return result;
      default:
        return a2;
    }
    if ( *(_DWORD *)a1 == v9 )
    {
      result = a1;
      if ( *(_BYTE *)(a1 + 4) == *((_BYTE *)a3 + 4) )
        return result;
    }
    a1 += 8;
    goto LABEL_23;
  }
  v6 = *a3;
  v7 = a1 + 32 * v4;
  while ( 1 )
  {
    if ( *(_DWORD *)a1 == v6 && *(_BYTE *)(a1 + 4) == *((_BYTE *)a3 + 4) )
      return a1;
    if ( v6 == *(_DWORD *)(a1 + 8) && *(_BYTE *)(a1 + 12) == *((_BYTE *)a3 + 4) )
      return a1 + 8;
    if ( v6 == *(_DWORD *)(a1 + 16) && *(_BYTE *)(a1 + 20) == *((_BYTE *)a3 + 4) )
      return a1 + 16;
    if ( v6 == *(_DWORD *)(a1 + 24) && *(_BYTE *)(a1 + 28) == *((_BYTE *)a3 + 4) )
      return a1 + 24;
    a1 += 32;
    if ( a1 == v7 )
    {
      v5 = (a2 - a1) >> 3;
      goto LABEL_9;
    }
  }
}
