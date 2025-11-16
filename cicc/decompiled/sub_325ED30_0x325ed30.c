// Function: sub_325ED30
// Address: 0x325ed30
//
__int64 __fastcall sub_325ED30(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx

  v4 = (a2 - a1) >> 6;
  v5 = (a2 - a1) >> 4;
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
        if ( *(_QWORD *)a1 == *a3 )
        {
          result = a1;
          if ( *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
            return result;
        }
        a1 += 16;
        break;
      case 1LL:
        v9 = *a3;
LABEL_23:
        result = a2;
        if ( *(_QWORD *)a1 == v9 && *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
          return a1;
        return result;
      default:
        return a2;
    }
    if ( *(_QWORD *)a1 == v9 )
    {
      result = a1;
      if ( *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
        return result;
    }
    a1 += 16;
    goto LABEL_23;
  }
  v6 = *a3;
  v7 = a1 + (v4 << 6);
  while ( 1 )
  {
    if ( *(_QWORD *)a1 == v6 && *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
      return a1;
    if ( v6 == *(_QWORD *)(a1 + 16) && *(_DWORD *)(a1 + 24) == *((_DWORD *)a3 + 2) )
      return a1 + 16;
    if ( v6 == *(_QWORD *)(a1 + 32) && *(_DWORD *)(a1 + 40) == *((_DWORD *)a3 + 2) )
      return a1 + 32;
    if ( v6 == *(_QWORD *)(a1 + 48) && *(_DWORD *)(a1 + 56) == *((_DWORD *)a3 + 2) )
      return a1 + 48;
    a1 += 64;
    if ( a1 == v7 )
    {
      v5 = (a2 - a1) >> 4;
      goto LABEL_9;
    }
  }
}
