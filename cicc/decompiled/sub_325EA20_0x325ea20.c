// Function: sub_325EA20
// Address: 0x325ea20
//
__int64 __fastcall sub_325EA20(__int64 a1, __int64 a2, __int64 *a3)
{
  signed __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // rcx

  v3 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
  v4 = v3 >> 2;
  if ( v3 >> 2 <= 0 )
  {
LABEL_9:
    switch ( v3 )
    {
      case 2LL:
        v8 = *a3;
        break;
      case 3LL:
        v8 = *a3;
        if ( *(_QWORD *)a1 == *a3 )
        {
          result = a1;
          if ( *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
            return result;
        }
        a1 += 40;
        break;
      case 1LL:
        v8 = *a3;
LABEL_23:
        result = a2;
        if ( *(_QWORD *)a1 == v8 && *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
          return a1;
        return result;
      default:
        return a2;
    }
    if ( *(_QWORD *)a1 == v8 )
    {
      result = a1;
      if ( *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
        return result;
    }
    a1 += 40;
    goto LABEL_23;
  }
  v5 = *a3;
  v6 = a1 + 160 * v4;
  while ( 1 )
  {
    if ( *(_QWORD *)a1 == v5 && *(_DWORD *)(a1 + 8) == *((_DWORD *)a3 + 2) )
      return a1;
    if ( v5 == *(_QWORD *)(a1 + 40) && *(_DWORD *)(a1 + 48) == *((_DWORD *)a3 + 2) )
      return a1 + 40;
    if ( v5 == *(_QWORD *)(a1 + 80) && *(_DWORD *)(a1 + 88) == *((_DWORD *)a3 + 2) )
      return a1 + 80;
    if ( v5 == *(_QWORD *)(a1 + 120) && *(_DWORD *)(a1 + 128) == *((_DWORD *)a3 + 2) )
      return a1 + 120;
    a1 += 160;
    if ( a1 == v6 )
    {
      v3 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
      goto LABEL_9;
    }
  }
}
