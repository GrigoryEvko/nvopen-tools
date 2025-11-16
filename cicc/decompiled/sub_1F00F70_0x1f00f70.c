// Function: sub_1F00F70
// Address: 0x1f00f70
//
__int64 __fastcall sub_1F00F70(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // rcx

  result = a1;
  v5 = a2 - result;
  v6 = (a2 - result) >> 6;
  v7 = v5 >> 4;
  if ( v6 <= 0 )
  {
LABEL_21:
    switch ( v7 )
    {
      case 2LL:
        v10 = *a3;
        break;
      case 3LL:
        v10 = *a3;
        if ( *(_QWORD *)result == *a3
          && *(_DWORD *)(result + 8) == *((_DWORD *)a3 + 2)
          && *(_DWORD *)(result + 12) == *((_DWORD *)a3 + 3) )
        {
          return result;
        }
        result += 16;
        break;
      case 1LL:
        v10 = *a3;
LABEL_26:
        if ( v10 == *(_QWORD *)result && *(_DWORD *)(result + 8) == *((_DWORD *)a3 + 2) )
        {
          if ( *(_DWORD *)(result + 12) != *((_DWORD *)a3 + 3) )
            return a2;
          return result;
        }
        return a2;
      default:
        return a2;
    }
    if ( *(_QWORD *)result == v10
      && *((_DWORD *)a3 + 2) == *(_DWORD *)(result + 8)
      && *(_DWORD *)(result + 12) == *((_DWORD *)a3 + 3) )
    {
      return result;
    }
    result += 16;
    goto LABEL_26;
  }
  v8 = *a3;
  v9 = result + (v6 << 6);
  while ( *(_QWORD *)result != v8
       || *(_DWORD *)(result + 8) != *((_DWORD *)a3 + 2)
       || *(_DWORD *)(result + 12) != *((_DWORD *)a3 + 3) )
  {
    if ( v8 == *(_QWORD *)(result + 16)
      && *((_DWORD *)a3 + 2) == *(_DWORD *)(result + 24)
      && *(_DWORD *)(result + 28) == *((_DWORD *)a3 + 3) )
    {
      result += 16;
      return result;
    }
    if ( v8 == *(_QWORD *)(result + 32)
      && *(_DWORD *)(result + 40) == *((_DWORD *)a3 + 2)
      && *(_DWORD *)(result + 44) == *((_DWORD *)a3 + 3) )
    {
      result += 32;
      return result;
    }
    if ( v8 == *(_QWORD *)(result + 48)
      && *(_DWORD *)(result + 56) == *((_DWORD *)a3 + 2)
      && *(_DWORD *)(result + 60) == *((_DWORD *)a3 + 3) )
    {
      result += 48;
      return result;
    }
    result += 64;
    if ( result == v9 )
    {
      v7 = (a2 - result) >> 4;
      goto LABEL_21;
    }
  }
  return result;
}
