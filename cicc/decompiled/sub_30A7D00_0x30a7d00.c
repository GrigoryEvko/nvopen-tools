// Function: sub_30A7D00
// Address: 0x30a7d00
//
unsigned __int64 __fastcall sub_30A7D00(_QWORD *a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx

  if ( *(_BYTE *)*(a1 - 4) == 25 )
    return 0;
  if ( !sub_B491E0((__int64)a1) )
  {
    v2 = *(a1 - 4);
    if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != a1[10] || (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      return 0;
  }
  if ( *(_QWORD **)(a1[5] + 56LL) == a1 + 3 )
    return 0;
  v3 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v3 )
    return 0;
  while ( 1 )
  {
    result = v3 - 24;
    if ( *(_BYTE *)result == 85 )
    {
      v4 = *(_QWORD *)(result - 32);
      if ( v4 )
      {
        if ( !*(_BYTE *)v4
          && *(_QWORD *)(v4 + 24) == *(_QWORD *)(result + 80)
          && (*(_BYTE *)(v4 + 33) & 0x20) != 0
          && *(_DWORD *)(v4 + 36) == 196 )
        {
          break;
        }
      }
    }
    if ( *(_QWORD *)(*(_QWORD *)(result + 40) + 56LL) != result + 24 )
    {
      v3 = *(_QWORD *)(result + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 )
        continue;
    }
    return 0;
  }
  return result;
}
