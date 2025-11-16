// Function: sub_28E5B80
// Address: 0x28e5b80
//
char __fastcall sub_28E5B80(__int64 a1, __int64 *a2)
{
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rsi

  result = sub_F57670(a1, a2);
  if ( result )
    return 0;
  v3 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)a1 != 85 )
  {
    if ( !v3 )
      return 1;
    if ( *(_BYTE *)v3 )
      return 1;
    v4 = *(_QWORD *)(v3 + 24);
    if ( v4 != *(_QWORD *)(a1 + 80) )
      return 1;
    goto LABEL_15;
  }
  if ( *(_BYTE *)v3 == 25 )
    return result;
  if ( *(_BYTE *)v3 )
    return 1;
  v4 = *(_QWORD *)(v3 + 24);
  if ( v4 == *(_QWORD *)(a1 + 80) )
  {
LABEL_15:
    if ( *(_DWORD *)(v3 + 36) == 151 )
      return result;
    if ( *(_BYTE *)a1 == 85 )
    {
      if ( (*(_BYTE *)(v3 + 33) & 0x20) != 0 && *(_DWORD *)(v3 + 36) == 149 )
        return result;
      goto LABEL_6;
    }
    return 1;
  }
LABEL_6:
  result = 1;
  if ( v4 == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
    return *(_DWORD *)(v3 + 36) != 150;
  return result;
}
