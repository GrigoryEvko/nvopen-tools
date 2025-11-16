// Function: sub_851F00
// Address: 0x851f00
//
__int64 __fastcall sub_851F00(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v5; // eax
  _BOOL4 v6; // edx
  char *v7; // rdi
  char *v8; // rsi

  v2 = *(_DWORD *)(a1 + 8);
  v3 = 0;
  if ( v2 != *(_DWORD *)(a2 + 8) )
    return 0;
  if ( v2 == 1 )
  {
    if ( *(_DWORD *)(a1 + 12) != *(_DWORD *)(a2 + 12) || *(_BYTE *)(a1 + 16) != *(_BYTE *)(a2 + 16) )
      return v3;
    v6 = 0;
  }
  else
  {
    if ( v2 != 2 )
      sub_721090();
    v5 = *(_DWORD *)(a1 + 12);
    v6 = v5 == 8;
    if ( v5 != *(_DWORD *)(a2 + 12) )
      return v3;
  }
  v7 = *(char **)(a1 + 24);
  v8 = *(char **)(a2 + 24);
  if ( !v7 || !*v7 )
  {
    v3 = 1;
    if ( !v8 )
      return v3;
    if ( !*v8 )
      return v3;
    v3 = 0;
    if ( !v7 )
      return v3;
    goto LABEL_9;
  }
  v3 = 0;
  if ( v8 )
  {
LABEL_9:
    if ( v6 )
    {
      v3 = 0;
      if ( *v7 == *v8 )
        return !sub_722E50(v7, v8, 1, 1, 0);
    }
    else
    {
      return strcmp(v7, v8) == 0;
    }
  }
  return v3;
}
