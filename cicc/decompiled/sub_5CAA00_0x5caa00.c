// Function: sub_5CAA00
// Address: 0x5caa00
//
int __fastcall sub_5CAA00(char *a1, __int64 a2)
{
  char v2; // dl
  int result; // eax
  char v4; // dl
  const char *v5; // rdx
  char v6; // si
  char v7; // dl
  const char *v8; // rdx
  char v9; // si
  char *v10; // [rsp+8h] [rbp-8h] BYREF

  v10 = a1;
  v2 = *a1;
  if ( *a1 == 115 )
    return 0;
  if ( v2 != 103 )
  {
    if ( v2 != 108 || !unk_4F077B4 )
      return 0;
    v7 = a1[1];
    if ( v7 != 120 )
    {
      if ( v7 == 99 )
      {
        if ( unk_4F077C4 == 2 )
          return 0;
      }
      else if ( v7 != 43 || unk_4F077C4 != 2 )
      {
        return 0;
      }
    }
    v8 = *(const char **)(a2 + 24);
    v9 = *(_BYTE *)(a2 + 9);
    v10 = a1 + 2;
    result = sub_5CA8C0(&v10, v9, v8);
    if ( result )
    {
      if ( *v10 == 40 )
        return sub_5C9690(unk_4F077A0, v10);
      return result;
    }
    return 0;
  }
  v4 = a1[1];
  if ( v4 == 120 )
  {
    if ( !unk_4F077B8 )
      return 0;
  }
  else if ( v4 == 99 )
  {
    if ( !unk_4F077C0 )
      return 0;
  }
  else if ( v4 != 43 || !unk_4F077BC )
  {
    return 0;
  }
  v5 = *(const char **)(a2 + 24);
  v6 = *(_BYTE *)(a2 + 9);
  v10 = a1 + 2;
  result = sub_5CA8C0(&v10, v6, v5);
  if ( !result )
    return 0;
  if ( *v10 == 40 )
    return sub_5C9690(unk_4F077A8, v10);
  return result;
}
