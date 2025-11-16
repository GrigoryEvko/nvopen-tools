// Function: sub_5CAB70
// Address: 0x5cab70
//
int __fastcall sub_5CAB70(char *a1, __int64 a2)
{
  int result; // eax
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v3 = a1;
  if ( *a1 != 99 )
    return 0;
  v3 = a1 + 1;
  if ( a1[1] == 43 )
  {
    v3 = a1 + 2;
    if ( unk_4F077C4 != 2 )
      return 0;
  }
  else if ( unk_4F077C4 == 2 )
  {
    return 0;
  }
  result = sub_5CA8C0(&v3, *(_BYTE *)(a2 + 9), *(const char **)(a2 + 24));
  if ( !result )
    return 0;
  if ( *v3 == 40 )
    return sub_5C9690(unk_4F07778, v3);
  return result;
}
