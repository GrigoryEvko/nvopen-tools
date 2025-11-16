// Function: sub_878690
// Address: 0x878690
//
__int64 __fastcall sub_878690(__int64 *a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rax
  const char *v3; // rbx
  const char *v4; // rsi
  int v5; // eax

  v1 = 0;
  v2 = *a1;
  if ( (*((_BYTE *)a1 + 18) & 2) != 0 )
  {
    if ( !v2 )
      return v1;
    v3 = *(const char **)(v2 + 8);
    if ( !v3 )
      return v1;
    v4 = *(const char **)(**(_QWORD **)a1[4] + 8LL);
    if ( v4 )
    {
      v5 = strcmp(v3, v4);
      v1 = 1;
      if ( !v5 )
        return v1;
    }
  }
  else
  {
    if ( !v2 )
      return v1;
    v3 = *(const char **)(v2 + 8);
    if ( !v3 )
      return v1;
  }
  return *v3 == 126;
}
