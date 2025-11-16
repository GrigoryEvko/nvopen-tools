// Function: sub_7B0680
// Address: 0x7b0680
//
_BOOL8 __fastcall sub_7B0680(const char **a1, const char **a2)
{
  const char *v4; // rdi
  const char *v5; // rsi
  int v6; // r8d
  _BOOL8 result; // rax
  const char *v8; // rdi
  const char *v9; // rsi
  int v10; // r8d
  const char *v11; // rdi
  const char *v12; // rsi

  v4 = *a1;
  v5 = *a2;
  if ( v4 == v5 || (v6 = strcmp(v4, v5), result = 0, !v6) )
  {
    v8 = a1[1];
    v9 = a2[1];
    if ( v8 == v9 || (v10 = strcmp(v8, v9), result = 0, !v10) )
    {
      v11 = a1[2];
      v12 = a2[2];
      result = 1;
      if ( v11 != v12 )
        return strcmp(v11, v12) == 0;
    }
  }
  return result;
}
