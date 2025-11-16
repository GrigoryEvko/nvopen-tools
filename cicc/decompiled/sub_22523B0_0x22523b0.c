// Function: sub_22523B0
// Address: 0x22523b0
//
__int64 __fastcall sub_22523B0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  const char *v10; // rbp
  const char *v11; // rsi
  const char *v13; // rsi

  v10 = *(const char **)(a1 + 8);
  if ( a5 != a7 )
  {
    v11 = *(const char **)(a4 + 8);
    if ( v11 == v10 )
    {
LABEL_5:
      *(_QWORD *)a8 = a5;
      *(_DWORD *)(a8 + 8) = a3;
      *(_DWORD *)(a8 + 16) = 1;
      return 0;
    }
    if ( *v10 == 42 )
      return 0;
LABEL_4:
    if ( !strcmp(v10, v11) )
      goto LABEL_5;
    return 0;
  }
  v13 = *(const char **)(a6 + 8);
  if ( v13 != v10 )
  {
    if ( *v10 == 42 )
    {
      if ( *(const char **)(a4 + 8) != v10 )
        return 0;
      goto LABEL_5;
    }
    if ( strcmp(*(const char **)(a1 + 8), v13) )
    {
      v11 = *(const char **)(a4 + 8);
      if ( v10 == v11 )
        goto LABEL_5;
      goto LABEL_4;
    }
  }
  *(_DWORD *)(a8 + 12) = a3;
  return 0;
}
