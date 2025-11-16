// Function: sub_E0DEF0
// Address: 0xe0def0
//
signed __int64 __fastcall sub_E0DEF0(char **a1, char a2)
{
  char *v3; // r8
  char *v4; // rsi
  int v5; // edx
  char *v6; // rax
  char *v8; // rax
  char *v9; // rcx

  v3 = a1[1];
  v4 = *a1;
  if ( !a2 )
  {
    v6 = *a1;
    goto LABEL_6;
  }
  if ( v4 == v3 )
    return 0;
  v5 = *v4;
  v6 = *a1;
  if ( *v4 == 110 )
  {
    v6 = v4 + 1;
    *a1 = v4 + 1;
LABEL_6:
    if ( v6 == v3 )
      return 0;
    v5 = *v6;
  }
  if ( (unsigned int)(v5 - 48) > 9 )
    return 0;
  v8 = v6 + 1;
  do
  {
    *a1 = v8;
    v9 = v8;
    if ( v3 == v8 )
      break;
    ++v8;
  }
  while ( (unsigned int)(*v9 - 48) <= 9 );
  return v9 - v4;
}
