// Function: sub_722110
// Address: 0x722110
//
char *__fastcall sub_722110(char *s)
{
  char v1; // al
  char *v2; // rbx
  char *v3; // r13
  __int64 v4; // rdx
  char v5; // r14
  char v7; // al
  char *v8; // rbx
  char *v9; // r15
  __int64 v10; // rdx

  v1 = *s;
  if ( !*s )
  {
    if ( !dword_4F07598 )
    {
      v5 = 1;
      v3 = 0;
      goto LABEL_9;
    }
    v9 = 0;
LABEL_21:
    v3 = v9;
    v5 = v9 == 0;
LABEL_9:
    if ( dword_4F07594 )
    {
      if ( v5 )
      {
        v3 = 0;
        if ( strlen(s) > 1 && s[1] == 58 )
          return s + 1;
      }
    }
    return v3;
  }
  v2 = s;
  v3 = 0;
  do
  {
    while ( v1 == 47 )
    {
      v3 = v2++;
      v1 = *v2;
      if ( !*v2 )
        goto LABEL_8;
    }
    v4 = 1;
    if ( v1 < 0 )
      v4 = (int)sub_721AB0(v2, 0, 0);
    v2 += v4;
    v1 = *v2;
  }
  while ( *v2 );
LABEL_8:
  v5 = v3 == 0;
  if ( !dword_4F07598 )
    goto LABEL_9;
  v7 = *s;
  if ( *s )
  {
    v8 = s;
    v9 = 0;
    do
    {
      while ( v7 == 92 )
      {
        v9 = v8++;
        v7 = *v8;
        if ( !*v8 )
          goto LABEL_19;
      }
      v10 = 1;
      if ( v7 < 0 )
        v10 = (int)sub_721AB0(v8, 0, 0);
      v8 += v10;
      v7 = *v8;
    }
    while ( *v8 );
LABEL_19:
    v5 |= v3 < v9;
  }
  else
  {
    v9 = 0;
  }
  if ( v5 )
    goto LABEL_21;
  return v3;
}
