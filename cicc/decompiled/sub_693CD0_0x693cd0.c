// Function: sub_693CD0
// Address: 0x693cd0
//
char *__fastcall sub_693CD0(char *s)
{
  char *v1; // r12
  const char *v2; // r13
  size_t v3; // rbx
  char *v5; // rbx
  char v6; // al
  char v7; // dl
  __int64 v8; // [rsp+8h] [rbp-28h] BYREF

  v1 = s;
  v2 = off_4B7D3F8;
  v3 = strlen(off_4B7D3F8);
  if ( !strncmp(s, v2, v3) )
  {
    v5 = &s[v3];
    v8 = 0;
    if ( sscanf(v5, "%lu_", &v8) == 1 )
    {
      v6 = *v5;
      do
      {
        if ( !v6 )
          goto LABEL_9;
        v7 = v6;
        v6 = *++v5;
      }
      while ( v7 != 95 );
      if ( !v6 )
      {
LABEL_9:
        v1 = v5;
        goto LABEL_2;
      }
      v1 = &v5[v8 + 1];
    }
  }
LABEL_2:
  if ( strlen(v1) > 2 && *v1 == 95 && v1[1] == 90 )
    return (char *)sub_8257B0(v1);
  else
    return v1;
}
