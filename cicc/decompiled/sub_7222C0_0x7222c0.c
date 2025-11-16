// Function: sub_7222C0
// Address: 0x7222c0
//
char *__fastcall sub_7222C0(char *s)
{
  char *v1; // r12
  char v2; // al
  char *v3; // rbx
  char *v4; // r13
  __int64 v5; // rdx
  char *v7; // rax

  v1 = s;
  if ( *s != 45 || s[1] )
  {
    v7 = sub_722110(s);
    if ( v7 )
      v1 = v7 + 1;
    v4 = 0;
    v2 = *v1;
    v3 = v1;
    if ( *v1 )
    {
      do
      {
LABEL_8:
        while ( v2 == 46 )
        {
          v4 = v3++;
          v2 = *v3;
          if ( !*v3 )
            goto LABEL_10;
        }
        v5 = 1;
        if ( v2 < 0 )
          v5 = (int)sub_721AB0(v3, 0, 0);
        v3 += v5;
        v2 = *v3;
      }
      while ( *v3 );
LABEL_10:
      if ( v4 )
        return v4;
    }
  }
  else
  {
    v2 = *s;
    v3 = s;
    v4 = 0;
    if ( *s )
      goto LABEL_8;
  }
  return &v1[strlen(v1)];
}
