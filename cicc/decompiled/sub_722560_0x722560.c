// Function: sub_722560
// Address: 0x722560
//
char *__fastcall sub_722560(char *s, const char *src)
{
  char *v2; // r12
  char v3; // al
  char *v4; // rbx
  char *v5; // r14
  __int64 v6; // rdx
  size_t v7; // r14
  size_t v8; // r14
  size_t v9; // rbx
  char *v10; // r15
  char *v12; // rax

  v2 = s;
  if ( *s == 45 && !s[1] )
  {
    v3 = *s;
    v4 = s;
    v5 = 0;
    if ( *s )
      goto LABEL_8;
    goto LABEL_16;
  }
  v12 = sub_722110(s);
  if ( v12 )
    v2 = v12 + 1;
  v5 = 0;
  v3 = *v2;
  v4 = v2;
  if ( !*v2 )
  {
LABEL_16:
    v7 = (size_t)&v2[strlen(v2) - 1];
    goto LABEL_12;
  }
  do
  {
LABEL_8:
    while ( v3 == 46 )
    {
      v5 = v4++;
      v3 = *v4;
      if ( !*v4 )
        goto LABEL_10;
    }
    v6 = 1;
    if ( v3 < 0 )
      v6 = (int)sub_721AB0(v4, 0, 0);
    v4 += v6;
    v3 = *v4;
  }
  while ( *v4 );
LABEL_10:
  if ( !v5 )
    goto LABEL_16;
  v7 = (size_t)(v5 - 1);
LABEL_12:
  v8 = v7 - (_QWORD)v2 + 1;
  v9 = strlen(src);
  v10 = (char *)sub_822B10(v9 + v8 + 1);
  memcpy(v10, v2, v8);
  memcpy(&v10[v8], src, v9);
  v10[v9 + v8] = 0;
  return v10;
}
