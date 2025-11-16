// Function: sub_7ABDF0
// Address: 0x7abdf0
//
__int64 **__fastcall sub_7ABDF0(char *s)
{
  const char *v1; // r12
  char v2; // al
  char v3; // al
  char *v4; // r13
  char *i; // rdx
  __int64 **v7; // [rsp+8h] [rbp-28h] BYREF

  v1 = s;
  v2 = *s;
  v7 = 0;
  if ( v2 == 58 )
  {
    v2 = s[1];
    v1 = s + 1;
  }
  if ( !v2 )
    return 0;
  do
  {
    if ( v2 == 32 )
    {
      do
        v3 = *++v1;
      while ( v3 == 32 );
      if ( !v3 )
        break;
    }
    v4 = strchr(v1, 58);
    if ( !v4 )
      v4 = (char *)&v1[strlen(v1)];
    for ( i = v4 - 1; *i == 32; --i )
      ;
    sub_7ABD00(&v7, v1, (_DWORD)i - (_DWORD)v1 + 1);
    if ( !*v4 )
      break;
    v2 = v4[1];
    v1 = v4 + 1;
  }
  while ( v2 );
  return v7;
}
