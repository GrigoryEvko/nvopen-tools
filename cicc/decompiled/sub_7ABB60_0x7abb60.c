// Function: sub_7ABB60
// Address: 0x7abb60
//
int __fastcall sub_7ABB60(char *s, char *a2)
{
  char *v3; // rax
  char *v4; // rbx
  __int64 *i; // rax
  char v6; // dl
  int *v7; // rax
  char *sa; // [rsp+8h] [rbp-38h]

  do
  {
    if ( unk_4F06440 )
    {
      if ( a2 )
      {
        v4 = s;
        if ( s < a2 )
        {
          do
          {
            if ( *v4 == 10 )
              break;
            ++v4;
          }
          while ( a2 != v4 );
LABEL_5:
          LODWORD(i) = fprintf(qword_4D04908, "%.*s", (_DWORD)v4 - (_DWORD)s, s);
          goto LABEL_6;
        }
      }
      else
      {
        sa = s;
        v3 = strchr(s, 10);
        s = sa;
        v4 = v3;
      }
    }
    else
    {
      v4 = a2;
    }
    if ( v4 )
      goto LABEL_5;
    if ( fputs(s, qword_4D04908) == -1 )
    {
      v7 = __errno_location();
      sub_6866A0(1514, *v7);
    }
    LODWORD(i) = putc(10, qword_4D04908);
LABEL_6:
    if ( a2 == v4 )
      break;
    for ( i = (__int64 *)unk_4F06440; ; i = (__int64 *)*i )
    {
      if ( (char *)i[2] == v4 )
      {
        v6 = *((_BYTE *)i + 50);
        if ( v6 != 10 )
          break;
      }
    }
    s = v4 + 2;
    if ( v6 )
    {
      LODWORD(i) = putc(v6, qword_4D04908);
      s = v4 + 1;
    }
  }
  while ( s != a2 );
  return (int)i;
}
