// Function: sub_722040
// Address: 0x722040
//
char *__fastcall sub_722040(int a1, const char *a2, const char *a3)
{
  const char *v4; // rdi
  struct dirent *v5; // rax
  char *d_name; // r12
  char v7; // al
  char *v8; // rbx
  const char *v9; // r14
  __int64 v10; // rdx

  if ( !a1 )
    goto LABEL_14;
  v4 = a2;
  if ( !a2 )
    v4 = ".";
  dirp = opendir(v4);
  v5 = readdir(dirp);
  if ( v5 )
  {
    while ( 1 )
    {
      d_name = v5->d_name;
      v7 = v5->d_name[0];
      if ( v7 )
      {
        v8 = d_name;
        v9 = 0;
        do
        {
          while ( v7 == 46 )
          {
            v9 = v8++;
            v7 = *v8;
            if ( !*v8 )
              goto LABEL_12;
          }
          v10 = 1;
          if ( v7 < 0 )
            v10 = (int)sub_721AB0(v8, 0, 0);
          v8 += v10;
          v7 = *v8;
        }
        while ( *v8 );
LABEL_12:
        if ( v9 && !strcmp(v9, a3) )
          break;
      }
LABEL_14:
      v5 = readdir(dirp);
      if ( !v5 )
        goto LABEL_15;
    }
  }
  else
  {
LABEL_15:
    d_name = 0;
    closedir(dirp);
  }
  return d_name;
}
