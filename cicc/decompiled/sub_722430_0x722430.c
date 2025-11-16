// Function: sub_722430
// Address: 0x722430
//
char *__fastcall sub_722430(char *s, int a2)
{
  __int64 *v2; // r15
  size_t v4; // r12
  __int64 v5; // rbx
  char *v6; // r14
  char *v8; // rax
  size_t v9; // rdi
  __int64 v10; // rax

  v2 = &qword_4F07920;
  if ( !a2 )
    v2 = (__int64 *)&unk_4F07670;
  if ( (*s != 45 || s[1]) && (v8 = sub_722110(s)) != 0 )
    v4 = v8 - s + 1;
  else
    v4 = 0;
  v5 = *v2;
  if ( !*v2 )
  {
LABEL_14:
    v9 = v4 + 1;
    if ( a2 )
    {
      v6 = (char *)sub_822B10(v9);
      if ( !v4 )
        goto LABEL_16;
    }
    else
    {
      v6 = (char *)sub_7247C0(v9);
      if ( !v4 )
      {
LABEL_16:
        v10 = qword_4F07940;
        v6[v4] = 0;
        if ( v10 )
          qword_4F07940 = *(_QWORD *)(v10 + 16);
        else
          v10 = sub_822B10(24);
        *(_QWORD *)(v10 + 16) = 0;
        *(_DWORD *)(v10 + 8) = 0;
        *(_QWORD *)v10 = v6;
        *(_QWORD *)(v10 + 16) = *v2;
        *v2 = v10;
        return v6;
      }
    }
    memcpy(v6, s, v4);
    goto LABEL_16;
  }
  while ( 1 )
  {
    v6 = *(char **)v5;
    if ( strlen(*(const char **)v5) == v4 && !strncmp(v6, s, v4) )
      return v6;
    v5 = *(_QWORD *)(v5 + 16);
    if ( !v5 )
      goto LABEL_14;
  }
}
