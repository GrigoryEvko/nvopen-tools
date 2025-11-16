// Function: sub_CBADF0
// Address: 0xcbadf0
//
size_t __fastcall sub_CBADF0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rcx
  unsigned int v7; // r12d
  int *v8; // rax
  char *v9; // r15
  size_t v10; // r12
  const char *v12; // r15
  int *v13; // rbx
  char s[112]; // [rsp+0h] [rbp-70h] BYREF

  v6 = a1;
  BYTE1(v6) = BYTE1(a1) & 0xFE;
  v7 = dword_4C5CEC0;
  if ( a1 == 255 )
  {
    if ( !dword_4C5CEC0 )
    {
LABEL_20:
      v10 = 2;
      v9 = "0";
LABEL_11:
      if ( !a4 )
        return v10;
      goto LABEL_8;
    }
    v12 = *(const char **)(a2 + 16);
    v13 = &dword_4C5CEC0;
    while ( strcmp(*((const char **)v13 + 1), v12) )
    {
      v7 = v13[6];
      v13 += 6;
      if ( !v7 )
        goto LABEL_20;
    }
    v9 = s;
    snprintf(s, 0x32u, "%d", v7);
LABEL_19:
    v10 = strlen(s) + 1;
    goto LABEL_11;
  }
  v8 = &dword_4C5CEC0;
  if ( !dword_4C5CEC0 )
  {
LABEL_9:
    if ( (a1 & 0x100) == 0 )
    {
LABEL_10:
      v9 = (char *)*((_QWORD *)v8 + 2);
      v10 = strlen(v9) + 1;
      goto LABEL_11;
    }
    v9 = s;
    snprintf(s, 0x32u, "REG_0x%x", v6);
    goto LABEL_19;
  }
  while ( (_DWORD)v6 != v7 )
  {
    v7 = v8[6];
    v8 += 6;
    if ( !v7 )
      goto LABEL_9;
  }
  if ( (a1 & 0x100) == 0 )
    goto LABEL_10;
  v9 = s;
  sub_CBF040(s, *((_QWORD *)v8 + 1), 50);
  v10 = strlen(s) + 1;
  if ( a4 )
LABEL_8:
    sub_CBF040(a3, v9, a4);
  return v10;
}
