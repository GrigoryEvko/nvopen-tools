// Function: sub_CB7550
// Address: 0xcb7550
//
__int64 __fastcall sub_CB7550(const char **a1, int a2)
{
  char *v2; // rcx
  const char *v3; // r13
  char *v4; // rax
  __int64 v5; // rdx
  const char *v6; // r14
  char **v7; // r15
  size_t v8; // rbx

  v2 = (char *)a1[1];
  v3 = *a1;
  v4 = (char *)*a1;
  v5 = v2 - *a1;
  if ( v5 <= 0 )
  {
LABEL_15:
    if ( !*((_DWORD *)a1 + 4) )
      *((_DWORD *)a1 + 4) = 7;
LABEL_14:
    *a1 = (const char *)&unk_4F85140;
    a1[1] = (const char *)&unk_4F85140;
    return 0;
  }
  while ( v5 == 1 || *v4 != a2 || v4[1] != 93 )
  {
    ++v4;
    v5 = v2 - v4;
    *a1 = v4;
    if ( v2 - v4 <= 0 )
      goto LABEL_15;
  }
  v6 = off_4C5C780;
  v7 = &off_4C5C780;
  v8 = v4 - v3;
  if ( off_4C5C780 )
  {
    while ( strncmp(v6, v3, v8) || strlen(v6) != v8 )
    {
      v6 = v7[2];
      v7 += 2;
      if ( !v6 )
        goto LABEL_11;
    }
    return *((unsigned __int8 *)v7 + 8);
  }
  else
  {
LABEL_11:
    if ( v8 != 1 )
    {
      if ( !*((_DWORD *)a1 + 4) )
        *((_DWORD *)a1 + 4) = 3;
      goto LABEL_14;
    }
    return *(unsigned __int8 *)v3;
  }
}
