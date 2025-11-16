// Function: sub_16E8EC0
// Address: 0x16e8ec0
//
__int64 __fastcall sub_16E8EC0(const char **a1, int a2)
{
  const char *v2; // r13
  unsigned __int64 v3; // rcx
  char *v4; // rdx
  const char *v5; // rax
  const char *v7; // r14
  size_t v8; // rbx
  char **v9; // r15

  v2 = *a1;
  v3 = (unsigned __int64)a1[1];
  v4 = (char *)*a1;
  if ( (unsigned __int64)*a1 >= v3 )
    goto LABEL_8;
  while ( 1 )
  {
    v5 = v4 + 1;
    if ( (unsigned __int64)(v4 + 1) < v3 )
      break;
    *a1 = v5;
    if ( v4 + 1 == (char *)v3 )
      goto LABEL_8;
LABEL_5:
    ++v4;
  }
  if ( *v4 != a2 || *v5 != 93 )
  {
    *a1 = v5;
    goto LABEL_5;
  }
  if ( (unsigned __int64)v4 >= v3 )
  {
LABEL_8:
    if ( !*((_DWORD *)a1 + 4) )
      *((_DWORD *)a1 + 4) = 7;
    goto LABEL_10;
  }
  v7 = off_4CD37A0;
  v8 = v4 - v2;
  if ( !off_4CD37A0 )
  {
LABEL_18:
    if ( v8 == 1 )
      return *(unsigned __int8 *)v2;
    if ( !*((_DWORD *)a1 + 4) )
      *((_DWORD *)a1 + 4) = 3;
LABEL_10:
    *a1 = (const char *)&unk_4FA17D0;
    a1[1] = (const char *)&unk_4FA17D0;
    return 0;
  }
  v9 = &off_4CD37A0;
  while ( strncmp(v7, v2, v8) || strlen(v7) != v8 )
  {
    v7 = v9[2];
    v9 += 2;
    if ( !v7 )
      goto LABEL_18;
  }
  return *((unsigned __int8 *)v9 + 8);
}
