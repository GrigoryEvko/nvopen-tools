// Function: sub_5CA8C0
// Address: 0x5ca8c0
//
int __fastcall sub_5CA8C0(char **a1, char a2, const char *a3)
{
  char *v3; // rbx
  char v4; // r13
  size_t v6; // r15
  int result; // eax
  int v8; // edx

  v3 = *a1;
  v4 = **a1;
  if ( !a3 )
    return v4 != 91;
  if ( (v4 & 0xDF) == 0x5B )
  {
    v6 = strlen(a3);
    result = strncmp(a3, v3 + 1, v6);
    if ( result )
    {
      return 0;
    }
    else
    {
      v8 = 93;
      if ( v4 != 91 )
        v8 = 125;
      if ( v3[v6 + 1] == v8 )
      {
        *a1 = &v3[v6 + 2];
        return 1;
      }
    }
  }
  else if ( !unk_4F077B4 || (result = 0, a2 != 5) )
  {
    result = unk_4F077B8;
    if ( unk_4F077B8 )
    {
      result = 1;
      if ( strcmp(a3, "gnu") )
        return strcmp(a3, "__gnu__") == 0;
    }
  }
  return result;
}
