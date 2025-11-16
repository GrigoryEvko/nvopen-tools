// Function: sub_30978E0
// Address: 0x30978e0
//
__int64 __fastcall sub_30978E0(__int64 a1, const void *a2, size_t a3)
{
  int v3; // eax
  unsigned int *i; // r15
  const char *v6; // rbx
  size_t v7; // rax
  size_t v8; // rbx
  size_t v9; // rax
  const char *v11; // [rsp+8h] [rbp-38h]

  v3 = 0;
  for ( i = (unsigned int *)&unk_49D6D60; ; v3 = *i )
  {
    if ( v3 )
    {
      v8 = (unsigned int)strlen(&byte_44CAF20[dword_44CAF10[v3 + 1]]);
      v11 = &byte_44CAF20[i[1]];
      v9 = strlen(v11);
      if ( v9 < v8 )
      {
        if ( !a3 )
          return i[10];
        goto LABEL_6;
      }
      v7 = v9 - v8;
      v6 = &v11[v8];
    }
    else
    {
      v6 = &byte_44CAF20[i[1]];
      v7 = strlen(v6);
    }
    if ( a3 == v7 && (!a3 || !memcmp(a2, v6, a3)) )
      return i[10];
LABEL_6:
    i += 20;
    if ( i == (unsigned int *)&off_49D87F0 )
      break;
  }
  return 0;
}
