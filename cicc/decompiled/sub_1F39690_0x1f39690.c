// Function: sub_1F39690
// Address: 0x1f39690
//
__int64 __fastcall sub_1F39690(__int64 a1, const char *a2, __int64 a3)
{
  unsigned __int8 v3; // r15
  const char *v4; // r12
  char v6; // r14
  const char *v7; // r13
  size_t v8; // rax
  int v9; // r13d
  int v11; // eax
  int v12; // ecx
  char *v13; // r15
  unsigned __int8 v14; // r14
  int v15; // [rsp+8h] [rbp-48h]
  unsigned int v16; // [rsp+Ch] [rbp-44h]
  char *endptr; // [rsp+18h] [rbp-38h] BYREF

  v3 = *a2;
  if ( *a2 )
  {
    v16 = 0;
    v4 = a2;
    v6 = 1;
    do
    {
      while ( 1 )
      {
        if ( v3 == 10
          || (v7 = *(const char **)(a3 + 40), v8 = strlen(v7), !strncmp(v4, v7, v8))
          || (v6 &= strncmp(v4, *(const char **)(a3 + 48), *(_QWORD *)(a3 + 56)) != 0) != 0 )
        {
          v6 = 1;
          if ( !isspace(v3) )
            break;
        }
        v3 = *++v4;
        if ( !v3 )
          return v16;
      }
      v9 = *(_DWORD *)(a3 + 24);
      if ( !memcmp(v4, ".space", 6u) )
      {
        v11 = strtol(v4 + 6, &endptr, 10);
        v12 = 0;
        v13 = endptr;
        if ( v11 >= 0 )
          v12 = v11;
        v15 = v12;
        v14 = *endptr;
        if ( *endptr == 10 )
          goto LABEL_19;
        while ( isspace(v14) )
        {
          endptr = ++v13;
          v14 = *v13;
          if ( *v13 == 10 )
            goto LABEL_19;
        }
        if ( !v14 || !strncmp(v13, *(const char **)(a3 + 48), *(_QWORD *)(a3 + 56)) )
LABEL_19:
          v9 = v15;
      }
      v3 = *++v4;
      v16 += v9;
      v6 = 0;
    }
    while ( v3 );
  }
  else
  {
    return 0;
  }
  return v16;
}
