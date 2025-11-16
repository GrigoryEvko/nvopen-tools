// Function: sub_703AC0
// Address: 0x703ac0
//
__int64 __fastcall sub_703AC0(const char **a1, _QWORD *a2, _QWORD *a3, unsigned int a4, _DWORD *a5)
{
  _QWORD *v5; // r10
  _QWORD *v6; // r12
  unsigned int v7; // ebx
  const char *v8; // r15
  char v9; // r13
  char *i; // r14
  unsigned int v11; // r12d
  const char *v12; // rdi
  int v13; // eax
  size_t v14; // rax
  const char *v16; // rdi
  char *s; // [rsp+8h] [rbp-48h]
  size_t n; // [rsp+10h] [rbp-40h]
  size_t na; // [rsp+10h] [rbp-40h]

  v5 = a3;
  v6 = a2;
  v7 = a4;
  v8 = *a1 + 1;
  *a1 = v8;
  v9 = *v8;
  for ( i = (char *)v8; *i != 93; v9 = *i )
  {
    if ( !v9 )
      break;
    *a1 = ++i;
  }
  if ( a4 )
  {
    if ( a3 )
    {
      v11 = 0;
      do
      {
        v12 = *(const char **)(v5[1] + 8LL);
        if ( v12 )
        {
          n = (size_t)v5;
          v13 = strncmp(v12, v8, i - v8);
          v5 = (_QWORD *)n;
          if ( !v13 )
          {
            v14 = strlen(v12);
            v5 = (_QWORD *)n;
            if ( i - v8 == v14 )
              return v11;
          }
        }
        v5 = (_QWORD *)*v5;
        ++v11;
      }
      while ( v5 );
    }
LABEL_12:
    *i = 0;
    v11 = -1;
    sub_6851A0(0x586u, a5, (__int64)v8);
    **a1 = v9;
    return v11;
  }
  if ( !a2 )
    goto LABEL_12;
  na = i - v8;
  while ( 1 )
  {
    v16 = (const char *)v6[1];
    if ( v16 )
    {
      s = (char *)v6[1];
      if ( !strncmp(v16, v8, na) && na == strlen(s) )
        return v7;
    }
    v6 = (_QWORD *)*v6;
    ++v7;
    if ( !v6 )
      goto LABEL_12;
  }
}
