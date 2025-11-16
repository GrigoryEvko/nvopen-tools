// Function: sub_130AAB0
// Address: 0x130aab0
//
unsigned __int64 __fastcall sub_130AAB0(char *a1, char **a2, unsigned int a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rcx
  bool v7; // si
  char *v8; // rbx
  char v9; // dl
  char *v10; // r14
  unsigned __int64 i; // r8
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  char v15; // di
  int *v16; // rax
  int *v17; // rax
  int v18; // r8d
  __int64 v19; // r9

  v3 = a3;
  v4 = a3;
  v7 = a3 > 0x24 || a3 == 1;
  v8 = a1;
  if ( v7 )
  {
LABEL_33:
    v16 = __errno_location();
    i = -1;
    *v16 = 22;
    if ( !a2 )
      return i;
LABEL_34:
    *a2 = a1;
    return i;
  }
  while ( 1 )
  {
    v9 = *v8;
    if ( *v8 == 43 )
      break;
    if ( v9 > 43 )
    {
      if ( v9 != 45 )
        goto LABEL_24;
      v7 = 1;
      break;
    }
    if ( v9 > 13 )
    {
      if ( v9 != 32 )
        goto LABEL_6;
    }
    else if ( v9 <= 8 )
    {
      goto LABEL_6;
    }
    ++v8;
  }
  v9 = *++v8;
LABEL_24:
  if ( v9 == 48 )
  {
    v15 = v8[1];
    if ( v15 != 88 )
    {
      if ( v15 <= 88 )
      {
        if ( (unsigned __int8)(v15 - 48) <= 7u )
        {
          v10 = v8;
          if ( (_DWORD)v3 )
          {
            if ( (_DWORD)v3 == 8 )
              v9 = *++v8;
          }
          else
          {
            v9 = *++v8;
            v4 = 8;
          }
          goto LABEL_8;
        }
LABEL_42:
        v10 = v8;
        i = 0;
        ++v8;
LABEL_38:
        if ( !a2 )
          return i;
        if ( v10 != v8 )
          goto LABEL_19;
        goto LABEL_34;
      }
      if ( v15 != 120 )
        goto LABEL_42;
    }
    v18 = (unsigned __int8)v8[2];
    if ( (unsigned __int8)(v18 - 48) <= 0x36u )
    {
      v19 = 0x7E0000007E03FFLL;
      if ( _bittest64(&v19, (unsigned int)(v18 - 48)) )
      {
        v10 = v8;
        if ( (_DWORD)v3 )
        {
          if ( (_DWORD)v3 == 16 )
          {
            v9 = v8[2];
            v8 += 2;
          }
        }
        else
        {
          v9 = v8[2];
          v8 += 2;
          v4 = 16;
        }
        goto LABEL_8;
      }
    }
  }
LABEL_6:
  v4 = 10;
  v10 = v8;
  if ( (_DWORD)v3 )
    v4 = v3;
LABEL_8:
  for ( i = 0; ; i = v12 )
  {
    LOBYTE(v13) = v9 - 48;
    if ( (unsigned __int8)(v9 - 48) <= 9u )
      goto LABEL_14;
    if ( (unsigned __int8)(v9 - 65) <= 0x19u )
    {
      LOBYTE(v13) = v9 - 55;
      goto LABEL_14;
    }
    if ( (unsigned __int8)(v9 - 97) > 0x19u )
      break;
    LOBYTE(v13) = v9 - 87;
LABEL_14:
    v13 = (char)v13;
    if ( v4 <= (char)v13 )
      break;
    v12 = i * v4 + v13;
    if ( i > v12 )
    {
      v17 = __errno_location();
      i = -1;
      *v17 = 34;
      goto LABEL_38;
    }
    v9 = *++v8;
  }
  if ( v7 )
    i = -(__int64)i;
  if ( v8 == v10 )
    goto LABEL_33;
  if ( !a2 )
    return i;
LABEL_19:
  *a2 = v8;
  return i;
}
