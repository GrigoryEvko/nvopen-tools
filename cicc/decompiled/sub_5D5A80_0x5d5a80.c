// Function: sub_5D5A80
// Address: 0x5d5a80
//
int __fastcall sub_5D5A80(__int64 a1, unsigned int a2)
{
  char *v3; // r14
  int result; // eax
  char v5; // r12
  FILE *v6; // r15
  char *v7; // rbx
  int v8; // r13d
  char *v9; // r13
  int v10; // edi
  int v11; // ebx
  int v12; // edi
  unsigned __int64 v13; // rcx
  char *v14; // rdi
  char *v15; // rbx
  int v16; // edi
  char v17; // al
  unsigned __int64 v18; // rcx
  char *v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi

  v3 = *(char **)(a1 + 8);
  if ( !v3
    || (*(_BYTE *)(a1 + 90) & 4) != 0 && ((*(_BYTE *)(a1 + 88) & 0x70) == 0x10 || (*(_WORD *)(a1 + 88) & 0x570) == 0)
    || (*(_DWORD *)(a1 + 88) & 0x80400) == 0x80000 )
  {
    return sub_5D34A0();
  }
  if ( (*(_BYTE *)(a1 + 88) & 0x50) == 0x10 )
  {
    v5 = *v3;
    v6 = stream;
    switch ( *v3 )
    {
      case '_':
        v17 = v3[1];
        if ( v17 == 65 )
        {
          if ( !strcmp(v3, "_Atomic") || !strcmp(v3, "_Alignof") || !strcmp(v3, "_Alignas") )
            goto LABEL_30;
          goto LABEL_82;
        }
        v18 = 6;
        v19 = "_Bool";
        if ( v17 == 66 )
          goto LABEL_81;
        v18 = 9;
        v19 = "_Complex";
        if ( v17 == 67 )
          goto LABEL_81;
        v18 = 9;
        v19 = "_Generic";
        if ( v17 == 71 )
          goto LABEL_81;
        v18 = 11;
        v19 = "_Imaginary";
        if ( v17 == 73 )
          goto LABEL_81;
        v18 = 10;
        v19 = "_Noreturn";
        if ( v17 == 78 )
          goto LABEL_81;
        if ( v17 == 83 )
        {
          v18 = 15;
          v19 = "_Static_assert";
LABEL_81:
          if ( !memcmp(v3, v19, v18) )
            goto LABEL_30;
        }
        else if ( v17 == 84 && !strcmp(v3, "_Thread_local") )
        {
          goto LABEL_30;
        }
LABEL_82:
        if ( !unk_4F068E4 || strcmp(v3, "__global") && strcmp(v3, "__symbolic") && strcmp(v3, "__hidden") )
          goto LABEL_27;
        goto LABEL_30;
      case 'a':
        v13 = 4;
        v14 = "asm";
        if ( !strcmp(v3, "auto") )
          goto LABEL_30;
        goto LABEL_26;
      case 'b':
        v13 = 6;
        v14 = "break";
        goto LABEL_26;
      case 'c':
        if ( !strcmp(v3, "case") )
          goto LABEL_30;
        if ( !strcmp(v3, "char") )
          goto LABEL_30;
        v13 = 9;
        v14 = "continue";
        if ( !strcmp(v3, "const") )
          goto LABEL_30;
        goto LABEL_26;
      case 'd':
        if ( !strcmp(v3, "default") || *v3 == 100 && v3[1] == 111 && !v3[2] )
          goto LABEL_30;
        v13 = 7;
        v14 = "double";
LABEL_26:
        if ( memcmp(v3, v14, v13) )
        {
LABEL_27:
          if ( qword_4CF7EB8 == v6 )
          {
LABEL_87:
            if ( !memcmp(v3, "__builtin_", 0xAu) && strcmp(v3, "__builtin_expect") )
              v3 += 10;
LABEL_10:
            v7 = v3 + 1;
            result = strlen(v3);
            v5 = *v3;
            v8 = result;
            if ( !*v3 )
              goto LABEL_14;
          }
          else
          {
            v7 = v3 + 1;
            v8 = strlen(v3);
          }
          while ( 1 )
          {
            ++v7;
            result = putc(v5, v6);
            v5 = *(v7 - 1);
            if ( !v5 )
              break;
            v6 = stream;
          }
LABEL_14:
          dword_4CF7F40 += v8;
          return result;
        }
LABEL_30:
        v15 = v3 + 1;
        putc(95, v6);
        ++dword_4CF7F40;
        putc(95, stream);
        ++dword_4CF7F40;
        result = putc(120, stream);
        v16 = *v3;
        ++dword_4CF7F40;
        for ( ; (_BYTE)v16; ++dword_4CF7F40 )
        {
          ++v15;
          result = putc(v16, stream);
          v16 = *(v15 - 1);
        }
        break;
      case 'e':
        if ( !strcmp(v3, "else") )
          goto LABEL_30;
        v13 = 7;
        v14 = "extern";
        if ( !strcmp(v3, "enum") )
          goto LABEL_30;
        goto LABEL_26;
      case 'f':
        if ( !strcmp(v3, "float") )
          goto LABEL_30;
        v13 = 8;
        v14 = "fortran";
        if ( !strcmp(v3, "for") )
          goto LABEL_30;
        goto LABEL_26;
      case 'g':
        v13 = 5;
        v14 = "goto";
        goto LABEL_26;
      case 'i':
        if ( *v3 == 105 && v3[1] == 102 && !v3[2] )
          goto LABEL_30;
        v13 = 4;
        v14 = "int";
        if ( !strcmp(v3, "inline") )
          goto LABEL_30;
        goto LABEL_26;
      case 'l':
        v13 = 5;
        v14 = "long";
        goto LABEL_26;
      case 'p':
        v13 = 7;
        v14 = "pascal";
        goto LABEL_26;
      case 'r':
        if ( !strcmp(v3, "register") )
          goto LABEL_30;
        v13 = 7;
        v14 = "return";
        if ( !strcmp(v3, "restrict") )
          goto LABEL_30;
        goto LABEL_26;
      case 's':
        if ( strcmp(v3, "short")
          && strcmp(v3, "signed")
          && strcmp(v3, "sizeof")
          && strcmp(v3, "static")
          && strcmp(v3, "struct")
          && strcmp(v3, "switch") )
        {
          goto LABEL_27;
        }
        goto LABEL_30;
      case 't':
        v13 = 8;
        v14 = "typedef";
        goto LABEL_26;
      case 'u':
        if ( !strcmp(v3, "union") )
          goto LABEL_30;
        v13 = 9;
        v14 = "unsigned";
        if ( !strcmp(v3, "unix") )
          goto LABEL_30;
        goto LABEL_26;
      case 'v':
        v13 = 9;
        v14 = "volatile";
        if ( !strcmp(v3, "void") )
          goto LABEL_30;
        goto LABEL_26;
      case 'w':
        v13 = 6;
        v14 = "while";
        goto LABEL_26;
      default:
        if ( qword_4CF7EB8 != stream )
          goto LABEL_10;
        goto LABEL_87;
    }
  }
  else
  {
    v9 = v3 + 1;
    result = *(_BYTE *)(a1 + 89) & 0xD;
    if ( (_BYTE)result == 1 )
    {
      if ( a2 )
      {
        putc(95, stream);
        ++dword_4CF7F40;
        putc(95, stream);
        ++dword_4CF7F40;
        sub_5D32F0(a2);
        putc(95, stream);
        v20 = *(unsigned int *)(a1 + 64);
        ++dword_4CF7F40;
        sub_5D32F0(v20);
        putc(95, stream);
        v21 = *(unsigned __int16 *)(a1 + 68);
        ++dword_4CF7F40;
        sub_5D32F0(v21);
        result = putc(95, stream);
        ++dword_4CF7F40;
      }
      v12 = *v3;
      if ( *v3 )
      {
        do
        {
          ++v9;
          result = putc(v12, stream);
          v12 = *(v9 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v12 );
      }
    }
    else
    {
      result = strlen(*(const char **)(a1 + 8));
      v10 = *v3;
      v11 = result;
      if ( *v3 )
      {
        do
        {
          ++v9;
          result = putc(v10, stream);
          v10 = *(v9 - 1);
        }
        while ( *(v9 - 1) );
      }
      dword_4CF7F40 += v11;
    }
  }
  return result;
}
