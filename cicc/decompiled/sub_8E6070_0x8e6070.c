// Function: sub_8E6070
// Address: 0x8e6070
//
char *__fastcall sub_8E6070(_BYTE *a1, _DWORD *a2, int *a3, _QWORD *a4, __int64 a5)
{
  char v5; // al
  char *v6; // r14
  char *v10; // rbx
  char v11; // si
  int v12; // eax
  char v13; // r10
  char v14; // dl
  char *v15; // r8
  char v16; // r10
  char v17; // dl
  char *v18; // r8
  char *i; // rdi
  char v20; // r10
  char *v21; // rdi
  char *v22; // r8
  char v23; // dl
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  _DWORD *v27; // r9
  __int64 v28; // r8
  const char *v29; // r13
  signed __int64 v30; // rax
  char v31; // r10
  char *v32; // rdi
  char *v33; // r8
  char v34; // dl
  void *v35; // rax
  char v36; // r10
  char *v37; // rdi
  char *v38; // r8
  char v39; // dl
  char *v40; // rdx
  char v41; // di
  char *v42; // r8
  char *v43; // rdx
  char v44; // di
  char *v45; // r8
  void *v46; // rax
  char *v47; // rdx
  char v48; // di
  char *v49; // r8
  char *v50; // r8
  char v51; // di
  char *v52; // rdx
  char v53; // di
  __int64 v54; // rdx
  signed __int64 v55; // rax
  char v56; // di
  __int64 v57; // rdx
  char v58; // cl
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rax
  char v62; // dl
  __int64 v63; // rax
  char v64; // dl
  char v65; // al
  __int64 v66; // rax
  char v67; // dl
  __int64 v69; // [rsp+8h] [rbp-38h]
  size_t n[5]; // [rsp+18h] [rbp-28h] BYREF

  *a2 = 2;
  *a4 = byte_3F871B3;
  *a3 = 0;
  v5 = *a1;
  if ( !*a1 )
  {
    v6 = 0;
    if ( !*(_DWORD *)(a5 + 24) )
    {
      ++*(_QWORD *)(a5 + 32);
      ++*(_QWORD *)(a5 + 48);
      *(_DWORD *)(a5 + 24) = 1;
    }
    return v6;
  }
  v10 = a1;
  v11 = a1[1];
  switch ( v5 )
  {
    case 'a':
      switch ( v11 )
      {
        case 'a':
          v6 = "&&";
          goto LABEL_7;
        case 'd':
          *a2 = 1;
          v6 = "&";
          v12 = *a3;
          goto LABEL_13;
        case 'n':
          v6 = "&";
          goto LABEL_7;
        case 'N':
          v6 = (char *)&unk_3F7C052;
          goto LABEL_7;
        case 'S':
          v6 = "=";
          goto LABEL_7;
        case 't':
          *a2 = 0;
          v6 = "alignof(";
          *a4 = ")";
          v12 = *a3;
          break;
        case 'w':
          *a2 = 1;
          v6 = "co_await";
          v12 = *a3;
          break;
        default:
          v6 = 0;
          if ( v11 != 122 )
            goto LABEL_7;
          v6 = "alignof(";
          *a4 = ")";
          *a2 = 1;
          v12 = *a3;
          break;
      }
      goto LABEL_13;
    case 'c':
      switch ( v11 )
      {
        case 'c':
          *a2 = 1;
          v6 = "const_cast";
          v12 = *a3;
          goto LABEL_13;
        case 'l':
          *a2 = 0;
          v6 = "()";
          goto LABEL_7;
        case 'm':
          v6 = ",";
          goto LABEL_7;
        case 'o':
          *a2 = 1;
          v6 = "~";
          v12 = *a3;
          break;
        default:
          v6 = 0;
          if ( v11 != 118 )
            goto LABEL_7;
          *a2 = 1;
          v6 = "cast";
          v12 = *a3;
          break;
      }
      goto LABEL_13;
    case 'd':
      switch ( v11 )
      {
        case 'a':
          *a2 = 1;
          v6 = "delete[] ";
          v12 = *a3;
          goto LABEL_13;
        case 'c':
          *a2 = 1;
          v6 = (char *)"dynamic_cast";
          v12 = *a3;
          goto LABEL_13;
        case 'e':
          *a2 = 1;
          v6 = "*";
          v12 = *a3;
          goto LABEL_13;
        case 'l':
          *a2 = 1;
          v6 = "delete ";
          v12 = *a3;
          goto LABEL_13;
      }
      v6 = ".*";
      if ( v11 != 115 )
      {
        v6 = "/";
        if ( v11 != 118 )
        {
          v6 = "/=";
          if ( v11 != 86 )
            v6 = 0;
        }
      }
      goto LABEL_7;
    case 'e':
      v6 = "^";
      if ( v11 != 111 )
      {
        v6 = "^=";
        if ( v11 != 79 )
        {
          v6 = "==";
          if ( v11 != 113 )
            v6 = 0;
        }
      }
      goto LABEL_7;
    case 'g':
      v6 = ">=";
      if ( v11 != 101 )
      {
        v6 = ">";
        if ( v11 != 116 )
          v6 = 0;
      }
      goto LABEL_7;
    case 'i':
      v6 = 0;
      if ( v11 != 120 )
        goto LABEL_7;
      v6 = "[";
      *a4 = "]";
      v12 = *a3;
      goto LABEL_13;
    case 'l':
      if ( v11 == 101 )
      {
        v6 = "<=";
        goto LABEL_7;
      }
      if ( v11 == 105 )
      {
        v24 = sub_8E5810(a1 + 2, (__int64 *)n, a5);
        v28 = a5;
        *v27 = 0;
        v29 = (const char *)v24;
        if ( !*(_DWORD *)(a5 + 24) )
        {
          if ( (__int64)n[0] <= 0 )
            goto LABEL_151;
          v6 = (char *)qword_4F605A8;
          if ( !qword_4F605A8 )
          {
            qword_4F605A0 = 128;
            v35 = (void *)malloc(128, n, v25, v26, a5, v27);
            v28 = a5;
            qword_4F605A8 = v35;
            v6 = (char *)v35;
            if ( v35 )
            {
LABEL_142:
              v69 = v28;
              v30 = strlen(v29);
              v28 = v69;
              if ( (__int64)n[0] <= v30 )
              {
LABEL_143:
                strcpy(v6, "\"\"");
                strncpy(v6 + 2, v29, n[0]);
                v6[n[0] + 2] = 0;
                *a3 = (_DWORD)v29 + LODWORD(n[0]) - (_DWORD)a1;
LABEL_144:
                v12 = *a3;
                goto LABEL_13;
              }
            }
LABEL_151:
            sub_8E5770(v28);
            goto LABEL_152;
          }
          if ( n[0] + 4 <= qword_4F605A0 )
            goto LABEL_142;
          qword_4F605A0 = n[0] + 4;
          v46 = (void *)realloc(qword_4F605A8);
          v28 = a5;
          qword_4F605A8 = v46;
          v6 = (char *)v46;
          if ( !v46 )
          {
            if ( *(_DWORD *)(a5 + 24) )
              goto LABEL_152;
            goto LABEL_151;
          }
          v55 = strlen(v29);
          if ( v55 >= (__int64)n[0] )
            goto LABEL_143;
          v28 = a5;
          if ( !*(_DWORD *)(a5 + 24) )
            goto LABEL_151;
        }
LABEL_152:
        v6 = 0;
        goto LABEL_144;
      }
      v6 = "<<";
      if ( v11 != 115 )
      {
        v6 = "<<=";
        if ( v11 != 83 )
        {
          v6 = "<";
          if ( v11 != 116 )
            v6 = 0;
        }
      }
LABEL_7:
      *a3 = 2;
      return v6;
    case 'm':
      switch ( v11 )
      {
        case 'i':
          v6 = "-";
          goto LABEL_7;
        case 'I':
          v6 = "-=";
          goto LABEL_7;
        case 'l':
          v6 = "*";
          goto LABEL_7;
        case 'L':
          v6 = "*=";
          goto LABEL_7;
      }
      v6 = 0;
      if ( v11 != 109 )
        goto LABEL_7;
      *a2 = 1;
      v6 = "--";
      v12 = *a3;
      goto LABEL_13;
    case 'n':
      switch ( v11 )
      {
        case 'a':
          v6 = "new[] ";
          goto LABEL_7;
        case 'e':
          v6 = "!=";
          goto LABEL_7;
        case 'g':
          *a2 = 1;
          v6 = "-";
          v12 = *a3;
          goto LABEL_13;
        case 't':
          *a2 = 1;
          v6 = (char *)&unk_3F6A4C5;
          v12 = *a3;
          goto LABEL_13;
        case 'w':
          v6 = "new ";
          goto LABEL_7;
      }
      v6 = 0;
      if ( v11 != 120 )
        goto LABEL_7;
      v6 = "noexcept(";
      *a4 = ")";
      *a2 = 1;
      v12 = *a3;
      goto LABEL_13;
    case 'o':
      v6 = "||";
      if ( v11 != 111 )
      {
        v6 = "|";
        if ( v11 != 114 )
        {
          v6 = "|=";
          if ( v11 != 82 )
            v6 = 0;
        }
      }
      goto LABEL_7;
    case 'p':
      switch ( v11 )
      {
        case 'l':
          v6 = "+";
          goto LABEL_7;
        case 'L':
          v6 = "+=";
          goto LABEL_7;
        case 'm':
          v6 = "->*";
          goto LABEL_7;
        case 'p':
          *a2 = 1;
          v6 = "++";
          v12 = *a3;
          goto LABEL_13;
        case 's':
          *a2 = 1;
          v6 = "+";
          v12 = *a3;
          goto LABEL_13;
      }
      v6 = "->";
      if ( v11 != 116 )
        v6 = 0;
      goto LABEL_7;
    case 'q':
      v6 = 0;
      if ( v11 != 117 )
        goto LABEL_7;
      *a2 = 3;
      v6 = "?";
      v12 = *a3;
      goto LABEL_13;
    case 'r':
      if ( v11 == 99 )
      {
        *a2 = 1;
        v6 = "reinterpret_cast";
        v12 = *a3;
        goto LABEL_13;
      }
      v6 = "%";
      if ( v11 != 109 )
      {
        v6 = "%=";
        if ( v11 != 77 )
        {
          v6 = ">>";
          if ( v11 != 115 )
          {
            v6 = ">>=";
            if ( v11 != 83 )
              v6 = 0;
          }
        }
      }
      goto LABEL_7;
    case 's':
      switch ( v11 )
      {
        case 'c':
          *a2 = 1;
          v6 = "static_cast";
          v12 = *a3;
          goto LABEL_13;
        case 's':
          v6 = "<=>";
          goto LABEL_7;
        case 't':
          *a2 = 0;
          v6 = "sizeof(";
          *a4 = ")";
          v12 = *a3;
          break;
        default:
          v6 = 0;
          if ( v11 != 122 )
            goto LABEL_7;
          v6 = "sizeof(";
          *a4 = ")";
          *a2 = 1;
          v12 = *a3;
          break;
      }
      goto LABEL_13;
    case 't':
      switch ( v11 )
      {
        case 'e':
          v6 = "typeid(";
          *a4 = ")";
          *a2 = 1;
          v12 = *a3;
          goto LABEL_13;
        case 'i':
          v6 = "typeid(";
          *a4 = ")";
          *a2 = 0;
          v12 = *a3;
          goto LABEL_13;
        case 'r':
          *a2 = 0;
          v6 = "throw";
          goto LABEL_7;
      }
      v6 = 0;
      if ( v11 != 119 )
        goto LABEL_7;
      *a2 = 1;
      v12 = *a3;
      v6 = "throw ";
      goto LABEL_13;
    case 'v':
      v13 = *a1;
      v14 = *a1;
      v15 = "18alignofe";
      while ( 2 )
      {
        ++a1;
        if ( v14 == v13 )
        {
          v14 = *v15++;
          if ( v14 )
          {
            v13 = *a1;
            continue;
          }
          v6 = "__alignof__(";
          *a4 = ")";
          *a2 = 1;
          *a3 = 11;
          return v6;
        }
        break;
      }
      v16 = v5;
      v17 = v5;
      v18 = "17alignof";
      for ( i = v10; ; v16 = *i )
      {
        ++i;
        if ( v17 != v16 )
          break;
        v17 = *v18++;
        if ( !v17 )
        {
          v6 = "__alignof__(";
          *a4 = ")";
          *a2 = 0;
          *a3 = 10;
          return v6;
        }
      }
      v20 = v5;
      v21 = v10;
      v22 = "19__uuidofe";
      v23 = v5;
      while ( 1 )
      {
        ++v21;
        if ( v23 != v20 )
          break;
        v23 = *v22++;
        if ( !v23 )
        {
          v6 = "__uuidof(";
          *a4 = ")";
          *a2 = 1;
          *a3 = 12;
          return v6;
        }
        v20 = *v21;
      }
      v31 = v5;
      v32 = v10;
      v33 = "18__uuidof";
      v34 = v5;
      while ( 1 )
      {
        ++v32;
        if ( v34 != v31 )
          break;
        v34 = *v33++;
        if ( !v34 )
        {
          v6 = "__uuidof(";
          *a4 = ")";
          *a2 = 0;
          *a3 = 11;
          return v6;
        }
        v31 = *v32;
      }
      v36 = v5;
      v37 = v10;
      v38 = "17typeide";
      v39 = v5;
      while ( 1 )
      {
        ++v37;
        if ( v39 != v36 )
          break;
        v39 = *v38++;
        if ( !v39 )
        {
          v6 = "typeid(";
          *a4 = ")";
          *a2 = 1;
          *a3 = 10;
          return v6;
        }
        v36 = *v37;
      }
      v40 = v10;
      v41 = v5;
      v42 = "16typeid";
      while ( *v40++ == v41 )
      {
        v41 = *v42++;
        if ( !v41 )
        {
          v6 = "typeid(";
          *a4 = ")";
          *a2 = 0;
          *a3 = 9;
          return v6;
        }
      }
      v43 = v10;
      v44 = v5;
      v45 = "19clitypeid";
      while ( *v43++ == v44 )
      {
        v44 = *v45++;
        if ( !v44 )
        {
          *a2 = 0;
          v6 = "::typeid";
          *a3 = 12;
          return v6;
        }
      }
      v47 = v10;
      v48 = v5;
      v49 = "23min";
      while ( *v47++ == v48 )
      {
        v48 = *v49++;
        if ( !v48 )
        {
          *a3 = 6;
          v6 = "<?";
          *a2 = 2;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v50 = v10;
      v51 = v5;
      v52 = "23max";
      while ( *v50++ == v51 )
      {
        v51 = *v52++;
        if ( !v51 )
        {
          *a3 = 6;
          v6 = ">?";
          *a2 = 2;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v53 = v5;
      v54 = 0;
      while ( v10[v54] == v53 )
      {
        v53 = aV18Real[++v54];
        if ( !v53 )
        {
          v6 = "__real(";
          *a4 = ")";
          *a3 = 11;
          *a2 = 1;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v56 = v5;
      v57 = 0;
      while ( v10[v57] == v56 )
      {
        v56 = aV18Imag[++v57];
        if ( !v56 )
        {
          v6 = "__imag(";
          *a4 = ")";
          *a3 = 11;
          *a2 = 1;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v58 = v5;
      v59 = 0;
      while ( v10[v59] == v58 )
      {
        v58 = aV19clihandle[++v59];
        if ( !v58 )
        {
          *a3 = 12;
          v6 = "%";
          *a2 = 1;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v60 = 0;
      while ( v10[v60] == v5 )
      {
        v5 = aV112clisafeCas[++v60];
        if ( !v5 )
        {
          *a3 = 16;
          v6 = "safe_cast";
          *a2 = 1;
          v12 = *a3;
          goto LABEL_13;
        }
      }
      v61 = 0;
      v62 = 57;
      while ( v10[v61 + 2] == v62 )
      {
        v62 = a9builtin[++v61];
        if ( !v62 )
        {
LABEL_210:
          *a2 = v11 - 48;
          v6 = aBuiltinOperati;
          if ( v10[2] == 57 )
          {
            aBuiltinOperati[18] = v10[10];
            v65 = v10[11];
            aBuiltinOperati[20] = 0;
            aBuiltinOperati[19] = v65;
            *a3 = 12;
          }
          else
          {
            aBuiltinOperati[18] = v10[11];
            aBuiltinOperati[19] = v10[12];
            aBuiltinOperati[20] = v10[13];
            *a3 = 14;
          }
          return v6;
        }
      }
      v63 = 0;
      v64 = 49;
      while ( v10[v63 + 2] == v64 )
      {
        v64 = a10builtin[++v63];
        if ( !v64 )
          goto LABEL_210;
      }
      v66 = 0;
      v67 = 49;
      do
      {
        if ( v10[v66 + 2] != v67 )
        {
LABEL_6:
          v6 = 0;
          goto LABEL_7;
        }
        v67 = a12clisubscript[++v66];
      }
      while ( v67 );
      v6 = 0;
      if ( (unsigned __int8)(v11 - 48) <= 9u )
      {
        *a3 = 16;
        v6 = "subscript";
        *a2 = v10[1] - 48;
        v12 = *a3;
LABEL_13:
        if ( v12 )
          return v6;
        goto LABEL_7;
      }
      goto LABEL_7;
    default:
      goto LABEL_6;
  }
}
