// Function: sub_8E9FF0
// Address: 0x8e9ff0
//
char *__fastcall sub_8E9FF0(__int64 a1, int a2, int a3, int a4, unsigned int a5, __int64 a6)
{
  int v6; // r10d
  int v9; // esi
  unsigned __int8 *i; // r12
  int v11; // ebx
  int v12; // r13d
  __int64 v13; // rdx
  unsigned __int8 *v14; // r11
  char *v15; // r14
  __int64 v16; // r9
  bool v17; // bl
  char v19; // cl
  char *v20; // rax
  char *v21; // rdx
  unsigned __int8 *v22; // rdi
  int v23; // edx
  unsigned __int8 *v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  char v30; // dl
  unsigned __int8 *v31; // r13
  unsigned __int8 v32; // bl
  char *v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rcx
  unsigned __int8 v37; // dl
  unsigned __int8 *v38; // rax
  __int64 v39; // rax
  char v40; // al
  __int64 v41; // rcx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  char *v44; // rsi
  unsigned __int8 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int8 *v49; // rcx
  size_t v50; // r14
  unsigned int v51; // r11d
  char *v52; // r10
  char *v53; // rbx
  __int64 v54; // rsi
  __int64 v55; // rax
  char *v56; // r10
  __int64 v57; // rdx
  int v58; // eax
  char *v59; // r13
  __int64 v60; // rcx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdx
  unsigned __int8 v63; // r13
  int v64; // eax
  __int64 v65; // rcx
  char v66; // dl
  char *v67; // rsi
  __int64 v68; // r14
  __int64 v69; // rax
  unsigned __int8 *v70; // rax
  char v71; // dl
  char *v72; // rsi
  char *v73; // rax
  int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // rsi
  char *v77; // rdi
  __int64 v78; // r9
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rdx
  unsigned __int64 v82; // rax
  __int64 v83; // rdx
  unsigned __int64 v84; // rax
  __int64 v85; // rdx
  unsigned __int64 v86; // rax
  unsigned __int8 *v87; // r13
  unsigned __int8 *v88; // r13
  unsigned __int8 *v89; // r11
  unsigned __int8 v90; // al
  unsigned __int8 *v91; // rdi
  unsigned __int8 *v92; // rax
  unsigned __int8 *v93; // rax
  unsigned __int8 *v94; // r13
  __int64 v95; // [rsp-10h] [rbp-B0h]
  int srcc; // [rsp+8h] [rbp-98h]
  int srcd; // [rsp+8h] [rbp-98h]
  int srca; // [rsp+8h] [rbp-98h]
  char *srce; // [rsp+8h] [rbp-98h]
  unsigned __int8 *srcf; // [rsp+8h] [rbp-98h]
  unsigned __int8 *srcg; // [rsp+8h] [rbp-98h]
  int srch; // [rsp+8h] [rbp-98h]
  unsigned __int8 *srcb; // [rsp+8h] [rbp-98h]
  unsigned __int8 *srci; // [rsp+8h] [rbp-98h]
  __int64 v108; // [rsp+28h] [rbp-78h] BYREF
  size_t n[14]; // [rsp+30h] [rbp-70h] BYREF

  v6 = a3;
  v9 = 0;
  for ( i = (unsigned __int8 *)a1; ; ++i )
  {
    v11 = *i;
    if ( (_BYTE)v11 == 75 )
    {
      v9 |= 1u;
      continue;
    }
    if ( (_BYTE)v11 != 86 )
      break;
    v9 |= 2u;
LABEL_5:
    ;
  }
  if ( (_BYTE)v11 == 114 )
  {
    v9 |= 4u;
    goto LABEL_5;
  }
  v12 = v9 | a2;
  if ( (_BYTE)v11 == 83 )
  {
    if ( i[1] != 116 )
    {
      v33 = (char *)sub_8EC8F0((_DWORD)i, 1, v12, a3, a4, 0, 0, a6);
      v16 = v95;
      v15 = v33;
      if ( *v33 == 73 )
      {
        v15 = sub_8E9020(v33, a6);
        goto LABEL_16;
      }
      if ( (unsigned __int8 *)a1 == i )
        return v15;
LABEL_20:
      if ( !*(_QWORD *)(a6 + 48) )
        sub_8E5DC0(a1, 3, 0, a5, a6, v16);
      return v15;
    }
    goto LABEL_41;
  }
  if ( (unsigned __int8)(v11 - 67) > 0xFu )
  {
    v19 = 85;
    v20 = (char *)i;
    v21 = "8__vector";
    if ( (_BYTE)v11 == 85 )
    {
      while ( 1 )
      {
        ++v20;
        if ( v19 != (_BYTE)v11 )
          break;
        LOBYTE(v11) = *v21++;
        if ( !(_BYTE)v11 )
          goto LABEL_41;
        v19 = *v20;
      }
      v44 = (char *)n;
      v45 = sub_8E5810(i + 1, (__int64 *)n, a6);
      v49 = v45;
      switch ( n[0] )
      {
        case 8uLL:
          v66 = 95;
          v67 = "_handle";
          while ( *v45++ == v66 )
          {
            v66 = *v67++;
            if ( !v66 )
            {
              v50 = 8;
              v51 = 1;
              v52 = 0;
              v53 = "^";
              goto LABEL_85;
            }
          }
          v70 = v49;
          v71 = 95;
          v72 = "_trkref";
          while ( *v70++ == v71 )
          {
            v71 = *v72++;
            if ( !v71 )
            {
              v50 = 8;
              v51 = 1;
              v52 = 0;
              v53 = "%";
              goto LABEL_85;
            }
          }
          v44 = (char *)v49;
          v46 = 95;
          v73 = "_vector";
          while ( *v44++ == (_BYTE)v46 )
          {
            v46 = (unsigned __int8)*v73++;
            if ( !(_BYTE)v46 )
            {
              if ( *(_QWORD *)(a6 + 32) )
              {
                v50 = 8;
                v51 = 0;
                v52 = 0;
                v53 = 0;
              }
              else
              {
                srcg = v49;
                v53 = 0;
                sub_8E5790("__attribute__((vector_size(?))) ", a6);
                v50 = n[0];
                v49 = srcg;
              }
              goto LABEL_85;
            }
          }
          break;
        case 0xEuLL:
          v46 = 95;
          v44 = "_interior_ptr";
          while ( *v45++ == (_BYTE)v46 )
          {
            v46 = (unsigned __int8)*v44++;
            if ( !(_BYTE)v46 )
            {
              if ( *(_QWORD *)(a6 + 32) )
              {
                v50 = 14;
                v51 = 0;
                v52 = 0;
                v53 = ">";
                goto LABEL_85;
              }
              srcb = v49;
              v76 = a6;
              v77 = "interior_ptr<";
              goto LABEL_177;
            }
          }
          break;
        case 9uLL:
          v46 = 95;
          v44 = "_pin_ptr";
          while ( *v45++ == (_BYTE)v46 )
          {
            v46 = (unsigned __int8)*v44++;
            if ( !(_BYTE)v46 )
            {
              if ( *(_QWORD *)(a6 + 32) )
              {
                v50 = 9;
                v51 = 0;
                v52 = 0;
                v53 = ">";
              }
              else
              {
                srcb = v49;
                v76 = a6;
                v77 = "pin_ptr<";
LABEL_177:
                sub_8E5790((unsigned __int8 *)v77, v76);
                v51 = 0;
                v50 = n[0];
                v49 = srcb;
                v52 = 0;
                v53 = ">";
              }
              goto LABEL_85;
            }
          }
          break;
        case 3uLL:
          v46 = 101;
          v44 = "ut";
          while ( *v45++ == (_BYTE)v46 )
          {
            v46 = (unsigned __int8)*v44++;
            if ( !(_BYTE)v46 )
            {
              if ( *(_QWORD *)(a6 + 32) )
              {
                v50 = 3;
                v51 = 0;
                v52 = 0;
                v53 = ")";
              }
              else
              {
                srci = v49;
                v53 = ")";
                sub_8E5790("__underlying_type(", a6);
                v50 = n[0];
                v49 = srci;
              }
              goto LABEL_85;
            }
          }
          break;
      }
      v50 = n[0];
      srcf = v49;
      v53 = (char *)malloc(n[0] + 1, v44, v46, v49, v47, v48);
      memcpy(v53, srcf, v50);
      v53[v50] = 0;
      v52 = v53;
      v49 = srcf;
      v51 = 1;
LABEL_85:
      v54 = 0;
      srce = v52;
      v55 = sub_8E9FF0(&v49[v50], 0, 1, v51, a5, a6);
      v56 = srce;
      v15 = (char *)v55;
      if ( v53 )
      {
        if ( !*(_QWORD *)(a6 + 32) )
        {
          v54 = a6;
          sub_8E5790((unsigned __int8 *)v53, a6);
        }
        if ( v56 )
          _libc_free(v56, v54);
      }
      goto LABEL_15;
    }
    goto LABEL_24;
  }
  v13 = 45057;
  if ( !_bittest64(&v13, (unsigned int)(v11 - 67)) )
  {
    switch ( (_BYTE)v11 )
    {
      case 'M':
        ++*(_QWORD *)(a6 + 32);
        v68 = sub_8E9FF0(i + 1, 0, 0, 0, 1, a6);
        sub_8EB260(i + 1, 0, 0, a6);
        --*(_QWORD *)(a6 + 32);
        v69 = sub_8E9FF0(v68, 0, 1, 1, a5, a6);
        ++*(_QWORD *)(a6 + 48);
        v15 = (char *)v69;
        if ( !*(_QWORD *)(a6 + 32) )
          sub_8E5790((unsigned __int8 *)" :: ", a6);
        sub_8E9FF0(i + 1, 0, 0, 0, 1, a6);
        sub_8EB260(i + 1, 0, 0, a6);
        --*(_QWORD *)(a6 + 48);
        if ( !*(_QWORD *)(a6 + 32) )
          sub_8E5790((unsigned __int8 *)"::*", a6);
        goto LABEL_15;
      case 'F':
        v37 = i[1];
        v38 = i;
LABEL_64:
        srca = v6;
        v39 = sub_8E9FF0(&v38[(v37 == 89) + 1], 0, 0, 1, a5, a6);
        v15 = (char *)sub_8EBA20(v39, 1, 0, a6);
        v40 = *v15;
        if ( *v15 == 82 || v40 == 79 )
          v40 = *++v15;
        if ( v40 == 69 )
        {
          ++v15;
        }
        else if ( !*(_DWORD *)(a6 + 24) )
        {
          ++*(_QWORD *)(a6 + 32);
          ++*(_QWORD *)(a6 + 48);
          *(_DWORD *)(a6 + 24) = 1;
        }
        if ( srca && !*(_QWORD *)(a6 + 32) )
        {
          v41 = *(_QWORD *)(a6 + 8);
          v42 = v41 + 1;
          if ( !*(_DWORD *)(a6 + 28) )
          {
            v43 = *(_QWORD *)(a6 + 16);
            if ( v43 > v42 )
            {
              *(_BYTE *)(*(_QWORD *)a6 + v41) = 40;
              v42 = *(_QWORD *)(a6 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a6 + 28) = 1;
              if ( v43 )
              {
                *(_BYTE *)(*(_QWORD *)a6 + v43 - 1) = 0;
                v42 = *(_QWORD *)(a6 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a6 + 8) = v42;
        }
        if ( *(_QWORD *)(a6 + 48) )
          return v15;
        v17 = 0;
LABEL_18:
        sub_8E5DC0((__int64)i, 3, 0, a5, a6, v16);
        goto LABEL_19;
      case 'D':
        v37 = i[1];
        if ( (v37 & 0xDF) == 0x4F )
        {
          if ( v37 == 79 )
          {
            ++*(_QWORD *)(a6 + 32);
            srch = v6;
            v38 = sub_8E74B0(i + 2, a6);
            v75 = *(_QWORD *)(a6 + 32);
            v6 = srch;
            *(_QWORD *)(a6 + 32) = v75 - 1;
            if ( *v38 == 69 )
            {
              v37 = v38[2];
              ++v38;
            }
            else
            {
              if ( !*(_DWORD *)(a6 + 24) )
              {
                ++*(_QWORD *)(a6 + 48);
                *(_DWORD *)(a6 + 24) = 1;
                *(_QWORD *)(a6 + 32) = v75;
              }
              v37 = v38[1];
            }
          }
          else if ( v37 == 111 )
          {
            v37 = i[3];
            v38 = i + 2;
          }
          else
          {
            v38 = i;
            if ( !*(_DWORD *)(a6 + 24) )
            {
              ++*(_QWORD *)(a6 + 32);
              ++*(_QWORD *)(a6 + 48);
              *(_DWORD *)(a6 + 24) = 1;
              v37 = i[1];
            }
          }
          goto LABEL_64;
        }
LABEL_41:
        sub_8E6E80(v12, 1, a6);
        v29 = 0;
        v30 = 85;
        do
        {
          if ( i[v29] != v30 )
          {
            v31 = i;
            goto LABEL_46;
          }
          v30 = aU8Vector[++v29];
        }
        while ( v30 );
        v31 = i + 10;
        if ( !*(_QWORD *)(a6 + 32) )
          sub_8E5790("__attribute__((vector_size(?))) ", a6);
LABEL_46:
        v32 = *v31;
        if ( islower(*v31) )
        {
          if ( v32 != 114 )
          {
            v15 = (char *)(v31 + 1);
            switch ( v32 )
            {
              case 'D':
                LOBYTE(v74) = v31[1];
                goto LABEL_168;
              case 'a':
                v59 = "signed char";
                goto LABEL_114;
              case 'b':
                v59 = "bool";
                goto LABEL_114;
              case 'c':
                v59 = "char";
                goto LABEL_114;
              case 'd':
                v59 = "double";
                goto LABEL_114;
              case 'e':
                v59 = "long double";
                goto LABEL_114;
              case 'f':
                v59 = "float";
                goto LABEL_114;
              case 'g':
                v59 = "__float128";
                goto LABEL_114;
              case 'h':
                v59 = "unsigned char";
                goto LABEL_114;
              case 'i':
                v59 = "int";
                goto LABEL_114;
              case 'j':
                v59 = "unsigned int";
                qword_4F605D0 = (__int64)"u";
                goto LABEL_114;
              case 'l':
                v59 = "long";
                qword_4F605D0 = (__int64)"l";
                goto LABEL_114;
              case 'm':
                v59 = "unsigned long";
                qword_4F605D0 = (__int64)"ul";
                goto LABEL_114;
              case 'n':
                v59 = "__int128";
                goto LABEL_114;
              case 'o':
                v59 = "unsigned __int128";
                goto LABEL_114;
              case 's':
                v59 = "short";
                goto LABEL_114;
              case 't':
                v59 = "unsigned short";
                goto LABEL_114;
              case 'u':
                v91 = v31 + 1;
                v59 = (char *)byte_3F871B3;
                v92 = sub_8E72C0(v91, 0, a6);
                v15 = (char *)v92;
                if ( *v92 == 73 )
                  v15 = sub_8E9020(v92, a6);
                goto LABEL_114;
              case 'v':
                v59 = "void";
                goto LABEL_114;
              case 'w':
                v59 = "wchar_t";
                goto LABEL_114;
              case 'x':
                v59 = "long long";
                qword_4F605D0 = (__int64)"ll";
                goto LABEL_114;
              case 'y':
                v59 = "unsigned long long";
                qword_4F605D0 = (__int64)"ull";
                goto LABEL_114;
              default:
                v59 = (char *)byte_3F871B3;
                if ( !*(_DWORD *)(a6 + 24) )
                {
                  ++*(_QWORD *)(a6 + 32);
                  ++*(_QWORD *)(a6 + 48);
                  *(_DWORD *)(a6 + 24) = 1;
                }
                goto LABEL_114;
            }
          }
          goto LABEL_162;
        }
        if ( v32 != 68 )
        {
          if ( v32 != 84 )
          {
LABEL_162:
            if ( !*(_QWORD *)(a6 + 32) )
              sub_8E5790(" ::", a6);
            v15 = sub_8E9510(v31, (__int64)n, 3, a6);
            goto LABEL_116;
          }
          v15 = (char *)sub_8E5C30((__int64)v31, a6);
          if ( *v15 == 73 && a5 )
          {
            if ( !*(_QWORD *)(a6 + 48) )
              sub_8E5DC0((__int64)v31, 4, 0, 0, a6, v78);
            v15 = sub_8E9020(v15, a6);
          }
LABEL_116:
          if ( a4 && !*(_QWORD *)(a6 + 32) )
          {
            v60 = *(_QWORD *)(a6 + 8);
            v61 = v60 + 1;
            if ( !*(_DWORD *)(a6 + 28) )
            {
              v62 = *(_QWORD *)(a6 + 16);
              if ( v62 > v61 )
              {
                *(_BYTE *)(*(_QWORD *)a6 + v60) = 32;
                v61 = *(_QWORD *)(a6 + 8) + 1LL;
              }
              else
              {
                *(_DWORD *)(a6 + 28) = 1;
                if ( v62 )
                {
                  *(_BYTE *)(*(_QWORD *)a6 + v62 - 1) = 0;
                  v61 = *(_QWORD *)(a6 + 8) + 1LL;
                }
              }
            }
            *(_QWORD *)(a6 + 8) = v61;
          }
          v63 = *i;
          v17 = a1 != (_QWORD)i;
          if ( islower(*i) )
          {
            if ( v63 == 117 || v63 == 114 )
              goto LABEL_16;
            if ( v63 != 68 )
            {
LABEL_19:
              if ( !v17 )
                return v15;
              goto LABEL_20;
            }
            LOBYTE(v64) = i[1];
          }
          else
          {
            if ( v63 != 68 )
              goto LABEL_16;
            v64 = i[1];
            if ( (unsigned __int8)(v64 - 84) > 0x25u )
              goto LABEL_19;
            v65 = 0x2110000021LL;
            if ( _bittest64(&v65, (unsigned int)(v64 - 84)) )
              goto LABEL_16;
          }
          if ( (_BYTE)v64 == 118 )
            goto LABEL_16;
          goto LABEL_19;
        }
        v74 = v31[1];
        if ( (unsigned __int8)(v74 - 84) <= 0x25u )
        {
          v79 = 0x2110000021LL;
          if ( _bittest64(&v79, (unsigned int)(v74 - 84)) )
          {
            if ( (_BYTE)v74 == 112 )
            {
              v88 = v31 + 2;
              v15 = (char *)sub_8E9FF0(v88, 0, 0, 0, 1, a6);
              if ( !*(_QWORD *)(a6 + 32) )
                sub_8E5790((unsigned __int8 *)"...", a6);
              sub_8EB260(v88, 0, 0, a6);
              goto LABEL_116;
            }
            v80 = *(_QWORD *)(a6 + 32);
            if ( (v74 & 0xDF) != 0x54 )
            {
              if ( (v74 & 0xDF) != 0x59 )
                goto LABEL_162;
              if ( !v80 )
              {
                sub_8E5790("typeof(", a6);
                LOBYTE(v74) = v31[1];
              }
              v87 = v31 + 2;
              if ( (_BYTE)v74 == 121 )
              {
                v15 = (char *)sub_8E9FF0(v87, 0, 0, 0, 1, a6);
                sub_8EB260(v87, 0, 0, a6);
              }
              else
              {
                v15 = (char *)sub_8E74B0(v87, a6);
              }
              goto LABEL_203;
            }
            if ( v80 )
            {
              if ( (_BYTE)v74 != 116 )
              {
LABEL_197:
                v15 = (char *)sub_8E74B0(v31 + 2, a6);
                if ( *(_QWORD *)(a6 + 32) )
                {
LABEL_209:
                  if ( *v15 == 69 )
                  {
                    ++v15;
                  }
                  else if ( !*(_DWORD *)(a6 + 24) )
                  {
                    ++*(_QWORD *)(a6 + 32);
                    ++*(_QWORD *)(a6 + 48);
                    *(_DWORD *)(a6 + 24) = 1;
                  }
                  goto LABEL_116;
                }
                if ( !*(_DWORD *)(a6 + 28) )
                {
                  v83 = *(_QWORD *)(a6 + 8);
                  v84 = *(_QWORD *)(a6 + 16);
                  if ( v84 > v83 + 1 )
                  {
                    *(_BYTE *)(*(_QWORD *)a6 + v83) = 41;
                  }
                  else
                  {
                    *(_DWORD *)(a6 + 28) = 1;
                    if ( v84 )
                      *(_BYTE *)(*(_QWORD *)a6 + v84 - 1) = 0;
                  }
                }
                ++*(_QWORD *)(a6 + 8);
LABEL_203:
                if ( !*(_QWORD *)(a6 + 32) )
                {
                  if ( !*(_DWORD *)(a6 + 28) )
                  {
                    v85 = *(_QWORD *)(a6 + 8);
                    v86 = *(_QWORD *)(a6 + 16);
                    if ( v86 > v85 + 1 )
                    {
                      *(_BYTE *)(*(_QWORD *)a6 + v85) = 41;
                    }
                    else
                    {
                      *(_DWORD *)(a6 + 28) = 1;
                      if ( v86 )
                        *(_BYTE *)(*(_QWORD *)a6 + v86 - 1) = 0;
                    }
                  }
                  ++*(_QWORD *)(a6 + 8);
                }
                goto LABEL_209;
              }
            }
            else
            {
              sub_8E5790((unsigned __int8 *)"decltype(", a6);
              if ( v31[1] != 116 )
              {
                if ( !*(_QWORD *)(a6 + 32) )
                {
                  if ( !*(_DWORD *)(a6 + 28) )
                  {
                    v81 = *(_QWORD *)(a6 + 8);
                    v82 = *(_QWORD *)(a6 + 16);
                    if ( v82 > v81 + 1 )
                    {
                      *(_BYTE *)(*(_QWORD *)a6 + v81) = 40;
                    }
                    else
                    {
                      *(_DWORD *)(a6 + 28) = 1;
                      if ( v82 )
                        *(_BYTE *)(*(_QWORD *)a6 + v82 - 1) = 0;
                    }
                  }
                  ++*(_QWORD *)(a6 + 8);
                }
                goto LABEL_197;
              }
            }
            v15 = (char *)sub_8E74B0(v31 + 2, a6);
            goto LABEL_203;
          }
        }
LABEL_168:
        v15 = (char *)(v31 + 2);
        switch ( (char)v74 )
        {
          case 'F':
            v89 = sub_8E5810(v31 + 2, &v108, a6);
            v90 = *v89;
            if ( *v89 == 98 )
            {
              v59 = "std::bfloat16_t";
              if ( v108 == 16 )
                goto LABEL_242;
            }
            else
            {
              if ( v90 == 120 )
              {
                v59 = "_Float32x";
                if ( v108 == 32 || (v59 = "_Float64x", v108 == 64) )
                {
LABEL_242:
                  v15 = (char *)(v89 + 1);
LABEL_114:
                  if ( !*(_QWORD *)(a6 + 32) )
                    sub_8E5790((unsigned __int8 *)v59, a6);
                  goto LABEL_116;
                }
                goto LABEL_248;
              }
              if ( v90 == 95 )
              {
                v59 = "_Float64";
                if ( v108 == 64 )
                  goto LABEL_242;
                if ( v108 > 64 )
                {
                  v59 = "_Float128";
                  if ( v108 == 128 )
                    goto LABEL_242;
                }
                else
                {
                  v59 = "_Float16";
                  if ( v108 == 16 )
                    goto LABEL_242;
                  v59 = "_Float32";
                  if ( v108 == 32 )
                    goto LABEL_242;
                }
LABEL_248:
                v59 = (char *)byte_3F871B3;
                if ( *(_DWORD *)(a6 + 24) )
                  goto LABEL_242;
                goto LABEL_249;
              }
            }
            if ( *(_DWORD *)(a6 + 24) )
            {
              v59 = (char *)byte_3F871B3;
              goto LABEL_242;
            }
LABEL_249:
            v59 = (char *)byte_3F871B3;
            sub_8E5770(a6);
            goto LABEL_242;
          case 'N':
            v59 = "__nullptr";
            goto LABEL_114;
          case 'a':
            v59 = "auto";
            goto LABEL_114;
          case 'c':
            v59 = (char *)"decltype(auto)";
            goto LABEL_114;
          case 'h':
            v59 = "__fp16";
            goto LABEL_114;
          case 'i':
            v59 = "char32_t";
            goto LABEL_114;
          case 'n':
            v59 = "::std::nullptr_t";
            goto LABEL_114;
          case 's':
            v59 = "char16_t";
            goto LABEL_114;
          case 'u':
            v59 = "char8_t";
            goto LABEL_114;
          case 'v':
            v93 = sub_8E5810(v31 + 2, &v108, a6);
            v15 = (char *)v93;
            if ( *v93 != 95 )
              goto LABEL_226;
            v94 = v93 + 1;
            v15 = (char *)sub_8EB9C0(v93 + 1, 1, 0, a6);
            if ( !*(_QWORD *)(a6 + 32) )
              sub_8E5790(" __attribute((vector_size(", a6);
            sprintf((char *)n, "%ld", v108);
            if ( !*(_QWORD *)(a6 + 32) )
              sub_8E5790((unsigned __int8 *)n, a6);
            if ( !*(_QWORD *)(a6 + 32) )
              sub_8E5790("*sizeof(", a6);
            ++*(_QWORD *)(a6 + 48);
            sub_8EB9C0(v94, 1, 0, a6);
            --*(_QWORD *)(a6 + 48);
            if ( *(_QWORD *)(a6 + 32) )
              goto LABEL_116;
            v59 = (char *)byte_3F871B3;
            sub_8E5790(")))) ", a6);
            goto LABEL_114;
          default:
LABEL_226:
            v59 = (char *)byte_3F871B3;
            if ( !*(_DWORD *)(a6 + 24) )
              sub_8E5770(a6);
            goto LABEL_114;
        }
    }
LABEL_24:
    v22 = i + 1;
    if ( (_BYTE)v11 != 65 )
      goto LABEL_41;
    v23 = i[1];
    if ( (unsigned int)(v23 - 48) <= 9 )
    {
      do
      {
        v58 = *++v22;
        v25 = v58;
      }
      while ( (unsigned int)(v58 - 48) <= 9 );
    }
    else
    {
      if ( (_BYTE)v23 == 95 )
        goto LABEL_166;
      ++*(_QWORD *)(a6 + 32);
      srcc = v6;
      v24 = sub_8E74B0(v22, a6);
      --*(_QWORD *)(a6 + 32);
      v6 = srcc;
      v25 = *v24;
      v22 = v24;
    }
    if ( v25 != 95 )
    {
      if ( !*(_DWORD *)(a6 + 24) )
      {
        ++*(_QWORD *)(a6 + 32);
        ++*(_QWORD *)(a6 + 48);
        *(_DWORD *)(a6 + 24) = 1;
      }
      goto LABEL_31;
    }
LABEL_166:
    ++v22;
LABEL_31:
    srcd = v6;
    v15 = (char *)sub_8E9FF0(v22, 0, 0, 1, a5, a6);
    if ( srcd && !*(_QWORD *)(a6 + 32) )
    {
      v26 = *(_QWORD *)(a6 + 8);
      v27 = v26 + 1;
      if ( !*(_DWORD *)(a6 + 28) )
      {
        v28 = *(_QWORD *)(a6 + 16);
        if ( v28 > v27 )
        {
          *(_BYTE *)(*(_QWORD *)a6 + v26) = 40;
          v27 = *(_QWORD *)(a6 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a6 + 28) = 1;
          if ( v28 )
          {
            *(_BYTE *)(*(_QWORD *)a6 + v28 - 1) = 0;
            v27 = *(_QWORD *)(a6 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a6 + 8) = v27;
    }
    goto LABEL_16;
  }
  v14 = i + 1;
  if ( (_BYTE)v11 == 67 )
  {
    if ( !*(_QWORD *)(a6 + 32) )
      sub_8E5790("_Complex ", a6);
    v15 = (char *)sub_8E9FF0(v14, 0, 1, 1, a5, a6);
  }
  else
  {
    v15 = (char *)sub_8E9FF0(i + 1, 0, 1, 1, a5, a6);
    if ( (_BYTE)v11 == 80 )
    {
      if ( *(_QWORD *)(a6 + 32) )
        goto LABEL_15;
      v34 = *(_QWORD *)(a6 + 8);
      v35 = v34 + 1;
      if ( !*(_DWORD *)(a6 + 28) )
      {
        v36 = *(_QWORD *)(a6 + 16);
        if ( v36 > v35 )
        {
          *(_BYTE *)(*(_QWORD *)a6 + v34) = 42;
          v35 = *(_QWORD *)(a6 + 8) + 1LL;
        }
        else
        {
LABEL_58:
          *(_DWORD *)(a6 + 28) = 1;
          if ( v36 )
          {
            *(_BYTE *)(*(_QWORD *)a6 + v36 - 1) = 0;
            v35 = *(_QWORD *)(a6 + 8) + 1LL;
          }
        }
      }
      goto LABEL_60;
    }
    if ( (_BYTE)v11 == 82 )
    {
      if ( !*(_QWORD *)(a6 + 32) )
      {
        v57 = *(_QWORD *)(a6 + 8);
        v35 = v57 + 1;
        if ( !*(_DWORD *)(a6 + 28) )
        {
          v36 = *(_QWORD *)(a6 + 16);
          if ( v36 <= v35 )
            goto LABEL_58;
          *(_BYTE *)(*(_QWORD *)a6 + v57) = 38;
          v35 = *(_QWORD *)(a6 + 8) + 1LL;
        }
LABEL_60:
        *(_QWORD *)(a6 + 8) = v35;
      }
    }
    else if ( (_BYTE)v11 == 79 && !*(_QWORD *)(a6 + 32) )
    {
      sub_8E5790((unsigned __int8 *)"&&", a6);
    }
  }
LABEL_15:
  sub_8E6E80(v12, 1, a6);
LABEL_16:
  if ( !*(_QWORD *)(a6 + 48) )
  {
    v17 = a1 != (_QWORD)i;
    goto LABEL_18;
  }
  return v15;
}
