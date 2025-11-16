// Function: sub_16EAF00
// Address: 0x16eaf00
//
unsigned __int64 __fastcall sub_16EAF00(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v8; // rcx
  unsigned __int64 result; // rax
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rdx
  int v14; // edx
  int v15; // r8d
  _BYTE *v16; // rax
  int v17; // ecx
  _BYTE *v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  signed __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  signed __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  signed __int64 v26; // rcx
  signed __int64 v27; // rdx
  signed __int64 v28; // rsi
  __int64 v29; // rdx
  signed __int64 v30; // rcx
  signed __int64 v31; // rdx
  signed __int64 v32; // rsi
  __int64 v33; // rdx
  _BYTE *v34; // rdx
  signed __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  signed __int64 v38; // rsi
  __int64 v39; // rax
  signed __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  signed __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // ecx
  int v47; // esi
  __int64 v48; // rdx
  signed __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  signed __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rbx
  signed __int64 v55; // rdx
  signed __int64 v56; // rsi
  __int64 v57; // rdx
  signed __int64 v58; // rax
  signed __int64 v59; // rdx
  __int64 v60; // rcx
  signed __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rax
  signed __int64 v65; // rcx
  signed __int64 v66; // rsi
  __int64 v67; // rcx
  __int64 v68; // rax
  signed __int64 v69; // rcx
  signed __int64 v70; // rsi
  __int64 v71; // rcx
  __int64 v72; // rax
  signed __int64 v73; // rcx
  signed __int64 v74; // rsi
  __int64 v75; // rcx
  __int64 v76; // rax
  signed __int64 v77; // rcx
  signed __int64 v78; // rsi
  __int64 v79; // rcx
  signed __int64 v80; // rax
  signed __int64 v81; // rdx
  __int64 v82; // rcx
  signed __int64 v83; // rsi
  __int64 v84; // rdx
  __int64 *v85; // rdx
  signed __int64 v86; // rax
  signed __int64 v87; // rdx
  __int64 v88; // rcx
  signed __int64 v89; // rsi
  __int64 v90; // rdx
  __int64 v91; // rdi
  unsigned __int64 v92; // rdx
  signed __int64 v93; // rsi
  __int64 v94; // rsi
  int v95; // r9d
  signed __int64 v96; // rdx
  signed __int64 v97; // rdi
  int v98; // ecx
  signed __int64 v99; // rsi
  __int64 v100; // rsi
  int v101; // eax
  __int64 v102; // [rsp+0h] [rbp-70h]
  unsigned __int64 v103; // [rsp+0h] [rbp-70h]
  __int64 v104; // [rsp+8h] [rbp-68h]
  __int64 v105; // [rsp+8h] [rbp-68h]
  __int64 v106; // [rsp+8h] [rbp-68h]
  int v107; // [rsp+8h] [rbp-68h]
  __int64 v108; // [rsp+8h] [rbp-68h]
  __int64 v109; // [rsp+8h] [rbp-68h]
  __int64 v110; // [rsp+8h] [rbp-68h]
  __int64 v111; // [rsp+8h] [rbp-68h]
  int v112; // [rsp+8h] [rbp-68h]
  int v113; // [rsp+8h] [rbp-68h]
  char *v114; // [rsp+8h] [rbp-68h]
  int v115; // [rsp+8h] [rbp-68h]
  __int64 v116; // [rsp+10h] [rbp-60h]
  __int64 v117; // [rsp+18h] [rbp-58h]
  int v118; // [rsp+24h] [rbp-4Ch]
  __int64 v119; // [rsp+28h] [rbp-48h]
  _DWORD v120[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v8 = *(_QWORD *)(a1 + 8);
  v118 = 1;
  v116 = 0;
  v117 = 0;
  v119 = *(_QWORD *)(a1 + 40);
  result = *(_QWORD *)a1;
  while ( 1 )
  {
    if ( result >= v8 )
      goto LABEL_50;
    v10 = *(char *)result;
    v11 = v119;
LABEL_4:
    if ( (_BYTE)v10 != 124 && v10 != a2 )
      break;
LABEL_118:
    if ( v11 == v119 )
      goto LABEL_50;
    result = *(_QWORD *)a1;
    v8 = *(_QWORD *)(a1 + 8);
    if ( *(_QWORD *)a1 >= v8 || *(_BYTE *)result != 124 )
      goto LABEL_198;
    *(_QWORD *)a1 = ++result;
    if ( v118 )
    {
      if ( *(_DWORD *)(a1 + 16) )
        goto LABEL_175;
      sub_16E9180((_QWORD *)a1, 2013265920, v11 - v119 + 1, v119, a5, a6);
      v116 = v119;
      v11 = *(_QWORD *)(a1 + 40);
      v117 = v119;
    }
    a6 = *(_DWORD *)(a1 + 16);
    v54 = v11 - v117;
    if ( a6 )
    {
      result = *(_QWORD *)a1;
      v8 = *(_QWORD *)(a1 + 8);
LABEL_175:
      v117 = v11 - 1;
LABEL_176:
      v119 = v11;
      v116 = v11;
      goto LABEL_134;
    }
    v55 = *(_QWORD *)(a1 + 32);
    if ( v11 >= v55 )
    {
      v56 = (v55 + 1) / 2 + ((v55 + 1 + ((unsigned __int64)(v55 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
      if ( v55 < v56 )
      {
        sub_16E90A0(a1, v56, v55, v8, a5, 0);
        v11 = *(_QWORD *)(a1 + 40);
      }
    }
    v57 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 40) = v11 + 1;
    *(_QWORD *)(v57 + 8 * v11) = v54 | 0x80000000LL;
    v11 = *(_QWORD *)(a1 + 40);
    LODWORD(a5) = *(_DWORD *)(a1 + 16);
    v117 = v11 - 1;
    if ( (_DWORD)a5 )
    {
      result = *(_QWORD *)a1;
      v8 = *(_QWORD *)(a1 + 8);
      goto LABEL_176;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v116) = (*(_QWORD *)(a1 + 40) - v116)
                                                 | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v116) & 0xF8000000LL;
    v58 = *(_QWORD *)(a1 + 40);
    v116 = v58;
    if ( *(_DWORD *)(a1 + 16) )
    {
      result = *(_QWORD *)a1;
      v8 = *(_QWORD *)(a1 + 8);
      v119 = *(_QWORD *)(a1 + 40);
    }
    else
    {
      v59 = *(_QWORD *)(a1 + 32);
      if ( v58 >= v59 )
      {
        v60 = (v59 + 1) / 2;
        v61 = v60 + ((v59 + 1 + ((unsigned __int64)(v59 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
        if ( v59 < v61 )
        {
          sub_16E90A0(a1, v61, v59, v60, 0, a6);
          v58 = *(_QWORD *)(a1 + 40);
        }
      }
      v62 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v58 + 1;
      *(_QWORD *)(v62 + 8 * v58) = 2281701376LL;
      v8 = *(_QWORD *)(a1 + 8);
      v119 = *(_QWORD *)(a1 + 40);
      result = *(_QWORD *)a1;
    }
LABEL_134:
    v118 = 0;
  }
  while ( 2 )
  {
    a5 = result + 1;
    *(_QWORD *)a1 = result + 1;
    v12 = (unsigned int)*(char *)result;
    v13 = (unsigned __int8)(*(_BYTE *)result - 36);
    switch ( *(_BYTE *)result )
    {
      case '$':
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v35 = *(_QWORD *)(a1 + 32);
          v36 = v11;
          if ( v11 >= v35 )
          {
            v37 = (v35 + 1) / 2;
            v38 = v37 + ((v35 + 1 + ((unsigned __int64)(v35 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v35 < v38 )
            {
              sub_16E90A0(a1, v38, v11, v37, a5, a6);
              v36 = *(_QWORD *)(a1 + 40);
            }
          }
          v39 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v36 + 1;
          *(_QWORD *)(v39 + 8 * v36) = 0x20000000;
        }
        a6 = 0;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 2u;
        ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 80LL);
        result = *(_QWORD *)a1;
        v8 = *(_QWORD *)(a1 + 8);
        goto LABEL_14;
      case '(':
        if ( a5 >= v8 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 8;
          *(_QWORD *)a1 = &unk_4FA17D0;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
        }
        v25 = ++*(_QWORD *)(*(_QWORD *)(a1 + 56) + 112LL);
        if ( v25 <= 9 )
          *(_QWORD *)(a1 + 8 * v25 + 64) = *(_QWORD *)(a1 + 40);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v26 = *(_QWORD *)(a1 + 40);
          v27 = *(_QWORD *)(a1 + 32);
          if ( v26 >= v27 )
          {
            v28 = (v27 + 1) / 2 + ((v27 + 1 + ((unsigned __int64)(v27 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v27 < v28 )
            {
              v104 = v25;
              sub_16E90A0(a1, v28, v27, v26, a5, a6);
              v26 = *(_QWORD *)(a1 + 40);
              v25 = v104;
            }
          }
          v29 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v26 + 1;
          *(_QWORD *)(v29 + 8 * v26) = v25 | 0x68000000;
        }
        if ( *(_QWORD *)a1 >= *(_QWORD *)(a1 + 8) || **(_BYTE **)a1 != 41 )
        {
          v105 = v25;
          sub_16EAF00(a1, 41);
          v25 = v105;
        }
        if ( v25 <= 9 )
          *(_QWORD *)(a1 + 8 * v25 + 144) = *(_QWORD *)(a1 + 40);
        a6 = *(_DWORD *)(a1 + 16);
        if ( a6 )
        {
          v34 = *(_BYTE **)a1;
          if ( *(_QWORD *)(a1 + 8) <= *(_QWORD *)a1 )
            goto LABEL_29;
        }
        else
        {
          v30 = *(_QWORD *)(a1 + 40);
          v31 = *(_QWORD *)(a1 + 32);
          if ( v30 >= v31 )
          {
            v32 = (v31 + 1) / 2 + ((v31 + 1 + ((unsigned __int64)(v31 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v31 < v32 )
            {
              v106 = v25;
              sub_16E90A0(a1, v32, v31, v30, a5, 0);
              v30 = *(_QWORD *)(a1 + 40);
              v25 = v106;
            }
          }
          v33 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v30 + 1;
          *(_QWORD *)(v33 + 8 * v30) = v25 | 0x70000000;
          v34 = *(_BYTE **)a1;
          if ( *(_QWORD *)a1 >= *(_QWORD *)(a1 + 8) )
          {
LABEL_83:
            LODWORD(a5) = *(_DWORD *)(a1 + 16);
            if ( !(_DWORD)a5 )
              *(_DWORD *)(a1 + 16) = 8;
            goto LABEL_29;
          }
        }
        result = (unsigned __int64)(v34 + 1);
        *(_QWORD *)a1 = v34 + 1;
        if ( *v34 != 41 )
          goto LABEL_83;
LABEL_13:
        v8 = *(_QWORD *)(a1 + 8);
        a6 = 0;
        goto LABEL_14;
      case ')':
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 8;
        goto LABEL_48;
      case '*':
      case '+':
      case '?':
        goto LABEL_46;
      case '.':
        if ( (*(_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL) & 8) != 0 )
        {
          v103 = v8;
          *(_QWORD *)a1 = v120;
          *(_QWORD *)(a1 + 8) = (char *)v120 + 3;
          v114 = (char *)(result + 1);
          v120[0] = 6097502;
          sub_16E97A0(a1, v12, v13, v8, a5, a6);
          LODWORD(a5) = (_DWORD)v114;
          v8 = v103;
          a6 = 0;
          *(_QWORD *)a1 = v114;
          result = (unsigned __int64)v114;
          *(_QWORD *)(a1 + 8) = v103;
        }
        else
        {
          a6 = *(_DWORD *)(a1 + 16);
          if ( a6 )
          {
            v8 = *(_QWORD *)(a1 + 8);
            ++result;
            a6 = 0;
          }
          else
          {
            v49 = *(_QWORD *)(a1 + 32);
            v50 = v11;
            if ( v11 >= v49 )
            {
              v51 = (v49 + 1) / 2;
              v52 = v51 + ((v49 + 1 + ((unsigned __int64)(v49 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v49 < v52 )
              {
                v107 = *(_DWORD *)(a1 + 16);
                sub_16E90A0(a1, v52, v49, v51, a5, 0);
                v50 = *(_QWORD *)(a1 + 40);
                a6 = v107;
              }
            }
            v53 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = v50 + 1;
            *(_QWORD *)(v53 + 8 * v50) = 671088640;
            result = *(_QWORD *)a1;
            v8 = *(_QWORD *)(a1 + 8);
          }
        }
        goto LABEL_14;
      case '[':
        sub_16E97A0(a1, v12, v13, v8, a5, a6);
        result = *(_QWORD *)a1;
        v8 = *(_QWORD *)(a1 + 8);
        a6 = 0;
        goto LABEL_14;
      case '\\':
        if ( a5 >= v8 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 5;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
          a5 = (__int64)&unk_4FA17D0;
        }
        *(_QWORD *)a1 = a5 + 1;
        v12 = (unsigned int)*(char *)a5;
        if ( (unsigned __int8)(*(_BYTE *)a5 - 49) > 8u )
          goto LABEL_12;
        v45 = (char)(*(_BYTE *)a5 - 48);
        v46 = (char)(*(_BYTE *)a5 - 48);
        v47 = *(_DWORD *)(a1 + 16);
        v48 = *(_QWORD *)(a1 + 8 * v45 + 144);
        if ( !v48 )
        {
          if ( !v47 )
            *(_DWORD *)(a1 + 16) = 6;
          goto LABEL_48;
        }
        if ( !v47 )
        {
          a5 = *(_QWORD *)(a1 + 32);
          v91 = v11;
          if ( v11 >= a5 )
          {
            v92 = (a5 + 1 + ((unsigned __int64)(a5 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL;
            v93 = v92 + (a5 + 1) / 2;
            if ( a5 < v93 )
            {
              v102 = v45;
              v112 = v45;
              sub_16E90A0(a1, v93, v92, v46, a5, a6);
              v91 = *(_QWORD *)(a1 + 40);
              v45 = v102;
              v46 = v112;
            }
          }
          v94 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v91 + 1;
          *(_QWORD *)(v94 + 8 * v91) = v46 | 0x38000000;
          v48 = *(_QWORD *)(a1 + 8 * v45 + 144);
        }
        v113 = v46;
        sub_16E9110((_QWORD *)a1, *(_QWORD *)(a1 + 8 * v45 + 64) + 1LL, v48, v46, a5, a6);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v96 = *(_QWORD *)(a1 + 40);
          v97 = *(_QWORD *)(a1 + 32);
          v98 = v113;
          if ( v96 >= v97 )
          {
            v99 = ((v97 + 1 + ((unsigned __int64)(v97 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v97 + 1) / 2;
            if ( v97 < v99 )
            {
              sub_16E90A0(a1, v99, v96, v113, a5, v95);
              v96 = *(_QWORD *)(a1 + 40);
              v98 = v113;
            }
          }
          v100 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v96 + 1;
          *(_QWORD *)(v100 + 8 * v96) = v98 | 0x40000000;
        }
        a6 = 0;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 120LL) = 1;
        result = *(_QWORD *)a1;
        v8 = *(_QWORD *)(a1 + 8);
LABEL_14:
        if ( result >= v8 )
        {
          v11 = *(_QWORD *)(a1 + 40);
          goto LABEL_49;
        }
        v10 = *(char *)result;
        if ( (unsigned __int8)(*(_BYTE *)result - 42) <= 1u || (_BYTE)v10 == 63 )
        {
          *(_QWORD *)a1 = result + 1;
          if ( !a6 )
            goto LABEL_34;
          goto LABEL_31;
        }
        if ( (_BYTE)v10 != 123 )
        {
          v11 = *(_QWORD *)(a1 + 40);
          goto LABEL_4;
        }
        if ( result + 1 >= v8 || (unsigned int)*(unsigned __int8 *)(result + 1) - 48 > 9 )
        {
          v11 = *(_QWORD *)(a1 + 40);
          goto LABEL_117;
        }
        *(_QWORD *)a1 = result + 1;
        if ( a6 )
        {
LABEL_31:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 13;
          *(_QWORD *)a1 = &unk_4FA17D0;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
LABEL_34:
          if ( (_BYTE)v10 == 43 )
          {
            LODWORD(a5) = *(_DWORD *)(a1 + 16);
            v63 = *(_QWORD *)(a1 + 40);
            if ( !(_DWORD)a5 )
            {
              sub_16E9180((_QWORD *)a1, 1207959552, v63 - v11 + 1, v11, 0, a6);
              v63 = *(_QWORD *)(a1 + 40);
              v72 = v63 - v11;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v73 = *(_QWORD *)(a1 + 32);
                if ( v63 >= v73 )
                {
                  v74 = (v73 + 1) / 2 + ((v73 + 1 + ((unsigned __int64)(v73 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v73 < v74 )
                  {
                    v110 = *(_QWORD *)(a1 + 40) - v11;
                    sub_16E90A0(a1, v74, v63, v73, a5, a6);
                    v63 = *(_QWORD *)(a1 + 40);
                    v72 = v110;
                  }
                }
                v75 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v63 + 1;
                *(_QWORD *)(v75 + 8 * v63) = v72 | 0x50000000;
                result = *(_QWORD *)a1;
                v8 = *(_QWORD *)(a1 + 8);
                v11 = *(_QWORD *)(a1 + 40);
                goto LABEL_40;
              }
            }
            goto LABEL_158;
          }
          if ( (char)v10 <= 43 )
          {
            if ( (_BYTE)v10 != 42 )
            {
LABEL_136:
              result = *(_QWORD *)a1;
              v8 = *(_QWORD *)(a1 + 8);
              v11 = *(_QWORD *)(a1 + 40);
              goto LABEL_40;
            }
            v63 = *(_QWORD *)(a1 + 40);
            if ( !*(_DWORD *)(a1 + 16) )
            {
              sub_16E9180((_QWORD *)a1, 1207959552, v63 - v11 + 1, v11, a5, a6);
              v63 = *(_QWORD *)(a1 + 40);
              v64 = v63 - v11;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v65 = *(_QWORD *)(a1 + 32);
                if ( v63 >= v65 )
                {
                  v66 = (v65 + 1) / 2 + ((v65 + 1 + ((unsigned __int64)(v65 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v65 < v66 )
                  {
                    v108 = *(_QWORD *)(a1 + 40) - v11;
                    sub_16E90A0(a1, v66, v63, v65, a5, a6);
                    v63 = *(_QWORD *)(a1 + 40);
                    v64 = v108;
                  }
                }
                v67 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v63 + 1;
                *(_QWORD *)(v67 + 8 * v63) = v64 | 0x50000000;
                v63 = *(_QWORD *)(a1 + 40);
                LODWORD(a5) = v63 - v11 + 1;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  sub_16E9180((_QWORD *)a1, 1476395008, *(_QWORD *)(a1 + 40) - v11 + 1, v11, a5, a6);
                  v63 = *(_QWORD *)(a1 + 40);
                  a6 = *(_DWORD *)(a1 + 16);
                  v68 = v63 - v11;
                  if ( !a6 )
                  {
                    v69 = *(_QWORD *)(a1 + 32);
                    if ( v63 >= v69 )
                    {
                      v70 = (v69 + 1) / 2 + ((v69 + 1 + ((unsigned __int64)(v69 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v69 < v70 )
                      {
                        v109 = *(_QWORD *)(a1 + 40) - v11;
                        sub_16E90A0(a1, v70, v63, v69, a5, 0);
                        v63 = *(_QWORD *)(a1 + 40);
                        v68 = v109;
                      }
                    }
                    v71 = *(_QWORD *)(a1 + 24);
                    *(_QWORD *)(a1 + 40) = v63 + 1;
                    *(_QWORD *)(v71 + 8 * v63) = v68 | 0x60000000;
                    result = *(_QWORD *)a1;
                    v8 = *(_QWORD *)(a1 + 8);
                    v11 = *(_QWORD *)(a1 + 40);
                    goto LABEL_40;
                  }
                }
              }
            }
LABEL_158:
            result = *(_QWORD *)a1;
            v8 = *(_QWORD *)(a1 + 8);
            v11 = v63;
            goto LABEL_40;
          }
          if ( (_BYTE)v10 != 63 )
          {
            if ( (_BYTE)v10 == 123 )
              goto LABEL_21;
            goto LABEL_136;
          }
          v19 = *(_QWORD *)(a1 + 40);
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_38;
          sub_16E9180((_QWORD *)a1, 2013265920, v19 - v11 + 1, v11, a5, a6);
          v19 = *(_QWORD *)(a1 + 40);
          v76 = v19 - v11;
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_38;
          v77 = *(_QWORD *)(a1 + 32);
          if ( v19 >= v77 )
          {
            v78 = (v77 + 1) / 2 + ((v77 + 1 + ((unsigned __int64)(v77 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v77 < v78 )
            {
              v111 = *(_QWORD *)(a1 + 40) - v11;
              sub_16E90A0(a1, v78, v19, v77, a5, a6);
              v19 = *(_QWORD *)(a1 + 40);
              v76 = v111;
            }
          }
          v79 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v19 + 1;
          *(_QWORD *)(v79 + 8 * v19) = v76 | 0x80000000LL;
          v19 = *(_QWORD *)(a1 + 40);
          if ( *(_DWORD *)(a1 + 16) )
          {
LABEL_38:
            v11 = v19;
          }
          else
          {
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v11) = (*(_QWORD *)(a1 + 40) - v11)
                                                        | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v11) & 0xF8000000LL;
            if ( *(_DWORD *)(a1 + 16) )
            {
              v11 = *(_QWORD *)(a1 + 40);
            }
            else
            {
              v80 = *(_QWORD *)(a1 + 40);
              v81 = *(_QWORD *)(a1 + 32);
              if ( v80 >= v81 )
              {
                v82 = (v81 + 1) / 2;
                v83 = v82 + ((v81 + 1 + ((unsigned __int64)(v81 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                if ( v81 < v83 )
                {
                  sub_16E90A0(a1, v83, v81, v82, a5, a6);
                  v80 = *(_QWORD *)(a1 + 40);
                }
              }
              v84 = *(_QWORD *)(a1 + 24);
              *(_QWORD *)(a1 + 40) = v80 + 1;
              *(_QWORD *)(v84 + 8 * v80) = 2281701376LL;
              v11 = *(_QWORD *)(a1 + 40);
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v85 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * (v11 - 1));
                *v85 = *v85 & 0xF8000000LL | 1;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  v86 = *(_QWORD *)(a1 + 40);
                  v87 = *(_QWORD *)(a1 + 32);
                  if ( v86 >= v87 )
                  {
                    v88 = (v87 + 1) / 2;
                    v89 = v88 + ((v87 + 1 + ((unsigned __int64)(v87 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                    if ( v87 < v89 )
                    {
                      sub_16E90A0(a1, v89, v87, v88, a5, a6);
                      v86 = *(_QWORD *)(a1 + 40);
                    }
                  }
                  v90 = *(_QWORD *)(a1 + 24);
                  *(_QWORD *)(a1 + 40) = v86 + 1;
                  *(_QWORD *)(v90 + 8 * v86) = 2415919106LL;
                  result = *(_QWORD *)a1;
                  v8 = *(_QWORD *)(a1 + 8);
                  v11 = *(_QWORD *)(a1 + 40);
                  goto LABEL_40;
                }
                goto LABEL_136;
              }
            }
          }
          result = *(_QWORD *)a1;
          v8 = *(_QWORD *)(a1 + 8);
LABEL_40:
          if ( result >= v8 )
            goto LABEL_49;
          v10 = *(char *)result;
          if ( (unsigned __int8)(*(_BYTE *)result - 42) <= 1u || (_BYTE)v10 == 63 )
            goto LABEL_46;
          if ( (_BYTE)v10 != 123 )
            goto LABEL_4;
          if ( v8 > result + 1 && (unsigned int)*(unsigned __int8 *)(result + 1) - 48 <= 9 )
          {
LABEL_46:
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 13;
LABEL_48:
            result = (unsigned __int64)&unk_4FA17D0;
            *(_QWORD *)a1 = &unk_4FA17D0;
            *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            goto LABEL_49;
          }
LABEL_117:
          if ( a2 == 123 )
            goto LABEL_118;
          continue;
        }
LABEL_21:
        v14 = sub_16E8DE0((unsigned __int8 **)a1);
        v16 = *(_BYTE **)a1;
        v17 = v14;
        if ( *(_QWORD *)a1 < *(_QWORD *)(a1 + 8) && *v16 == 44 )
        {
          *(_QWORD *)a1 = v16 + 1;
          v17 = 256;
          if ( (unsigned int)(unsigned __int8)v16[1] - 48 <= 9 )
          {
            v115 = v14;
            v101 = sub_16E8DE0((unsigned __int8 **)a1);
            v14 = v115;
            v17 = v101;
            if ( v115 > v101 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 10;
              *(_QWORD *)a1 = &unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            }
          }
        }
        sub_16E9280(a1, v11, v14, v17, v15);
        v18 = *(_BYTE **)a1;
        v8 = *(_QWORD *)(a1 + 8);
        if ( *(_QWORD *)a1 >= v8 )
          goto LABEL_156;
        if ( *v18 == 125 )
        {
          result = (unsigned __int64)(v18 + 1);
          v11 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)a1 = result;
          goto LABEL_40;
        }
        while ( 1 )
        {
          *(_QWORD *)a1 = ++v18;
          if ( v8 <= (unsigned __int64)v18 )
            break;
          if ( *v18 == 125 )
          {
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 10;
            goto LABEL_29;
          }
        }
LABEL_156:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 9;
LABEL_29:
        result = (unsigned __int64)&unk_4FA17D0;
        v11 = *(_QWORD *)(a1 + 40);
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
LABEL_49:
        if ( v11 != v119 )
        {
LABEL_198:
          v119 = v11;
          goto LABEL_53;
        }
LABEL_50:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 14;
        result = (unsigned __int64)&unk_4FA17D0;
        *(_QWORD *)a1 = &unk_4FA17D0;
        *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
LABEL_53:
        if ( !v118 && !*(_DWORD *)(a1 + 16) )
        {
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v116) = (v119 - v116)
                                                       | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v116) & 0xF8000000LL;
          result = *(_QWORD *)(a1 + 40);
          v20 = result - v117;
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v21 = *(_QWORD *)(a1 + 32);
            if ( (__int64)result >= v21 )
            {
              v22 = (v21 + 1 + ((unsigned __int64)(v21 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL;
              v23 = v22 + (v21 + 1) / 2;
              if ( v21 < v23 )
              {
                sub_16E90A0(a1, v23, v22, v21, a5, a6);
                result = *(_QWORD *)(a1 + 40);
              }
            }
            v24 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v24 + 8 * result) = v20 | 0x90000000;
          }
        }
        return result;
      case '^':
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v40 = *(_QWORD *)(a1 + 32);
          v41 = v11;
          if ( v11 >= v40 )
          {
            v42 = (v40 + 1) / 2;
            v43 = v42 + ((v40 + 1 + ((unsigned __int64)(v40 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v40 < v43 )
            {
              sub_16E90A0(a1, v43, v11, v42, a5, a6);
              v41 = *(_QWORD *)(a1 + 40);
            }
          }
          v44 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v41 + 1;
          *(_QWORD *)(v44 + 8 * v41) = 402653184;
        }
        a6 = 1;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 1u;
        ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 76LL);
        result = *(_QWORD *)a1;
        v8 = *(_QWORD *)(a1 + 8);
        goto LABEL_14;
      case '{':
        if ( a5 < v8 && (unsigned int)*(unsigned __int8 *)(result + 1) - 48 <= 9 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 13;
          *(_QWORD *)a1 = &unk_4FA17D0;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
        }
        goto LABEL_12;
      case '|':
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 14;
        goto LABEL_48;
      default:
        v13 = (unsigned int)(v12 - 36);
LABEL_12:
        sub_16EA3B0(a1, v12, v13, v8, a5, a6);
        result = *(_QWORD *)a1;
        goto LABEL_13;
    }
  }
}
