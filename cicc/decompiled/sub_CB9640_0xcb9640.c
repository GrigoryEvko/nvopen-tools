// Function: sub_CB9640
// Address: 0xcb9640
//
unsigned __int8 *__fastcall sub_CB9640(__int64 a1, int a2)
{
  __int64 v4; // rcx
  unsigned __int8 *result; // rax
  int v6; // edx
  __int64 v7; // r14
  unsigned __int8 *v8; // r8
  __int64 v9; // rsi
  int v10; // r9d
  int v11; // eax
  unsigned __int8 *v12; // rcx
  int v13; // edx
  int v14; // r8d
  unsigned __int8 *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  signed __int64 v20; // rcx
  signed __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  signed __int64 v24; // rdx
  signed __int64 v25; // rcx
  signed __int64 v26; // rsi
  __int64 v27; // rcx
  signed __int64 v28; // rdx
  signed __int64 v29; // rcx
  signed __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned __int8 *v32; // rdx
  signed __int64 v33; // rdx
  __int64 v34; // rax
  signed __int64 v35; // rsi
  __int64 v36; // rdx
  signed __int64 v37; // rdx
  __int64 v38; // rax
  signed __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // ecx
  int v43; // esi
  __int64 v44; // rdx
  signed __int64 v45; // rdx
  __int64 v46; // rax
  signed __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rbx
  signed __int64 v50; // rax
  signed __int64 v51; // rsi
  __int64 v52; // rax
  signed __int64 v53; // rax
  signed __int64 v54; // rdx
  signed __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rax
  signed __int64 v59; // rcx
  signed __int64 v60; // rsi
  __int64 v61; // rcx
  __int64 v62; // rax
  signed __int64 v63; // rcx
  signed __int64 v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // rax
  signed __int64 v67; // rcx
  signed __int64 v68; // rsi
  __int64 v69; // rcx
  __int64 v70; // rax
  signed __int64 v71; // rcx
  signed __int64 v72; // rsi
  __int64 v73; // rcx
  signed __int64 v74; // rax
  signed __int64 v75; // rdx
  signed __int64 v76; // rsi
  __int64 v77; // rdx
  __int64 *v78; // rdx
  signed __int64 v79; // rax
  signed __int64 v80; // rdx
  signed __int64 v81; // rsi
  __int64 v82; // rdx
  signed __int64 v83; // r8
  __int64 v84; // rdi
  signed __int64 v85; // rsi
  __int64 v86; // rsi
  signed __int64 v87; // rdx
  signed __int64 v88; // rdi
  int v89; // ecx
  signed __int64 v90; // rsi
  __int64 v91; // rax
  int v92; // eax
  __int64 v93; // [rsp+0h] [rbp-60h]
  __int64 v94; // [rsp+0h] [rbp-60h]
  __int64 v95; // [rsp+8h] [rbp-58h]
  __int64 v96; // [rsp+8h] [rbp-58h]
  __int64 v97; // [rsp+8h] [rbp-58h]
  int v98; // [rsp+8h] [rbp-58h]
  __int64 v99; // [rsp+8h] [rbp-58h]
  __int64 v100; // [rsp+8h] [rbp-58h]
  __int64 v101; // [rsp+8h] [rbp-58h]
  __int64 v102; // [rsp+8h] [rbp-58h]
  int v103; // [rsp+8h] [rbp-58h]
  int v104; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v105; // [rsp+8h] [rbp-58h]
  int v106; // [rsp+8h] [rbp-58h]
  __int64 v107; // [rsp+10h] [rbp-50h]
  __int64 v108; // [rsp+18h] [rbp-48h]
  int v109; // [rsp+24h] [rbp-3Ch]
  __int64 v110; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v109 = 1;
  v107 = 0;
  v108 = 0;
  v110 = *(_QWORD *)(a1 + 40);
  result = *(unsigned __int8 **)a1;
  while ( 1 )
  {
    if ( v4 - (__int64)result <= 0 )
      goto LABEL_51;
    v6 = (char)*result;
    v7 = v110;
LABEL_4:
    if ( (_BYTE)v6 != 124 && v6 != a2 )
      break;
LABEL_113:
    if ( v7 == v110 )
      goto LABEL_51;
    v4 = *(_QWORD *)(a1 + 8);
    result = *(unsigned __int8 **)a1;
    if ( v4 - *(_QWORD *)a1 <= 0 || *result != 124 )
      goto LABEL_199;
    *(_QWORD *)a1 = ++result;
    if ( v109 )
    {
      if ( *(_DWORD *)(a1 + 16) )
        goto LABEL_176;
      sub_CB7820((_QWORD *)a1, 2013265920, v7 - v110 + 1, v110);
      v107 = v110;
      v7 = *(_QWORD *)(a1 + 40);
      v108 = v110;
    }
    v49 = v7 - v108;
    if ( *(_DWORD *)(a1 + 16) )
    {
      v4 = *(_QWORD *)(a1 + 8);
      result = *(unsigned __int8 **)a1;
LABEL_176:
      v108 = v7 - 1;
LABEL_177:
      v110 = v7;
      v107 = v7;
      goto LABEL_129;
    }
    v50 = *(_QWORD *)(a1 + 32);
    if ( v50 <= v7 )
    {
      v51 = (v50 + 1) / 2 + ((v50 + 1 + ((unsigned __int64)(v50 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
      if ( v50 < v51 )
      {
        sub_CB7740(a1, v51);
        v7 = *(_QWORD *)(a1 + 40);
      }
    }
    v52 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 40) = v7 + 1;
    *(_QWORD *)(v52 + 8 * v7) = v49 | 0x80000000LL;
    v7 = *(_QWORD *)(a1 + 40);
    v108 = v7 - 1;
    if ( *(_DWORD *)(a1 + 16) )
    {
      v4 = *(_QWORD *)(a1 + 8);
      result = *(unsigned __int8 **)a1;
      goto LABEL_177;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v107) = (*(_QWORD *)(a1 + 40) - v107)
                                                 | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v107) & 0xF8000000LL;
    v53 = *(_QWORD *)(a1 + 40);
    v107 = v53;
    if ( *(_DWORD *)(a1 + 16) )
    {
      v4 = *(_QWORD *)(a1 + 8);
      result = *(unsigned __int8 **)a1;
      v110 = *(_QWORD *)(a1 + 40);
    }
    else
    {
      v54 = *(_QWORD *)(a1 + 32);
      if ( v53 >= v54 )
      {
        v55 = (v54 + 1) / 2 + ((v54 + 1 + ((unsigned __int64)(v54 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
        if ( v54 < v55 )
        {
          sub_CB7740(a1, v55);
          v53 = *(_QWORD *)(a1 + 40);
        }
      }
      v56 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v53 + 1;
      *(_QWORD *)(v56 + 8 * v53) = 2281701376LL;
      v4 = *(_QWORD *)(a1 + 8);
      v110 = *(_QWORD *)(a1 + 40);
      result = *(unsigned __int8 **)a1;
    }
LABEL_129:
    v109 = 0;
  }
  while ( 2 )
  {
    v8 = result + 1;
    *(_QWORD *)a1 = result + 1;
    v9 = (unsigned int)(char)*result;
    switch ( *result )
    {
      case '$':
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v33 = *(_QWORD *)(a1 + 32);
          v34 = v7;
          if ( v33 <= v7 )
          {
            v35 = (v33 + 1) / 2 + ((v33 + 1 + ((unsigned __int64)(v33 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v33 < v35 )
            {
              sub_CB7740(a1, v35);
              v34 = *(_QWORD *)(a1 + 40);
            }
          }
          v36 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v34 + 1;
          *(_QWORD *)(v36 + 8 * v34) = 0x20000000;
        }
        v10 = 0;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 2u;
        ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 80LL);
        v4 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        goto LABEL_14;
      case '(':
        if ( v4 - (__int64)v8 <= 0 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 8;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        v23 = ++*(_QWORD *)(*(_QWORD *)(a1 + 56) + 112LL);
        if ( v23 <= 9 )
          *(_QWORD *)(a1 + 8 * v23 + 64) = *(_QWORD *)(a1 + 40);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v24 = *(_QWORD *)(a1 + 40);
          v25 = *(_QWORD *)(a1 + 32);
          if ( v24 >= v25 )
          {
            v26 = (v25 + 1) / 2 + ((v25 + 1 + ((unsigned __int64)(v25 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v25 < v26 )
            {
              v95 = v23;
              sub_CB7740(a1, v26);
              v24 = *(_QWORD *)(a1 + 40);
              v23 = v95;
            }
          }
          v27 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v24 + 1;
          *(_QWORD *)(v27 + 8 * v24) = v23 | 0x68000000;
        }
        if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) <= 0 || **(_BYTE **)a1 != 41 )
        {
          v96 = v23;
          sub_CB9640(a1, 41);
          v23 = v96;
        }
        if ( v23 <= 9 )
          *(_QWORD *)(a1 + 8 * v23 + 144) = *(_QWORD *)(a1 + 40);
        if ( *(_DWORD *)(a1 + 16) )
        {
          v4 = *(_QWORD *)(a1 + 8);
          v32 = *(unsigned __int8 **)a1;
          if ( v4 - *(_QWORD *)a1 <= 0 )
            goto LABEL_30;
        }
        else
        {
          v28 = *(_QWORD *)(a1 + 40);
          v29 = *(_QWORD *)(a1 + 32);
          if ( v28 >= v29 )
          {
            v30 = (v29 + 1) / 2 + ((v29 + 1 + ((unsigned __int64)(v29 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v29 < v30 )
            {
              v97 = v23;
              sub_CB7740(a1, v30);
              v28 = *(_QWORD *)(a1 + 40);
              v23 = v97;
            }
          }
          v31 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v28 + 1;
          *(_QWORD *)(v31 + 8 * v28) = v23 | 0x70000000;
          v4 = *(_QWORD *)(a1 + 8);
          v32 = *(unsigned __int8 **)a1;
          if ( v4 - *(_QWORD *)a1 <= 0 )
          {
LABEL_81:
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 8;
            goto LABEL_30;
          }
        }
        result = v32 + 1;
        *(_QWORD *)a1 = v32 + 1;
        if ( *v32 != 41 )
          goto LABEL_81;
LABEL_13:
        v10 = 0;
        goto LABEL_14;
      case ')':
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 8;
        goto LABEL_49;
      case '*':
      case '+':
      case '?':
        goto LABEL_47;
      case '.':
        if ( (*(_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL) & 8) != 0 )
        {
          *(_QWORD *)a1 = &unk_3F6AD74;
          *(_QWORD *)(a1 + 8) = &unk_3F6AD77;
          v94 = v4;
          v105 = result + 1;
          sub_CB7E40(a1, v9);
          v4 = v94;
          v10 = 0;
          *(_QWORD *)a1 = v105;
          result = v105;
          *(_QWORD *)(a1 + 8) = v94;
        }
        else
        {
          v10 = *(_DWORD *)(a1 + 16);
          if ( v10 )
          {
            v4 = *(_QWORD *)(a1 + 8);
            ++result;
            v10 = 0;
          }
          else
          {
            v45 = *(_QWORD *)(a1 + 32);
            v46 = v7;
            if ( v45 <= v7 )
            {
              v47 = (v45 + 1) / 2 + ((v45 + 1 + ((unsigned __int64)(v45 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v45 < v47 )
              {
                v98 = *(_DWORD *)(a1 + 16);
                sub_CB7740(a1, v47);
                v46 = *(_QWORD *)(a1 + 40);
                v10 = v98;
              }
            }
            v48 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = v46 + 1;
            *(_QWORD *)(v48 + 8 * v46) = 671088640;
            v4 = *(_QWORD *)(a1 + 8);
            result = *(unsigned __int8 **)a1;
          }
        }
        goto LABEL_14;
      case '[':
        sub_CB7E40(a1, v9);
        v4 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        v10 = 0;
        goto LABEL_14;
      case '\\':
        if ( v4 - (__int64)v8 <= 0 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 5;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
          v8 = byte_4F85140;
        }
        *(_QWORD *)a1 = v8 + 1;
        v9 = (unsigned int)(char)*v8;
        if ( (unsigned __int8)(*v8 - 49) > 8u )
          goto LABEL_12;
        v41 = (char)(*v8 - 48);
        v42 = (char)(*v8 - 48);
        v43 = *(_DWORD *)(a1 + 16);
        v44 = *(_QWORD *)(a1 + 8 * v41 + 144);
        if ( !v44 )
        {
          if ( !v43 )
            *(_DWORD *)(a1 + 16) = 6;
          goto LABEL_49;
        }
        if ( !v43 )
        {
          v83 = *(_QWORD *)(a1 + 32);
          v84 = v7;
          if ( v83 <= v7 )
          {
            v85 = ((v83 + 1 + ((unsigned __int64)(v83 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v83 + 1) / 2;
            if ( v83 < v85 )
            {
              v93 = v41;
              v103 = v41;
              sub_CB7740(a1, v85);
              v84 = *(_QWORD *)(a1 + 40);
              v41 = v93;
              v42 = v103;
            }
          }
          v86 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v84 + 1;
          *(_QWORD *)(v86 + 8 * v84) = v42 | 0x38000000;
          v44 = *(_QWORD *)(a1 + 8 * v41 + 144);
        }
        v104 = v42;
        sub_CB77B0((_QWORD *)a1, *(_QWORD *)(a1 + 8 * v41 + 64) + 1LL, v44);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v87 = *(_QWORD *)(a1 + 40);
          v88 = *(_QWORD *)(a1 + 32);
          v89 = v104;
          if ( v87 >= v88 )
          {
            v90 = ((v88 + 1 + ((unsigned __int64)(v88 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v88 + 1) / 2;
            if ( v88 < v90 )
            {
              sub_CB7740(a1, v90);
              v87 = *(_QWORD *)(a1 + 40);
              v89 = v104;
            }
          }
          v91 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v87 + 1;
          *(_QWORD *)(v91 + 8 * v87) = v89 | 0x40000000;
        }
        v10 = 0;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 120LL) = 1;
        v4 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
LABEL_14:
        if ( v4 - (__int64)result <= 0 )
        {
          v7 = *(_QWORD *)(a1 + 40);
          goto LABEL_50;
        }
        v6 = (char)*result;
        if ( (unsigned __int8)(*result - 42) <= 1u || (_BYTE)v6 == 63 )
        {
          *(_QWORD *)a1 = result + 1;
          if ( !v10 )
            goto LABEL_35;
          goto LABEL_32;
        }
        if ( v4 - (_QWORD)result == 1 || (_BYTE)v6 != 123 )
        {
          v7 = *(_QWORD *)(a1 + 40);
          goto LABEL_4;
        }
        if ( (unsigned int)result[1] - 48 > 9 )
        {
          v7 = *(_QWORD *)(a1 + 40);
          goto LABEL_112;
        }
        *(_QWORD *)a1 = result + 1;
        if ( v10 )
        {
LABEL_32:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 13;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
LABEL_35:
          if ( (_BYTE)v6 == 43 )
          {
            v57 = *(_QWORD *)(a1 + 40);
            if ( !*(_DWORD *)(a1 + 16) )
            {
              sub_CB7820((_QWORD *)a1, 1207959552, v57 - v7 + 1, v7);
              v57 = *(_QWORD *)(a1 + 40);
              v66 = v57 - v7;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v67 = *(_QWORD *)(a1 + 32);
                if ( v57 >= v67 )
                {
                  v68 = (v67 + 1) / 2 + ((v67 + 1 + ((unsigned __int64)(v67 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v67 < v68 )
                  {
                    v101 = *(_QWORD *)(a1 + 40) - v7;
                    sub_CB7740(a1, v68);
                    v57 = *(_QWORD *)(a1 + 40);
                    v66 = v101;
                  }
                }
                v69 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v57 + 1;
                *(_QWORD *)(v69 + 8 * v57) = v66 | 0x50000000;
                v4 = *(_QWORD *)(a1 + 8);
                result = *(unsigned __int8 **)a1;
                v7 = *(_QWORD *)(a1 + 40);
                goto LABEL_41;
              }
            }
            goto LABEL_151;
          }
          if ( (char)v6 <= 43 )
          {
            if ( (_BYTE)v6 != 42 )
            {
LABEL_131:
              v4 = *(_QWORD *)(a1 + 8);
              result = *(unsigned __int8 **)a1;
              v7 = *(_QWORD *)(a1 + 40);
              goto LABEL_41;
            }
            v57 = *(_QWORD *)(a1 + 40);
            if ( !*(_DWORD *)(a1 + 16) )
            {
              sub_CB7820((_QWORD *)a1, 1207959552, v57 - v7 + 1, v7);
              v57 = *(_QWORD *)(a1 + 40);
              v58 = v57 - v7;
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v59 = *(_QWORD *)(a1 + 32);
                if ( v57 >= v59 )
                {
                  v60 = (v59 + 1) / 2 + ((v59 + 1 + ((unsigned __int64)(v59 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                  if ( v59 < v60 )
                  {
                    v99 = *(_QWORD *)(a1 + 40) - v7;
                    sub_CB7740(a1, v60);
                    v57 = *(_QWORD *)(a1 + 40);
                    v58 = v99;
                  }
                }
                v61 = *(_QWORD *)(a1 + 24);
                *(_QWORD *)(a1 + 40) = v57 + 1;
                *(_QWORD *)(v61 + 8 * v57) = v58 | 0x50000000;
                v57 = *(_QWORD *)(a1 + 40);
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  sub_CB7820((_QWORD *)a1, 1476395008, *(_QWORD *)(a1 + 40) - v7 + 1, v7);
                  v57 = *(_QWORD *)(a1 + 40);
                  v62 = v57 - v7;
                  if ( !*(_DWORD *)(a1 + 16) )
                  {
                    v63 = *(_QWORD *)(a1 + 32);
                    if ( v57 >= v63 )
                    {
                      v64 = (v63 + 1) / 2 + ((v63 + 1 + ((unsigned __int64)(v63 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                      if ( v63 < v64 )
                      {
                        v100 = *(_QWORD *)(a1 + 40) - v7;
                        sub_CB7740(a1, v64);
                        v57 = *(_QWORD *)(a1 + 40);
                        v62 = v100;
                      }
                    }
                    v65 = *(_QWORD *)(a1 + 24);
                    *(_QWORD *)(a1 + 40) = v57 + 1;
                    *(_QWORD *)(v65 + 8 * v57) = v62 | 0x60000000;
                    v4 = *(_QWORD *)(a1 + 8);
                    result = *(unsigned __int8 **)a1;
                    v7 = *(_QWORD *)(a1 + 40);
                    goto LABEL_41;
                  }
                }
              }
            }
LABEL_151:
            v4 = *(_QWORD *)(a1 + 8);
            result = *(unsigned __int8 **)a1;
            v7 = v57;
            goto LABEL_41;
          }
          if ( (_BYTE)v6 != 63 )
          {
            if ( (_BYTE)v6 == 123 )
              goto LABEL_21;
            goto LABEL_131;
          }
          v18 = *(_QWORD *)(a1 + 40);
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_39;
          sub_CB7820((_QWORD *)a1, 2013265920, v18 - v7 + 1, v7);
          v18 = *(_QWORD *)(a1 + 40);
          v70 = v18 - v7;
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_39;
          v71 = *(_QWORD *)(a1 + 32);
          if ( v18 >= v71 )
          {
            v72 = (v71 + 1) / 2 + ((v71 + 1 + ((unsigned __int64)(v71 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v71 < v72 )
            {
              v102 = *(_QWORD *)(a1 + 40) - v7;
              sub_CB7740(a1, v72);
              v18 = *(_QWORD *)(a1 + 40);
              v70 = v102;
            }
          }
          v73 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v18 + 1;
          *(_QWORD *)(v73 + 8 * v18) = v70 | 0x80000000LL;
          v18 = *(_QWORD *)(a1 + 40);
          if ( *(_DWORD *)(a1 + 16) )
          {
LABEL_39:
            v7 = v18;
          }
          else
          {
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v7) = (*(_QWORD *)(a1 + 40) - v7)
                                                       | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v7) & 0xF8000000LL;
            if ( *(_DWORD *)(a1 + 16) )
            {
              v7 = *(_QWORD *)(a1 + 40);
            }
            else
            {
              v74 = *(_QWORD *)(a1 + 40);
              v75 = *(_QWORD *)(a1 + 32);
              if ( v74 >= v75 )
              {
                v76 = (v75 + 1) / 2 + ((v75 + 1 + ((unsigned __int64)(v75 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                if ( v75 < v76 )
                {
                  sub_CB7740(a1, v76);
                  v74 = *(_QWORD *)(a1 + 40);
                }
              }
              v77 = *(_QWORD *)(a1 + 24);
              *(_QWORD *)(a1 + 40) = v74 + 1;
              *(_QWORD *)(v77 + 8 * v74) = 2281701376LL;
              v7 = *(_QWORD *)(a1 + 40);
              if ( !*(_DWORD *)(a1 + 16) )
              {
                v78 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8 * (v7 - 1));
                *v78 = *v78 & 0xF8000000LL | 1;
                if ( !*(_DWORD *)(a1 + 16) )
                {
                  v79 = *(_QWORD *)(a1 + 40);
                  v80 = *(_QWORD *)(a1 + 32);
                  if ( v79 >= v80 )
                  {
                    v81 = (v80 + 1) / 2 + ((v80 + 1 + ((unsigned __int64)(v80 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
                    if ( v80 < v81 )
                    {
                      sub_CB7740(a1, v81);
                      v79 = *(_QWORD *)(a1 + 40);
                    }
                  }
                  v82 = *(_QWORD *)(a1 + 24);
                  *(_QWORD *)(a1 + 40) = v79 + 1;
                  *(_QWORD *)(v82 + 8 * v79) = 2415919106LL;
                  v4 = *(_QWORD *)(a1 + 8);
                  result = *(unsigned __int8 **)a1;
                  v7 = *(_QWORD *)(a1 + 40);
                  goto LABEL_41;
                }
                goto LABEL_131;
              }
            }
          }
          v4 = *(_QWORD *)(a1 + 8);
          result = *(unsigned __int8 **)a1;
LABEL_41:
          if ( v4 - (__int64)result <= 0 )
            goto LABEL_50;
          v6 = (char)*result;
          if ( (unsigned __int8)(*result - 42) <= 1u || (_BYTE)v6 == 63 )
          {
LABEL_47:
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 13;
LABEL_49:
            result = byte_4F85140;
            *(_QWORD *)a1 = byte_4F85140;
            *(_QWORD *)(a1 + 8) = byte_4F85140;
            goto LABEL_50;
          }
          if ( v4 - (_QWORD)result == 1 || (_BYTE)v6 != 123 )
            goto LABEL_4;
          if ( (unsigned int)result[1] - 48 <= 9 )
            goto LABEL_47;
LABEL_112:
          if ( a2 == 123 )
            goto LABEL_113;
          continue;
        }
LABEL_21:
        v11 = sub_CB7460((unsigned __int8 **)a1);
        v12 = *(unsigned __int8 **)a1;
        v13 = v11;
        v14 = v11;
        if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 0 && *v12 == 44 )
        {
          v14 = 256;
          *(_QWORD *)a1 = v12 + 1;
          if ( (unsigned int)v12[1] - 48 <= 9 )
          {
            v106 = v11;
            v92 = sub_CB7460((unsigned __int8 **)a1);
            v13 = v106;
            v14 = v92;
            if ( v106 > v92 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 10;
              *(_QWORD *)a1 = byte_4F85140;
              *(_QWORD *)(a1 + 8) = byte_4F85140;
            }
          }
        }
        sub_CB7920(a1, v7, v13, v14);
        v4 = *(_QWORD *)(a1 + 8);
        v15 = *(unsigned __int8 **)a1;
        if ( v4 - *(_QWORD *)a1 <= 0 )
          goto LABEL_152;
        if ( *v15 == 125 )
        {
          result = v15 + 1;
          v7 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)a1 = result;
          goto LABEL_41;
        }
        v16 = v15 + 1;
        while ( 1 )
        {
          *(_QWORD *)a1 = v16;
          v17 = v16;
          if ( v4 - (__int64)v16 <= 0 )
            break;
          ++v16;
          if ( *v17 == 125 )
          {
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 10;
            goto LABEL_30;
          }
        }
LABEL_152:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 9;
LABEL_30:
        result = byte_4F85140;
        v7 = *(_QWORD *)(a1 + 40);
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
LABEL_50:
        if ( v7 != v110 )
        {
LABEL_199:
          v110 = v7;
          goto LABEL_54;
        }
LABEL_51:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 14;
        result = byte_4F85140;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
LABEL_54:
        if ( !v109 && !*(_DWORD *)(a1 + 16) )
        {
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v107) = (v110 - v107)
                                                       | *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v107) & 0xF8000000LL;
          result = *(unsigned __int8 **)(a1 + 40);
          v19 = (unsigned __int64)&result[-v108];
          if ( !*(_DWORD *)(a1 + 16) )
          {
            v20 = *(_QWORD *)(a1 + 32);
            if ( (__int64)result >= v20 )
            {
              v21 = ((v20 + 1 + ((unsigned __int64)(v20 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v20 + 1) / 2;
              if ( v20 < v21 )
              {
                sub_CB7740(a1, v21);
                result = *(unsigned __int8 **)(a1 + 40);
              }
            }
            v22 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = result + 1;
            *(_QWORD *)(v22 + 8LL * (_QWORD)result) = v19 | 0x90000000;
          }
        }
        return result;
      case '^':
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v37 = *(_QWORD *)(a1 + 32);
          v38 = v7;
          if ( v37 <= v7 )
          {
            v39 = (v37 + 1) / 2 + ((v37 + 1 + ((unsigned __int64)(v37 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v37 < v39 )
            {
              sub_CB7740(a1, v39);
              v38 = *(_QWORD *)(a1 + 40);
            }
          }
          v40 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v38 + 1;
          *(_QWORD *)(v40 + 8 * v38) = 402653184;
        }
        v10 = 1;
        *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 1u;
        ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 76LL);
        v4 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        goto LABEL_14;
      case '{':
        if ( v4 - (__int64)v8 > 0 && (unsigned int)result[1] - 48 <= 9 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 13;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        goto LABEL_12;
      case '|':
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 14;
        goto LABEL_49;
      default:
LABEL_12:
        sub_CB8AB0(a1, v9);
        v4 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        goto LABEL_13;
    }
  }
}
