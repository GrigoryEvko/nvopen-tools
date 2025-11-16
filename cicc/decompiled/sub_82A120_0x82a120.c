// Function: sub_82A120
// Address: 0x82a120
//
__int64 __fastcall sub_82A120(__int64 a1, __int64 a2)
{
  char v4; // al
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned __int8 v9; // al
  _QWORD *v10; // r15
  _QWORD *v11; // r13
  unsigned int v12; // r14d
  unsigned int v13; // eax
  __int64 v15; // r8
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int8 v19; // di
  __int64 v20; // rsi
  int v21; // ecx
  __int64 v22; // r14
  __int64 v23; // rsi
  char v24; // cl
  __int64 v25; // rcx
  unsigned __int8 v26; // al
  char v27; // dl
  __int64 v28; // rcx
  char v29; // dl
  __m128i *v30; // r14
  __m128i *v31; // rax
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rdi
  int v35; // esi
  __int64 *v36; // rax
  __int64 v37; // rcx
  unsigned int v38; // edx
  __int64 *v39; // rax
  unsigned int v40; // edx
  unsigned int v41; // esi
  __int64 v42; // rdx
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // r14
  __int64 v45; // rdi
  __int64 v46; // rsi
  char v47; // si
  __int64 *v48; // r15
  __int64 v49; // r10
  __int64 *v50; // rax
  __int64 v51; // r9
  __int64 v52; // rdi
  __int64 v53; // rsi
  int v54; // eax
  __int64 v55; // r8
  __int64 v56; // rsi
  __int64 v57; // r10
  __int64 v58; // r9
  char v59; // al
  __int64 v60; // r14
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rax
  char i; // dl
  __int64 v65; // r14
  char j; // cl
  __int64 v67; // r15
  __int64 v68; // r14
  __int64 v69; // rdi
  char v70; // al
  __int64 v71; // r11
  __int64 v72; // rcx
  __int64 v73; // rdi
  __int64 v74; // rsi
  __int64 *v75; // r15
  __int64 *v76; // r14
  __int64 v77; // rax
  __int64 v78; // rdx
  __int128 v79; // rdi
  _BOOL4 v80; // r15d
  __int64 v81; // r11
  int v82; // r14d
  __int64 v83; // rdi
  __int64 v84; // rcx
  __int64 v85; // rax
  char v86; // al
  __int64 v87; // rax
  __int64 v88; // r13
  bool v89; // dl
  char v90; // r9
  char v91; // r8
  __int64 v92; // rdi
  __int64 v93; // rsi
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // [rsp+8h] [rbp-48h]
  _BOOL4 v98; // [rsp+10h] [rbp-40h]
  __int64 v99; // [rsp+10h] [rbp-40h]
  _BOOL4 v100; // [rsp+18h] [rbp-38h]
  int v101; // [rsp+18h] [rbp-38h]
  __int64 v102; // [rsp+18h] [rbp-38h]
  __int64 v103; // [rsp+18h] [rbp-38h]

  v4 = *(_BYTE *)(a1 + 145);
  if ( (v4 & 0x40) != 0 )
  {
    v100 = (*(_BYTE *)(a1 + 100) & 2) != 0;
    v98 = (*(_BYTE *)(a2 + 100) & 2) != 0;
    if ( !unk_4D0430C )
      goto LABEL_3;
  }
  else
  {
    v98 = 0;
    v100 = 0;
    if ( !unk_4D0430C )
      goto LABEL_3;
  }
  if ( qword_4D0495C && (*(_BYTE *)(a1 + 32) || *(_BYTE *)(a2 + 32))
    || (v10 = *(_QWORD **)(a1 + 120), v11 = *(_QWORD **)(a2 + 120), !v10) )
  {
LABEL_3:
    if ( ((*(_BYTE *)(a2 + 145) ^ v4) & 0x80u) != 0 )
    {
      if ( v4 >= 0 )
        return (unsigned int)-1;
    }
    else
    {
      if ( !dword_4D04304 )
      {
        v29 = *(_BYTE *)(a1 + 32);
        if ( v29 != *(_BYTE *)(a2 + 32) )
          goto LABEL_76;
        if ( !v29 )
        {
          v12 = sub_829160(*(_QWORD *)(a1 + 8), a2);
          if ( v12 )
            return v12;
          v4 = *(_BYTE *)(a1 + 145);
        }
      }
      if ( (v4 & 0x40) != 0 )
      {
        v30 = sub_829BD0(*(_QWORD *)(a2 + 8));
        v31 = sub_829BD0(*(_QWORD *)(a1 + 8));
        v12 = sub_826E60((__int64 *)(a1 + 88), (__int64 *)(a2 + 88), 1, (__int64)v31, (__int64)v30);
        if ( v12 )
          return v12;
      }
      if ( v100 && sub_829C30(a2, a1) )
        return (unsigned int)-1;
      if ( !v98 || !sub_829C30(a1, a2) )
      {
        v5 = *(_QWORD *)(a1 + 8);
        v6 = *(_QWORD *)(a2 + 8);
        if ( v5
          && v6
          && (*(_BYTE *)(v5 + 81) & 0x10) != 0
          && (*(_BYTE *)(v6 + 81) & 0x10) != 0
          && !*(_BYTE *)(a1 + 32)
          && !*(_BYTE *)(a2 + 32)
          && unk_4D04460
          && ((unsigned int)sub_826D00(a1) || (unsigned int)sub_826D00(a2))
          && !dword_4D04964 )
        {
          if ( *(_BYTE *)(v5 + 80) == 16 )
            v5 = **(_QWORD **)(v5 + 88);
          if ( *(_BYTE *)(v5 + 80) == 24 )
            v5 = *(_QWORD *)(v5 + 88);
          if ( *(_BYTE *)(v6 + 80) == 16 )
            v6 = **(_QWORD **)(v6 + 88);
          v7 = *(_QWORD *)(v5 + 88);
          if ( *(_BYTE *)(v6 + 80) == 24 )
            v6 = *(_QWORD *)(v6 + 88);
          v8 = *(_QWORD *)(v6 + 88);
          if ( *(_BYTE *)(v7 + 174) == 1 )
          {
            v82 = sub_72F500(v7, 0, 0, 1, 0);
            if ( *(_BYTE *)(v8 + 174) == 1 && (unsigned int)sub_72F500(v8, 0, 0, 1, 0) )
            {
              if ( !v82 )
                return (unsigned int)-1;
            }
            else if ( v82 )
            {
              return 1;
            }
          }
          else if ( *(_BYTE *)(v8 + 174) == 1 && (unsigned int)sub_72F500(*(_QWORD *)(v6 + 88), 0, 0, 1, 0) )
          {
            return (unsigned int)-1;
          }
        }
        v9 = *(_BYTE *)(a1 + 145);
        if ( ((v9 ^ *(_BYTE *)(a2 + 145)) & 4) != 0 )
        {
          if ( (v9 & 4) != 0 )
            return (unsigned int)-1;
          return 1;
        }
        v15 = dword_4D04304;
        if ( !dword_4D04304 )
          goto LABEL_50;
        v29 = *(_BYTE *)(a1 + 32);
        v16 = *(_BYTE *)(a2 + 32);
        if ( v29 == v16 )
        {
          if ( v29 )
          {
            if ( !dword_4F077BC )
              goto LABEL_53;
            v32 = *(_QWORD *)(a1 + 120);
            v33 = *(_QWORD *)(a2 + 120);
            if ( !v32 || !v33 )
              goto LABEL_53;
LABEL_91:
            if ( (*(_BYTE *)(v32 + 84) & 2) != 0
              && (*(_BYTE *)(v33 + 84) & 2) != 0
              && *(_BYTE *)(v32 + 15) != *(_BYTE *)(v33 + 15)
              && (*(_BYTE *)(a1 + 145) & 2) == 0
              && (*(_BYTE *)(a2 + 145) & 2) == 0 )
            {
              if ( (unsigned int)sub_8D32E0(*(_QWORD *)(v32 + 32))
                && (v83 = sub_8D46C0(*(_QWORD *)(v32 + 32)), (*(_BYTE *)(v83 + 140) & 0xFB) == 8) )
              {
                v101 = sub_8D4C10(v83, dword_4F077C4 != 2);
                v35 = 0;
                if ( !(unsigned int)sub_8D32E0(*(_QWORD *)(v33 + 32))
                  || (v35 = 0, v34 = sub_8D46C0(*(_QWORD *)(v33 + 32)), (*(_BYTE *)(v34 + 140) & 0xFB) != 8) )
                {
LABEL_101:
                  if ( v35 == v101 )
                    goto LABEL_102;
                  goto LABEL_51;
                }
              }
              else
              {
                if ( !(unsigned int)sub_8D32E0(*(_QWORD *)(v33 + 32))
                  || (v34 = sub_8D46C0(*(_QWORD *)(v33 + 32)), (*(_BYTE *)(v34 + 140) & 0xFB) != 8) )
                {
LABEL_102:
                  if ( *(_BYTE *)(v32 + 15) )
                    return (unsigned int)-1;
                  return 1;
                }
                v101 = 0;
              }
              v35 = sub_8D4C10(v34, dword_4F077C4 != 2);
              goto LABEL_101;
            }
LABEL_51:
            if ( !*(_BYTE *)(a1 + 32) )
            {
LABEL_54:
              v17 = *(_QWORD *)(a1 + 8);
              v18 = *(_QWORD *)(a2 + 8);
              if ( !v17 || !v18 )
                goto LABEL_71;
              v19 = *(_BYTE *)(v17 + 80);
              v20 = *(_QWORD *)(a1 + 8);
              v21 = v19;
              if ( v19 == 16 )
              {
                v20 = **(_QWORD **)(v17 + 88);
                v21 = *(unsigned __int8 *)(v20 + 80);
              }
              if ( (_BYTE)v21 == 24 )
              {
                v20 = *(_QWORD *)(v20 + 88);
                v21 = *(unsigned __int8 *)(v20 + 80);
              }
              v15 = (unsigned int)(v21 - 10);
              if ( (unsigned __int8)(v21 - 10) <= 1u )
              {
                v22 = *(_QWORD *)(v20 + 88);
              }
              else
              {
                if ( (_BYTE)v21 != 20 )
                  goto LABEL_71;
                v22 = *(_QWORD *)(*(_QWORD *)(v20 + 88) + 176LL);
              }
              v15 = *(unsigned __int8 *)(v18 + 80);
              v23 = *(_QWORD *)(a2 + 8);
              v24 = *(_BYTE *)(v18 + 80);
              if ( (_BYTE)v15 == 16 )
              {
                v23 = **(_QWORD **)(v18 + 88);
                v24 = *(_BYTE *)(v23 + 80);
              }
              if ( v24 == 24 )
              {
                v23 = *(_QWORD *)(v23 + 88);
                v24 = *(_BYTE *)(v23 + 80);
              }
              if ( (unsigned __int8)(v24 - 10) <= 1u )
              {
                v25 = *(_QWORD *)(v23 + 88);
              }
              else
              {
                if ( v24 != 20 )
                  goto LABEL_71;
                v25 = *(_QWORD *)(*(_QWORD *)(v23 + 88) + 176LL);
              }
              if ( *(_BYTE *)(v22 + 174) != *(_BYTE *)(v25 + 174) )
              {
LABEL_71:
                v26 = *(_BYTE *)(a1 + 145);
                v27 = v26 ^ *(_BYTE *)(a2 + 145);
                LOBYTE(v28) = v26;
                if ( (v27 & 0x20) == 0 )
                  goto LABEL_72;
                v28 = (unsigned int)qword_4F077B4;
                if ( !(_DWORD)qword_4F077B4 )
                {
LABEL_135:
                  v43 = *(_QWORD *)(a2 + 120);
                  v44 = *(_QWORD *)(a1 + 120);
                  if ( v43 && v44 )
                  {
                    do
                    {
                      v45 = *(_QWORD *)(v44 + 32);
                      v46 = *(_QWORD *)(v43 + 32);
                      if ( v45 != v46 && !(unsigned int)sub_8D97D0(v45, v46, 0, v28, v15) )
                      {
                        LOBYTE(v28) = *(_BYTE *)(a1 + 145);
                        v27 = v28 ^ *(_BYTE *)(a2 + 145);
                        goto LABEL_72;
                      }
                      v44 = *(_QWORD *)v44;
                      v43 = *(_QWORD *)v43;
                    }
                    while ( v44 && v43 );
                    v26 = *(_BYTE *)(a1 + 145);
                  }
                  if ( !(v43 | v44) )
                  {
LABEL_126:
                    if ( (v26 & 0x20) != 0 )
                      return (unsigned int)-1;
                    return 1;
                  }
                  LOBYTE(v28) = v26;
                  v27 = v26 ^ *(_BYTE *)(a2 + 145);
LABEL_72:
                  if ( (v27 & 1) != 0 )
                  {
                    if ( (v28 & 1) != 0 )
                      return (unsigned int)-1;
                    return 1;
                  }
                  if ( (v27 & 2) != 0 )
                  {
                    if ( (v28 & 2) != 0 )
                      return (unsigned int)-1;
                    return 1;
                  }
                  v55 = *(_QWORD *)(a1 + 8);
                  v56 = *(_QWORD *)(a2 + 8);
                  v57 = v55;
                  v58 = v56;
                  if ( !v55 )
                  {
LABEL_186:
                    if ( !dword_4F077BC )
                      return 0;
                    if ( (_DWORD)qword_4F077B4 )
                      return 0;
                    if ( qword_4F077A8 <= 0x1116Fu )
                      return 0;
                    v87 = *(_QWORD *)(a1 + 8);
                    v88 = *(_QWORD *)(a2 + 8);
                    v89 = v88 != 0 && v87 != 0;
                    if ( !v89 )
                      return 0;
                    v90 = *(_BYTE *)(v87 + 80);
                    v91 = *(_BYTE *)(v88 + 80);
                    if ( v90 == 16 && (*(_BYTE *)(v87 + 96) & 4) != 0 )
                    {
                      if ( v91 != 16 )
                      {
                        v92 = *(_QWORD *)(a1 + 112);
                        v93 = *(_QWORD *)(a2 + 112);
                        if ( v92 )
                        {
                          if ( v93 )
                            goto LABEL_260;
                          v95 = *(_QWORD *)(a2 + 8);
                          goto LABEL_275;
                        }
                        goto LABEL_280;
                      }
                    }
                    else
                    {
                      v89 = 0;
                      if ( v91 != 16 )
                        return 0;
                    }
                    if ( v89 == ((*(_BYTE *)(v88 + 96) & 4) != 0) )
                      return 0;
                    v92 = *(_QWORD *)(a1 + 112);
                    v93 = *(_QWORD *)(a2 + 112);
                    if ( v92 )
                    {
                      if ( v93 )
                        goto LABEL_260;
                      goto LABEL_274;
                    }
                    if ( v90 != 16 )
                    {
LABEL_268:
                      if ( *(_BYTE *)(v87 + 80) == 24 )
                        v87 = *(_QWORD *)(v87 + 88);
                      v94 = *(_QWORD *)(v87 + 88);
                      if ( *(_BYTE *)(v87 + 80) == 20 )
                        v94 = *(_QWORD *)(v94 + 176);
                      v92 = *(_QWORD *)(v94 + 152);
                      if ( v93 )
                      {
LABEL_260:
                        if ( (unsigned int)sub_8DE890(v92, v93, 128, 0) )
                        {
                          if ( *(_BYTE *)(v88 + 80) != 16 || (*(_BYTE *)(v88 + 96) & 4) == 0 )
                            return (unsigned int)-1;
                          return 1;
                        }
                        return 0;
                      }
                      v95 = *(_QWORD *)(a2 + 8);
                      if ( v91 != 16 )
                      {
LABEL_275:
                        if ( *(_BYTE *)(v95 + 80) == 24 )
                          v95 = *(_QWORD *)(v95 + 88);
                        v96 = *(_QWORD *)(v95 + 88);
                        if ( *(_BYTE *)(v95 + 80) == 20 )
                          v96 = *(_QWORD *)(v96 + 176);
                        v93 = *(_QWORD *)(v96 + 152);
                        goto LABEL_260;
                      }
LABEL_274:
                      v95 = **(_QWORD **)(v88 + 88);
                      goto LABEL_275;
                    }
LABEL_280:
                    v87 = **(_QWORD **)(v87 + 88);
                    goto LABEL_268;
                  }
                  if ( !v56 )
                    goto LABEL_212;
                  v59 = *(_BYTE *)(v55 + 80);
                  v60 = *(_QWORD *)(v55 + 88);
                  if ( (unsigned __int8)(v59 - 10) <= 1u )
                  {
                    if ( *(_BYTE *)(v60 + 174) != 7 )
                      goto LABEL_168;
                    v69 = *(_QWORD *)(v55 + 88);
                  }
                  else
                  {
                    if ( v59 != 20 )
                      goto LABEL_168;
                    v69 = *(_QWORD *)(v60 + 176);
                    if ( *(_BYTE *)(v69 + 174) != 7 )
                      goto LABEL_170;
                  }
                  v70 = *(_BYTE *)(v56 + 80);
                  if ( (unsigned __int8)(v70 - 10) <= 1u )
                  {
                    v71 = *(_QWORD *)(v56 + 88);
                    goto LABEL_204;
                  }
                  if ( v70 == 20 )
                  {
                    v71 = *(_QWORD *)(*(_QWORD *)(v56 + 88) + 176LL);
LABEL_204:
                    v72 = *(unsigned __int8 *)(v71 + 193);
                    if ( (((unsigned __int8)v72 ^ *(_BYTE *)(v69 + 193)) & 0x10) != 0 )
                    {
                      if ( (v72 & 0x10) == 0 )
                        return (unsigned int)-1;
                      return 1;
                    }
                    v103 = *(_QWORD *)(v56 + 88);
                    if ( (*(_BYTE *)(v69 + 193) & 0x10) != 0 )
                    {
                      v80 = sub_829490(v69);
                      if ( v80 != sub_829490(v81) )
                      {
                        if ( !v80 )
                          return (unsigned int)-1;
                        return 1;
                      }
                      v84 = *(_QWORD *)(v60 + 416);
                      v85 = *(_QWORD *)(v103 + 416);
                      if ( v84 )
                      {
                        if ( v85 )
                        {
                          v86 = *(_BYTE *)(v85 + 80);
                          if ( (*(_BYTE *)(v84 + 80) == 20) != (v86 == 20) )
                          {
                            if ( v86 != 20 )
                              return (unsigned int)-1;
                            return 1;
                          }
                        }
                      }
                      goto LABEL_168;
                    }
                    v73 = *(_QWORD *)(v69 + 152);
                    v74 = *(_QWORD *)(v71 + 152);
                    if ( v73 == v74 || (unsigned int)sub_8D97D0(v73, v74, 0, v72, v55) )
                      return 1;
                    v55 = *(_QWORD *)(a1 + 8);
                    v56 = *(_QWORD *)(a2 + 8);
LABEL_212:
                    v57 = v55;
                    v58 = v56;
                    if ( !v55 || !v56 )
                      goto LABEL_186;
                  }
LABEL_168:
                  if ( *(_BYTE *)(v55 + 80) == 16 )
                    v57 = **(_QWORD **)(v55 + 88);
LABEL_170:
                  if ( *(_BYTE *)(v57 + 80) == 24 )
                    v57 = *(_QWORD *)(v57 + 88);
                  if ( *(_BYTE *)(v56 + 80) == 16 )
                    v58 = **(_QWORD **)(v56 + 88);
                  if ( *(_BYTE *)(v58 + 80) == 24 )
                    v58 = *(_QWORD *)(v58 + 88);
                  v61 = *(_QWORD *)(v57 + 88);
                  if ( *(_BYTE *)(v57 + 80) == 20 )
                    v61 = *(_QWORD *)(v61 + 176);
                  v62 = *(_QWORD *)(v58 + 88);
                  if ( *(_BYTE *)(v58 + 80) == 20 )
                    v62 = *(_QWORD *)(v62 + 176);
                  v63 = *(_QWORD *)(v61 + 152);
                  for ( i = *(_BYTE *)(v63 + 140); i == 12; i = *(_BYTE *)(v63 + 140) )
                    v63 = *(_QWORD *)(v63 + 160);
                  v65 = *(_QWORD *)(v62 + 152);
                  for ( j = *(_BYTE *)(v65 + 140); j == 12; j = *(_BYTE *)(v65 + 140) )
                    v65 = *(_QWORD *)(v65 + 160);
                  if ( j == 7
                    && i == 7
                    && ((*(_BYTE *)(*(_QWORD *)(v65 + 168) + 20LL) & 2) != 0
                     || (*(_BYTE *)(*(_QWORD *)(v63 + 168) + 20LL) & 2) != 0) )
                  {
                    if ( (*(_BYTE *)(*(_QWORD *)(v63 + 168) + 20LL) & 2) == 0 )
                      return (unsigned int)-1;
                    if ( (*(_BYTE *)(*(_QWORD *)(v65 + 168) + 20LL) & 2) == 0 )
                      return 1;
                    v75 = sub_736C60(19, *(__int64 **)(v63 + 104));
                    v76 = sub_736C60(19, *(__int64 **)(v65 + 104));
                    while ( v75 )
                    {
                      if ( !v76 )
                        return 1;
                      v77 = v75[4];
                      v78 = v76[4];
                      if ( !v77 )
                        goto LABEL_186;
                      if ( !v78 )
                        goto LABEL_186;
                      if ( *(_BYTE *)(v77 + 10) != 5 )
                        goto LABEL_186;
                      if ( *(_BYTE *)(v78 + 10) != 5 )
                        goto LABEL_186;
                      *((_QWORD *)&v79 + 1) = *(_QWORD *)(v78 + 40);
                      *(_QWORD *)&v79 = *(_QWORD *)(v77 + 40);
                      if ( !(unsigned int)sub_7386E0(v79, 4u) )
                        goto LABEL_186;
                      v75 = sub_736C60(19, (__int64 *)*v75);
                      v76 = sub_736C60(19, (__int64 *)*v76);
                    }
                    if ( v76 )
                      return (unsigned int)-1;
                  }
                  goto LABEL_186;
                }
LABEL_125:
                if ( qword_4F077A0 )
                  goto LABEL_126;
                goto LABEL_135;
              }
              v47 = *(_BYTE *)(v25 + 194) & 0x40;
              if ( (*(_BYTE *)(v22 + 194) & 0x40) != 0 )
              {
                v48 = *(__int64 **)(a1 + 120);
                do
                  v22 = *(_QWORD *)(v22 + 232);
                while ( (*(_BYTE *)(v22 + 194) & 0x40) != 0 );
                if ( v47 )
                  goto LABEL_146;
              }
              else
              {
                if ( v47 )
                {
                  v48 = *(__int64 **)(a1 + 120);
                  do
LABEL_146:
                    v25 = *(_QWORD *)(v25 + 232);
                  while ( (*(_BYTE *)(v25 + 194) & 0x40) != 0 );
                  goto LABEL_147;
                }
                if ( !dword_4F077BC )
                  goto LABEL_71;
                if ( (_DWORD)qword_4F077B4 )
                {
                  v26 = *(_BYTE *)(a1 + 145);
                  v27 = v26 ^ *(_BYTE *)(a2 + 145);
                  v28 = v26;
                  if ( (v27 & 0x20) == 0 )
                    goto LABEL_72;
                  goto LABEL_125;
                }
                if ( qword_4F077A8 <= 0x1116Fu
                  || (*(_BYTE *)(v22 + 89) & 4) == 0
                  || (v19 != 16 || (*(_BYTE *)(v17 + 96) & 4) == 0)
                  && ((_BYTE)v15 != 16 || (*(_BYTE *)(v18 + 96) & 4) == 0) )
                {
                  v26 = *(_BYTE *)(a1 + 145);
                  v27 = v26 ^ *(_BYTE *)(a2 + 145);
                  v28 = v26;
                  if ( (v27 & 0x20) == 0 )
                    goto LABEL_72;
                  goto LABEL_135;
                }
                v48 = *(__int64 **)(a1 + 120);
              }
LABEL_147:
              if ( v48 )
              {
                v49 = **(_QWORD **)(*(_QWORD *)(v22 + 152) + 168LL);
                v50 = *(__int64 **)(*(_QWORD *)(v25 + 152) + 168LL);
                if ( !*((_BYTE *)v48 + 15) )
                {
                  v51 = *v50;
                  goto LABEL_150;
                }
                v48 = (__int64 *)*v48;
                v51 = *v50;
                if ( v48 )
                {
LABEL_150:
                  LOBYTE(v15) = v51 == 0;
                  LOBYTE(v50) = v49 == 0;
                  v15 = (unsigned int)v50 | (unsigned int)v15;
                  do
                  {
                    if ( (_BYTE)v15 )
                    {
                      if ( v51 != v49 )
                        goto LABEL_71;
                    }
                    else
                    {
                      v52 = *(_QWORD *)(v49 + 8);
                      v53 = *(_QWORD *)(v51 + 8);
                      if ( v52 != v53 )
                      {
                        v97 = v49;
                        v99 = v51;
                        v102 = v25;
                        v54 = sub_8D97D0(v52, v53, 0, v25, v15);
                        v25 = v102;
                        v51 = v99;
                        v49 = v97;
                        v15 = 0;
                        if ( !v54 )
                          goto LABEL_71;
                      }
                    }
                    v48 = (__int64 *)*v48;
                  }
                  while ( v48 );
                }
              }
              v67 = *(_QWORD *)(*(_QWORD *)(v22 + 40) + 32LL);
              v68 = *(_QWORD *)(*(_QWORD *)(v25 + 40) + 32LL);
              if ( sub_8D5CE0(v67, v68) )
                return 1;
              if ( sub_8D5CE0(v68, v67) )
                return (unsigned int)-1;
              goto LABEL_71;
            }
            v16 = *(_BYTE *)(a2 + 32);
LABEL_53:
            if ( v16 )
            {
              v36 = *(__int64 **)(a1 + 120);
              v37 = 0;
              while ( v36 )
              {
                v38 = *((_DWORD *)v36 + 6);
                v36 = (__int64 *)*v36;
                if ( (unsigned int)v37 < v38 )
                  v37 = v38;
              }
              v39 = *(__int64 **)(a2 + 120);
              if ( v39 )
              {
                v40 = 0;
                do
                {
                  v41 = *((_DWORD *)v39 + 6);
                  v39 = (__int64 *)*v39;
                  if ( v40 < v41 )
                    v40 = v41;
                }
                while ( v39 );
                if ( (unsigned int)v37 < v40 )
                  v37 = v40;
              }
              v42 = 4 * (unsigned int)((*(_BYTE *)(a1 + 145) & 2) != 0);
              if ( (*(_BYTE *)(a2 + 145) & 2) != 0 )
                v42 = (4 * ((*(_BYTE *)(a1 + 145) & 2) != 0)) | 8u;
              v12 = sub_8B2440(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8), v42, v37);
              if ( v12 )
                return v12;
            }
            goto LABEL_54;
          }
          v12 = sub_829160(*(_QWORD *)(a1 + 8), a2);
          if ( v12 )
            return v12;
LABEL_50:
          if ( !dword_4F077BC )
            goto LABEL_51;
          v32 = *(_QWORD *)(a1 + 120);
          v33 = *(_QWORD *)(a2 + 120);
          if ( !v32 || !v33 )
            goto LABEL_51;
          goto LABEL_91;
        }
LABEL_76:
        if ( v29 )
          return (unsigned int)-1;
      }
    }
    return 1;
  }
  v12 = 0;
  do
  {
    while ( 1 )
    {
      v13 = sub_827710((__int64)v10, (__int64)v11);
      if ( v13 )
        break;
LABEL_37:
      v10 = (_QWORD *)*v10;
      v11 = (_QWORD *)*v11;
      if ( !v10 )
        goto LABEL_41;
    }
    if ( v12 )
    {
      if ( v13 != v12 )
        goto LABEL_42;
      goto LABEL_37;
    }
    v10 = (_QWORD *)*v10;
    v11 = (_QWORD *)*v11;
    v12 = v13;
  }
  while ( v10 );
LABEL_41:
  if ( !v12 )
  {
LABEL_42:
    v4 = *(_BYTE *)(a1 + 145);
    goto LABEL_3;
  }
  return v12;
}
