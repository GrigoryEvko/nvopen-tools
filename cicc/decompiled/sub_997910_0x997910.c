// Function: sub_997910
// Address: 0x997910
//
__int64 __fastcall sub_997910(__int64 a1, char a2, __int64 (__fastcall *a3)(__int64, unsigned __int8 *), __int64 a4)
{
  _QWORD *v5; // rdx
  char v6; // cl
  __int64 result; // rax
  __int64 v8; // r15
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  char v12; // dl
  __int64 v13; // rdi
  bool v14; // zf
  unsigned __int8 v15; // al
  __int64 v16; // rdi
  unsigned __int8 v17; // al
  unsigned __int8 **v18; // rdx
  unsigned __int8 *v19; // r14
  unsigned __int8 **v20; // rdx
  __int64 v21; // r14
  unsigned int v22; // eax
  unsigned __int8 v23; // r15
  unsigned int v24; // ecx
  unsigned __int8 *v25; // rdi
  unsigned __int8 v26; // dl
  __int64 v27; // rdx
  __int64 v28; // r10
  __int64 v29; // r9
  int v30; // r8d
  int v31; // edi
  char v32; // r11
  unsigned __int8 *v33; // r15
  __int64 v34; // rax
  unsigned __int8 *v35; // rdi
  unsigned __int8 *v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int8 v39; // r14
  unsigned __int8 *v40; // rdi
  unsigned __int8 *v41; // r14
  char v42; // al
  char v43; // al
  unsigned __int8 v44; // dl
  int v45; // eax
  unsigned __int8 *v46; // r8
  unsigned __int8 *v47; // r9
  char v48; // al
  int v49; // eax
  unsigned __int8 *v50; // r8
  unsigned __int8 *v51; // rdi
  _BYTE *v52; // rdi
  char v53; // al
  unsigned __int8 **v54; // rax
  unsigned __int8 *v55; // r8
  unsigned __int8 *v56; // r8
  __int64 v57; // rax
  char v58; // al
  _BYTE *v59; // rax
  bool v60; // cl
  char v61; // al
  char v62; // al
  unsigned int v63; // ebx
  unsigned __int8 *v64; // rax
  unsigned __int8 *v65; // r14
  __int64 v66; // rdx
  __int64 v67; // rcx
  unsigned __int8 *v68; // rdi
  __int64 v69; // rdx
  unsigned __int8 *v70; // rdi
  unsigned int v71; // [rsp+10h] [rbp-130h]
  unsigned int v72; // [rsp+10h] [rbp-130h]
  unsigned __int8 v73; // [rsp+10h] [rbp-130h]
  bool v74; // [rsp+10h] [rbp-130h]
  unsigned __int8 v75; // [rsp+18h] [rbp-128h]
  unsigned int v76; // [rsp+18h] [rbp-128h]
  unsigned __int8 v77; // [rsp+18h] [rbp-128h]
  unsigned int v78; // [rsp+18h] [rbp-128h]
  int v79; // [rsp+18h] [rbp-128h]
  __int64 v80; // [rsp+20h] [rbp-120h]
  __int64 v81; // [rsp+20h] [rbp-120h]
  unsigned int v82; // [rsp+20h] [rbp-120h]
  unsigned int v83; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v84; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v85; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v86; // [rsp+20h] [rbp-120h]
  unsigned int v87; // [rsp+20h] [rbp-120h]
  unsigned int v88; // [rsp+20h] [rbp-120h]
  int v89; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v90; // [rsp+20h] [rbp-120h]
  unsigned int v91; // [rsp+20h] [rbp-120h]
  __int64 v92; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v93; // [rsp+20h] [rbp-120h]
  char v94; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v97; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int8 *v98; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int8 **v99; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int8 **v100; // [rsp+58h] [rbp-E8h]
  _QWORD *v101; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v102; // [rsp+68h] [rbp-D8h]
  _QWORD v103[8]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v104; // [rsp+B0h] [rbp-90h] BYREF
  __int64 *v105; // [rsp+B8h] [rbp-88h]
  __int64 v106; // [rsp+C0h] [rbp-80h]
  int v107; // [rsp+C8h] [rbp-78h]
  char v108; // [rsp+CCh] [rbp-74h]
  char v109; // [rsp+D0h] [rbp-70h] BYREF

  v103[0] = a1;
  v101 = v103;
  v104 = 0;
  v106 = 8;
  v107 = 0;
  v108 = 1;
  v105 = (__int64 *)&v109;
  v5 = v103;
  v6 = 1;
  v102 = 0x800000001LL;
  LODWORD(result) = 1;
  while ( 1 )
  {
    v8 = v5[(unsigned int)result - 1];
    LODWORD(v102) = result - 1;
    if ( v6 )
    {
      v9 = v105;
      v10 = HIDWORD(v106);
      v11 = &v105[HIDWORD(v106)];
      if ( v105 != v11 )
      {
        while ( v8 != *v9 )
        {
          if ( v11 == ++v9 )
            goto LABEL_27;
        }
LABEL_7:
        result = (unsigned int)v102;
        goto LABEL_8;
      }
LABEL_27:
      if ( HIDWORD(v106) < (unsigned int)v106 )
      {
        v10 = (unsigned int)++HIDWORD(v106);
        *v11 = v8;
        ++v104;
        if ( !a2 )
          goto LABEL_12;
LABEL_29:
        sub_9854E0((unsigned __int8 *)v8, a3, a4);
        v99 = 0;
        v10 = 30;
        v100 = &v98;
        if ( (unsigned __int8)sub_996420(&v99, 30, (unsigned __int8 *)v8) )
        {
          v10 = (__int64)a3;
          sub_9854E0(v98, a3, a4);
        }
        goto LABEL_12;
      }
    }
    v10 = v8;
    sub_C8CC70(&v104, v8);
    v6 = v108;
    if ( !v12 )
      goto LABEL_7;
    if ( a2 )
      goto LABEL_29;
LABEL_12:
    if ( *(_BYTE *)v8 <= 0x1Cu )
      goto LABEL_25;
    v13 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v13 + 16);
    v10 = 1;
    v14 = (unsigned __int8)sub_BCAC40(v13, 1) == 0;
    v15 = *(_BYTE *)v8;
    if ( v14 )
      goto LABEL_45;
    if ( v15 == 57 )
    {
      if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
        v18 = *(unsigned __int8 ***)(v8 - 8);
      else
        v18 = (unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
      if ( *v18 )
      {
        v19 = v18[4];
        v97 = *v18;
        if ( v19 )
          goto LABEL_40;
LABEL_44:
        v15 = *(_BYTE *)v8;
LABEL_45:
        if ( v15 <= 0x1Cu )
          goto LABEL_25;
      }
      v16 = *(_QWORD *)(v8 + 8);
      goto LABEL_19;
    }
    if ( v15 != 86 )
      goto LABEL_45;
    v16 = *(_QWORD *)(v8 + 8);
    v80 = *(_QWORD *)(v8 - 96);
    if ( *(_QWORD *)(v80 + 8) == v16 && **(_BYTE **)(v8 - 32) <= 0x15u )
    {
      v19 = *(unsigned __int8 **)(v8 - 64);
      if ( (unsigned __int8)sub_AC30F0(*(_QWORD *)(v8 - 32)) )
      {
        v97 = (unsigned __int8 *)v80;
        if ( v19 )
          goto LABEL_40;
      }
      goto LABEL_44;
    }
LABEL_19:
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    v10 = 1;
    v14 = (unsigned __int8)sub_BCAC40(v16, 1) == 0;
    v17 = *(_BYTE *)v8;
    if ( v14 )
      goto LABEL_53;
    if ( v17 == 58 )
    {
      if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
        v20 = *(unsigned __int8 ***)(v8 - 8);
      else
        v20 = (unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
      if ( !*v20 )
        goto LABEL_25;
      v19 = v20[4];
      v97 = *v20;
      if ( !v19 )
        goto LABEL_52;
    }
    else
    {
      if ( v17 != 86 )
        goto LABEL_53;
      v81 = *(_QWORD *)(v8 - 96);
      if ( *(_QWORD *)(v81 + 8) != *(_QWORD *)(v8 + 8) )
        goto LABEL_25;
      v52 = *(_BYTE **)(v8 - 64);
      if ( *v52 > 0x15u )
        goto LABEL_25;
      v19 = *(unsigned __int8 **)(v8 - 32);
      if ( !(unsigned __int8)sub_AD7A80(v52) || (v97 = (unsigned __int8 *)v81, !v19) )
      {
LABEL_52:
        v17 = *(_BYTE *)v8;
LABEL_53:
        if ( v17 > 0x1Cu )
        {
          if ( v17 != 82 )
            goto LABEL_151;
          if ( *(_QWORD *)(v8 - 64) )
          {
            v97 = *(unsigned __int8 **)(v8 - 64);
            v21 = *(_QWORD *)(v8 - 32);
            if ( v21 )
            {
              v22 = sub_B53900(v8);
              v23 = *(_BYTE *)v21;
              v24 = v22;
              if ( v22 - 32 > 1 )
              {
                v25 = v97;
                if ( a2 )
                {
                  v82 = v22;
                  sub_9854E0(v97, a3, a4);
                  v10 = (__int64)a3;
                  sub_9854E0((unsigned __int8 *)v21, a3, a4);
                  v24 = v82;
                }
                else
                {
                  if ( v23 > 0x15u )
                    goto LABEL_60;
                  v10 = (__int64)a3;
                  v88 = v22;
                  sub_9854E0(v97, a3, a4);
                  v24 = v88;
                }
                v25 = v97;
                v26 = *v97;
                if ( v23 != 17 )
                  goto LABEL_61;
                if ( v26 == 42 )
                {
                  v56 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                  if ( v56 )
                  {
                    v98 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                    if ( **((_BYTE **)v97 - 4) != 17 )
                    {
                      v26 = *v97;
                      goto LABEL_88;
                    }
LABEL_148:
                    v10 = (__int64)a3;
                    v91 = v24;
                    sub_9854E0(v56, a3, a4);
                    v24 = v91;
                  }
                }
                else
                {
LABEL_88:
                  if ( v26 == 58 && (v97[1] & 2) != 0 )
                  {
                    v56 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                    if ( v56 )
                    {
                      v98 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                      if ( **((_BYTE **)v97 - 4) == 17 )
                        goto LABEL_148;
                    }
                  }
                }
                v83 = v24;
                v42 = sub_B532A0(v24);
                v25 = v97;
                v24 = v83;
                v26 = *v97;
                if ( !v42 )
                  goto LABEL_61;
                LOBYTE(v10) = *v97;
                if ( v26 != 57 )
                  goto LABEL_91;
                v46 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                if ( v46 )
                {
                  v98 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                  v47 = (unsigned __int8 *)*((_QWORD *)v97 - 4);
                  if ( v47 )
                    goto LABEL_99;
                  LOBYTE(v10) = *v97;
                  v26 = *v97;
LABEL_91:
                  if ( (_BYTE)v10 != 58 )
                  {
LABEL_92:
                    v71 = v83;
                    v75 = v26;
                    v84 = v97;
                    v43 = sub_987880(v97);
                    v44 = v75;
                    v25 = v84;
                    v24 = v71;
                    v10 = (unsigned __int8)v10;
                    if ( v43 )
                    {
                      if ( (unsigned __int8)v10 > 0x1Cu )
                        goto LABEL_94;
                      v45 = *((unsigned __int16 *)v84 + 1);
LABEL_95:
                      if ( v45 == 13 && (v25[1] & 2) != 0 && (v46 = (unsigned __int8 *)*((_QWORD *)v25 - 8)) != 0 )
                      {
                        v98 = (unsigned __int8 *)*((_QWORD *)v25 - 8);
                        v47 = (unsigned __int8 *)*((_QWORD *)v25 - 4);
                        if ( v47 )
                          goto LABEL_99;
                        v44 = *v25;
                      }
                      else
                      {
                        v44 = v10;
                      }
                    }
                    v72 = v24;
                    v77 = v44;
                    v58 = sub_987880(v25);
                    v26 = v77;
                    v24 = v72;
                    if ( !v58 )
                    {
LABEL_61:
                      if ( v26 != 78 )
                        goto LABEL_63;
                      v27 = *((_QWORD *)v25 - 4);
                      v28 = *((_QWORD *)v25 + 1);
                      v29 = *(_QWORD *)(v27 + 8);
                      v30 = *(unsigned __int8 *)(v28 + 8);
                      v10 = (unsigned int)(v30 - 17);
                      v31 = *(unsigned __int8 *)(v29 + 8);
                      v32 = (unsigned int)v10 <= 1;
                      LOBYTE(v10) = (unsigned int)(v31 - 17) <= 1;
                      if ( v32 != (_BYTE)v10 )
                        goto LABEL_63;
                      if ( (unsigned int)(v31 - 17) <= 1 )
                      {
                        if ( *(_DWORD *)(v28 + 32) != *(_DWORD *)(v29 + 32) )
                          goto LABEL_63;
                        LOBYTE(v10) = (_BYTE)v30 == 18;
                        if ( ((_BYTE)v30 == 18) != ((_BYTE)v31 == 18) )
                          goto LABEL_63;
                      }
                      v98 = (unsigned __int8 *)v27;
                      if ( v24 != 40 )
                      {
                        if ( v24 == 38 )
                        {
                          v10 = v21;
                          v99 = 0;
                          v53 = sub_995B10(&v99, v21);
                          goto LABEL_125;
                        }
                        goto LABEL_63;
                      }
                      if ( *(_BYTE *)v21 > 0x15u )
                        goto LABEL_63;
                      if ( !(unsigned __int8)sub_AC30F0(v21) )
                      {
                        if ( *(_BYTE *)v21 == 17 )
                        {
                          if ( *(_DWORD *)(v21 + 32) <= 0x40u )
                          {
                            v53 = *(_QWORD *)(v21 + 24) == 0;
                          }
                          else
                          {
                            v89 = *(_DWORD *)(v21 + 32);
                            v53 = v89 == (unsigned int)sub_C444A0(v21 + 24);
                          }
LABEL_125:
                          if ( v53 )
                            goto LABEL_126;
LABEL_63:
                          if ( v23 == 17 )
                          {
                            v25 = v97;
                            goto LABEL_65;
                          }
LABEL_26:
                          result = (unsigned int)v102;
                          goto LABEL_41;
                        }
                        v92 = *(_QWORD *)(v21 + 8);
                        if ( (unsigned int)*(unsigned __int8 *)(v92 + 8) - 17 > 1 )
                          goto LABEL_63;
                        v10 = 0;
                        v59 = (_BYTE *)sub_AD7630(v21, 0);
                        v60 = 0;
                        if ( !v59 || *v59 != 17 )
                        {
                          if ( *(_BYTE *)(v92 + 8) == 17 )
                          {
                            v79 = *(_DWORD *)(v92 + 32);
                            if ( v79 )
                            {
                              v94 = a2;
                              v63 = 0;
                              while ( 1 )
                              {
                                v10 = v63;
                                v74 = v60;
                                v64 = (unsigned __int8 *)sub_AD69F0(v21, v63);
                                if ( !v64 )
                                  break;
                                v10 = *v64;
                                v60 = v74;
                                if ( (_BYTE)v10 != 13 )
                                {
                                  if ( (_BYTE)v10 != 17 )
                                    break;
                                  v60 = sub_9867B0((__int64)(v64 + 24));
                                  if ( !v60 )
                                    break;
                                }
                                if ( v79 == ++v63 )
                                {
                                  a2 = v94;
                                  goto LABEL_173;
                                }
                              }
                              a2 = v94;
                            }
                          }
                          goto LABEL_63;
                        }
                        v60 = sub_9867B0((__int64)(v59 + 24));
LABEL_173:
                        if ( !v60 )
                          goto LABEL_63;
                      }
LABEL_126:
                      v10 = (__int64)v98;
                      a3(a4, v98);
                      goto LABEL_63;
                    }
LABEL_102:
                    if ( v26 <= 0x1Cu )
                      v49 = *((unsigned __int16 *)v25 + 1);
                    else
LABEL_103:
                      v49 = v26 - 29;
                    if ( v49 == 15 && (v25[1] & 2) != 0 )
                    {
                      v50 = (unsigned __int8 *)*((_QWORD *)v25 - 8);
                      if ( v50 )
                      {
                        v10 = (__int64)a3;
                        v51 = (unsigned __int8 *)*((_QWORD *)v25 - 8);
                        v87 = v24;
                        v98 = v50;
                        sub_9854E0(v51, a3, a4);
                        v25 = v97;
                        v24 = v87;
                        v26 = *v97;
                        goto LABEL_61;
                      }
                    }
LABEL_60:
                    v26 = *v25;
                    goto LABEL_61;
                  }
                  v46 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                  if ( v46 )
                  {
                    v98 = (unsigned __int8 *)*((_QWORD *)v97 - 8);
                    v47 = (unsigned __int8 *)*((_QWORD *)v97 - 4);
                    if ( !v47 )
                    {
                      LOBYTE(v10) = *v97;
                      v26 = *v97;
                      goto LABEL_92;
                    }
LABEL_99:
                    v76 = v24;
                    v85 = v47;
                    sub_9854E0(v46, a3, a4);
                    v10 = (__int64)a3;
                    sub_9854E0(v85, a3, a4);
                    v86 = v97;
                    v48 = sub_987880(v97);
                    v25 = v86;
                    v24 = v76;
                    if ( v48 )
                    {
                      if ( !v86 )
                      {
                        v26 = MEMORY[0];
                        goto LABEL_61;
                      }
                      v26 = *v86;
                      goto LABEL_102;
                    }
                    goto LABEL_60;
                  }
                }
                v78 = v83;
                v93 = v97;
                v73 = v26;
                v61 = sub_987880(v97);
                v25 = v93;
                v24 = v78;
                v10 = (unsigned __int8)v10;
                if ( v61 )
                {
LABEL_94:
                  v45 = (unsigned __int8)v10 - 29;
                  goto LABEL_95;
                }
                v62 = sub_987880(v93);
                v25 = v93;
                v24 = v78;
                v26 = v73;
                if ( v62 )
                  goto LABEL_103;
LABEL_65:
                v33 = v25;
LABEL_66:
                if ( *v33 == 85 )
                {
                  v34 = *((_QWORD *)v33 - 4);
                  if ( v34 )
                  {
                    if ( !*(_BYTE *)v34 && *(_QWORD *)(v34 + 24) == *((_QWORD *)v33 + 10) && *(_DWORD *)(v34 + 36) == 66 )
                    {
                      v35 = *(unsigned __int8 **)&v33[-32 * (*((_DWORD *)v33 + 1) & 0x7FFFFFF)];
                      if ( v35 )
                      {
LABEL_72:
                        v98 = v35;
LABEL_73:
                        v10 = (__int64)a3;
                        sub_9854E0(v35, a3, a4);
                        result = (unsigned int)v102;
                        goto LABEL_41;
                      }
                    }
                  }
                }
                goto LABEL_26;
              }
              sub_9854E0(v97, a3, a4);
              v10 = (__int64)a3;
              sub_9854E0((unsigned __int8 *)v21, a3, a4);
              if ( v23 != 17 )
                goto LABEL_26;
              v33 = v97;
              v39 = *v97;
              if ( *v97 <= 0x1Cu )
                goto LABEL_66;
              if ( (unsigned int)v39 - 54 <= 2 )
              {
                v54 = (unsigned __int8 **)sub_986520((__int64)v97);
                v55 = *v54;
                if ( *v54 )
                {
                  v98 = *v54;
                  v90 = v55;
                  if ( **(_BYTE **)(sub_986520((__int64)v33) + 32) == 17 )
                  {
                    v10 = (__int64)a3;
                    sub_9854E0(v90, a3, a4);
                    v33 = v97;
                    goto LABEL_66;
                  }
                  v39 = *v33;
                }
              }
              if ( v39 == 57 )
              {
                v40 = (unsigned __int8 *)*((_QWORD *)v33 - 8);
                if ( !v40 )
                  goto LABEL_66;
                v98 = (unsigned __int8 *)*((_QWORD *)v33 - 8);
                v41 = (unsigned __int8 *)*((_QWORD *)v33 - 4);
                if ( v41 )
                  goto LABEL_84;
                v39 = *v33;
              }
              if ( v39 != 58 )
                goto LABEL_66;
              v40 = (unsigned __int8 *)*((_QWORD *)v33 - 8);
              if ( !v40 )
                goto LABEL_66;
              v98 = (unsigned __int8 *)*((_QWORD *)v33 - 8);
              v41 = (unsigned __int8 *)*((_QWORD *)v33 - 4);
              if ( !v41 )
                goto LABEL_66;
LABEL_84:
              sub_9854E0(v40, a3, a4);
              v10 = (__int64)a3;
              sub_9854E0(v41, a3, a4);
              v33 = v97;
              goto LABEL_66;
            }
            v17 = *(_BYTE *)v8;
            if ( *(_BYTE *)v8 > 0x1Cu )
            {
LABEL_151:
              if ( v17 == 83 )
              {
                if ( *(_QWORD *)(v8 - 64) )
                {
                  v97 = *(unsigned __int8 **)(v8 - 64);
                  v65 = *(unsigned __int8 **)(v8 - 32);
                  if ( v65 )
                  {
                    sub_B53900(v8);
                    v68 = v97;
                    if ( a2 )
                    {
                      sub_9854E0(v97, a3, a4);
                      sub_9854E0(v65, a3, a4);
                      v68 = v97;
                    }
                    else if ( *v65 <= 0x15u )
                    {
                      sub_9854E0(v97, a3, a4);
                      v68 = v97;
                    }
                    v10 = (__int64)v68;
                    v99 = &v97;
                    if ( (unsigned __int8)sub_995E90(&v99, (unsigned __int64)v68, v66, v67, (__int64)&v99) )
                    {
                      v10 = (__int64)a3;
                      sub_9854E0(v97, a3, a4);
                    }
                    if ( *v97 == 85 )
                    {
                      v69 = *((_QWORD *)v97 - 4);
                      if ( v69 )
                      {
                        if ( !*(_BYTE *)v69
                          && *(_QWORD *)(v69 + 24) == *((_QWORD *)v97 + 10)
                          && *(_DWORD *)(v69 + 36) == 170 )
                        {
                          v70 = *(unsigned __int8 **)&v97[-32 * (*((_DWORD *)v97 + 1) & 0x7FFFFFF)];
                          if ( v70 )
                          {
                            v97 = *(unsigned __int8 **)&v97[-32 * (*((_DWORD *)v97 + 1) & 0x7FFFFFF)];
                            v10 = (__int64)a3;
                            sub_9854E0(v70, a3, a4);
                            result = (unsigned int)v102;
                            goto LABEL_41;
                          }
                        }
                      }
                    }
                    goto LABEL_26;
                  }
                }
              }
              else if ( v17 == 85 )
              {
                v57 = *(_QWORD *)(v8 - 32);
                if ( v57 )
                {
                  if ( !*(_BYTE *)v57 && *(_QWORD *)(v57 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v57 + 36) == 207 )
                  {
                    v35 = *(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
                    if ( v35 )
                    {
                      v97 = *(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
                      if ( *(_BYTE *)v8 == 85 )
                        goto LABEL_73;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_25:
        if ( !a2 )
        {
          if ( *(_BYTE *)v8 == 67 )
          {
            v35 = *(unsigned __int8 **)(v8 - 32);
            if ( v35 )
              goto LABEL_72;
          }
          v10 = 30;
          v99 = 0;
          v100 = &v98;
          if ( (unsigned __int8)sub_996420(&v99, 30, (unsigned __int8 *)v8) )
          {
            v37 = (unsigned int)v102;
            v19 = v98;
            v38 = (unsigned int)v102 + 1LL;
            if ( v38 > HIDWORD(v102) )
              goto LABEL_118;
            goto LABEL_77;
          }
        }
        goto LABEL_26;
      }
    }
LABEL_40:
    result = (unsigned int)v102;
    if ( !a2 )
    {
      v36 = v97;
      if ( (unsigned __int64)(unsigned int)v102 + 1 > HIDWORD(v102) )
      {
        v10 = (__int64)v103;
        sub_C8D5F0(&v101, v103, (unsigned int)v102 + 1LL, 8);
        result = (unsigned int)v102;
      }
      v101[result] = v36;
      v37 = (unsigned int)(v102 + 1);
      v38 = v37 + 1;
      LODWORD(v102) = v102 + 1;
      if ( v37 + 1 > (unsigned __int64)HIDWORD(v102) )
      {
LABEL_118:
        v10 = (__int64)v103;
        sub_C8D5F0(&v101, v103, v38, 8);
        v37 = (unsigned int)v102;
      }
LABEL_77:
      v101[v37] = v19;
      result = (unsigned int)(v102 + 1);
      LODWORD(v102) = v102 + 1;
    }
LABEL_41:
    v6 = v108;
LABEL_8:
    if ( !(_DWORD)result )
      break;
    v5 = v101;
  }
  if ( !v6 )
    result = _libc_free(v105, v10);
  if ( v101 != v103 )
    return _libc_free(v101, v10);
  return result;
}
