// Function: sub_27BA8D0
// Address: 0x27ba8d0
//
__int64 *__fastcall sub_27BA8D0(__int64 *a1, unsigned __int8 *a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 *v6; // r14
  __int64 v8; // r13
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 *v16; // rax
  unsigned __int8 **v17; // rax
  char *v18; // r13
  unsigned __int8 *v19; // r12
  unsigned __int8 **v20; // rbx
  __int64 *v21; // rbx
  __int64 *v22; // r15
  unsigned __int64 *v23; // r13
  __int64 v24; // r14
  __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 *v27; // rdi
  char v28; // dl
  __int64 v29; // rcx
  __int64 v30; // r12
  unsigned __int8 **v31; // r15
  __int64 v32; // rdx
  __int64 v33; // r12
  unsigned __int8 **v34; // r12
  char *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int8 *v41; // r13
  unsigned __int8 **v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // r14
  unsigned __int64 *v48; // r12
  __int64 v49; // r13
  _QWORD *v50; // rax
  _QWORD *v51; // rbx
  char v52; // dl
  const char *v53; // rbx
  unsigned __int64 *v54; // r14
  __int64 v55; // r15
  __int64 v56; // rdx
  _QWORD *v57; // rax
  _QWORD *v58; // rax
  unsigned int v59; // ebx
  __int64 *v60; // rdx
  unsigned int v61; // edi
  _QWORD *v62; // rdx
  unsigned __int8 **v63; // rax
  _QWORD *v64; // rax
  _QWORD *v65; // rbx
  int v66; // eax
  unsigned int v67; // r10d
  __int64 *v68; // rsi
  int v69; // edi
  unsigned int v70; // r15d
  unsigned int v71; // esi
  __int64 v72; // r11
  __int64 *v73; // rdi
  __int64 *v74; // [rsp+10h] [rbp-350h]
  __int64 ***v75; // [rsp+18h] [rbp-348h]
  _QWORD *v76; // [rsp+20h] [rbp-340h]
  _QWORD *v77; // [rsp+20h] [rbp-340h]
  unsigned __int8 **v78; // [rsp+28h] [rbp-338h]
  __int64 ***v79; // [rsp+28h] [rbp-338h]
  __int64 *v80; // [rsp+30h] [rbp-330h]
  __int64 *v81; // [rsp+30h] [rbp-330h]
  __int64 *v82; // [rsp+38h] [rbp-328h]
  __int64 v83; // [rsp+40h] [rbp-320h]
  const char *v84; // [rsp+40h] [rbp-320h]
  __int64 *v85; // [rsp+48h] [rbp-318h]
  __int64 *v86; // [rsp+58h] [rbp-308h] BYREF
  _QWORD v87[2]; // [rsp+60h] [rbp-300h] BYREF
  char v88; // [rsp+70h] [rbp-2F0h]
  __int64 v89; // [rsp+80h] [rbp-2E0h] BYREF
  unsigned __int8 *v90; // [rsp+88h] [rbp-2D8h]
  __int64 v91; // [rsp+90h] [rbp-2D0h]
  unsigned int v92; // [rsp+98h] [rbp-2C8h]
  const char *v93; // [rsp+A0h] [rbp-2C0h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-2B8h]
  const char *v95; // [rsp+B0h] [rbp-2B0h]
  __int16 v96; // [rsp+C0h] [rbp-2A0h]
  __int64 *v97; // [rsp+D0h] [rbp-290h] BYREF
  __int64 v98; // [rsp+D8h] [rbp-288h]
  _QWORD v99[16]; // [rsp+E0h] [rbp-280h] BYREF
  __int64 *v100; // [rsp+160h] [rbp-200h] BYREF
  __int64 v101; // [rsp+168h] [rbp-1F8h]
  _BYTE v102[128]; // [rsp+170h] [rbp-1F0h] BYREF
  __int64 v103; // [rsp+1F0h] [rbp-170h] BYREF
  __int64 *v104; // [rsp+1F8h] [rbp-168h]
  __int64 v105; // [rsp+200h] [rbp-160h]
  int v106; // [rsp+208h] [rbp-158h]
  char v107; // [rsp+20Ch] [rbp-154h]
  char v108; // [rsp+210h] [rbp-150h] BYREF
  const char *v109; // [rsp+290h] [rbp-D0h] BYREF
  char *v110; // [rsp+298h] [rbp-C8h]
  __int64 v111; // [rsp+2A0h] [rbp-C0h]
  int v112; // [rsp+2A8h] [rbp-B8h]
  char v113; // [rsp+2ACh] [rbp-B4h]
  char v114; // [rsp+2B0h] [rbp-B0h] BYREF
  char v115; // [rsp+2B1h] [rbp-AFh]

  v4 = (__int64)(a3 - 3);
  v6 = a1;
  v85 = (__int64 *)a2;
  if ( !a3 )
    v4 = 0;
  v8 = v4;
  if ( sub_98ED70(a2, 0, v4, *a1, 0) )
    return v85;
  sub_27B9170((__int64)v87, (__int64)a2, (__int64 **)*a1);
  if ( !v88 )
  {
    v115 = 1;
    v109 = "gw.freeze";
    v114 = 3;
    v64 = sub_BD2C40(72, unk_3F10A14);
    v65 = v64;
    if ( v64 )
      sub_B549F0((__int64)v64, (__int64)a2, (__int64)&v109, 0, 0);
    sub_B44150(v65, *(_QWORD *)(v8 + 40), a3, a4);
    return v65;
  }
  if ( *a2 <= 0x15u )
  {
    v115 = 1;
    v47 = v87[0];
    v109 = "gw.freeze";
    v114 = 3;
    v48 = (unsigned __int64 *)v87[0];
    v49 = v87[1];
    v50 = sub_BD2C40(72, unk_3F10A14);
    v51 = v50;
    if ( v50 )
      sub_B549F0((__int64)v50, (__int64)a2, (__int64)&v109, 0, 0);
    if ( !v47 )
      BUG();
    sub_B44150(v51, *(_QWORD *)(v47 + 16), v48, v49);
    return v51;
  }
  v99[0] = a2;
  v104 = (__int64 *)&v108;
  v97 = v99;
  v110 = &v114;
  v100 = (__int64 *)v102;
  v101 = 0x1000000000LL;
  v98 = 0x1000000001LL;
  v12 = 1;
  v103 = 0;
  v105 = 16;
  v106 = 0;
  v107 = 1;
  v109 = 0;
  v111 = 16;
  v112 = 0;
  v113 = 1;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v83 = v8;
  do
  {
    v13 = v97;
    v14 = v12;
    v15 = v97[v12 - 1];
    LODWORD(v98) = v12 - 1;
    if ( !v107 )
      goto LABEL_28;
    v16 = v104;
    v14 = HIDWORD(v105);
    v13 = &v104[HIDWORD(v105)];
    if ( v104 != v13 )
    {
      do
      {
        if ( v15 == *v16 )
          goto LABEL_13;
        ++v16;
      }
      while ( v13 != v16 );
    }
    if ( HIDWORD(v105) < (unsigned int)v105 )
    {
      ++HIDWORD(v105);
      *v13 = v15;
      ++v103;
    }
    else
    {
LABEL_28:
      a2 = (unsigned __int8 *)v15;
      sub_C8CC70((__int64)&v103, v15, (__int64)v13, v14, v10, v11);
      if ( !v28 )
        goto LABEL_13;
    }
    a2 = 0;
    if ( sub_98ED70((unsigned __int8 *)v15, 0, v83, *v6, 0) )
      goto LABEL_13;
    if ( *(_BYTE *)v15 > 0x1Cu )
    {
      a2 = 0;
      if ( !sub_98CD60((unsigned __int8 *)v15, 0) )
      {
        v81 = (__int64 *)v15;
        v30 = 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
        {
          v31 = *(unsigned __int8 ***)(v15 - 8);
          v78 = &v31[(unsigned __int64)v30 / 8];
        }
        else
        {
          v78 = (unsigned __int8 **)v15;
          v31 = (unsigned __int8 **)(v15 - v30);
        }
        v32 = v30 >> 5;
        v33 = v30 >> 7;
        if ( v33 )
        {
          v34 = &v31[16 * v33];
          while ( 1 )
          {
            a2 = *v31;
            if ( **v31 > 0x1Cu )
            {
              sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
              if ( !(_BYTE)v95 )
                goto LABEL_42;
            }
            a2 = v31[4];
            if ( *a2 > 0x1Cu )
            {
              sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
              if ( !(_BYTE)v95 )
              {
                v31 += 4;
                goto LABEL_42;
              }
            }
            a2 = v31[8];
            if ( *a2 > 0x1Cu )
            {
              sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
              if ( !(_BYTE)v95 )
              {
                v31 += 8;
                goto LABEL_42;
              }
            }
            a2 = v31[12];
            if ( *a2 > 0x1Cu )
            {
              sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
              if ( !(_BYTE)v95 )
                break;
            }
            v31 += 16;
            if ( v31 == v34 )
            {
              v32 = ((char *)v78 - (char *)v31) >> 5;
              goto LABEL_121;
            }
          }
          v31 += 12;
LABEL_42:
          if ( v78 == v31 )
            goto LABEL_43;
          goto LABEL_72;
        }
LABEL_121:
        if ( v32 != 2 )
        {
          if ( v32 != 3 )
          {
            if ( v32 != 1 )
              goto LABEL_43;
            goto LABEL_124;
          }
          a2 = *v31;
          if ( **v31 > 0x1Cu )
          {
            sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
            if ( !(_BYTE)v95 )
              goto LABEL_42;
          }
          v31 += 4;
        }
        a2 = *v31;
        if ( **v31 > 0x1Cu )
        {
          sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
          if ( !(_BYTE)v95 )
            goto LABEL_42;
        }
        v31 += 4;
LABEL_124:
        a2 = *v31;
        if ( **v31 > 0x1Cu )
        {
          sub_27B9170((__int64)&v93, (__int64)a2, (__int64 **)*v6);
          if ( !(_BYTE)v95 )
            goto LABEL_42;
        }
LABEL_43:
        if ( v113 )
        {
          v35 = v110;
          v29 = HIDWORD(v111);
          v32 = (__int64)&v110[8 * HIDWORD(v111)];
          if ( v110 != (char *)v32 )
          {
            while ( v15 != *(_QWORD *)v35 )
            {
              v35 += 8;
              if ( (char *)v32 == v35 )
                goto LABEL_127;
            }
LABEL_48:
            v36 = 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
            v37 = v15 - v36;
            if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
            {
              v37 = *(_QWORD *)(v15 - 8);
              v81 = (__int64 *)(v37 + v36);
            }
            if ( (__int64 *)v37 != v81 )
            {
              v79 = (__int64 ***)v6;
              v38 = (__int64 *)v37;
              while ( 1 )
              {
                v41 = (unsigned __int8 *)*v38;
                if ( *(_BYTE *)*v38 > 0x15u )
                {
                  v39 = (unsigned int)v98;
                  v29 = HIDWORD(v98);
                  v40 = (unsigned int)v98 + 1LL;
                  if ( v40 > HIDWORD(v98) )
                  {
                    a2 = (unsigned __int8 *)v99;
                    sub_C8D5F0((__int64)&v97, v99, v40, 8u, v10, v11);
                    v39 = (unsigned int)v98;
                  }
                  v37 = (__int64)v97;
                  v97[v39] = (__int64)v41;
                  LODWORD(v98) = v98 + 1;
                }
                else
                {
                  if ( !v107 )
                    goto LABEL_87;
                  v42 = (unsigned __int8 **)v104;
                  v29 = HIDWORD(v105);
                  v37 = (__int64)&v104[HIDWORD(v105)];
                  if ( v104 != (__int64 *)v37 )
                  {
                    while ( v41 != *v42 )
                    {
                      if ( (unsigned __int8 **)v37 == ++v42 )
                        goto LABEL_96;
                    }
                    goto LABEL_62;
                  }
LABEL_96:
                  if ( HIDWORD(v105) < (unsigned int)v105 )
                  {
                    ++HIDWORD(v105);
                    *(_QWORD *)v37 = v41;
                    ++v103;
                  }
                  else
                  {
LABEL_87:
                    sub_C8CC70((__int64)&v103, *v38, v37, v29, v10, v11);
                    if ( !v52 )
                      goto LABEL_62;
                  }
                  a2 = 0;
                  if ( !sub_98ED70(v41, 0, v83, (__int64)*v79, 0) )
                  {
                    sub_27B9170((__int64)&v93, (__int64)v41, *v79);
                    v53 = v93;
                    v54 = (unsigned __int64 *)v93;
                    v55 = v94;
                    v93 = sub_BD5D20((__int64)v41);
                    v95 = ".gw.fr";
                    v96 = 773;
                    v94 = v56;
                    v57 = sub_BD2C40(72, unk_3F10A14);
                    if ( v57 )
                    {
                      v76 = v57;
                      sub_B549F0((__int64)v57, (__int64)v41, (__int64)&v93, 0, 0);
                      v57 = v76;
                    }
                    if ( !v53 )
                      BUG();
                    v77 = v57;
                    sub_B44150(v57, *((_QWORD *)v53 + 2), v54, v55);
                    v58 = v77;
                    if ( v92 )
                    {
                      v59 = v92 - 1;
                      v11 = 1;
                      v60 = 0;
                      v61 = (v92 - 1) & (((unsigned int)v41 >> 4) ^ ((unsigned int)v41 >> 9));
                      v29 = (__int64)&v90[16 * v61];
                      v10 = *(_QWORD *)v29;
                      if ( *(unsigned __int8 **)v29 == v41 )
                      {
LABEL_94:
                        v62 = (_QWORD *)(v29 + 8);
                        goto LABEL_95;
                      }
                      while ( v10 != -4096 )
                      {
                        if ( v10 == -8192 && !v60 )
                          v60 = (__int64 *)v29;
                        v67 = v11 + 1;
                        v11 = v61 + (unsigned int)v11;
                        v61 = v59 & v11;
                        v29 = (__int64)&v90[16 * (v59 & (unsigned int)v11)];
                        v10 = *(_QWORD *)v29;
                        if ( v41 == *(unsigned __int8 **)v29 )
                          goto LABEL_94;
                        v11 = v67;
                      }
                      if ( !v60 )
                        v60 = (__int64 *)v29;
                      ++v89;
                      v29 = (unsigned int)(v91 + 1);
                      if ( 4 * (int)v29 < 3 * v92 )
                      {
                        v10 = v92 >> 3;
                        if ( v92 - HIDWORD(v91) - (unsigned int)v29 <= (unsigned int)v10 )
                        {
                          sub_27BA6F0((__int64)&v89, v92);
                          if ( !v92 )
                            goto LABEL_178;
                          v68 = 0;
                          v69 = 1;
                          v70 = (v92 - 1) & (((unsigned int)v41 >> 4) ^ ((unsigned int)v41 >> 9));
                          v29 = (unsigned int)(v91 + 1);
                          v58 = v77;
                          v60 = (__int64 *)&v90[16 * v70];
                          v10 = *v60;
                          if ( v41 != (unsigned __int8 *)*v60 )
                          {
                            while ( v10 != -4096 )
                            {
                              if ( v10 == -8192 && !v68 )
                                v68 = v60;
                              v11 = (unsigned int)(v69 + 1);
                              v70 = (v92 - 1) & (v69 + v70);
                              v60 = (__int64 *)&v90[16 * v70];
                              v10 = *v60;
                              if ( v41 == (unsigned __int8 *)*v60 )
                                goto LABEL_147;
                              ++v69;
                            }
                            if ( v68 )
                              v60 = v68;
                          }
                        }
LABEL_147:
                        LODWORD(v91) = v29;
                        if ( *v60 != -4096 )
                          --HIDWORD(v91);
                        *v60 = (__int64)v41;
                        v62 = v60 + 1;
                        *v62 = 0;
LABEL_95:
                        *v62 = v58;
LABEL_62:
                        v37 = v92;
                        a2 = v90;
                        if ( v92 )
                        {
                          v29 = (v92 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                          v43 = (__int64 *)&v90[16 * v29];
                          v10 = *v43;
                          if ( v41 == (unsigned __int8 *)*v43 )
                          {
LABEL_64:
                            v37 = (__int64)&v90[16 * v92];
                            if ( v43 != (__int64 *)v37 )
                            {
                              v44 = v43[1];
                              if ( *v38 )
                              {
                                v29 = v38[2];
                                v37 = v38[1];
                                *(_QWORD *)v29 = v37;
                                if ( v37 )
                                {
                                  v29 = v38[2];
                                  *(_QWORD *)(v37 + 16) = v29;
                                }
                              }
                              *v38 = v44;
                              if ( v44 )
                              {
                                v37 = *(_QWORD *)(v44 + 16);
                                v29 = v44 + 16;
                                v38[1] = v37;
                                if ( v37 )
                                {
                                  a2 = (unsigned __int8 *)(v38 + 1);
                                  *(_QWORD *)(v37 + 16) = v38 + 1;
                                }
                                v38[2] = v29;
                                *(_QWORD *)(v44 + 16) = v38;
                              }
                            }
                          }
                          else
                          {
                            v66 = 1;
                            while ( v10 != -4096 )
                            {
                              v11 = (unsigned int)(v66 + 1);
                              v29 = (v92 - 1) & (v66 + (_DWORD)v29);
                              v43 = (__int64 *)&v90[16 * (unsigned int)v29];
                              v10 = *v43;
                              if ( v41 == (unsigned __int8 *)*v43 )
                                goto LABEL_64;
                              v66 = v11;
                            }
                          }
                        }
                        goto LABEL_55;
                      }
                    }
                    else
                    {
                      ++v89;
                    }
                    sub_27BA6F0((__int64)&v89, 2 * v92);
                    if ( !v92 )
                    {
LABEL_178:
                      LODWORD(v91) = v91 + 1;
                      BUG();
                    }
                    v29 = (unsigned int)(v91 + 1);
                    v58 = v77;
                    v71 = (v92 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                    v60 = (__int64 *)&v90[16 * v71];
                    v72 = *v60;
                    if ( v41 != (unsigned __int8 *)*v60 )
                    {
                      v10 = 1;
                      v73 = 0;
                      while ( v72 != -4096 )
                      {
                        if ( v72 == -8192 && !v73 )
                          v73 = v60;
                        v11 = (unsigned int)(v10 + 1);
                        v71 = (v92 - 1) & (v10 + v71);
                        v60 = (__int64 *)&v90[16 * v71];
                        v72 = *v60;
                        if ( v41 == (unsigned __int8 *)*v60 )
                          goto LABEL_147;
                        v10 = (unsigned int)v11;
                      }
                      if ( v73 )
                        v60 = v73;
                    }
                    goto LABEL_147;
                  }
                }
LABEL_55:
                v38 += 4;
                if ( v81 == v38 )
                {
                  v6 = (__int64 *)v79;
                  break;
                }
              }
            }
LABEL_13:
            v12 = v98;
            continue;
          }
LABEL_127:
          if ( HIDWORD(v111) < (unsigned int)v111 )
          {
            v29 = (unsigned int)++HIDWORD(v111);
            *(_QWORD *)v32 = v15;
            ++v109;
            goto LABEL_48;
          }
        }
        a2 = (unsigned __int8 *)v15;
        sub_C8CC70((__int64)&v109, v15, v32, v29, v10, v11);
        goto LABEL_48;
      }
    }
LABEL_72:
    v45 = (unsigned int)v101;
    v46 = (unsigned int)v101 + 1LL;
    if ( v46 > HIDWORD(v101) )
    {
      a2 = v102;
      sub_C8D5F0((__int64)&v100, v102, v46, 8u, v10, v11);
      v45 = (unsigned int)v101;
    }
    v100[v45] = v15;
    v12 = v98;
    LODWORD(v101) = v101 + 1;
  }
  while ( v12 );
  v17 = (unsigned __int8 **)v110;
  if ( v113 )
    v18 = &v110[8 * HIDWORD(v111)];
  else
    v18 = &v110[8 * (unsigned int)v111];
  if ( v110 != v18 )
  {
    while ( 1 )
    {
      v19 = *v17;
      v20 = v17;
      if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v18 == (char *)++v17 )
        goto LABEL_20;
    }
    while ( v18 != (char *)v20 )
    {
      sub_B44F30(v19);
      sub_B44B50((__int64 *)v19, (__int64)a2);
      sub_B44A60((__int64)v19);
      v63 = v20 + 1;
      if ( v20 + 1 == (unsigned __int8 **)v18 )
        break;
      while ( 1 )
      {
        v19 = *v63;
        v20 = v63;
        if ( (unsigned __int64)*v63 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v18 == (char *)++v63 )
          goto LABEL_20;
      }
    }
  }
LABEL_20:
  v21 = v100;
  v80 = &v100[(unsigned int)v101];
  if ( v80 == v100 )
  {
    v74 = v85;
  }
  else
  {
    v75 = (__int64 ***)v6;
    v74 = v85;
    do
    {
      v22 = (__int64 *)*v21;
      sub_27B9170((__int64)&v93, *v21, *v75);
      v23 = (unsigned __int64 *)v93;
      v24 = v94;
      v84 = v93;
      v93 = sub_BD5D20((__int64)v22);
      v95 = ".gw.fr";
      v96 = 773;
      v94 = v25;
      v26 = sub_BD2C40(72, unk_3F10A14);
      v27 = v26;
      if ( v26 )
      {
        v82 = v26;
        sub_B549F0((__int64)v26, (__int64)v22, (__int64)&v93, 0, 0);
        v27 = v82;
      }
      v86 = v27;
      if ( !v84 )
        BUG();
      sub_B44150(v27, *((_QWORD *)v84 + 2), v23, v24);
      if ( v22 == v85 )
        v74 = v86;
      ++v21;
      v93 = (const char *)&v86;
      sub_BD79D0(v22, v86, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_27B8DE0, (__int64)&v93);
    }
    while ( v80 != v21 );
  }
  sub_C7D6A0((__int64)v90, 16LL * v92, 8);
  if ( v100 != (__int64 *)v102 )
    _libc_free((unsigned __int64)v100);
  if ( !v113 )
    _libc_free((unsigned __int64)v110);
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  if ( !v107 )
    _libc_free((unsigned __int64)v104);
  return v74;
}
