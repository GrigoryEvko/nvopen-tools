// Function: sub_7C6D90
// Address: 0x7c6d90
//
_QWORD *__fastcall sub_7C6D90(__int64 a1, int a2, _DWORD *a3, __int64 a4, _QWORD *a5)
{
  char v5; // ch
  __int64 v6; // rax
  char v7; // bl
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 *v10; // rbx
  __int64 j; // r12
  __int64 *v12; // r13
  __int64 *v13; // rax
  _QWORD *v14; // r15
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned int *v17; // rsi
  __int64 v18; // rdx
  __int64 i; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r15
  unsigned __int8 v23; // di
  char v24; // al
  __int64 *v25; // rax
  __int64 v26; // r11
  char v27; // al
  __int64 v28; // r13
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 ***v35; // rax
  __int64 **v36; // rbx
  __int64 v37; // rax
  __int64 **v38; // rcx
  int v39; // eax
  bool v40; // al
  char v41; // al
  char v42; // al
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rax
  _QWORD *v49; // r15
  __int64 v50; // r10
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // rax
  __int64 *v55; // r13
  __int64 v56; // rbx
  char v57; // si
  char v58; // dl
  __int64 v59; // rbx
  char v60; // r12
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  _QWORD *v66; // rax
  __int64 v67; // r11
  __int64 v68; // rsi
  __int64 v69; // rdi
  char v70; // al
  _QWORD *v71; // rax
  __int64 v72; // r11
  _QWORD *v73; // r10
  char v74; // al
  __int64 v75; // rax
  _QWORD *v76; // r10
  char v77; // dl
  __int64 v78; // rax
  _BYTE *v79; // rax
  __int64 *v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r8
  __int64 v84; // rsi
  __int64 *v85; // rax
  __int64 v86; // r8
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // r11
  __int64 v90; // rax
  int v91; // eax
  __int64 v92; // r11
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  bool v100; // [rsp+1Ch] [rbp-C4h]
  __int64 v101; // [rsp+20h] [rbp-C0h]
  __int64 v102; // [rsp+28h] [rbp-B8h]
  unsigned int v103; // [rsp+30h] [rbp-B0h]
  char v104; // [rsp+36h] [rbp-AAh]
  char v105; // [rsp+37h] [rbp-A9h]
  bool v107; // [rsp+40h] [rbp-A0h]
  __int64 *v108; // [rsp+48h] [rbp-98h]
  _QWORD *v109; // [rsp+48h] [rbp-98h]
  __int64 v110; // [rsp+48h] [rbp-98h]
  int v111; // [rsp+50h] [rbp-90h]
  __int64 *v112; // [rsp+50h] [rbp-90h]
  int v113; // [rsp+58h] [rbp-88h]
  __int64 v114; // [rsp+60h] [rbp-80h]
  int v115; // [rsp+68h] [rbp-78h]
  __int64 v116; // [rsp+68h] [rbp-78h]
  unsigned int v117; // [rsp+68h] [rbp-78h]
  __int64 v118; // [rsp+68h] [rbp-78h]
  unsigned int v119; // [rsp+68h] [rbp-78h]
  __int64 v120; // [rsp+68h] [rbp-78h]
  __int64 v121; // [rsp+68h] [rbp-78h]
  __int64 *v122; // [rsp+68h] [rbp-78h]
  __int64 v123; // [rsp+68h] [rbp-78h]
  __int64 v124; // [rsp+68h] [rbp-78h]
  _QWORD *v125; // [rsp+68h] [rbp-78h]
  __int64 v126; // [rsp+70h] [rbp-70h]
  __int64 v127; // [rsp+70h] [rbp-70h]
  __int64 v128; // [rsp+70h] [rbp-70h]
  __int64 v129; // [rsp+70h] [rbp-70h]
  __int64 v130; // [rsp+78h] [rbp-68h]
  int v131; // [rsp+78h] [rbp-68h]
  _QWORD *v132; // [rsp+78h] [rbp-68h]
  _QWORD *v133; // [rsp+78h] [rbp-68h]
  __int64 v134; // [rsp+78h] [rbp-68h]
  __int64 v135; // [rsp+78h] [rbp-68h]
  __int64 v136; // [rsp+78h] [rbp-68h]
  _QWORD *v137; // [rsp+78h] [rbp-68h]
  _QWORD *v138; // [rsp+78h] [rbp-68h]
  int v139; // [rsp+88h] [rbp-58h] BYREF
  int v140; // [rsp+8Ch] [rbp-54h] BYREF
  int v141; // [rsp+90h] [rbp-50h] BYREF
  int v142; // [rsp+94h] [rbp-4Ch] BYREF
  _BYTE *v143; // [rsp+98h] [rbp-48h] BYREF
  unsigned __int64 v144; // [rsp+A0h] [rbp-40h] BYREF
  _QWORD v145[7]; // [rsp+A8h] [rbp-38h] BYREF

  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v7 = *(_BYTE *)(v6 + 12);
  v104 = v7 & 1;
  *(_BYTE *)(v6 + 12) = v7 | 1;
  *a5 = -1;
  v8 = *(_QWORD *)(a1 + 88);
  v126 = **(_QWORD **)(v8 + 32);
  if ( v126 )
  {
    if ( a2 && (*(_BYTE *)(v126 + 56) & 0x10) == 0 )
      v126 = *(_QWORD *)v126;
  }
  else
  {
    *a3 = 1;
  }
  v100 = 0;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 && *(char *)(*(_QWORD *)(a1 + 64) + 177LL) < 0 )
    v100 = v5 >= 0;
  v102 = 0;
  v101 = v126;
  if ( *(_BYTE *)(a1 + 80) == 19 && (*(_BYTE *)(v8 + 266) & 1) != 0 )
  {
    v80 = *(__int64 **)(*(_QWORD *)(a1 + 88) + 208LL);
    if ( v80 )
    {
      v81 = *v80;
      switch ( *(_BYTE *)(*v80 + 80) )
      {
        case 4:
        case 5:
          v95 = *(_QWORD *)(*(_QWORD *)(v81 + 96) + 80LL);
          break;
        case 6:
          v95 = *(_QWORD *)(*(_QWORD *)(v81 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v95 = *(_QWORD *)(*(_QWORD *)(v81 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v95 = *(_QWORD *)(v81 + 88);
          break;
        default:
          BUG();
      }
      v102 = 0;
      v101 = **(_QWORD **)(v95 + 32);
    }
    else
    {
      v102 = *(_QWORD *)(v8 + 200);
      v94 = *(_QWORD *)(v102 + 88);
      if ( (*(_BYTE *)(v94 + 160) & 6) == 0 )
        v101 = **(_QWORD **)(v94 + 32);
    }
  }
  v9 = 0;
  v10 = 0;
  v115 = 0;
  j = v101;
  v114 = 0;
  v108 = 0;
  v111 = 0;
  v107 = !v100;
  v130 = 0;
LABEL_8:
  if ( dword_4F04C44 != -1
    || (v78 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v78 + 6) & 6) != 0)
    || *(_BYTE *)(v78 + 4) == 12 )
  {
    v12 = (__int64 *)(j | v126);
    if ( !(j | v126) )
    {
      v126 = 0;
      v49 = (_QWORD *)v130;
      j = 0;
      v105 = 0;
      goto LABEL_156;
    }
  }
  if ( j )
  {
    v13 = v10;
    v14 = (_QWORD *)v130;
    v15 = (_QWORD *)v126;
    while ( 1 )
    {
      if ( *(char *)(j + 56) >= 0 )
      {
        v130 = (__int64)v14;
        v126 = (__int64)v15;
        v10 = v13;
        goto LABEL_22;
      }
      v13 = sub_725090(3u);
      if ( !v14 )
        v14 = v13;
      if ( v9 )
        *v9 = (__int64)v13;
      j = *(_QWORD *)j;
      if ( v15 )
        v15 = (_QWORD *)*v15;
      if ( !j )
        break;
      v9 = v13;
    }
    v126 = (__int64)v15;
    v9 = v13;
    v10 = v13;
    v130 = (__int64)v14;
  }
LABEL_22:
  v16 = (unsigned __int64)&v144;
  v17 = 0;
  v12 = 0;
  for ( i = (unsigned int)sub_868D90(&v144, 0, 0, 0, 1); ; i = 0 )
  {
    while ( 1 )
    {
      LOBYTE(v18) = v126 == 0 || j == 0;
      v105 = v18;
      if ( (_BYTE)v18 )
        break;
      if ( (_DWORD)i )
      {
        v22 = v10;
        while ( v12 )
        {
          if ( !j )
          {
            v49 = (_QWORD *)v130;
            sub_867030(v144);
            goto LABEL_156;
          }
LABEL_28:
          v23 = 0;
          ++*(_BYTE *)(qword_4F061C8 + 75LL);
          v24 = *(_BYTE *)(*(_QWORD *)(j + 8) + 80LL);
          if ( v24 != 3 )
            v23 = (v24 != 2) + 1;
          v116 = *(_QWORD *)(j + 8);
          v25 = sub_725090(v23);
          v26 = v116;
          v22 = v25;
          v27 = *((_BYTE *)v25 + 8);
          if ( v27 )
          {
            if ( v27 == 1 )
            {
              v50 = *(_QWORD *)(*(_QWORD *)(v116 + 88) + 128LL);
              if ( (*(_BYTE *)(j + 72) & 1) != 0 && v107 )
              {
                v17 = (unsigned int *)v116;
                v50 = sub_8AE7F0(a1, v116, j, v130, 0, 0);
              }
              if ( word_4F06418[0] == 287 )
              {
                v120 = v50;
                v54 = sub_7BE320(0, v17);
                if ( !v54 )
                {
                  v17 = 0;
                  sub_867630(v144, 0);
                  v16 = v144;
                  v18 = (unsigned int)sub_866C00(v144);
                  --*(_BYTE *)(qword_4F061C8 + 75LL);
                  goto LABEL_83;
                }
                v16 = v54[4];
                v12 = (__int64 *)*v54;
                v22 = v54;
                *v54 = 0;
                v29 = 0;
                if ( *(_BYTE *)(v16 + 173) == 1 )
                {
                  sub_712540((const __m128i *)v16, v120, 1, 0, &v142, v145);
                  v29 = 0;
                }
              }
              else if ( v12 )
              {
                v16 = v12[4];
                v51 = (__int64 *)*v12;
                *v12 = 0;
                if ( *(_BYTE *)(v16 + 173) == 1 )
                {
                  v122 = v51;
                  v22 = v12;
                  sub_712540((const __m128i *)v16, v50, 1, 0, &v142, v145);
                  v12 = v122;
                }
                else
                {
                  v22 = v12;
                  v12 = v51;
                }
                v29 = 1;
              }
              else
              {
                v121 = v50;
                v143 = sub_724D80(0);
                sub_6D6050(v121, (__int64)v143, v130, v101);
                v16 = (unsigned __int64)v143;
                if ( sub_72A990((__int64)v143) )
                {
                  sub_6851C0(0x1DBu, v145);
                  v16 = (unsigned __int64)v143;
                  sub_72C970((__int64)v143);
                }
                v22[4] = (__int64)v143;
                v29 = 0;
              }
            }
            else
            {
              v44 = *(_QWORD *)(j + 64);
              v16 = *(_QWORD *)(v44 + 104);
              if ( (*(_BYTE *)(v44 + 266) & 4) != 0 && v107 )
              {
                v52 = sub_8B0DE0(a1, j, v130);
                v26 = v116;
                v22[5] = v52;
                v16 = v52;
              }
              if ( v12 )
              {
                v16 = 2877;
                sub_6854C0(0xB3Du, (FILE *)v145, v26);
                v45 = *(_QWORD *)(*(_QWORD *)(((__int64 (*)(void))sub_87F550)() + 88) + 104LL);
              }
              else
              {
                v45 = sub_7C68A0((__int64 *)v16, (FILE *)v145, 0, 0);
              }
              v22[4] = v45;
              v12 = 0;
              v29 = 0;
            }
          }
          else
          {
            if ( v12 )
            {
              sub_6854C0(0xB3Cu, (FILE *)v145, v116);
              v28 = sub_72C930();
            }
            else
            {
              v28 = sub_65CFF0(0, 0);
            }
            v16 = v28;
            if ( (unsigned int)sub_8DD0E0(v28, &v139, &v140, &v141, &v142) )
            {
              if ( v140 )
              {
                v16 = 468;
                sub_6851C0(0x1D4u, v145);
              }
              else if ( v139 )
              {
                v16 = 510;
                sub_6851C0(0x1FEu, v145);
              }
              else if ( v141 )
              {
                v16 = 1660;
                sub_6851C0(0x67Cu, v145);
              }
              else
              {
                v16 = 1658;
                sub_6851C0(0x67Au, v145);
              }
              v28 = sub_72C930();
            }
            v22[4] = v28;
            v29 = 0;
            v12 = 0;
          }
          v17 = (unsigned int *)v130;
          if ( !v130 )
            v17 = (unsigned int *)v22;
          v130 = (__int64)v17;
          if ( v9 )
            *v9 = (__int64)v22;
          ++v114;
          --*(_BYTE *)(qword_4F061C8 + 75LL);
          if ( !v29 )
          {
            v16 = v144;
            v17 = 0;
            v47 = sub_867630(v144, 0);
            v22[2] = v47;
            if ( v47 )
            {
              v48 = v108;
              *((_BYTE *)v22 + 24) |= 0x10u;
              if ( !v108 )
                v48 = v22;
              v108 = v48;
            }
          }
          if ( !v12 || (v18 = 1, (*(_BYTE *)(j + 56) & 0x10) == 0) )
          {
            v16 = v144;
            v18 = (unsigned int)sub_866C00(v144) != 0;
          }
          if ( v144 )
          {
            if ( *(_QWORD *)(v144 + 16) )
            {
              if ( unk_4F04C48 != -1 )
              {
                v16 = (unsigned __int64)qword_4F04C68;
                if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 6) & 0x10) != 0 && !*((_BYTE *)v22 + 8) )
                {
                  v30 = v22[4];
                  if ( *(_BYTE *)(v30 + 140) != 14 || (*(_BYTE *)(v30 + 161) & 1) != 0 )
                  {
                    v16 = (unsigned __int64)v22;
                    v117 = v18;
                    v31 = sub_892B20(v22);
                    v18 = v117;
                    v32 = (unsigned int *)v31;
                    if ( v31 )
                    {
                      if ( unk_4F04C48 != -1 && qword_4F04C68[0] + 776LL * unk_4F04C48 )
                      {
                        v103 = v117;
                        v33 = qword_4F04C68[0] + 776LL * unk_4F04C48;
                        while ( 1 )
                        {
                          if ( *(_BYTE *)(v33 + 4) == 9 )
                          {
                            v35 = *(__int64 ****)(v33 + 408);
                            if ( v35 )
                            {
                              v36 = *v35;
                              if ( *v35 )
                              {
                                v118 = v33;
                                v37 = sub_892BC0(*v35);
                                v16 = v32[1];
                                v33 = v118;
                                if ( *(_DWORD *)(v37 + 4) == (_DWORD)v16 )
                                  break;
                              }
                            }
                          }
                          v34 = *(int *)(v33 + 552);
                          if ( (_DWORD)v34 != -1 )
                          {
                            v33 = qword_4F04C68[0] + 776 * v34;
                            if ( v33 )
                              continue;
                          }
                          v18 = v103;
                          goto LABEL_71;
                        }
                        v17 = (unsigned int *)*v32;
                        v18 = v103;
                        v38 = v36;
                        if ( (unsigned int)v17 <= 1 )
                          goto LABEL_69;
                        v39 = 1;
                        do
                        {
                          ++v39;
                          v38 = (__int64 **)*v38;
                          if ( (_DWORD)v17 == v39 )
                            break;
                        }
                        while ( v38 );
                        if ( v38 )
                        {
LABEL_69:
                          v40 = ((_BYTE)v38[7] & 0x10) != 0;
                          goto LABEL_72;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
LABEL_71:
          v40 = 0;
LABEL_72:
          v41 = (32 * v40) | v22[3] & 0xDF;
          *((_BYTE *)v22 + 24) = v41;
          i = *(unsigned __int8 *)(j + 56);
          if ( (i & 0x50) != 0 )
          {
            v42 = v41 | 8;
            *((_BYTE *)v22 + 24) = v42;
            i = *(unsigned __int8 *)(j + 56);
            if ( (v42 & 0x10) == 0 )
              goto LABEL_86;
          }
          else if ( (v41 & 0x10) == 0 )
          {
            goto LABEL_78;
          }
          if ( (i & 0x10) != 0 )
          {
            if ( !v126 )
            {
              i &= 0x50u;
              v9 = v22;
              if ( (_BYTE)i == 16 )
                goto LABEL_83;
              goto LABEL_78;
            }
            v43 = *(_BYTE *)(v126 + 56);
            if ( (v43 & 0x10) != 0 )
            {
              i &= 0x50u;
              if ( (_BYTE)i == 16 )
                goto LABEL_81;
LABEL_78:
              j = *(_QWORD *)j;
              goto LABEL_79;
            }
          }
          if ( word_4F06418[0] == 67 )
          {
            v119 = v18;
            v9 = v22;
            sub_7B8B50(v16, v17, v18, i, v20, v21);
            v17 = 0;
            v16 = 1;
            v46 = sub_7BF3A0(1u, 0);
            v18 = v119;
            *v22 = v46;
            for ( j = v46; j; j = *(_QWORD *)j )
              v9 = (__int64 *)j;
            goto LABEL_83;
          }
LABEL_86:
          i &= 0x50u;
          if ( (_BYTE)i != 16 )
            goto LABEL_78;
LABEL_79:
          v9 = v22;
          if ( v126 )
          {
            v43 = *(_BYTE *)(v126 + 56);
LABEL_81:
            v9 = v22;
            if ( (v43 & 0x50) != 0x10 )
              v126 = *(_QWORD *)v126;
          }
LABEL_83:
          if ( !(_DWORD)v18 )
          {
            v115 = 1;
            v10 = v22;
            goto LABEL_145;
          }
        }
        if ( !v111 )
        {
          if ( j )
          {
            v111 = 0;
            if ( (*(_BYTE *)(j + 56) & 0x10) != 0 )
            {
              v16 = 3;
              v22 = sub_725090(3u);
              v53 = (__int64 *)v130;
              if ( !v130 )
                v53 = v22;
              v130 = (__int64)v53;
              if ( v9 )
                *v9 = (__int64)v22;
              v9 = v22;
              v111 = ((*(_BYTE *)(j + 56) >> 6) ^ 1) & 1;
            }
          }
        }
        if ( word_4F06418[0] == 42 )
        {
          if ( !dword_4F07770 || (sub_7BC010(v16), word_4F06418[0] != 44) )
          {
LABEL_110:
            v145[0] = *(_QWORD *)&dword_4F063F8;
            if ( !j )
              goto LABEL_111;
            goto LABEL_28;
          }
        }
        else if ( word_4F06418[0] != 44 )
        {
          goto LABEL_110;
        }
        if ( !v130 )
        {
LABEL_111:
          v10 = v22;
LABEL_112:
          sub_867030(v144);
          v115 = 1;
          goto LABEL_113;
        }
        if ( v22 == v9 && v111 )
        {
          v111 = 1;
          v10 = v22;
          goto LABEL_112;
        }
        goto LABEL_110;
      }
      if ( !v12 )
      {
LABEL_113:
        if ( !(unsigned int)sub_7BE800(0x43u, v17, v18, i, v20, v21) )
        {
          v113 = 0;
          v49 = (_QWORD *)v130;
          v12 = 0;
          v105 = v111 & (j != 0);
          goto LABEL_157;
        }
        goto LABEL_8;
      }
      i = 0;
    }
    if ( (_DWORD)i )
      break;
LABEL_145:
    if ( !v12 )
      goto LABEL_113;
    if ( !j )
    {
      v105 = 0;
      v49 = (_QWORD *)v130;
      goto LABEL_156;
    }
  }
  v49 = (_QWORD *)v130;
  if ( word_4F06418[0] == 44 && v115 )
    sub_6851C0(0x380u, dword_4F07508);
  sub_867030(v144);
  v105 = v111 & (j != 0);
LABEL_156:
  v113 = 1;
LABEL_157:
  if ( v105 )
  {
    j = *(_QWORD *)j;
    if ( v126 )
    {
      v126 = *(_QWORD *)v126;
      v105 = v126 == 0;
    }
  }
  else
  {
    v105 = v126 == 0;
  }
  if ( !unk_4D04854 && (!dword_4F077BC || qword_4F077A8 > 0x9D07u) )
    goto LABEL_162;
  if ( !j )
    goto LABEL_178;
  if ( v105 )
  {
    v126 = j;
  }
  else
  {
LABEL_162:
    if ( !j || v105 )
      goto LABEL_178;
  }
  v112 = v12;
  v55 = v108;
  v131 = 0;
  v56 = v126;
  while ( 2 )
  {
    v57 = *(_BYTE *)(j + 56) & 0x10;
    v58 = *(_BYTE *)(v56 + 56) & 0x10;
    if ( v58 && v102 && (*(_BYTE *)(j + 56) & 1) != 0 )
    {
      if ( !v57 )
      {
        if ( v55 )
        {
          j = *(_QWORD *)j;
LABEL_169:
          if ( !j )
            goto LABEL_177;
          goto LABEL_170;
        }
        v67 = j;
        v123 = v102;
LABEL_190:
        if ( !v131 )
          *a5 = v114;
        if ( a2 )
        {
LABEL_177:
          v12 = v112;
          goto LABEL_178;
        }
        if ( (*(_BYTE *)(v67 + 56) & 4) != 0 )
        {
          v134 = v67;
          sub_8AEC90(v123, v67, v101);
          v67 = v134;
        }
        v68 = *(_QWORD *)(j + 8);
        v69 = 0;
        v70 = *(_BYTE *)(v68 + 80);
        if ( v70 != 3 )
          v69 = (unsigned int)(v70 != 2) + 1;
        v127 = v67;
        v71 = sub_725090(v69);
        v72 = v127;
        v73 = v71;
        v74 = *((_BYTE *)v71 + 8);
        if ( v74 )
        {
          v77 = *(_BYTE *)(v127 + 56) & 1;
          if ( v74 == 2 )
          {
            if ( v77 )
            {
              v82 = *(_QWORD *)(j + 64);
              v83 = *(_QWORD *)(v82 + 104);
              if ( (*(_BYTE *)(v82 + 266) & 4) != 0 && !v100 )
              {
                v138 = v73;
                v96 = sub_8B0DE0(a1, j, v49);
                v73 = v138;
                v72 = v127;
                v83 = v96;
                v138[5] = v96;
              }
              v84 = v72;
              v128 = v83;
              v109 = v73;
              v135 = v72;
              v85 = (__int64 *)sub_89A690(v123, v72, v49);
              v76 = v109;
              v86 = v128;
              v109[4] = v85;
              v87 = *(_QWORD *)(v135 + 64);
              if ( (*(_BYTE *)(v87 + 266) & 2) != 0 )
              {
                v136 = *v85;
                v129 = **(_QWORD **)(v87 + 104);
                v88 = sub_8794A0(v86, v84, v87);
                v76 = v109;
                v89 = v88;
                v90 = *(_QWORD *)(v136 + 88);
                if ( (*(_BYTE *)(v90 + 160) & 2) == 0 )
                {
                  v124 = v89;
                  v91 = sub_89B3C0(**(_QWORD **)(v89 + 32), **(_QWORD **)(v90 + 32), 0, 4, 0, 8);
                  v92 = v124;
                  v76 = v109;
                  if ( !v91 )
                  {
                    v125 = v109;
                    v110 = v92;
                    sub_686C60(0x3E7u, (FILE *)&dword_4F063F8, v136, v129);
                    *(_BYTE *)(v110 + 266) &= ~2u;
                    v76 = v125;
                  }
                }
              }
            }
            else
            {
              v137 = v73;
              v93 = sub_87F550(v69, v68);
              v76 = v137;
              v137[4] = *(_QWORD *)(*(_QWORD *)(v93 + 88) + 104LL);
            }
          }
          else
          {
            v133 = v73;
            if ( v77 )
            {
              sub_8AE7F0(v123, v68, v127, v49, 1, &v143);
              v76 = v133;
              v133[4] = v143;
            }
            else
            {
              v79 = sub_72C9A0();
              v76 = v133;
              v143 = v79;
              v133[4] = v79;
            }
          }
        }
        else
        {
          v132 = v73;
          if ( (*(_BYTE *)(v127 + 56) & 1) != 0 )
            v75 = sub_89A4B0(v123, v127, v49);
          else
            v75 = sub_72C930();
          v76 = v132;
          v132[4] = v75;
        }
        if ( !v49 )
          v49 = v76;
        if ( v9 )
          *v9 = (__int64)v76;
        v131 = 1;
        v9 = v76;
        v58 = *(_BYTE *)(v56 + 56) & 0x10;
        goto LABEL_175;
      }
LABEL_183:
      v66 = sub_725090(3u);
      if ( !v49 )
        v49 = v66;
      if ( v9 )
        *v9 = (__int64)v66;
      v9 = v66;
      v58 = *(_BYTE *)(v56 + 56) & 0x10;
      goto LABEL_175;
    }
    if ( v57 )
      goto LABEL_183;
    if ( (*(_BYTE *)(v56 + 56) & 1) != 0 )
    {
      if ( !v55 )
      {
        v67 = v56;
        v123 = a1;
        goto LABEL_190;
      }
LABEL_175:
      j = *(_QWORD *)j;
      if ( v58 )
        goto LABEL_169;
      v56 = *(_QWORD *)v56;
      if ( !j )
        goto LABEL_177;
LABEL_170:
      if ( !v56 )
        goto LABEL_177;
      continue;
    }
    break;
  }
  if ( v55 )
    goto LABEL_175;
  v12 = v112;
  if ( !v131 )
    sub_6854E0(0x1BAu, a1);
  *a3 = 1;
LABEL_178:
  if ( v113 && (word_4F06418[0] != 44 || v12) )
  {
    sub_6854C0(0x1BBu, (FILE *)&dword_4F063F8, a1);
    v59 = qword_4F061C8;
    v60 = *(_BYTE *)(qword_4F061C8 + 75LL);
    *(_BYTE *)(qword_4F061C8 + 75LL) = 0;
    sub_7BE180(443, (__int64)&dword_4F063F8, v61, v62, v63, v64);
    *(_BYTE *)(v59 + 75) = v60;
    *a3 = 1;
  }
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) = v104
                                                            | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12)
                                                            & 0xFE;
  return v49;
}
