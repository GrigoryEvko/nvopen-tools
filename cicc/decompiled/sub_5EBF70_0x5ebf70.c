// Function: sub_5EBF70
// Address: 0x5ebf70
//
__int64 __fastcall sub_5EBF70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  _BYTE *v5; // rax
  int v6; // eax
  __int64 v7; // rdi
  __int64 *v8; // rax
  __int64 v9; // rdx
  char *v10; // rcx
  char v11; // bl
  char *v12; // rax
  int v13; // r12d
  int v14; // r14d
  __int64 v15; // rdx
  unsigned __int16 v16; // ax
  _DWORD *v17; // rcx
  __int64 result; // rax
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // r10
  _BYTE *v22; // rax
  int v23; // eax
  __int64 v24; // rdx
  char v25; // bl
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // r11
  __int64 v29; // r11
  char v30; // al
  __int64 i; // rdx
  int v32; // eax
  void *v33; // r8
  __int64 v34; // r11
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // r9
  __int64 j; // r9
  int v40; // r9d
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rax
  char v44; // al
  __int64 v45; // rax
  _BOOL4 v46; // edx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r12
  char v50; // al
  __int64 v51; // rax
  char v52; // al
  char v53; // al
  _QWORD *v54; // rax
  int v55; // ebx
  _QWORD *v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // r12
  int v59; // r14d
  char v60; // dl
  __int64 *v61; // r12
  char v62; // al
  char v63; // al
  _QWORD *v64; // r12
  _QWORD *v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rbx
  _QWORD *v69; // r14
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rax
  _QWORD *v74; // rcx
  __int64 v75; // r14
  _QWORD *v76; // rbx
  __int64 **v77; // rdx
  __int64 *k; // rax
  __int64 v79; // rdx
  char v80; // cl
  __int64 v81; // rdx
  char v82; // cl
  int v83; // eax
  char v84; // al
  char v85; // al
  int v86; // eax
  __int64 v87; // rax
  int v88; // eax
  int v89; // eax
  __int64 v90; // r12
  __int64 v91; // rbx
  __int64 v92; // rdi
  __int64 v93; // rax
  int v94; // eax
  __int64 v95; // rdx
  _QWORD *v96; // [rsp+0h] [rbp-100h]
  _QWORD *v97; // [rsp+8h] [rbp-F8h]
  __int64 *v98; // [rsp+10h] [rbp-F0h]
  __int64 v99; // [rsp+18h] [rbp-E8h]
  bool v100; // [rsp+20h] [rbp-E0h]
  __int64 v101; // [rsp+20h] [rbp-E0h]
  __int64 v102; // [rsp+20h] [rbp-E0h]
  __int64 v103; // [rsp+30h] [rbp-D0h]
  __int64 v104; // [rsp+30h] [rbp-D0h]
  __int64 *v105; // [rsp+30h] [rbp-D0h]
  __int64 v106; // [rsp+30h] [rbp-D0h]
  __int64 v107; // [rsp+30h] [rbp-D0h]
  __int64 v108; // [rsp+30h] [rbp-D0h]
  __int64 v109; // [rsp+30h] [rbp-D0h]
  __int64 v110; // [rsp+30h] [rbp-D0h]
  __int64 v111; // [rsp+30h] [rbp-D0h]
  __int64 *v112; // [rsp+38h] [rbp-C8h]
  __int64 v113; // [rsp+40h] [rbp-C0h]
  bool v114; // [rsp+40h] [rbp-C0h]
  __int64 v115; // [rsp+48h] [rbp-B8h]
  __int64 v116; // [rsp+50h] [rbp-B0h]
  unsigned int v117; // [rsp+50h] [rbp-B0h]
  int v118; // [rsp+50h] [rbp-B0h]
  __int64 v119; // [rsp+50h] [rbp-B0h]
  __int64 v120; // [rsp+58h] [rbp-A8h]
  int v121; // [rsp+60h] [rbp-A0h]
  __int64 v122; // [rsp+60h] [rbp-A0h]
  __int16 v123; // [rsp+68h] [rbp-98h]
  _QWORD *v124; // [rsp+68h] [rbp-98h]
  _QWORD *v125; // [rsp+70h] [rbp-90h]
  __int64 v126; // [rsp+70h] [rbp-90h]
  char *v127; // [rsp+78h] [rbp-88h]
  __int64 v128; // [rsp+78h] [rbp-88h]
  __int64 v129; // [rsp+78h] [rbp-88h]
  __int64 *v130; // [rsp+80h] [rbp-80h]
  __int64 v131; // [rsp+80h] [rbp-80h]
  __int16 v132; // [rsp+8Ch] [rbp-74h]
  __int16 v133; // [rsp+8Eh] [rbp-72h]
  char v134; // [rsp+90h] [rbp-70h]
  __int64 v135; // [rsp+90h] [rbp-70h]
  int v136; // [rsp+90h] [rbp-70h]
  __int64 v137; // [rsp+90h] [rbp-70h]
  __int64 v138; // [rsp+90h] [rbp-70h]
  __int64 v139; // [rsp+90h] [rbp-70h]
  __int64 v140; // [rsp+90h] [rbp-70h]
  unsigned __int8 v141; // [rsp+90h] [rbp-70h]
  __int64 v142; // [rsp+90h] [rbp-70h]
  __int64 v143; // [rsp+90h] [rbp-70h]
  __int64 v144; // [rsp+90h] [rbp-70h]
  __int64 v145; // [rsp+90h] [rbp-70h]
  __int64 v146; // [rsp+90h] [rbp-70h]
  _BYTE *v147; // [rsp+98h] [rbp-68h]
  char v148; // [rsp+ABh] [rbp-55h] BYREF
  int v149; // [rsp+ACh] [rbp-54h] BYREF
  _QWORD *v150; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v151; // [rsp+B8h] [rbp-48h] BYREF
  __int64 v152; // [rsp+C0h] [rbp-40h] BYREF
  _QWORD v153[7]; // [rsp+C8h] [rbp-38h] BYREF

  v4 = *(_QWORD *)a1;
  v120 = a1;
  v150 = 0;
  v147 = (_BYTE *)v4;
  if ( *(_BYTE *)(v4 + 140) == 11 )
  {
    a2 = (__int64)dword_4F07508;
    a1 = 241;
    sub_6851C0(241, dword_4F07508);
    v112 = 0;
  }
  else
  {
    v112 = *(__int64 **)(v4 + 168);
  }
  sub_7B8B50(a1, a2, a3, a4);
  v5 = v147;
  if ( v147[140] == 12 )
  {
    do
      v5 = (_BYTE *)*((_QWORD *)v5 + 20);
    while ( v5[140] == 12 );
  }
  else
  {
    v5 = v147;
  }
  v134 = 1;
  v113 = 0;
  v133 = 0;
  v115 = 0;
  v99 = *(_QWORD *)(*(_QWORD *)v5 + 96LL);
  v132 = 0;
  do
  {
    ++*(_BYTE *)(unk_4F061C8 + 75LL);
    v6 = sub_869470(&v151);
    v132 -= ((*(_BYTE *)(v120 + 8) & 0x40) == 0) - 1;
    if ( v6 )
    {
      do
      {
        v7 = 4;
        v8 = (__int64 *)sub_5CC190(4);
        v130 = v8;
        if ( v8 )
        {
          v7 = (__int64)v8;
          sub_5CF700(v8);
        }
        v10 = "private";
        v11 = 2 * (v147[140] == 9);
        v12 = "public";
        if ( v147[140] == 9 )
          v12 = "private";
        ++v133;
        v13 = 0;
        v14 = 0;
        v127 = v12;
        v153[0] = *(_QWORD *)&dword_4F063F8;
        while ( 1 )
        {
          v16 = word_4F06418[0];
          if ( word_4F06418[0] == 164 )
            break;
          v15 = (unsigned int)word_4F06418[0] - 157;
          if ( (unsigned __int16)(word_4F06418[0] - 157) > 2u )
            goto LABEL_21;
LABEL_13:
          if ( v14 )
          {
            a2 = (__int64)&dword_4F063F8;
            v7 = 242;
            sub_6851C0(242, &dword_4F063F8);
          }
          else
          {
            v11 = 0;
            if ( v16 != 159 )
              v11 = (v16 != 158) + 1;
          }
          sub_7B8B50(v7, a2, v15, v10);
          v14 = 1;
        }
        if ( v13 )
        {
LABEL_29:
          a2 = (__int64)&dword_4F063F8;
          v7 = 240;
          sub_6851C0(240, &dword_4F063F8);
        }
        sub_7B8B50(v7, a2, v9, v10);
        v16 = word_4F06418[0];
        if ( word_4F06418[0] == 164 )
          goto LABEL_29;
        v15 = (unsigned int)word_4F06418[0] - 157;
        v13 = 1;
        if ( (unsigned __int16)(word_4F06418[0] - 157) <= 2u )
          goto LABEL_13;
LABEL_21:
        v17 = &dword_4F077C4;
        if ( dword_4F077C4 == 2 )
        {
          if ( v16 == 1 && (unk_4D04A11 & 2) != 0 )
            goto LABEL_34;
          a2 = 0;
          v7 = (__int64)&loc_1000000;
          if ( (unsigned int)sub_7C0F00(&loc_1000000, 0) )
            goto LABEL_34;
LABEL_23:
          if ( unk_4F07740 && word_4F06418[0] == 18 )
            goto LABEL_34;
          sub_6851D0(40);
          a2 = 0;
          sub_867630(v151, 0);
          continue;
        }
        if ( v16 != 1 )
          goto LABEL_23;
LABEL_34:
        v149 = 0;
        v121 = qword_4F063F0;
        v123 = WORD2(qword_4F063F0);
        v152 = *(_QWORD *)&dword_4F063F8;
        if ( (unsigned __int8)v13 | (unsigned __int8)v134 ^ 1 && unk_4D04324 )
        {
          v7 = (__int64)v153;
          a2 = 882;
          sub_684AB0(v153, 882);
        }
        if ( unk_4F07740 && word_4F06418[0] == 18 )
        {
          v143 = unk_4D04A38;
          sub_7B8B50(v7, a2, v15, v17);
          v29 = v143;
          v30 = *(_BYTE *)(v143 + 140);
          if ( v30 == 12 )
          {
            v79 = v143;
            do
            {
              v79 = *(_QWORD *)(v79 + 160);
              v80 = *(_BYTE *)(v79 + 140);
            }
            while ( v80 == 12 );
            v81 = v143;
            if ( !v80 )
              goto LABEL_72;
            do
            {
              v81 = *(_QWORD *)(v81 + 160);
              v82 = *(_BYTE *)(v81 + 140);
            }
            while ( v82 == 12 );
          }
          else
          {
            v82 = *(_BYTE *)(v143 + 140);
            if ( !v30 )
              goto LABEL_72;
          }
          if ( unk_4F04C44 != -1
            || (a2 = (__int64)qword_4F04C68,
                v95 = qword_4F04C68[0] + 776LL * dword_4F04C64,
                (*(_BYTE *)(v95 + 6) & 6) != 0)
            || *(_BYTE *)(v95 + 4) == 12 )
          {
            if ( (*(_BYTE *)(v143 + 186) & 8) != 0 )
            {
              v19 = v143;
              *(_BYTE *)(v99 + 180) = (v112 != 0) | *(_BYTE *)(v99 + 180) & 0xFE;
              j = sub_7CFE40(v143);
              v115 = v143;
              if ( v147[140] == 11 )
                goto LABEL_72;
              v117 = 1;
              goto LABEL_91;
            }
          }
          if ( (unsigned __int8)(v82 - 9) > 1u )
          {
            sub_6851C0(262, &v152);
            goto LABEL_72;
          }
          if ( v147[140] == 11 )
            goto LABEL_72;
          v117 = 1;
LABEL_58:
          for ( i = v29; v30 == 12; v30 = *(_BYTE *)(i + 140) )
            i = *(_QWORD *)(i + 160);
          if ( v30 == 11 )
          {
            a2 = (__int64)dword_4F07508;
            v19 = 3377;
            sub_6851C0(3377, dword_4F07508);
LABEL_71:
            if ( !v117 )
              goto LABEL_73;
LABEL_72:
            a2 = 0;
            sub_867630(v151, 0);
            v134 = 0;
            continue;
          }
          if ( dword_4F077C4 == 2 )
          {
            v110 = v29;
            v146 = i;
            v88 = sub_8D23B0(i);
            i = v146;
            v29 = v110;
            if ( v88 )
            {
              v89 = sub_8D3A70(v146);
              i = v146;
              v29 = v110;
              if ( v89 )
              {
                a2 = 0;
                sub_8AD220(v146, 0);
                v29 = v110;
                i = v146;
              }
            }
          }
          if ( (*(_BYTE *)(i + 176) & 1) == 0 )
          {
            v19 = i;
            v135 = v29;
            v103 = i;
            v32 = sub_8D23B0(i);
            v34 = v135;
            if ( v32 )
            {
              v136 = 1;
              v24 = v103;
              goto LABEL_65;
            }
LABEL_234:
            for ( j = v34; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            v115 = v34;
            if ( (*(_BYTE *)(j + 179) & 4) == 0 )
              goto LABEL_91;
            v147[179] |= 4u;
            if ( v14 )
              goto LABEL_92;
            goto LABEL_238;
          }
          a2 = (__int64)&dword_4F063F8;
          v108 = v29;
          v145 = i;
          sub_6851C0(1904, &dword_4F063F8);
          v19 = v145;
          v86 = sub_8D23B0(v145);
          v24 = v145;
          v34 = v108;
          if ( !v86 )
            goto LABEL_71;
          v136 = 0;
LABEL_65:
          if ( !dword_4F077BC )
            goto LABEL_70;
          v35 = *(_QWORD *)(*(_QWORD *)(v24 + 168) + 152LL);
          if ( v35 && (*(_BYTE *)(v35 + 29) & 0x20) == 0 )
          {
            v101 = v24;
            v106 = v34;
            v83 = sub_8DD3B0(v34);
            v34 = v106;
            v24 = v101;
            if ( !v83 )
            {
              if ( !dword_4F077BC )
                goto LABEL_70;
              goto LABEL_68;
            }
          }
          else
          {
LABEL_68:
            if ( dword_4F077B4
              || !qword_4F077A8
              || (v102 = v24, v111 = v34, v94 = sub_8DC060(v34), v34 = v111, v24 = v102, !v94) )
            {
LABEL_70:
              v137 = v24;
              a2 = (__int64)dword_4F07508;
              v19 = (unsigned int)sub_67F240(v24);
              sub_685A50(v19, dword_4F07508, v137, 8);
              goto LABEL_71;
            }
          }
          a2 = (__int64)dword_4F07508;
          v19 = 1599;
          v107 = v34;
          sub_684B30(1599, dword_4F07508);
          v34 = v107;
          if ( !v136 )
            goto LABEL_71;
          goto LABEL_234;
        }
        a2 = 7;
        v19 = (__int64)&loc_1040000;
        v21 = sub_7BF130(&loc_1040000, 7, &v149);
        if ( !v21 )
          goto LABEL_48;
        if ( dword_4F04C64 != -1 )
        {
          v22 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
          if ( (v22[7] & 1) != 0 && (unk_4F04C44 != -1 || (v22[6] & 6) != 0 || v22[4] == 12) )
          {
            v19 = v21;
            v138 = v21;
            a2 = (__int64)&unk_4D04A08;
            sub_867130(v21, &unk_4D04A08, 0, 0);
            v21 = v138;
          }
        }
        v23 = *(unsigned __int8 *)(v21 + 80);
        v24 = (unsigned int)(v23 - 4);
        if ( (unsigned __int8)(v23 - 4) <= 1u )
          goto LABEL_55;
        if ( (_BYTE)v23 != 3 )
        {
          if ( (*(_BYTE *)(v21 + 81) & 0x20) == 0 )
            goto LABEL_48;
LABEL_73:
          v25 = 0;
          v26 = 0;
LABEL_49:
          sub_7B8B50(v19, a2, v24, v20);
          a2 = 0;
          v27 = sub_867630(v151, 0);
          if ( v27 )
            goto LABEL_50;
LABEL_207:
          v134 = 0;
          continue;
        }
        v19 = *(_QWORD *)(v21 + 88);
        v139 = v21;
        v36 = sub_8D3A70(v19);
        v21 = v139;
        v117 = v36;
        if ( v36 )
        {
LABEL_55:
          if ( (*(_BYTE *)(v21 + 81) & 0x20) != 0 )
            goto LABEL_73;
          v28 = *(_QWORD *)(v21 + 88);
          *(_BYTE *)(v28 + 88) |= 4u;
          if ( unk_4D04A10 < 0 )
          {
            a2 = 406;
            v109 = v28;
            v119 = v21;
            v19 = unk_4F07470;
            sub_685440(unk_4F07470, 406, unk_4D04A18);
            v21 = v119;
            v28 = v109;
          }
          if ( v147[140] == 11 )
            goto LABEL_73;
          a2 = v21;
          v116 = v28;
          sub_8767A0(4, v21, &unk_4D04A08, 1);
          v29 = v116;
          if ( dword_4F077C4 == 2 && unk_4D04A18 && (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
          {
            v40 = 0;
LABEL_89:
            a2 = 0;
            v19 = (__int64)&unk_4D04A00;
            v118 = v40;
            v140 = v29;
            sub_8841F0(&unk_4D04A00, 0, 0, 0);
            v29 = v140;
            if ( v118 )
            {
              v117 = 0;
              j = v140;
              goto LABEL_91;
            }
          }
          v30 = *(_BYTE *)(v29 + 140);
          v117 = 0;
          goto LABEL_58;
        }
        if ( (*(_BYTE *)(v139 + 81) & 0x20) != 0 )
          goto LABEL_73;
        if ( *(_BYTE *)(v139 + 80) != 3 )
          goto LABEL_48;
        v19 = *(_QWORD *)(v139 + 88);
        v24 = *(unsigned __int8 *)(v19 + 140);
        if ( (_BYTE)v24 == 12 )
        {
          v37 = *(_QWORD *)(v139 + 88);
          do
          {
            v37 = *(_QWORD *)(v37 + 160);
            v24 = *(unsigned __int8 *)(v37 + 140);
          }
          while ( (_BYTE)v24 == 12 );
        }
        if ( !(_BYTE)v24 )
          goto LABEL_73;
        if ( *(_BYTE *)(sub_8D21C0(v19) + 140) != 14 )
        {
LABEL_48:
          a2 = (__int64)dword_4F07508;
          v25 = 0;
          v26 = 0;
          sub_6851C0(262, dword_4F07508);
          v19 = (__int64)&unk_4D04A00;
          sub_8767B0(&unk_4D04A00);
          goto LABEL_49;
        }
        if ( unk_4F04C44 == -1 )
        {
          v87 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( (*(_BYTE *)(v87 + 6) & 6) == 0 && *(_BYTE *)(v87 + 4) != 12 )
          {
            a2 = (__int64)dword_4F07508;
            v19 = 264;
            v25 = 0;
            v26 = 0;
            sub_6851C0(264, dword_4F07508);
            goto LABEL_49;
          }
        }
        v115 = *(_QWORD *)(v139 + 88);
        *(_BYTE *)(v115 + 88) |= 4u;
        *(_BYTE *)(v99 + 180) = (v112 != 0) | *(_BYTE *)(v99 + 180) & 0xFE;
        v19 = v115;
        v38 = sub_7CFE40(v115);
        if ( v147[140] == 11 )
          goto LABEL_73;
        a2 = v139;
        v19 = 4;
        v104 = v38;
        sub_8767A0(4, v139, &unk_4D04A08, 1);
        v33 = &unk_4D04A00;
        j = v104;
        if ( dword_4F077C4 == 2 )
        {
          v24 = unk_4D04A18;
          if ( unk_4D04A18 )
          {
            if ( (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
            {
              v29 = v104;
              v40 = 1;
              goto LABEL_89;
            }
          }
        }
LABEL_91:
        if ( v14 )
        {
LABEL_92:
          v41 = *v112;
          if ( *v112 )
            goto LABEL_93;
          goto LABEL_239;
        }
LABEL_238:
        a2 = 261;
        v19 = 3;
        v144 = j;
        sub_6849F0(3, 261, dword_4F07508, v127);
        j = v144;
        v41 = *v112;
        if ( *v112 )
        {
LABEL_93:
          a2 = v13 & 1;
          v24 = 0;
          v20 = dword_4F07588;
          while ( 1 )
          {
            while ( 1 )
            {
              v43 = *(_QWORD *)(v41 + 40);
              if ( v43 == j )
                break;
              if ( j )
              {
                if ( v43 )
                {
                  if ( (_DWORD)v20 )
                  {
                    v42 = *(_QWORD *)(v43 + 32);
                    if ( *(_QWORD *)(j + 32) == v42 )
                    {
                      if ( v42 )
                        break;
                    }
                  }
                }
              }
              v41 = *(_QWORD *)v41;
              if ( !v41 )
                goto LABEL_105;
            }
            v44 = *(_BYTE *)(v41 + 96);
            if ( (v44 & 1) != 0 )
            {
              a2 = (__int64)dword_4F07508;
              v19 = 263;
              sub_6851C0(263, dword_4F07508);
              goto LABEL_71;
            }
            if ( (v44 & 2) != 0 && (_BYTE)a2 )
              break;
            *(_BYTE *)(v41 + 96) |= 4u;
            v41 = *(_QWORD *)v41;
            v24 = 1;
            if ( !v41 )
            {
LABEL_105:
              v141 = v24 & 1;
              goto LABEL_106;
            }
          }
          sub_5EBB30(v41, 0, 0, v11);
          *(_BYTE *)(v41 + 96) |= 1u;
          *(_QWORD *)(v41 + 48) = v115;
          *(_WORD *)(v41 + 98) = v133;
          *(_QWORD *)(v41 + 72) = v152;
          if ( !v11 )
            *(_BYTE *)(v41 + 97) |= 1u;
          *(_QWORD *)(v41 + 80) = v153[0];
          *(_QWORD *)(v41 + 88) = qword_4F063F0;
          *(_QWORD *)(v113 + 8) = v41;
          if ( v130 )
            sub_5CEC90(v130, v41, 37);
          a2 = v120;
          v19 = v41;
          sub_5E4E60(v41, (__int64 *)v120, v132);
          v113 = v41;
          goto LABEL_71;
        }
LABEL_239:
        v141 = 0;
LABEL_106:
        v128 = j;
        v45 = sub_725160(v19, a2, v24, v20, v33);
        v26 = v45;
        *(_QWORD *)(v45 + 40) = v128;
        *(_QWORD *)(v45 + 48) = v115;
        *(_QWORD *)(v45 + 56) = v147;
        *(_QWORD *)(v45 + 72) = v152;
        *(_WORD *)(v45 + 98) = v133;
        *(_WORD *)(v45 + 96) = *(_WORD *)(v45 + 96) & 0xFEFA | ((v11 == 0) << 8) | (4 * v141 + 1);
        if ( v13 )
          *(_BYTE *)(v45 + 96) |= 2u;
        sub_5E4E60(v45, (__int64 *)v120, v132);
        *(_QWORD *)(v26 + 80) = v153[0];
        if ( v117 )
        {
          *(_DWORD *)(v26 + 88) = v121;
          *(_WORD *)(v26 + 92) = v123;
        }
        else
        {
          *(_QWORD *)(v26 + 88) = qword_4F063F0;
        }
        if ( v130 )
          sub_5CEC90(v130, v26, 37);
        v100 = (*(_BYTE *)(v26 + 96) & 2) != 0;
        v46 = v100;
        v131 = *(_QWORD *)(v26 + 40);
        v47 = *(_QWORD *)(v26 + 56);
        v142 = v47;
        v105 = *(__int64 **)(v47 + 168);
        if ( *(_BYTE *)(v47 + 140) == 12 )
        {
          do
            v47 = *(_QWORD *)(v47 + 160);
          while ( *(_BYTE *)(v47 + 140) == 12 );
        }
        else
        {
          v47 = *(_QWORD *)(v26 + 56);
        }
        v129 = *(_QWORD *)(*(_QWORD *)v47 + 96LL);
        v48 = *(_QWORD *)(v26 + 40);
        if ( *(_BYTE *)(v131 + 140) == 12 )
        {
          do
            v48 = *(_QWORD *)(v48 + 160);
          while ( *(_BYTE *)(v48 + 140) == 12 );
        }
        else
        {
          v48 = *(_QWORD *)(v26 + 40);
        }
        v49 = *(_QWORD *)(*(_QWORD *)v48 + 96LL);
        v50 = *(_BYTE *)(v120 + 9) & 0x10;
        if ( !unk_4D04418 || v100 || v11 )
        {
          *(_BYTE *)(v120 + 8) |= 6u;
          if ( v50 )
            goto LABEL_122;
          if ( v100 )
          {
LABEL_215:
            *(_BYTE *)(v120 + 9) |= 0x10u;
            goto LABEL_122;
          }
LABEL_120:
          if ( (*(_BYTE *)(v49 + 176) & 1) != 0 || !*(_QWORD *)(v49 + 16) && *(_QWORD *)(v49 + 8) )
          {
            v93 = sub_5EB340(v49);
            v46 = v100;
            if ( !v93 || (*(_BYTE *)(*(_QWORD *)(v93 + 88) + 194LL) & 2) == 0 )
              goto LABEL_215;
          }
        }
        else
        {
          *(_BYTE *)(v120 + 8) |= 4u;
          if ( !v50 )
            goto LABEL_120;
        }
LABEL_122:
        if ( unk_4D04464 )
        {
          if ( (*(_BYTE *)(v49 + 176) & 8) != 0 )
            *(_BYTE *)(v120 + 10) |= 0x20u;
          if ( *(_QWORD *)(v49 + 32) )
            *(_BYTE *)(v120 + 10) |= 0x40u;
        }
        v51 = *(_QWORD *)(v49 + 24);
        if ( v51 )
        {
          if ( (*(_BYTE *)(v49 + 177) & 2) != 0 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v51 + 88) + 206LL) & 0x10) != 0 )
              *(_BYTE *)(v120 + 9) |= 0x80u;
          }
          else
          {
            *(_BYTE *)(v120 + 9) |= 0x40u;
          }
        }
        v52 = *(_BYTE *)(v49 + 179);
        if ( (v52 & 4) != 0 )
        {
          *(_BYTE *)(v129 + 179) |= 4u;
          v52 = *(_BYTE *)(v49 + 179);
        }
        if ( (v52 & 8) != 0 )
        {
          *(_BYTE *)(v129 + 179) |= 8u;
          v52 = *(_BYTE *)(v49 + 179);
        }
        if ( (v52 & 0x10) != 0 )
        {
          *(_BYTE *)(v129 + 179) |= 0x10u;
          v52 = *(_BYTE *)(v49 + 179);
        }
        if ( (v52 & 0x20) != 0 )
          *(_BYTE *)(v129 + 179) |= 0x20u;
        if ( v46 )
        {
          *(_DWORD *)(v129 + 176) = *(_DWORD *)(v129 + 176) & 0xFFF81FFF | 0x78000;
          if ( *(char *)(v49 + 179) < 0 )
            *(_BYTE *)(v129 + 179) |= 0x80u;
          goto LABEL_141;
        }
        if ( (*(_BYTE *)(v131 + 177) & 0x20) == 0 )
        {
          v84 = *(_BYTE *)(v49 + 177);
          if ( (v84 & 0x40) == 0 )
          {
            *(_BYTE *)(v129 + 177) &= ~0x40u;
            v84 = *(_BYTE *)(v49 + 177);
          }
          if ( v84 < 0 )
            *(_BYTE *)(v129 + 177) |= 0x80u;
          if ( (*(_BYTE *)(v49 + 178) & 1) != 0 )
            *(_BYTE *)(v129 + 178) |= 1u;
          if ( (*(_BYTE *)(v49 + 177) & 0x20) == 0 )
            *(_BYTE *)(v129 + 177) &= ~0x20u;
          v85 = *(_BYTE *)(v49 + 178);
          if ( (v85 & 2) != 0 )
          {
            *(_BYTE *)(v129 + 178) |= 2u;
            v85 = *(_BYTE *)(v49 + 178);
          }
          if ( (v85 & 4) != 0 )
            *(_BYTE *)(v129 + 178) |= 4u;
        }
        if ( *(char *)(v49 + 179) < 0 )
          *(_BYTE *)(v129 + 179) |= 0x80u;
        if ( (*(_BYTE *)(v131 + 176) & 0x10) != 0 )
        {
LABEL_141:
          *(_BYTE *)(v142 + 176) |= 0x10u;
          *(_BYTE *)(v120 + 8) |= 2u;
        }
        if ( (*(_BYTE *)(v131 + 177) & 1) != 0 )
        {
          *(_BYTE *)(v142 + 177) |= 1u;
          *(_BYTE *)(v120 + 8) |= 2u;
        }
        if ( *(char *)(v49 + 181) >= 0 )
          *(_BYTE *)(v129 + 181) &= ~0x80u;
        v53 = *(_BYTE *)(v131 + 176);
        if ( (v53 & 4) != 0 )
        {
          *(_BYTE *)(v142 + 176) |= 4u;
          v53 = *(_BYTE *)(v131 + 176);
        }
        if ( (v53 & 8) != 0 )
          *(_BYTE *)(v142 + 176) |= 8u;
        if ( *(char *)(v131 + 179) < 0 )
          *(_BYTE *)(v142 + 179) |= 0x80u;
        if ( (*(_BYTE *)(v49 + 180) & 1) != 0 || (*(_BYTE *)(v131 + 177) & 0xB0) == 0x30 )
          *(_BYTE *)(v129 + 180) |= 1u;
        v54 = sub_5EBB30(v26, 0, 0, v11);
        v55 = 0;
        v125 = v54;
        v124 = v56;
        if ( **(_QWORD **)(v131 + 168) )
        {
          v57 = **(_QWORD **)(v131 + 168);
          v58 = v26;
          v59 = 0;
          while ( 1 )
          {
            v60 = *(_BYTE *)(v57 + 96);
            if ( v60 >= 0 && *(_QWORD *)(v57 + 120) )
              v59 = 1;
            if ( (v60 & 1) == 0 )
              goto LABEL_157;
            if ( (*(_BYTE *)(v57 + 96) & 2) == 0 || (*(_BYTE *)(*(_QWORD *)(v57 + 112) + 24LL) & 1) != 0 )
            {
              sub_5EBC80(v57, v58, v125, v124, (__int64 *)&v150, v142);
LABEL_157:
              v57 = *(_QWORD *)v57;
              if ( !v57 )
                goto LABEL_165;
            }
            else
            {
              v57 = *(_QWORD *)v57;
              v59 = 1;
              if ( !v57 )
              {
LABEL_165:
                v55 = v59;
                v26 = v58;
                break;
              }
            }
          }
        }
        if ( v150 )
          *v150 = v26;
        else
          *v105 = v26;
        v150 = (_QWORD *)v26;
        if ( v113 )
          *(_QWORD *)(v113 + 8) = v26;
        else
          v105[1] = v26;
        sub_5E6390(v142, v26);
        if ( v55 )
        {
          v61 = **(__int64 ***)(v131 + 168);
          if ( v61 )
          {
            v122 = v26;
            v96 = v125;
            while ( 1 )
            {
              v62 = *((_BYTE *)v61 + 96);
              if ( v62 < 0 || !v61[15] && ((v62 & 3) != 3 || (*(_BYTE *)(v61[14] + 24) & 1) != 0) )
                goto LABEL_201;
              v126 = sub_8E5650(v61);
              v63 = *((_BYTE *)v61 + 96);
              if ( (v63 & 3) == 3 )
              {
                v77 = (__int64 **)v61[14];
                if ( ((_BYTE)v77[3] & 1) == 0 )
                {
                  for ( k = *v77; (k[3] & 1) == 0; k = (__int64 *)*k )
                    ;
                  sub_5EBB30(v126, v96, v124, *((_BYTE *)k + 25));
                  v63 = *((_BYTE *)v61 + 96);
                }
              }
              if ( v63 < 0 || !v61[15] )
                goto LABEL_201;
              v114 = 0;
              v98 = v61;
              v64 = (_QWORD *)v61[15];
              if ( *(char *)(v126 + 96) >= 0 )
                v114 = *(_QWORD *)(v126 + 120) != 0;
              while ( 2 )
              {
                v67 = v64[3];
                v68 = v122;
                if ( v67 )
                  v68 = sub_8E5650(v67);
                if ( !v114 || (v69 = *(_QWORD **)(v126 + 120)) == 0 )
                {
LABEL_180:
                  v65 = (_QWORD *)sub_725050();
                  v65[2] = v64[2];
                  v66 = v64[1];
                  v65[3] = v68;
                  v65[1] = v66;
                  v65[4] = v64[4];
                  sub_5E4860(v126, v65);
                  goto LABEL_181;
                }
                while ( 1 )
                {
                  v70 = v69[2];
                  v71 = v64[2];
                  if ( v70 != v71 )
                  {
                    if ( *(_WORD *)(v70 + 224) > *(_WORD *)(v71 + 224) )
                      goto LABEL_180;
                    goto LABEL_188;
                  }
                  v72 = v69[3];
                  if ( v69[1] == v64[1] && v64[3] == v72 || (unsigned int)sub_8D5D50(v68, v72) )
                    goto LABEL_181;
                  if ( (unsigned int)sub_8D5D50(v69[3], v68) )
                    break;
LABEL_188:
                  v69 = (_QWORD *)*v69;
                  if ( !v69 )
                    goto LABEL_180;
                }
                v73 = v64[1];
                v74 = (_QWORD *)*v69;
                v69[3] = v68;
                v69[1] = v73;
                v69[4] = v64[4];
                if ( v74 )
                {
                  v97 = v69;
                  v75 = v68;
                  v76 = v74;
                  do
                  {
                    if ( v76[2] != v64[2] )
                      break;
                    if ( (unsigned int)sub_8D5D50(v76[3], v75) )
                      *v97 = *v76;
                    else
                      v97 = v76;
                    v76 = (_QWORD *)*v76;
                  }
                  while ( v76 );
                }
LABEL_181:
                v64 = (_QWORD *)*v64;
                if ( v64 )
                  continue;
                break;
              }
              v61 = v98;
LABEL_201:
              v61 = (__int64 *)*v61;
              if ( !v61 )
              {
                v26 = v122;
                break;
              }
            }
          }
        }
        if ( !v105[10] && !v100 )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v131 + 168) + 80LL) || (*(_BYTE *)(v131 + 176) & 0x50) != 0 )
          {
            v90 = *(_QWORD *)(*(_QWORD *)(v26 + 56) + 168LL);
            *(_QWORD *)(v90 + 24) = v26;
            v91 = *(_QWORD *)(*(_QWORD *)(v26 + 40) + 168LL);
            v92 = *(_QWORD *)(v91 + 80);
            if ( v92 )
              *(_QWORD *)(v90 + 80) = sub_8E5650(v92);
            else
              *(_QWORD *)(v90 + 80) = v26;
            *(_WORD *)(v90 + 44) = *(_WORD *)(v91 + 44);
          }
          if ( (*(_BYTE *)(v129 + 178) & 0x10) != 0 )
            sub_5E48B0(v131);
        }
        v19 = v131;
        a2 = (__int64)&v148;
        v25 = 1;
        sub_8D9610(v131, &v148);
        v20 = v142;
        *(_WORD *)(v142 + 180) = *(_WORD *)(v142 + 180) & 0xFC3F
                               | ((v148 & 0xF | (*(_WORD *)(v142 + 180) >> 6) & 0xF) << 6);
        v24 = v117;
        v113 = v26;
        if ( !v117 )
          goto LABEL_49;
        a2 = 0;
        v27 = sub_867630(v151, 0);
        if ( !v27 )
          goto LABEL_207;
LABEL_50:
        if ( !v25 )
          goto LABEL_207;
        *(_QWORD *)(v26 + 120) = v27;
        *(_BYTE *)(v26 + 96) |= 0x80u;
        v147[177] |= 0x20u;
        v134 = 0;
      }
      while ( (unsigned int)sub_866C00(v151) );
    }
    --*(_BYTE *)(unk_4F061C8 + 75LL);
    result = sub_7BE800(67);
  }
  while ( (_DWORD)result );
  return result;
}
