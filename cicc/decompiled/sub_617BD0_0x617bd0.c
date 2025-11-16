// Function: sub_617BD0
// Address: 0x617bd0
//
__int64 *__fastcall sub_617BD0(int a1, __int64 a2)
{
  char *v3; // rax
  unsigned int v4; // r15d
  __int64 v5; // rdi
  unsigned int v6; // r13d
  char v7; // r14
  int v8; // edx
  _BOOL4 v9; // eax
  _BOOL4 v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  char *v13; // rdi
  bool v14; // zf
  char *v15; // rax
  int v16; // eax
  __int64 *result; // rax
  char *v18; // r14
  size_t v19; // rax
  int v20; // esi
  __int64 v21; // r15
  _BYTE *v22; // rdx
  char v23; // al
  _BYTE *v24; // rcx
  __int64 v25; // r14
  __int64 v26; // rax
  char *v27; // rax
  char *v28; // r15
  __int64 v29; // rax
  int v30; // eax
  char *v31; // rsi
  int v32; // r9d
  __int64 v33; // r15
  unsigned int v34; // r14d
  bool v35; // r11
  int v36; // eax
  _BOOL4 v37; // eax
  __int64 v38; // r13
  char *v39; // rdi
  unsigned __int8 v40; // dl
  __int64 v41; // rax
  const char *v42; // r14
  const char *v43; // r13
  char *v44; // r15
  size_t v45; // rax
  size_t v46; // r8
  _QWORD *v47; // rdx
  __int64 v48; // rax
  char *v49; // r15
  __int64 v50; // r13
  int v51; // eax
  const char *v52; // r12
  __int64 v53; // r8
  unsigned int v54; // eax
  int v55; // edx
  __int64 v56; // r13
  unsigned int i; // r15d
  int v58; // eax
  __int64 v59; // r14
  const char *v60; // rdi
  __int64 v61; // r15
  const char **v62; // r14
  int v63; // r13d
  __int64 v64; // rax
  char *v65; // r15
  __int64 *v66; // r13
  __int64 v67; // rax
  int v68; // eax
  int v69; // r9d
  char *v70; // rsi
  __int64 v71; // r15
  unsigned int v72; // r14d
  bool v73; // r12
  int v74; // eax
  __int64 v75; // r13
  char *v76; // rdi
  __int64 v77; // r13
  __int64 v78; // rax
  char *v79; // r15
  __int64 v80; // r14
  __int64 v81; // rax
  unsigned int v82; // eax
  int v83; // edi
  __int64 v84; // rsi
  __int64 *v85; // rcx
  int v86; // eax
  const char *v87; // r13
  __int64 v88; // r13
  _DWORD *v89; // rax
  __int64 v90; // rax
  int v91; // eax
  unsigned int v92; // edx
  int v93; // eax
  const char *v94; // rdi
  const char *v95; // r13
  char **v96; // r12
  unsigned int *v97; // r14
  unsigned __int8 v98; // r14
  int v99; // r13d
  char *v100; // r12
  unsigned int v101; // r15d
  __int64 v102; // rax
  unsigned int v103; // eax
  size_t v104; // r14
  __int64 v105; // rax
  char *v106; // r14
  __int64 v107; // r13
  __int64 v108; // rax
  unsigned int v109; // eax
  int v110; // edi
  unsigned int v111; // esi
  __int64 *v112; // rcx
  unsigned int v113; // edx
  __int64 *v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rax
  unsigned int v117; // edx
  __int64 *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  _QWORD *v122; // rdi
  const char *v123; // r13
  const char *v124; // r12
  __int64 *v125; // r15
  __int64 v126; // r13
  __int64 v127; // rax
  size_t *v128; // rax
  __int64 v129; // r14
  __int64 v130; // rdi
  __int64 v131; // rax
  _QWORD *v132; // rdx
  _QWORD *v133; // r9
  _QWORD *v134; // rsi
  __int64 v135; // [rsp+0h] [rbp-C0h]
  int v136; // [rsp+8h] [rbp-B8h]
  int v137; // [rsp+8h] [rbp-B8h]
  int v138; // [rsp+8h] [rbp-B8h]
  __int64 v139; // [rsp+8h] [rbp-B8h]
  bool v140; // [rsp+10h] [rbp-B0h]
  int v141; // [rsp+10h] [rbp-B0h]
  int v142; // [rsp+10h] [rbp-B0h]
  _QWORD *v143; // [rsp+10h] [rbp-B0h]
  __int64 v144; // [rsp+10h] [rbp-B0h]
  __int64 v145; // [rsp+18h] [rbp-A8h]
  __int64 v146; // [rsp+20h] [rbp-A0h]
  __int64 v147; // [rsp+28h] [rbp-98h]
  __int64 v148; // [rsp+30h] [rbp-90h]
  unsigned int v149; // [rsp+38h] [rbp-88h]
  int v150; // [rsp+3Ch] [rbp-84h]
  int nb; // [rsp+40h] [rbp-80h]
  int n; // [rsp+40h] [rbp-80h]
  size_t nc; // [rsp+40h] [rbp-80h]
  size_t na; // [rsp+40h] [rbp-80h]
  __int64 v156; // [rsp+58h] [rbp-68h] BYREF
  char *s2; // [rsp+60h] [rbp-60h] BYREF
  size_t v158; // [rsp+68h] [rbp-58h] BYREF
  __int64 v159[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v160[8]; // [rsp+80h] [rbp-40h] BYREF

  v156 = 0;
  unk_4F063FC = 1;
  dword_4F063F8 = 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  sub_610260();
  sub_614680();
  sub_610110();
  sub_720BA0(unk_4F076B0);
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v145 = 0;
LABEL_2:
  while ( 1 )
  {
    v3 = sub_6140E0(a1, a2);
    if ( !v3 )
      break;
    v4 = (unsigned __int8)v3[17];
    v5 = *((unsigned int *)v3 + 8);
    v6 = *(_DWORD *)v3;
    v7 = v3[17];
    if ( (_DWORD)v5 )
    {
      sub_8526E0(v5, v6, (unsigned __int8)v3[17], src);
      if ( v6 > 0x126 )
LABEL_6:
        sub_721090(v5);
    }
    switch ( v6 )
    {
      case 1u:
      case 2u:
        dword_4D04964 = 1;
        if ( v6 == 1 )
        {
          byte_4F07472[0] = 8;
          v40 = 8;
          unk_4F07471 = 7;
        }
        else
        {
          byte_4F07472[0] = 5;
          v40 = 5;
          unk_4F07471 = 5;
        }
        if ( unk_4F07481 > v40 )
          unk_4F07481 = v40;
        continue;
      case 3u:
        unk_4D04944 = 1;
        unk_4D0493C = 1;
        unk_4D04930 = 0;
        continue;
      case 4u:
        unk_4D04944 = 1;
        unk_4D0493C = 1;
        unk_4D04930 = 1;
        continue;
      case 5u:
        unk_4D04938 = 1;
        continue;
      case 6u:
        unk_4D04934 = 1;
        continue;
      case 7u:
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 1;
        dword_4CF8024 = 1;
        continue;
      case 8u:
        unk_4D04944 = 1;
        unk_4D0493C = 0;
        qword_4D04914 = 1;
        continue;
      case 9u:
        qword_4D04914 = 0x100000000LL;
        continue;
      case 0xAu:
        unk_4D048F8 = 1;
        unk_4D048FC = 1;
        continue;
      case 0xBu:
        unk_4D04950 = v4;
        continue;
      case 0xCu:
        qword_4D0495C = 0x100000000LL;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0xDu:
        qword_4D0495C = 1;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0xEu:
        unk_4D048FC = 1;
        unk_4D048F8 = 1;
        continue;
      case 0xFu:
        unk_4F06B98 = v4;
        continue;
      case 0x10u:
        if ( src )
        {
          if ( !strcmp(src, "none") )
          {
            unk_4D04734 = 0;
          }
          else if ( !strcmp(src, "all") )
          {
            unk_4D04734 = 1;
          }
          else if ( !strcmp(src, "used") )
          {
            unk_4D04734 = 2;
          }
          else
          {
            if ( strcmp(src, "local") )
              sub_684920(576);
            unk_4D04734 = 3;
          }
        }
        continue;
      case 0x11u:
        unk_4D0472C = v4;
        continue;
      case 0x12u:
        unk_4D048C8 = 1 - ((v4 == 0) - 1);
        continue;
      case 0x13u:
        unk_4D04748 = v4;
        continue;
      case 0x14u:
        unk_4D04744 = 1;
        continue;
      case 0x15u:
        v42 = off_4B6EB08[0];
        v43 = off_4B6EB10;
        if ( strlen(aAug202025) != 11 )
          goto LABEL_737;
        fprintf(
          stderr,
          "lgenfe: \n"
          "Portions Copyright (c) 2005, 2024- %s NVIDIA Corporation\n"
          "Portions Copyright (c) 1988-2018, 2024 Edison Design Group, Inc.\n"
          "Portions Copyright (c) 2007-2018, 2024 University of Illinois at Urbana-Champaign.\n"
          "Based on Edison Design Group C/C++ Front End, version %s (%s %s)\n",
          &aAug202025[7],
          "6.6",
          v43,
          v42);
        fputc(10, qword_4F07510);
        v150 = 1;
        continue;
      case 0x16u:
        unk_4F07481 = 7;
        continue;
      case 0x17u:
        unk_4F07480 = 5;
        continue;
      case 0x18u:
        unk_4F07481 = 4;
        continue;
      case 0x19u:
        if ( dword_4F077C4 != 2 )
          continue;
        if ( dword_4CF8024 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0x1Au:
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0x1Bu:
        dword_4D048B8 = v4;
        continue;
      case 0x1Cu:
        unk_4D048C4 = 1;
        continue;
      case 0x1Du:
      case 0xBBu:
        if ( *src != 45 || src[1] )
        {
          v41 = sub_721290(src);
          sub_7209B0(v41, v6 == 187);
        }
        else
        {
          v156 = qword_4F076A0;
          unk_4F07680 = 0;
        }
        continue;
      case 0x1Eu:
        sub_615C50((__int64)src, &qword_4D04750, (_QWORD **)&qword_4CFDDD0, 0);
        continue;
      case 0x1Fu:
        sub_615C50((__int64)src, &qword_4D04750, (_QWORD **)&qword_4CFDDD0, 1);
        continue;
      case 0x20u:
        unk_4F07478 = sub_60F910((unsigned __int8 *)src);
        if ( !unk_4F07478 )
          sub_684920(578);
        continue;
      case 0x21u:
        v146 = sub_721290(src);
        continue;
      case 0x22u:
        v147 = sub_721290(src);
        continue;
      case 0x23u:
        v148 = sub_721290(src);
        continue;
      case 0x24u:
        v145 = sub_721290(src);
        continue;
      case 0x25u:
      case 0x6Bu:
        continue;
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x29u:
      case 0x2Au:
        v18 = src;
        v19 = strlen(src);
        v20 = 0;
        v21 = sub_822B10(v19 + 1);
        v22 = (_BYTE *)v21;
        while ( 2 )
        {
          v23 = *v18++;
          if ( v23 == 32 )
          {
            v24 = v22;
LABEL_178:
            v22 = v24;
            continue;
          }
          break;
        }
        if ( !v23 || v23 == 44 )
        {
          ++v20;
          v23 = 0;
        }
        *v22 = v23;
        v24 = v22 + 1;
        if ( *(v18 - 1) )
          goto LABEL_178;
        v98 = byte_3A05158[v6 - 38];
        if ( v20 )
        {
          v142 = a1;
          v99 = 0;
          v100 = (char *)v21;
          v101 = v98;
          do
          {
            v104 = strlen(v100);
            if ( (unsigned int)(unsigned __int8)*v100 - 48 <= 9 )
            {
              v102 = sub_60F910((unsigned __int8 *)v100);
              v103 = sub_67D2F0(v102);
              if ( (unsigned int)sub_67D850(v103, v101, 1) )
                sub_684920(614);
            }
            else if ( (unsigned int)sub_67D8B0(v100, v101, 1) )
            {
              sub_684920(613);
            }
            v100 += v104 + 1;
            ++v99;
          }
          while ( v99 != v20 );
          a1 = v142;
        }
        break;
      case 0x2Bu:
        unk_4D04728 = v4;
        continue;
      case 0x2Cu:
        unk_4D04510 = sub_721290(src);
        continue;
      case 0x2Du:
        unk_4F068BC = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0x2Eu:
        unk_4D04718 = 1;
        continue;
      case 0x2Fu:
        unk_4D0471C = 1;
        continue;
      case 0x30u:
        unk_4D04714 = 1;
        continue;
      case 0x31u:
        unk_4D04724 = 1;
        continue;
      case 0x32u:
        unk_4D04720 = 1;
        continue;
      case 0x33u:
        unk_4D04710 = 1;
        continue;
      case 0x34u:
        unk_4D0470C = 1;
        continue;
      case 0x35u:
        unk_4D04708 = 1;
        continue;
      case 0x36u:
        unk_4D04704 = sub_60F910((unsigned __int8 *)src) != 0;
        continue;
      case 0x37u:
        unk_4D04700 = 1;
        continue;
      case 0x38u:
        unk_4D046E0 = sub_721290(src);
        continue;
      case 0x39u:
        unk_4D046D8 = sub_721290(src);
        continue;
      case 0x3Au:
        unk_4D046D0 = src;
        continue;
      case 0x3Bu:
        unk_4D046B8 = 1;
        continue;
      case 0x3Cu:
        unk_4D046B4 = 1;
        continue;
      case 0x3Du:
        unk_4D046B0 = 1;
        continue;
      case 0x3Eu:
        v44 = src;
        v159[0] = (__int64)v160;
        if ( !src )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v45 = strlen(src);
        v158 = v45;
        v46 = v45;
        if ( v45 > 0xF )
        {
          nc = v45;
          v121 = sub_22409D0(v159, &v158, 0);
          v46 = nc;
          v159[0] = v121;
          v122 = (_QWORD *)v121;
          v160[0] = v158;
        }
        else
        {
          if ( v45 == 1 )
          {
            LOBYTE(v160[0]) = *v44;
            v47 = v160;
            goto LABEL_238;
          }
          if ( !v45 )
          {
            v47 = v160;
            goto LABEL_238;
          }
          v122 = v160;
        }
        memcpy(v122, v44, v46);
        v45 = v158;
        v47 = (_QWORD *)v159[0];
LABEL_238:
        v159[1] = v45;
        *((_BYTE *)v47 + v45) = 0;
        sub_617930(&qword_4D04680, (__int64)v159);
        if ( (_QWORD *)v159[0] != v160 )
          j_j___libc_free_0(v159[0], v160[0] + 1LL);
        continue;
      case 0x3Fu:
        unk_4D046B0 = 0;
        continue;
      case 0x40u:
        unk_4D04660 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0x41u:
        unk_4D04658 = 1;
        continue;
      case 0x42u:
        unk_4D04654 = 1;
        continue;
      case 0x43u:
        unk_4D04654 = 0;
        continue;
      case 0x44u:
        if ( !strcmp(src, "global") )
        {
          unk_4D04650 = 1;
          unk_4D0463C = 1;
        }
        else if ( !strcmp(src, "shared") )
        {
          unk_4D0464C = 1;
          unk_4D0463C = 1;
        }
        else if ( !strcmp(src, "constant") )
        {
          unk_4D04648 = 1;
          unk_4D0463C = 1;
        }
        else if ( !strcmp(src, "local") )
        {
          unk_4D04644 = 1;
          unk_4D0463C = 1;
        }
        else if ( !strcmp(src, "generic") )
        {
          unk_4D04640 = 1;
          unk_4D0463C = 1;
        }
        else
        {
          if ( strcmp(src, "all") )
            sub_684920(3664);
          unk_4D04650 = 1;
          unk_4D0464C = 1;
          unk_4D04648 = 1;
          unk_4D04644 = 1;
          unk_4D04640 = 1;
          unk_4D0463C = 1;
        }
        continue;
      case 0x45u:
        unk_4D04638 = sub_60F910((unsigned __int8 *)src) != 0;
        continue;
      case 0x46u:
        v86 = sub_60F910((unsigned __int8 *)src);
        if ( dword_4D04634 > v86 )
          sub_6849E0(3672);
        dword_4D04634 = v86;
        continue;
      case 0x47u:
        unk_4D04630 = 1;
        continue;
      case 0x48u:
        unk_4D0462C = 1;
        continue;
      case 0x49u:
        unk_4D04628 = 1;
        continue;
      case 0x4Au:
        unk_4D0465C = 1;
        continue;
      case 0x4Bu:
        unk_4D046F8 = 1;
        continue;
      case 0x4Cu:
        HIDWORD(qword_4D046F0) = 1;
        continue;
      case 0x4Du:
        LODWORD(qword_4D046F0) = 1;
        continue;
      case 0x4Eu:
        unk_4D046E8 = 1;
        continue;
      case 0x4Fu:
        unk_4D046FC = 1;
        continue;
      case 0x50u:
        unk_4D04624 = 1;
        unk_4F06B10 = 8;
        unk_4F06B08 = 8;
        unk_4F06A51 = 8;
        unk_4F06A60 = 7;
        unk_4F06872 = 7;
        continue;
      case 0x51u:
        unk_4D04620 = 1;
        unk_4F06B10 = 4;
        unk_4F06B08 = 4;
        unk_4F06A51 = 10;
        unk_4F06A60 = 9;
        unk_4F06872 = 9;
        continue;
      case 0x52u:
        v87 = src;
        if ( !strcmp(src, "compute_75") )
        {
          unk_4D045E8 = 75;
        }
        else if ( !strcmp(src, "compute_80") )
        {
          unk_4D045E8 = 80;
        }
        else if ( !strcmp(src, "compute_86") )
        {
          unk_4D045E8 = 86;
        }
        else if ( !strcmp(src, "compute_87") )
        {
          unk_4D045E8 = 87;
        }
        else if ( !strcmp(src, "compute_88") )
        {
          unk_4D045E8 = 88;
        }
        else if ( !strcmp(src, "compute_89") )
        {
          unk_4D045E8 = 89;
        }
        else if ( !strcmp(src, "compute_90") )
        {
          unk_4D045E8 = 90;
        }
        else if ( !strcmp(src, "compute_90a") )
        {
          unk_4D045E8 = 90;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(src, "compute_100") )
        {
          unk_4D045E8 = 100;
        }
        else if ( !strcmp(src, "compute_100a") )
        {
          unk_4D045E8 = 100;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(src, "compute_100f") )
        {
          unk_4D045E8 = 100;
          unk_4D045E4 = 1;
          unk_4D045E0 = 1;
        }
        else if ( !strcmp(src, "compute_103") )
        {
          unk_4D045E8 = 103;
        }
        else if ( !strcmp(src, "compute_103a") )
        {
          unk_4D045E8 = 103;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(src, "compute_103f") )
        {
          unk_4D045E8 = 103;
          unk_4D045E4 = 1;
          unk_4D045E0 = 1;
        }
        else if ( !strcmp(v87, "compute_110") )
        {
          unk_4D045E8 = 110;
        }
        else if ( !strcmp(v87, "compute_110a") )
        {
          unk_4D045E8 = 110;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(v87, "compute_110f") )
        {
          unk_4D045E8 = 110;
          unk_4D045E4 = 1;
          unk_4D045E0 = 1;
        }
        else if ( !strcmp(v87, "compute_120") )
        {
          unk_4D045E8 = 120;
        }
        else if ( !strcmp(v87, "compute_120a") )
        {
          unk_4D045E8 = 120;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(v87, "compute_120f") )
        {
          unk_4D045E8 = 120;
          unk_4D045E4 = 1;
          unk_4D045E0 = 1;
        }
        else if ( !strcmp(v87, "compute_121") )
        {
          unk_4D045E8 = 121;
        }
        else if ( !strcmp(v87, "compute_121a") )
        {
          unk_4D045E8 = 121;
          unk_4D045E4 = 1;
        }
        else if ( !strcmp(v87, "compute_121f") )
        {
          unk_4D045E8 = 121;
          unk_4D045E4 = 1;
          unk_4D045E0 = 1;
        }
        else
        {
          unk_4D045E8 = 0;
          unk_4D045E4 = 0;
          unk_4D045E0 = 0;
        }
        continue;
      case 0x53u:
        unk_4D045F8 = 1;
        continue;
      case 0x54u:
        unk_4D045F4 = 1;
        continue;
      case 0x55u:
        unk_4D045F0 = 1;
        continue;
      case 0x56u:
        unk_4D04600 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0x57u:
        unk_4D04614 = 1;
        continue;
      case 0x58u:
        unk_4D04610 = 1;
        continue;
      case 0x59u:
        unk_4D0460C = 1;
        continue;
      case 0x5Au:
        unk_4D045A4 = 1;
        continue;
      case 0x5Bu:
        unk_4D0459C = 1;
        continue;
      case 0x5Cu:
        unk_4D04598 = v4;
        continue;
      case 0x5Du:
        HIDWORD(qword_4D045BC) = 1;
        continue;
      case 0x5Eu:
        unk_4D045CC = 1;
        continue;
      case 0x5Fu:
        unk_4D045C8 = 1;
        continue;
      case 0x60u:
        unk_4D045C4 = 1;
        continue;
      case 0x61u:
        unk_4D045B8 = 1;
        continue;
      case 0x62u:
        unk_4D045B4 = 1;
        unk_4D045B0 = 1;
        unk_4D045AC = 1;
        continue;
      case 0x63u:
        unk_4D045A8 = 1;
        continue;
      case 0x64u:
        unk_4D04588 = sub_721290(src);
        continue;
      case 0x65u:
        unk_4D04580 = sub_721290(src);
        continue;
      case 0x66u:
        unk_4D04578 = sub_721290(src);
        continue;
      case 0x67u:
        unk_4D04570 = sub_721290(src);
        continue;
      case 0x68u:
        unk_4D04590 = sub_721290(src);
        continue;
      case 0x69u:
        dword_4D045A0 = 1;
        continue;
      case 0x6Au:
        unk_4F06A80 = 0;
        unk_4F06A68 = 4;
        unk_4F06A64 = 4;
        unk_4F06A58 = 0xFFFFFFFFLL;
        unk_4F06A51 = 6;
        unk_4F06A60 = 5;
        unk_4F069C8 = 4;
        unk_4F069C0 = 4;
        unk_4F069B8 = 8;
        unk_4F069B4 = 4;
        unk_4F069A8 = 4;
        unk_4F069A4 = 4;
        unk_4F06B10 = 4;
        unk_4F06B08 = 4;
        unk_4F069FC = 4;
        continue;
      case 0x6Cu:
        unk_4D0461C = sub_60F910((unsigned __int8 *)src) != 0;
        continue;
      case 0x6Du:
        unk_4D04618 = sub_60F910((unsigned __int8 *)src) != 0;
        continue;
      case 0x6Eu:
        v123 = off_4B6EB08[0];
        v124 = off_4B6EB10;
        if ( strlen(aAug202025) == 11 )
        {
          fprintf(
            stderr,
            "lgenfe: \n"
            "Portions Copyright (c) 2005-%s NVIDIA Corporation\n"
            "Portions Copyright (c) 1988-2016 Edison Design Group, Inc.\n"
            "Portions Copyright (c) 2007-2016 University of Illinois at Urbana-Champaign.\n"
            "Based on Edison Design Group C/C++ Front End, version %s (%s %s)\n",
            &aAug202025[7],
            "6.6",
            v124,
            v123);
          fputc(10, stderr);
          exit(1);
        }
LABEL_737:
        sub_6849E0(3457);
      case 0x6Fu:
        unk_4D04568 = sub_721290(src);
        continue;
      case 0x70u:
        unk_4D04560 = sub_721290(src);
        continue;
      case 0x71u:
        unk_4D0455C = 1;
        continue;
      case 0x72u:
        unk_4D04558 = 1;
        dword_4D043DC = 1;
        continue;
      case 0x73u:
        unk_4D04550 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0x74u:
        unk_4D04548 = 1;
        dword_4D043DC = 1;
        continue;
      case 0x75u:
        unk_4D04544 = 1;
        continue;
      case 0x76u:
        unk_4D04540 = 1;
        continue;
      case 0x77u:
        unk_4D0453C = 1;
        continue;
      case 0x78u:
        unk_4D04538 = 1;
        continue;
      case 0x79u:
        unk_4D04534 = 1;
        continue;
      case 0x7Au:
        unk_4D04530 = 1;
        continue;
      case 0x7Bu:
        unk_4D0452C = 1;
        continue;
      case 0x7Cu:
        unk_4D04528 = 1;
        continue;
      case 0x7Du:
        unk_4D04520 = 1;
        unk_4F06A80 = 0;
        unk_4F06A7C = 1;
        unk_4F06930 = 113;
        continue;
      case 0x7Eu:
        unk_4F06A80 = 0;
        unk_4F06A7C = 1;
        continue;
      case 0x7Fu:
        unk_4D0451C = 1;
        continue;
      case 0x80u:
        unk_4D04518 = 1;
        continue;
      case 0x81u:
        unk_4D046C8 = 1;
        continue;
      case 0x82u:
        unk_4D046C4 = 1;
        continue;
      case 0x83u:
        unk_4D046C0 = 1;
        continue;
      case 0x84u:
        unk_4D046BC = 1;
        continue;
      case 0x85u:
        byte_4F06B90[0] = byte_4B6DF80[byte_4F06B90[0]];
        continue;
      case 0x86u:
        dword_4D04504 = 1;
        unk_4D04508 = 1;
        v88 = sub_721290(src);
        unk_4D044F0 = v88;
        unk_4D044E8 = 0;
        v89 = &dword_4D04500;
        goto LABEL_513;
      case 0x87u:
        dword_4D04500 = 1;
        v88 = sub_721290(src);
        unk_4D044F8 = v88;
        unk_4D04508 = 1;
        unk_4D044E8 = 0;
        v89 = &dword_4D04504;
LABEL_513:
        *v89 = 0;
        if ( !(unsigned int)sub_720EA0(v88) )
          sub_614590(v88);
        continue;
      case 0x88u:
        unk_4D044E8 = 1;
        dword_4D04500 = 0;
        dword_4D04504 = 0;
        unk_4D04508 = 1;
        continue;
      case 0x89u:
        unk_4D044E4 = v4 == 0;
        continue;
      case 0x8Au:
        unk_4D044E0 = v4;
        continue;
      case 0x8Bu:
        qword_4D044D8 = (char *)sub_721290(src);
        if ( !(unsigned int)sub_721580(qword_4D044D8) )
          sub_684920(680);
        continue;
      case 0x8Cu:
        unk_4D044C8 = v4;
        continue;
      case 0x8Du:
        unk_4D044B4 = v4;
        continue;
      case 0x8Eu:
        unk_4D043A4 = v4;
        continue;
      case 0x8Fu:
        v90 = sub_60F910((unsigned __int8 *)src);
        if ( !(unsigned int)sub_7A7520(v90, &unk_4D0438C) )
          sub_6849E0(660);
        continue;
      case 0x90u:
        unk_4D04388 = v4;
        continue;
      case 0x91u:
        unk_4D04380 = v4;
        continue;
      case 0x92u:
        unk_4D04378 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0x93u:
        unk_4D0436C = v4;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0x94u:
        unk_4F0746C = v4;
        continue;
      case 0x95u:
        unk_4D0435C = v4;
        continue;
      case 0x96u:
        v149 = v4;
        continue;
      case 0x97u:
        unk_4D0484C = v4;
        continue;
      case 0x98u:
        unk_4D04358 = 1;
        continue;
      case 0x99u:
        unk_4D0439C = v4;
        continue;
      case 0x9Au:
        unk_4D04844 = v4;
        continue;
      case 0x9Bu:
        unk_4D04840 = v4;
        continue;
      case 0x9Cu:
        unk_4D04838 = v4;
        continue;
      case 0x9Du:
        unk_4D04834 = v4;
        continue;
      case 0x9Eu:
        unk_4D04354 = v4;
        continue;
      case 0x9Fu:
        unk_4D04830 = v4;
        continue;
      case 0xA0u:
        unk_4D0482C = v4;
        unk_4D04828 = v4;
        continue;
      case 0xA1u:
        unk_4D04760 = v4;
        continue;
      case 0xA3u:
        dword_4CF8040 = 1;
        continue;
      case 0xA4u:
        unk_4D04350 = v4;
        continue;
      case 0xA5u:
        unk_4D04348 = v4;
        continue;
      case 0xA6u:
        unk_4F0697C = v4;
        continue;
      case 0xA7u:
        unk_4D04340 = v4;
        continue;
      case 0xA8u:
        unk_4D04338 = v4;
        continue;
      case 0xA9u:
        unk_4F07468 = v4 == 0;
        continue;
      case 0xAAu:
        unk_4D04334 = v4;
        continue;
      case 0xABu:
        unk_4D0475C = v4;
        continue;
      case 0xACu:
        dword_4D04824 = v4;
        continue;
      case 0xADu:
        unk_4D0432C = v4;
        if ( v4 )
          sub_721160();
        continue;
      case 0xAEu:
        unk_4D04324 = v4;
        continue;
      case 0xAFu:
        unk_4D047EC = v4;
        continue;
      case 0xB0u:
        unk_4D047E4 = v4;
        continue;
      case 0xB1u:
        unk_4D04318 = v4;
        continue;
      case 0xB2u:
        unk_4D0430C = v4;
        continue;
      case 0xB3u:
      case 0xB4u:
        v25 = sub_721290(src);
        v26 = sub_7AFC00();
        *(_QWORD *)(v26 + 8) = v25;
        if ( v6 == 180 )
        {
          if ( unk_4F076D0 )
            *qword_4F076C0 = v26;
          else
            unk_4F076D0 = v26;
          qword_4F076C0 = (_QWORD *)v26;
        }
        else
        {
          if ( unk_4F076D8 )
            *qword_4F076C8 = v26;
          else
            unk_4F076D8 = v26;
          qword_4F076C8 = (_QWORD *)v26;
        }
        continue;
      case 0xB5u:
        v64 = sub_60F910((unsigned __int8 *)src);
        if ( v64 )
          unk_4D042F0 = v64;
        else
          unk_4D042F0 = -1;
        continue;
      case 0xB6u:
        unk_4D047E0 = v4;
        continue;
      case 0xB7u:
        unk_4D047DC = v4;
        continue;
      case 0xB8u:
        unk_4D047D8 = v4;
        continue;
      case 0xB9u:
        unk_4D047D0 = v4;
        unk_4D047CC = v4;
        continue;
      case 0xBAu:
        unk_4D0479C = v4;
        continue;
      case 0xBCu:
        unk_4D04798 = v4;
        continue;
      case 0xBDu:
        unk_4D04794 = v4;
        continue;
      case 0xBEu:
        unk_4D04788 = v4;
        continue;
      case 0xBFu:
        unk_4D04780 = v4;
        continue;
      case 0xC0u:
        unk_4D042C0 = src;
        continue;
      case 0xC1u:
        unk_4D0477C = v4;
        continue;
      case 0xC2u:
        unk_4D04344 = v4;
        continue;
      case 0xC3u:
        unk_4D047C8 = v4;
        continue;
      case 0xC4u:
        unk_4D042AC = v4;
        continue;
      case 0xC5u:
        dword_4D047B0 = v4;
        continue;
      case 0xC6u:
        unk_4F07778 = 199901;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0xC7u:
        unk_4F07778 = 199000;
        unk_4D0436C = 0;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0xC8u:
        dword_4D04278 = v4;
        if ( v4 )
          unk_4D04274 = 1;
        continue;
      case 0xC9u:
        unk_4F068FC = v4;
        continue;
      case 0xCAu:
        HIDWORD(qword_4F077B4) = v4;
        dword_4F077C0 = v4;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0xCBu:
        HIDWORD(qword_4F077B4) = v4;
        dword_4F077BC = v4;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0xCCu:
        qword_4F077A8 = sub_60F910((unsigned __int8 *)src);
        HIDWORD(qword_4F077B4) = 1;
        if ( (unsigned __int64)(qword_4F077A8 - 30200LL) > 0xECC47 )
          sub_684920(1358);
        continue;
      case 0xCDu:
        unk_4D04320 = 1;
        continue;
      case 0xCEu:
        unk_4F072F0 = v7;
        continue;
      case 0xCFu:
        LODWORD(qword_4F077B4) = v4;
        continue;
      case 0xD0u:
        unk_4F077A0 = sub_60F910((unsigned __int8 *)src);
        LODWORD(qword_4F077B4) = 1;
        if ( unk_4F077A0 > 0xF423Fu )
          sub_684920(3301);
        continue;
      case 0xD1u:
        unk_4F077B0 = v4;
        continue;
      case 0xD2u:
        unk_4D04298 = v4;
        continue;
      case 0xD3u:
        v91 = sub_60F910((unsigned __int8 *)src);
        v92 = v91 & 0xFFFFFFFE;
        v14 = v91 == 1;
        v93 = 2;
        if ( !v14 )
          v93 = v92;
        unk_4F07474 = v93;
        continue;
      case 0xD4u:
        v94 = off_4A43F20;
        v95 = src;
        if ( !off_4A43F20 )
          goto LABEL_772;
        n = a1;
        v96 = &off_4A43F20;
        v97 = 0;
        do
        {
          if ( !strcmp(v94, v95) )
            v97 = (unsigned int *)v96[1];
          v94 = v96[2];
          v96 += 2;
        }
        while ( v94 );
        a1 = n;
        if ( !v97 )
LABEL_772:
          sub_684920(1157);
        *v97 = v4;
        continue;
      case 0xD5u:
        qword_4F07578 = (char *)sub_721290(src);
        if ( !(unsigned int)sub_721580(qword_4F07578) )
          sub_684920(1334);
        continue;
      case 0xD7u:
        unk_4D04220 = v4;
        unk_4D04224 = v4;
        continue;
      case 0xD8u:
        unk_4D04384 = v4;
        continue;
      case 0xD9u:
        unk_4D04314 = v4;
        continue;
      case 0xDAu:
        unk_4D042F8 = v4;
        continue;
      case 0xDBu:
        unk_4F07460 = v4;
        continue;
      case 0xDCu:
        dword_4D047AC = v4;
        continue;
      case 0xDDu:
        unk_4D043A8 = v4;
        continue;
      case 0xDEu:
        unk_4D044B0 = v4;
        continue;
      case 0xDFu:
        if ( dword_4CF8024 )
        {
          if ( dword_4F077C4 != 2 )
          {
            unk_4F07778 = 199711;
            continue;
          }
          if ( v4 )
            unk_4F07778 = 201103;
          else
            unk_4F07778 = 199711;
        }
        else if ( v4 )
        {
          unk_4F07778 = 201103;
        }
        else
        {
          unk_4F07778 = 199711;
        }
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0xE0u:
        unk_4D04944 = 1;
        unk_4D0493C = 0;
        unk_4D04910 = 1;
        continue;
      case 0xE1u:
        fprintf(
          qword_4F07510,
          "/* Configuration data for Edison Design Group C/C++ Front End */\n/* version %s, built on %s at %s. */\n\n",
          "6.6",
          off_4B6EB10,
          off_4B6EB08[0]);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_CHANGES_FOR_ARRAY_NEW_AND_DELETE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_CHANGES_FOR_CONSTRUCTION_VTBLS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_CHANGES_FOR_COVARIANT_VIRTUAL_FUNC_RETURN", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_CHANGES_FOR_PLACEMENT_DELETE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_CHANGES_FOR_RTTI", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ABI_COMPATIBILITY_VERSION", 9999);
        fprintf(qword_4F07510, "#define %s %td\n", "ABORT_ON_INIT_COMPONENT_LEAKAGE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ACCEPT_GNU_CARRIAGE_RETURN_LINE_TERMINATOR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ACCEPT_UNRECOGNIZED_GNU_ASM_OPERANDS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ADDRS_NOT_IN_SAME_ARRAY_CAN_BE_COMPARED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ADDR_OF_BIT_FIELD_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ADD_BRACES_TO_AVOID_DANGLING_ELSE_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ALIAS_DIRECTIVE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_ADDR_OF_REGISTER_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_CPPCLI_AND_CPPCX_WITH_LOWERING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_ELLIPSIS_ONLY_PARAM_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "ALLOW_ENABLING_OF_SSI_MODE");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "ALLOW_HIDDEN_NAMES_IN_IL_WITH_IL_LOWERING");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "ALLOW_HOST_FP_TOO_SMALL_FOR_LARGEST_FIXED_POINT_TYPE");
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_NON_INT_BIT_FIELD_BASE_TYPE_IN_GENERATED_C", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_SOURCE_SEQUENCE_LISTS_WITH_IL_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ALLOW_VOID_QUESTION_OPERAND_IN_GENERATED_C", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ALL_TEMPLATE_INFO_IN_IL", 0);
        fprintf(qword_4F07510, "/*      %s %td */\n", "ALTERNATE_IL_FILE_FORMAT", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ALWAYS_SET_MULTIBYTE_LOCALE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "APPROXIMATE_QUADMATH", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ARRAY_NEW_AND_DELETE_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ASM_FUNCTION_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ASSIGNMENT_TO_THIS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ASSIGN_STRING_LITERAL_SEQUENCE_NUMBERS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ASSUME_LITTLE_ENDIAN_IFC_MODULES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MODULE_MAX_LINE_NUMBER", 250000);
        fprintf(qword_4F07510, "#define %s %td\n", "ASSUME_REFERENCES_CANNOT_BE_NULL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ASSUME_THIS_CANNOT_BE_NULL_IN_CONDITIONAL_OPERATORS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ATT_PREPROCESSING_EXTENSIONS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "AUTOMATIC_TEMPLATE_INSTANTIATION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "BACKSLASH_CAN_OCCUR_AS_PART_OF_MULTIBYTE_CHAR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "BACKSLASH_IS_ALSO_DIR_SEPARATOR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "BACK_END_IS_CP_GEN_BE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "BACK_END_IS_C_GEN_BE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "BACK_END_SHOULD_BE_CALLED", 1);
        fprintf(qword_4F07510, "/*      %s %td */\n", "BITS_IN_AN_INTEGER_VALUE", 128);
        fprintf(qword_4F07510, "#define %s %td\n", "BOOL_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "BSEARCH_QSORT_FUNCTION_IS_EXTERN_C", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "BUILTIN_FUNCTIONS_ENABLED", "1");
        fprintf(qword_4F07510, "#define %s %s\n", "BUILTIN_VA_LIST_OVERRIDE_TYPE_NAME", "\"__gnuc_va_list\"");
        fprintf(qword_4F07510, "#define %s %td\n", "BUILTIN_VA_START_TAKES_ADDRESS_OF_VARIABLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "C99_IL_EXTENSIONS_SUPPORTED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "CFRONT_2_1_OBJECT_CODE_COMPATIBILITY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "CFRONT_3_0_OBJECT_CODE_COMPATIBILITY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "CFRONT_GLOBAL_VS_MEMBER_NAME_LOOKUP_BUG", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "CHECKING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "CHECK_SWITCH_DEFAULT_UNEXPECTED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "CLANG_VERSION_STRING", "\"\\\"%v \\\"\"");
        fprintf(qword_4F07510, "#define %s %td\n", "CLASS_TEMPLATE_INSTANTIATIONS_IN_SOURCE_SEQUENCE_LISTS", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CLR_FALLBACK_VERSION");
        fprintf(qword_4F07510, "#define %s %td\n", "COLUMN_NUMBER_IN_BRIEF_DIAGNOSTICS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "COMPILE_MULTIPLE_SOURCE_FILES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "COMPILE_MULTIPLE_TRANSLATION_UNITS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "COMPOUND_LITERAL_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CP_GEN_BE_TARGET_MATCHES_SOURCE_DIALECT");
        fprintf(qword_4F07510, "#define %s %td\n", "CPPCLI_ENABLING_POSSIBLE", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CPPCLI_PORTABLE_ASSEMBLY_PATH");
        fprintf(qword_4F07510, "#define %s %td\n", "CPPCX_ENABLING_POSSIBLE", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CPPCX_INCLUDE_PATH");
        fprintf(qword_4F07510, "#define %s %td\n", "CPP11_IL_EXTENSIONS_SUPPORTED", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CUSTOM_NAME_LINKAGE_KINDS");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CUSTOM_NAME_LINKAGE_KIND_NAMES");
        fprintf(qword_4F07510, "#define %s %td\n", "C_ANACHRONISMS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "C_GEN_BE_GENERATES_ANSI_C", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "CLANG_IS_GENERATED_CODE_TARGET", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "CLANG_TARGET_VERSION_NUMBER");
        fprintf(qword_4F07510, "#define %s %td\n", "COROUTINE_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEBUG", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DECL_MODIFIERS_IN_USE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ADDRESS_OF_ELLIPSIS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ADD_MATCH_NOTES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_ANACHRONISMS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_COPY_ASSIGNMENT_OP_WITH_BASE_CLASS_PARAM", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_DOLLAR_IN_ID_CHARS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_ELLIPSIS_ONLY_PARAM_IN_C_MODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_NONCONST_CALL_ANACHRONISM", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_NONCONST_REF_ANACHRONISM", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALLOW_NONSTANDARD_ANONYMOUS_UNIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALTERNATIVE_TOKENS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ALWAYS_FOLD_CALLS_TO_BUILTIN_CONSTANT_P", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ARG_DEPENDENT_LOOKUP", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ARRAY_NEW_AND_DELETE_ENABLED", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_AUTOMATIC_INSTANTIATION_MODE");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_AUTO_STORAGE_CLASS_SPECIFIER_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_AUTO_TYPE_SPECIFIER_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_BOOL_IS_KEYWORD", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_BRIEF_DIAGNOSTICS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_C99_MODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CHECK_CONCATENATIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CHECK_FOR_BYTE_ORDER_MARK", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CHECK_PRINTF_SCANF_POSITIONAL_ARGS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CLANG_COMPATIBILITY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CLANG_VERSION", 90100);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CLASS_NAME_INJECTION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_COMPOUND_LITERALS_ALLOWED", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_COMPRESS_MANGLED_NAMES");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CONTEXT_LIMIT", 10);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPPCLI_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPPCX_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPP11_DEPENDENT_NAME_PROCESSING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPP_MODE", 199711);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPP11_SFINAE_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPP11_SFINAE_IGNORE_ACCESS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_C_AND_CPP_FUNCTION_TYPES_ARE_DISTINCT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_CPPCLI_CPPCX_VERSION", 1800);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DEPENDENT_LOOKUP_FINDS_STATIC_FUNCTIONS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DEPENDENT_NAME_PROCESSING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DEPRECATED_STRING_LITERAL_CONV_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DESIGNATORS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DIAG_OVERRIDE_DOES_NOT_AFFECT_SFINAE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DISABLE_ACCESS_CHECKING_IN_MICROSOFT_ENUM_BASES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DISPLAY_ERROR_CONTEXT_ON_CATASTROPHE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DISPLAY_ERROR_NUMBER", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DISPLAY_TEMPLATE_TYPEDEFS_IN_DIAGNOSTICS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DISTINCT_TEMPLATE_SIGNATURES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_DO_LATE_OVL_RES_TIEBREAKER", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_EDG_BASE", "\"\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "DEFAULT_EDG_COLORS",
          "\"error=01;31:warning=01;35:note=01;36:locus=01:quote=01:range1=32\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EMBEDDED_C_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EMULATE_GNU_ABI_BUGS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EMULATE_GNU_VALUE_INITIALIZATION_BUGS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EMULATE_MSVC_VALUE_INITIALIZATION_BUGS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ENABLE_COLORIZED_DIAGNOSTICS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXCEPTIONS_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXPLICIT_KEYWORD_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXPORT_TEMPLATE_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXTENDED_DESIGNATORS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXTENDED_VARIADIC_MACROS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_EXTERN_INLINE_ALLOWED", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_FAR_CODE_POINTERS");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_FAR_DATA_POINTERS");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_FIXED_POINT_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_FLOAT_KIND_FOR_FLOAT80", "fk_long_double");
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_FLOAT_KIND_FOR_FLOAT128", "fk_float128");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_FLOATING_POINT_TEMPLATE_PARAMETERS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_FRIEND_INJECTION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GCC_CONST_VARIABLES_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GENERIC_ARITY_OVERLOAD_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GNU_ABI_VERSION", 30200);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GNU_COMPATIBILITY", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GNU_STDC_ZERO_IN_SYSTEM_HEADERS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GNU_VERSION", 80100);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GNU_VISIBILITY_ATTRIBUTE_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_GUIDING_DECLS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_IMPLICIT_NOEXCEPT_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_IMPLICIT_TEMPLATE_INCLUSION_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_IMPLICIT_TYPENAME_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_IMPLICIT_USING_STD", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_IMPL_CONV_BETWEEN_C_AND_CPP_FUNCTION_PTRS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_INCLUDE_FILE_SUFFIX_LIST", "\"::stdh:\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_INLINE_STATEMENT_LIMIT", 100);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_INSTANTIATIONS_PERMITTED_IN_CLASS_SRC_SEQ_LIST");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "DEFAULT_INSTANTIATION_FILE_SUFFIX_LIST",
          "\"c:C:cpp:CPP:cxx:CXX:cc\"");
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_INSTANTIATION_MODE", "tim_none");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_LAMBDAS_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_LONG_PRESERVING_RULES", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_MACRO_POSITIONS_IN_DIAGNOSTICS");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MAX_COST_CONSTEXPR_CALL", 2000000);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MAX_DEPTH_CONSTEXPR_CALL", 512);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MAX_MANGLED_NAME_LENGTH", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MAX_PENDING_INSTANTIATIONS", 200);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MICROSOFT_64BIT_POINTER_EXTENSIONS_ENABLED", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_MICROSOFT_BUGS");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MICROSOFT_COMPATIBILITY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MICROSOFT_EXTENSIONS", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_MICROSOFT_MODE");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MICROSOFT_VERSION", 1926);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MODULES_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MODULE_IMPORT_DIAG_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MS_PERMISSIVE", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_MSVC_EXECUTION_CHARACTER_SET", "\"1252\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_MULTIBYTE_CHARS_IN_SOURCE_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NAMED_ADDRESS_SPACES_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NAMED_REGISTERS_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NAMESPACES_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NEAR_AND_FAR_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NONSTANDARD_DEFAULT_ARG_DEDUCTION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NONSTANDARD_INSTANTIATION_LOOKUP", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NONSTANDARD_QUALIFIER_DEDUCTION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NONSTANDARD_USING_DECL_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NO_ACCESS_CHECK_ON_FRIEND_DECLARATOR_IDS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NULLPTR_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_NULL_CHARS_ALLOWED_IN_SOURCE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_OLD_SPECIALIZATIONS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_OLD_SPECIALIZATIONS_FOR_GENERATED_INSTANCES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_OPERATOR_OVERLOADING_ON_ENUMS", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_OUTPUT_MODE", "om_text");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_INCOGNITO", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_PASS_STDARG_REFERENCES_TO_GENERATED_CODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_POINTER_TO_MEMBER_CALL_OPTIMIZATION_ALLOWED", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_PREALLOCATED_PCH_MEM_SIZE");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RANGE_BASED_FOR_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RECORD_FORM_OF_NAME_REFERENCE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_REFLECTION_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_REMOVE_QUALIFIERS_FROM_PARAM_TYPES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_REMOVE_UNNEEDED_ENTITIES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RESTRICT_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RIGHT_SHIFT_CAN_BE_ANGLE_BRACKETS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RTTI_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_RVALUE_REFERENCES_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_SINGLE_REF_QUAL_OVL_RES_TIEBREAKER", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_SPECIAL_SUBSCRIPT_COST", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_STDC_ZERO_IN_SYSTEM_HEADERS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_STRING_LITERALS_ARE_CONST", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_SUN_COMPATIBILITY");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_SUN_LINKER_SCOPE_ALLOWED");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_SUPPRESS_DEFERRAL_ON_PARTIAL_SPEC_MEMBERS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_SVR4_C_MODE", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEFAULT_TARGET_CONFIGURATION_NAME");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_THREAD_LOCAL_STORAGE_SPECIFIER_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_TMPDIR", "\"/tmp\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_TRIGRAPHS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_TYPENAME_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_TYPE_INFO_IN_NAMESPACE_STD", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_TYPE_TRAITS_HELPERS_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_ULITERALS_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_UNICODE_SOURCE_KIND", "usk_utf8");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_UPC_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_USE_NONSTANDARD_FOR_INIT_SCOPE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_USE_PREDEFINED_MACRO_FILE", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "DEFAULT_USR_INCLUDE", "\"/usr/include\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_VARIADIC_MACROS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_VARIADIC_TEMPLATES_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_VA_LIST_IN_STD_NAMESPACE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_VLA_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_WARNING_ON_FOR_INIT_DIFFERENCE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_WARNING_ON_LOSSY_CONVERSION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_WARNING_ON_NON_TEMPLATE_FRIEND", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFAULT_WCHAR_T_IS_KEYWORD", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_ARRAY_NEW_AND_DELETE_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_FEATURE_TEST_MACRO_OPERATORS_IN_ALL_MODES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_BOOL_IS_KEYWORD", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_CHAR16_T_AND_CHAR32_T_ARE_KEYWORDS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_EXCEPTIONS_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_LONG_LONG_IS_DISABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_PLACEMENT_DELETE_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_RTTI_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_VARIADIC_TEMPLATES_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_MACRO_WHEN_WCHAR_T_IS_KEYWORD", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DEFINE_STDC_IN_MICROSOFT_MODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DELETE_CAN_BE_FOLDED_INTO_DTOR", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DELETED_COPY_FUNCTION_CLEARS_BITWISE_COPY_FLAG", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DEMO_VERSION_ID");
        fprintf(qword_4F07510, "#define %s %s\n", "DIRECTORY_SEPARATOR", "'/'");
        fprintf(qword_4F07510, "#define %s %s\n", "DIRECTORY_SEPARATOR_STRING", "\"/\"");
        fprintf(qword_4F07510, "#define %s %td\n", "DIRECT_ERROR_OUTPUT_TO_STDOUT", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DOING_SOURCE_ANALYSIS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DO_FULL_PORTABLE_EH_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DO_IL_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DO_NOT_ASSUME_QUESTION_IS_LARGER_THAN_END_OF_LINE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "DO_RETURN_VALUE_OPTIMIZATION_IN_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DO_UNORDERED_EH_PROCESSING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "DRIVER_COMPATIBILITY_VERSION", 9999);
        fprintf(qword_4F07510, "#define %s %td\n", "DUMP_CONFIG_ENABLED", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "DUMP_LOWERED_EH_CONSTRUCTS_IN_C_GEN_BE");
        fprintf(qword_4F07510, "#define %s %td\n", "DUPLICATE_SPECIAL_STATICS_IN_INSTANTIATION_SLICES", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "EDG_AUXILIARY_INFO_DIR_NAME", "\"lib\"");
        fprintf(qword_4F07510, "#define %s %s\n", "EDG_MAIN", "lgenfe_main");
        fprintf(qword_4F07510, "#define %s %td\n", "EDG_MULTIBYTE_CHAR_TEST_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "EDG_NATIVE_MULTIBYTE_TEST_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "EDG_WIN32", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "EMBEDDED_C_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ENABLE_TRANS_UNIT_TEST_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "END_OF_LINE_COMMENTS_ALLOWED_IN_C_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ENSURE_LOWERED_TYPE_LIST_ORDERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ENTRY_NUMBER_SHARES_BITS_IN_PREFIX", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "ERROR_SEVERITY_EXPLICIT_IN_ERROR_MESSAGES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "EXC_SPEC_IN_FUNC_TYPE_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "EXIT_ON_INTERNAL_ERROR", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "EXPENSIVE_CHECKING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "EXPLICITLY_UNROLL_CRITICAL_LOOPS", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "EXPORTED_TEMPLATE_FILE_SUFFIX");
        fprintf(qword_4F07510, "#define %s %td\n", "EXPORT_ENABLING_POSSIBLE", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "EXPORT_INFO_FILE_NAME");
        fprintf(qword_4F07510, "#define %s %td\n", "EXTRA_SOURCE_POSITIONS_IN_IL", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FAVOR_CONSTANT_RESULT_FOR_NONSTATIC_INIT", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "FILE_NAME_FOR_STDIN", "\"-\"");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "FIXED_ADDRESS_FOR_MMAP");
        fprintf(qword_4F07510, "#define %s %td\n", "FIXED_POINT_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "FLOAT128_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FLOAT80_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FORCE_STORES_OF_VARS_MODIFIED_IN_TRY_BLOCKS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FORCE_VARIABLE_DEFINITION_VIA_ZEROING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FREE_MEMORY_REGIONS_EARLY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "FP_HAS_LONG_DOUBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FP_LONG_DOUBLE_IS_80BIT_EXTENDED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FP_LONG_DOUBLE_IS_BINARY128", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "FP_LONG_DOUBLE_IS_BINARY64", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "FP_USE_EMULATION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FRIEND_AND_MEMBER_DEFINITIONS_MAY_BE_MOVED_OUT_OF_CLASS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FULLY_RESOLVED_MACRO_POSITIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "FUNC_AVAILABLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "FUNCTION_PROTOTYPE_INSTANTIATION_DEFERRAL_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GCC_BUILTIN_VARARGS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GCC_BUILTIN_VARARGS_IN_GENERATED_CODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GCC_IS_GENERATED_CODE_TARGET", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "GCC_VERSION_STRING", "\"\\\"EDG %m %v mode\\\"\"");
        fprintf(qword_4F07510, "#define %s %td\n", "GENERATE_EH_TABLES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GENERATE_LINKAGE_SPEC_BLOCKS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GENERATE_MICROSOFT_IF_EXISTS_ENTRIES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "GENERATE_SOURCE_SEQUENCE_LISTS", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "GEN_C_FILE_SUFFIX", "\".int.c\"");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "GEN_CPP_FILE_SUFFIX");
        fprintf(qword_4F07510, "#define %s %td\n", "GEN_EXTRA_LINE_ID_INFO", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "GET_DEFINITION_OF_CLASS_NEEDED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_EXTENSIONS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_FUNCTION_MULTIVERSIONING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_INIT_PRIORITY_ATTRIBUTE_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_NAKED_ATTRIBUTE_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_TARGET_VERSION_NUMBER", 100300);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_VECTOR_TYPES_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_VISIBILITY_ATTRIBUTE_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_X86_ASM_EXTENSIONS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "GNU_X86_ATTRIBUTES_ALLOWED", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "GUARD_MACRO2_FOR_VA_LIST");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "GUARD_MACRO_FOR_VA_LIST");
        fprintf(qword_4F07510, "#define %s %td\n", "HANDLE_VIRTUAL_BASES_IN_COMPLETE_CTOR_DTORS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "HANDLE_VIRTUAL_BASES_IN_SUBOBJECT_CTOR_DTORS", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "HOSTID");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "HOSTID2");
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_ALIGNMENT_REQUIRED", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_ALLOCATION_INCREMENT", 0x10000);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_IL_ENTRY_PREFIX_ALIGNMENT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_FP_VALUE_IS_128BIT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_HAS_FLOAT16_TYPE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_TARGET_ENDIAN_MISMATCH_OKAY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "HOST_POINTER_ALIGNMENT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "IA64_ABI", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "IDENTIFIER_STRINGS_ALLOW_MULTIBYTE_CHARS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "IDENT_DIRECTIVE_AND_PRAGMA", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "IGNORE_CARRIAGE_RETURN_IN_SOURCE", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "IL_FILE_SUFFIX");
        fprintf(qword_4F07510, "#define %s %s\n", "IL_LOWERING_INIT_ROUTINE_PREFIX", "\"__sti__\"");
        fprintf(qword_4F07510, "#define %s %td\n", "IL_SHOULD_BE_WRITTEN_TO_FILE", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "IL_VERSION_NUMBER", "\"6.6\"");
        fprintf(qword_4F07510, "#define %s %td\n", "IL_WALK_NEEDED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "IMPL_CONV_BETWEEN_C_AND_CPP_FUNCTION_PTRS_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "IMPLEMENTATION_SUPPORTS_MULTIPLE_THREADS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_COMMENTS_IN_ASM_FUNC_BODY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_EDG_TEST_ATTRIBUTES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_EDG_TEST_NAMED_ADDRESS_SPACES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_EDG_TEST_NAMED_REGISTERS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_EDG_TEST_PRAGMAS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INCLUDE_UNRECOGNIZED_PRAGMAS_IN_IL", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "INDICATE_CLEANUP_STATE_IN_UNREACHABLE_CODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INSTANTIATE_BEFORE_PCH_CREATION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INSTANTIATE_EXTERN_INLINE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INSTANTIATE_INLINE_VARIABLES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INSTANTIATE_TEMPLATES_EVERYWHERE_USED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "INSTANTIATION_BY_IMPLICIT_INCLUSION", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "INSTANTIATION_FILE_SUFFIX");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "INSTANTIATION_FLAGS_IN_TEMPLATE_INFO_FILE");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "INSTANTIATION_REQUEST_LINES_RESERVED");
        fprintf(qword_4F07510, "#define %s %td\n", "INT128_EXTENSIONS_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "INTEGER_VALUE_REPR_IS_A_HOST_INTEGER", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ISSUE_WARNING_ON_LONG_DOUBLE_AS_DOUBLE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "KEEP_OBJECT_LIFETIME_INFO_IN_LOWERED_IL_WHEN_EH_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "KEEP_TEMPLATE_ARG_EXPR_THAT_CAUSES_INSTANTIATION", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "LARGE_IL_FILE_SUPPORT");
        fprintf(qword_4F07510, "#define %s %td\n", "LAZY_INITIALIZATION_USES_WEAK_REFERENCES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LINKER_CAN_DISCARD_DUPLICATE_DEFINITIONS", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "LOCALE_TO_SET_WHEN_MULTIBYTE_CHARS_ENABLED");
        fprintf(qword_4F07510, "#define %s %td\n", "LONG_DOUBLE_AS_DOUBLE_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "LONG_LONG_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_CLASS_RVALUE_ADJUST", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_COMPLEX", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_DESIGNATED_INITIALIZERS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_EXTERN_INLINE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_FIXED_POINT", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_IFUNC", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_LVALUE_RETURNING_OPERATIONS", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "LOWER_MICROSOFT_NONCONSTANT_AGGREGATE");
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_STRING_LITERALS_TO_NON_CONST", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWER_VARIABLE_LENGTH_ARRAYS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWERING_NORMALIZES_BOOLEAN_CONTROLLING_EXPRESSIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "LOWERING_REMOVES_UNNEEDED_CONSTRUCTIONS_AND_DESTRUCTIONS", 0);
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_ARRAY_NEW_AND_DELETE_ENABLED",
          "\"__ARRAY_OPERATORS\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_BOOL_IS_KEYWORD", "\"_BOOL\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_CHAR16_T_AND_CHAR32_T_ARE_KEYWORDS",
          "\"__CHAR16_T_AND_CHAR32_T\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_EXCEPTIONS_ENABLED", "\"__EXCEPTIONS\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_IA64_ABI", "\"__EDG_IA64_ABI\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_IA64_CTORS_DTORS_RETURN_THIS",
          "\"__EDG_IA64_ABI_VARIANT_CTORS_AND_DTORS_RETURN_THIS\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_IA64_USE_INT_STATIC_INIT_GUARD",
          "\"__EDG_IA64_ABI_USE_INT_STATIC_INIT_GUARD\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_IMPLICITLY_USING_STD",
          "\"__EDG_IMPLICIT_USING_STD\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_LONG_LONG_IS_DISABLED", "\"__NO_LONG_LONG\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_PLACEMENT_DELETE_ENABLED",
          "\"__PLACEMENT_DELETE\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_RTTI_ENABLED", "\"__RTTI\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_RUNTIME_USES_NAMESPACES",
          "\"__EDG_RUNTIME_USES_NAMESPACES\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_TYPE_TRAITS_HELPERS_ENABLED",
          "\"__EDG_TYPE_TRAITS_ENABLED\"");
        fprintf(
          qword_4F07510,
          "#define %s %s\n",
          "MACRO_DEFINED_WHEN_VARIADIC_TEMPLATES_ENABLED",
          "\"__VARIADIC_TEMPLATES\"");
        fprintf(qword_4F07510, "#define %s %s\n", "MACRO_DEFINED_WHEN_WCHAR_T_IS_KEYWORD", "\"_WCHAR_T\"");
        fprintf(qword_4F07510, "#define %s %td\n", "MACRO_INVOCATION_TREE_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MAINTAIN_ALLOCATION_SEQUENCE_NUMBER", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MAINTAIN_NEEDED_FLAGS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "MAKE_ALL_FUNCTIONS_UNPROTOTYPED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MAKE_FRONT_END_CALLABLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "MANGLE_ALL_NAMES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_CHAR16_T_ENCODING_LENGTH", 2);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_ERROR_OUTPUT_LINE_LENGTH", 79);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_INCLUDE_FILES_OPEN_AT_ONCE", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_MULTIBYTE_CHAR_LENGTH", 16);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "MAX_SIZEOF_LARGEST_FIXED_POINT");
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_SIZEOF_LARGEST_INTEGER", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_TOTAL_PENDING_INSTANTIATIONS", 256);
        fprintf(qword_4F07510, "#define %s %td\n", "MAX_UNUSED_ALL_MODE_INSTANTIATIONS", 200);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "METADATA_IMPORT_BUFFER_ALLOCATION_INCREMENT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "METADATA_IMPORT_BUFFER_SIZE");
        fprintf(qword_4F07510, "#define %s %td\n", "MICROSOFT_DIALECT_IS_GENERATED_CODE_TARGET", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MICROSOFT_EXTENSIONS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MICROSOFT_MODE_TYPE_INFO_IN_NAMESPACE_STD", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MINIMAL_INLINING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "MIN_GNU_VERSION", 30200);
        fprintf(qword_4F07510, "#define %s %td\n", "MSVC_IS_GENERATED_CODE_TARGET", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "MSVC_TARGET_VERSION_NUMBER", 1926);
        fprintf(qword_4F07510, "#define %s %td\n", "MULTIBYTE_CHARS_IN_SOURCE_SUPPORTED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "NAMED_ADDRESS_SPACES_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NAMED_REGISTERS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NATIVE_MULTIBYTE_CHARS_SUPPORTED_WITH_UNICODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NEAR_AND_FAR_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NEED_DECLARATIVE_WALK", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NEED_IL_DISPLAY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NEED_NAME_MANGLING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "NEW_AND_DELETE_FOR_ARRAY_CAN_BE_FOLDED_INTO_RUNTIME_ROUTINE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "NEW_CAN_BE_FOLDED_INTO_CTOR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NONCLASS_TEMPLATE_INSTANTIATIONS_IN_SOURCE_SEQUENCE_LISTS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NO_USR_INCLUDE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "NO_VLA_DIMENSION_TEMPORARIES_IN_FUNCTION_PROTOTYPES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "NULL_POINTER_IS_ZERO", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "NUM_BITS_FOR_CHARACTER_KIND", 3);
        fprintf(qword_4F07510, "#define %s %td\n", "NUM_BITS_FOR_EXPR_NODE_KIND", 8);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "NUM_BITS_FOR_NAMED_ADDRESS_SPACE");
        fprintf(qword_4F07510, "#define %s %td\n", "NUM_BITS_FOR_NAME_LINKAGE", 3);
        fprintf(qword_4F07510, "#define %s %td\n", "NUM_BITS_FOR_STDC_PRAGMA_VALUE", 2);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "NUM_NAMED_REGISTERS");
        fprintf(qword_4F07510, "#define %s %s\n", "OBJECT_FILE_SUFFIX", "\".o\"");
        fprintf(qword_4F07510, "#define %s %td\n", "OLD_STYLE_PREPROCESSING_IN_CFRONT_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ONE_INSTANTIATION_PER_OBJECT", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "ONLY_MANGLE_TYPES_NEEDED_FOR_EXTERNAL_NAMES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "OPTIMIZE_VIRTUAL_FUNCTION_CALLS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "OVERWRITE_FREED_MEM_BLOCKS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "PARENS_IN_IL", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "PASS_ELEM_COUNT_TO_RUNTIME_AS_PTRDIFF_T");
        fprintf(qword_4F07510, "#define %s %td\n", "PCH_DECL_SEQ_THRESHOLD", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "PCH_FILE_SUFFIX", "\".pch\"");
        fprintf(qword_4F07510, "#define %s %td\n", "PRAGMA_DEFINE_TYPE_INFO_IS_REQUIRED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "PRAGMA_WEAK_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "PREDEFINED_MACRO_FILE_NAME", "\"predefined_macros.txt\"");
        fprintf(qword_4F07510, "#define %s %td\n", "PRESERVE_SOURCE_SEQUENCE_LISTS_WITH_IL_LOWERING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "PRESERVE_TOP_LEVEL_CASTS_TO_VOID_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "PRINTF_FORMAT_FOR_HEX_INTEGER_VALUE", "\"%llx\"");
        fprintf(qword_4F07510, "#define %s %s\n", "PRINTF_FORMAT_FOR_HOST_LARGE_INTEGER", "\"%td\"");
        fprintf(qword_4F07510, "#define %s %s\n", "PRINTF_FORMAT_FOR_HOST_LARGE_UNSIGNED", "\"%zu\"");
        fprintf(qword_4F07510, "#define %s %s\n", "PRINTF_FORMAT_FOR_SIGNED_INTEGER_VALUE", "\"%lld\"");
        fprintf(qword_4F07510, "#define %s %s\n", "PRINTF_FORMAT_FOR_UNSIGNED_INTEGER_VALUE", "\"%llu\"");
        fprintf(qword_4F07510, "#define %s %td\n", "PROMOTE_LOCAL_ENTITIES_TO_FILE_SCOPE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "PROTOTYPED_INT_ARGS_PASSED_LIKE_UNPROTOTYPED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "PROTOTYPE_INSTANTIATIONS_IN_IL", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "PTR_TO_INCOMP_ARRAY_ARITHMETIC_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "PTR_TO_MEMBER_REPR_SUPPORTS_CAST_FROM_VIRTUAL_BASE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "QUESTION_MARK_CAN_OCCUR_AS_PART_OF_MULTIBYTE_CHAR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "READ_CPPCLI_PORTABLE_ASSEMBLIES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "READ_SOURCE_IN_BINARY_MODE_FOR_MSDOS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECOGNIZE_MICROSOFT_ATTRIBUTES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_BACKING_EXPRS_WITH_IL_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_BIT_FIELD_CONTAINER_OFFSETS_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_HIDDEN_NAMES_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_MACROS_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_MACRO_INVOCATIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_RAW_ASM_OPERAND_DESCRIPTIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_SCOPE_DEPTH_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_TEMPLATE_STRINGS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "RECORD_UNRECOGNIZED_ATTRIBUTES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "REDEFINE_EXTNAME_PRAGMA_ENABLED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "REDUCE_BACKING_EXPRESSION_USE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "REF_DESTRUCTORS_FOR_PARAMETER_VARIABLES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "REFLECTION_ENABLING_POSSIBLE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "REMOVE_INLINE_BODIES_FROM_CLASS_TEMPLATE_DEFINITIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "REPLACE_SPECIAL_CHARACTERS_IN_MANGLED_NAMES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "REPRESENT_C11_GENERIC_CONSTRUCT_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "REWRITE_UCN_ESCAPE_CHAR_IN_LOWERING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RTTI_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RUNTIME_SUPPORTS_ARRAY_LENGTH_CHECK", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RUNTIME_SUPPORTS_SIZED_DEALLOCATION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RUNTIME_USES_NAMESPACES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "RUNTIME_USES_TYPENAME", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SAME_REPR_INTS_INTERCHANGEABLE_IN_IL", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SEPARATE_ROUTINES_FOR_FILE_SCOPE_DYNAMIC_INITS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SEQUENCING_DIAGNOSTICS_ENABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "STACK_REFERENCED_INCLUDE_DIRECTORIES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "STATEMENTS_INSERTED_FOR_INLINING_HAVE_INVOCATION_POSITION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "STAT_AVAILABLE", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "STAT_FIRST_PARAM_IS_CONST");
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_HOSTED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_IEC_559", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_IEC_559_COMPLEX", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_ISO_10646", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "STDC_ISO_10646_VALUE");
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_MB_MIGHT_NEQ_WC", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "STDC_ZERO_IN_NONSTRICT_MODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SUN_EXTENSIONS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SUN_IS_GENERATED_CODE_TARGET", 0);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "SUN_TARGET_VERSION_NUMBER");
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_ARRAY_STATIC_IN_GENERATED_CODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_CONST_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_MICROSOFT_ATTRIBUTE_PROCESSING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_NEAR_AND_FAR_IN_GENERATED_CODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_RESTRICT_IN_GENERATED_CODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SUPPRESS_TYPEINFO_VARIABLES_WHEN_RTTI_DISABLED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "SVR4_TRAP_NULL_POINTER_REFERENCES", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_ALERT_CHAR", "'\\007'");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_DOUBLE", 8);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_FAR_POINTER");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_FLOAT", 4);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_FLOAT128", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_FLOAT80", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_INT", 4);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_INT128", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_LONG", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_LONG_DOUBLE", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_LONG_LONG", 8);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_NEAR_POINTER");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_POINTER", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_PTR_TO_DATA_MEMBER", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_PTR_TO_MEMBER_FUNCTION", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_PTR_TO_VIRTUAL_BASE_CLASS", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_SHORT", 2);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_SIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_ALIGNOF_UNSIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALIGNOF_VIRTUAL_FUNCTION_INFO", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ALL_POINTERS_SAME_SIZE", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_BACKSPACE_CHAR", "'\\b'");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_BIT_FIELD_AFFECTS_UNION_ALIGNMENT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_BIT_FIELD_CONTAINER_SIZE", -1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_BOOL_INT_KIND", "((an_integer_kind)ik_char)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_C_BOOL_INT_KIND", "((an_integer_kind)ik_unsigned_char)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_CARR_RETURN_CHAR", "'\\r'");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_CASE_SENSITIVE_EXTERNAL_NAMES", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_CHAR16_T_INT_KIND", "((an_integer_kind)ik_unsigned_short)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_CHAR32_T_INT_KIND", "((an_integer_kind)ik_unsigned_int)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_CHAR_BIT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_CHAR_CONSTANT_FIRST_CHAR_MOST_SIGNIFICANT", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_CPP_COMPILER_DOES_NOT_VISIBLY_INJECT_FRIEND_NAMES");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DBL_MANT_DIG", 53);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DBL_MAX_EXP", 1024);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DBL_MIN_EXP", -1021);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DEFAULT_NEW_ALIGNMENT", 16);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_DELTA_INT_KIND", "((an_integer_kind)ik_long)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DOUBLE_FIELD_ALIGNMENT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_DUAL_ALIGNMENTS_FOR_BUILTIN_TYPES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ENUM_BIT_FIELDS_ARE_ALWAYS_UNSIGNED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ENUM_TYPES_CAN_BE_SMALLER_THAN_INT", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_ESC_CHAR", "'\\033'");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_ETS_FLAG_TYPE_INT_KIND", "((an_integer_kind)ik_unsigned_int)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_EXTERNAL_NAMES_GET_UNDERSCORE_ADDED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FIELD_ALLOC_SEQUENCE_EQUALS_DECL_SEQUENCE", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FLOAT_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FLOAT128_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FLOAT80_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT_MANT_DIG", 24);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT_MAX_EXP", 128);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT_MIN_EXP", -125);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT128_MANT_DIG", 113);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT128_MAX_EXP", 0x4000);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT128_MIN_EXP", -16381);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT80_MANT_DIG", 64);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT80_MAX_EXP", 0x4000);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FLT80_MIN_EXP", -16381);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_FORCE_ONE_BIT_BIT_FIELD_TO_BE_UNSIGNED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_FORM_FEED_CHAR", "'\\f'");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_SIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_FRACTIONAL_BITS_FOR_UNSIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_HAS_IEEE_FLOATING_POINT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_HAS_SIGNED_CHARS", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_HORIZ_TAB_CHAR", "'\\t'");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_HOST_STRING_CHAR_BIT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_USE_GUARD_ACQUIRE_RELEASE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_USE_INT_STATIC_INIT_GUARD", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_USE_VARIANT_ARRAY_COOKIES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_USE_VARIANT_PTR_TO_MEMBER_FUNCTION_REPR", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_VARIANT_CTORS_AND_DTORS_RETURN_THIS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_IA64_ABI_VARIANT_KEY_FUNCTION", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_IA64_VTABLE_ENTRY_INT_KIND", "((an_integer_kind)ik_long)");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_INT_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_INT128_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_JMP_BUF_ELEMENTS_ARE_FLOAT", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_JMP_BUF_ELEMENT_FLOAT_KIND", "((a_float_kind)fk_long_double)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_JMP_BUF_ELEMENT_INT_KIND", "((an_integer_kind)ik_long)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_JMP_BUF_NUM_ELEMENTS", 25);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_SETJMP_FUNC", "\"_setjmp\"");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_LDBL_MANT_DIG", 64);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_LDBL_MAX_EXP", 0x4000);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_LDBL_MIN_EXP", -16381);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_LIBGCC_CMP_RETURN_MODE", "((a_type_mode_kind)tmk_DI)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_LIBGCC_SHIFT_COUNT_MODE", "((a_type_mode_kind)tmk_DI)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_LITTLE_ENDIAN", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_LONG_DOUBLE_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_LONG_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_LONG_LONG_FIELD_ALIGNMENT", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MAXIMUM_INTRINSIC_ALIGNMENT", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MAXIMUM_PACK_ALIGNMENT", 0x80000000LL);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MAX_BASE_CLASS_OFFSET", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MAX_CLASS_OBJECT_SIZE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MICROSOFT_BIT_FIELD_ALLOCATION", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MICROSOFT_PTR_TO_MEMBER_SIZING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MINIMUM_PACK_ALIGNMENT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_MINIMUM_STRUCT_ALIGNMENT", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_NEWLINE_CHAR", "'\\n'");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_NONNEGATIVE_ENUM_BIT_FIELD_IS_UNSIGNED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_NO_ERROR_ON_INTEGER_OVERFLOW", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_NULL_IS_ALL_BITS_ZERO", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_OPTIMIZE_EMPTY_BASE_CLASS_LAYOUT", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_PAD_ALLOCATED_EMPTY_BASE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_PAD_BIT_FIELDS_LARGER_THAN_BASE_TYPE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_PLAIN_INT_BIT_FIELD_IS_UNSIGNED", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_POINTER_MODE", "((a_type_mode_kind)tmk_DI)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_PTRDIFF_T_INT_KIND", "((an_integer_kind)ik_long)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_REGION_NUMBER_INT_KIND", "((an_integer_kind)ik_unsigned_short)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_REUSE_TAIL_PADDING", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_RIGHT_SHIFT_IS_ARITHMETIC", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_RUNTIME_ELEM_COUNT_INT_KIND");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SHORT_FIELD_ALIGNMENT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIGNIF_CHARS_IN_EXTERNAL_NAME", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_DOUBLE", 8);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_FAR_POINTER");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_FLOAT", 4);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_FLOAT80", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_FLOAT128", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_INT", 4);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_INT128", 16);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_LARGEST_FIXED_POINT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_LARGEST_ATOMIC", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_LARGEST_INTEGER", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_LONG", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_LONG_DOUBLE", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_LONG_LONG", 8);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_NEAR_POINTER");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_POINTER", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_PTR_TO_DATA_MEMBER", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_PTR_TO_MEMBER_FUNCTION", 16);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_PTR_TO_VIRTUAL_BASE_CLASS", 8);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_SHORT", 2);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_SIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_LONG_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_LONG_FRACT");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_SHORT_ACCUM");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TARG_SIZEOF_UNSIGNED_SHORT_FRACT");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SIZEOF_VIRTUAL_FUNCTION_INFO", 8);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_SIZE_T_INT_KIND", "((an_integer_kind)ik_unsigned_long)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_SSIZE_T_INT_KIND", "((an_integer_kind)ik_long)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SUPPORTS_X86_64", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SUPPORTS_ARM64", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_SUPPORTS_ARM32", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_TOO_LARGE_SHIFT_COUNT_IS_TAKEN_MODULO_SIZE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_UNNAMED_BIT_FIELD_AFFECTS_STRUCT_ALIGNMENT", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_UNWIND_WORD_MODE", "((a_type_mode_kind)tmk_DI)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_USER_CONTROL_OF_STRUCT_PACKING_AFFECTS_BASE_CLASSES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_USER_CONTROL_OF_STRUCT_PACKING_AFFECTS_BIT_FIELDS", 1);
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_VAR_HANDLE_INT_KIND", "((an_integer_kind)ik_unsigned_short)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_VERT_TAB_CHAR", "'\\013'");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_VIRTUAL_FUNCTION_INDEX_INT_KIND", "((an_integer_kind)ik_short)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_WCHAR_T_INT_KIND", "((an_integer_kind)ik_int)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_WINT_T_INT_KIND", "((an_integer_kind)ik_unsigned_int)");
        fprintf(qword_4F07510, "#define %s %s\n", "TARG_WORD_MODE", "((a_type_mode_kind)tmk_DI)");
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ZERO_WIDTH_BIT_FIELD_AFFECTS_STRUCT_ALIGNMENT", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "TARG_ZERO_WIDTH_BIT_FIELD_ALIGNMENT", -1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TEMPLATE_INFO_FILE_SUFFIX");
        fprintf(qword_4F07510, "#define %s %td\n", "TEMPLATE_STATIC_DATA_MEMBER_INIT_GUARD_CODE", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TEMPORARILY_EXTEND_USE_OF_SSI_CP_GEN_BE");
        fprintf(qword_4F07510, "#define %s %td\n", "THREAD_LOCAL_STORAGE_SPECIFIER_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TIE_DEFAULT_GNU_ABI_VERSION_TO_GNU_VERSION", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "TRACK_INTERPRETER_ALLOCATIONS", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "TYPE_FOR_AN_FP_VALUE_PART", "unsigned int");
        fprintf(qword_4F07510, "#define %s %s\n", "TYPE_FOR_AN_INTEGER_VALUE", "unsigned long long");
        fprintf(qword_4F07510, "/*      %s not defined */\n", "TYPE_FOR_A_FIXED_POINT_VALUE");
        fprintf(qword_4F07510, "#define %s %s\n", "TYPE_FOR_A_SIGNED_INTEGER_VALUE", "long long");
        fprintf(qword_4F07510, "#define %s %s\n", "TYPE_FOR_PREFIX_ENTRY_NUMBER", "unsigned int");
        fprintf(qword_4F07510, "#define %s %s\n", "TYPE_FOR_TARG_ALIGNMENT", "unsigned int");
        fprintf(qword_4F07510, "#define %s %s\n", "UCN_ESCAPE_REWRITE_CHAR", "'_'");
        fprintf(qword_4F07510, "#define %s %td\n", "UNICODE_SOURCE_SUPPORTED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "UNICODE_VULNERABILITY_DETECTION_SUPPORTED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "UNIQUE_FILE_IDENTIFIER_AVAILABLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "UPC_EXTENSIONS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_BOOL_FOR_BOOLEAN_IN_CPLUSPLUS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_CCTOR_TO_PASS_CLASS_TO_ELLIPSIS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_DOUBLE_FOR_HOST_FP_VALUE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_EMPTY_STRUCT_IN_GENERATED_C", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_ENUMS_IN_BITFIELDS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_FIXED_ADDRESS_FOR_MMAP", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_FLOAT128_FOR_HOST_FP_VALUE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_HEX_FP_CONSTANTS_IN_GENERATED_CODE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_HOST_FP_CONVERSION_ROUTINES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_INIT_SECTION_IN_GENERATED_C", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_LAZY_INITIALIZATION_FOR_THREAD_LOCAL_VARIABLES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_LONG_DOUBLE_FOR_HOST_FP_VALUE", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_MMAP_FOR_MEMORY_REGIONS", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_MMAP_FOR_MODULES", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_OWN_SJIS_MULTIBYTE_CHAR_PROCESSING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_PATCH_INIT_STARTUP", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_QUADMATH_LIBRARY", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_SOFTFLOAT", 1);
        fprintf(qword_4F07510, "/*      %s not defined */\n", "USE_TEMPLATE_INFO_FILE");
        fprintf(qword_4F07510, "#define %s %td\n", "USE_VIRTUAL_FUNCTIONS", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USE_X86_FUNCTION_MULTIVERSIONING", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USING_DECLARATIONS_IN_GENERATED_CODE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "USING_DRIVER", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "USING_KAI_INLINER", 0);
        fprintf(qword_4F07510, "#define %s %s\n", "VERSION_NUMBER", "\"6.6\"");
        fprintf(qword_4F07510, "#define %s %td\n", "VERSION_NUMBER_FOR_MACRO", 606);
        fprintf(qword_4F07510, "#define %s %td\n", "VLA_ALLOWED", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "VLA_DEALLOCATIONS_IN_IL", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "VLA_DEALLOCATION_REQUIRED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "WCHAR_T_ENABLING_POSSIBLE", 1);
        fprintf(qword_4F07510, "#define %s %td\n", "WINDOWS_PATHS_ALLOWED", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "WRITE_CPPCLI_PORTABLE_ASSEMBLIES", 0);
        fprintf(qword_4F07510, "#define %s %td\n", "WRITE_SIGNOFF_MESSAGE", 1);
        sub_88CDC0();
        v150 = 1;
        continue;
      case 0xE2u:
        sub_88BDE0(src);
        v150 = 1;
        continue;
      case 0xE3u:
        unk_4F06AA8 = 0;
        continue;
      case 0xE4u:
        unk_4F06AA8 = 1;
        continue;
      case 0xE5u:
        unk_4D0420C = v4;
        continue;
      case 0xE6u:
        if ( !strcmp(src, "UTF-8") )
        {
          unk_4F076FC = 1;
        }
        else if ( !strcmp(src, "UTF-16") )
        {
          unk_4F076FC = (unk_4F07580 == 0) + 2;
        }
        else if ( !strcmp(src, "UTF-16LE") )
        {
          unk_4F076FC = 2;
        }
        else if ( !strcmp(src, "UTF-16BE") )
        {
          unk_4F076FC = 3;
        }
        else
        {
          if ( strcmp(src, "none") )
            sub_684920(1672);
          unk_4F076FC = 0;
        }
        continue;
      case 0xE7u:
        unk_4D0448C = v4;
        continue;
      case 0xE8u:
        unk_4D04474 = v4;
        continue;
      case 0xE9u:
        unk_4D04470 = v4;
        continue;
      case 0xEAu:
        dword_4D0446C = v4;
        continue;
      case 0xEBu:
        unk_4F0775C = v4;
        continue;
      case 0xECu:
        unk_4F07758 = v4;
        continue;
      case 0xEDu:
        unk_4D047C4 = v4;
        continue;
      case 0xEEu:
        unk_4F0773C = v4;
        continue;
      case 0xEFu:
        unk_4D04374 = 0;
        unk_4D04370 = 1;
        continue;
      case 0xF0u:
        unk_4D044BC = v4;
        continue;
      case 0xF1u:
        unk_4F072F1 = v7;
        continue;
      case 0xF2u:
        unk_4F04D9C = 1;
        continue;
      case 0xF3u:
        unk_4F07738 = v4;
        continue;
      case 0xF4u:
        unk_4F07734 = v4;
        continue;
      case 0xF5u:
        unk_4F07730 = v4;
        continue;
      case 0xF6u:
        unk_4D04408 = v4;
        continue;
      case 0xF7u:
        unk_4F07778 = 199711;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0xF8u:
        dword_4D04360 = v4;
        continue;
      case 0xF9u:
        dword_4D048B0 = v4;
        continue;
      case 0xFAu:
        dword_4D04434 = v4;
        continue;
      case 0xFBu:
        unk_4D042E8 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0xFCu:
        unk_4D042E0 = sub_60F910((unsigned __int8 *)src);
        continue;
      case 0xFDu:
        unk_4D048A0 = v4;
        continue;
      case 0xFEu:
        unk_4D04200 = v4;
        continue;
      case 0xFFu:
        unk_4F0771C = v4;
        continue;
      case 0x100u:
        unk_4F07718 = v4;
        continue;
      case 0x101u:
        unk_4D041FC = v4;
        continue;
      case 0x102u:
        HIDWORD(qword_4D043AC) = v4;
        continue;
      case 0x103u:
        unk_4F07778 = 201402;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0x104u:
        unk_4F07778 = 201112;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0x105u:
        unk_4F07778 = 201710;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0x106u:
        unk_4F07778 = 202311;
        if ( dword_4CF8024 && dword_4F077C4 == 2 )
          goto LABEL_321;
        dword_4F077C4 = 0;
        dword_4CF8024 = 1;
        continue;
      case 0x107u:
        unk_4F0770C = v4;
        continue;
      case 0x108u:
        unk_4F06BA8 = sub_88CD00(src);
        if ( unk_4F06BA8 == -1 )
          sub_684920(2664);
        sub_88CD10();
        continue;
      case 0x109u:
        unk_4F07778 = 201703;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0x10Au:
        unk_4D041F0 = v4;
        continue;
      case 0x10Bu:
        unk_4D047A8 = 1;
        continue;
      case 0x10Cu:
        unk_4F06978 = v4;
        continue;
      case 0x10Du:
        unk_4D04818 = v4;
        continue;
      case 0x10Eu:
        unk_4F07778 = 202002;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
          goto LABEL_321;
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0x10Fu:
        unk_4F07778 = 202302;
        if ( dword_4CF8024 && dword_4F077C4 != 2 )
LABEL_321:
          sub_6849E0(1027);
        dword_4F077C4 = 2;
        dword_4CF8024 = 1;
        continue;
      case 0x111u:
        unk_4D041B4 = v4;
        continue;
      case 0x112u:
        dword_4D041AC = v4;
        continue;
      case 0x113u:
        v48 = sub_721290(src);
        sub_720930(v48, 0, &unk_4F07698, &unk_4F07690);
        continue;
      case 0x114u:
        sub_6145B0(src, (unsigned __int64 *)&s2, &v158);
        if ( v158 )
        {
          v49 = s2;
          v159[0] = sub_721290(v158);
          v50 = qword_4D048F0;
          v51 = sub_887620(v49);
          v137 = a1;
          v52 = v49;
          v53 = *(_QWORD *)v50;
          LODWORD(v50) = *(_DWORD *)(v50 + 8);
          v54 = v50 & v51;
          v55 = v50;
          v56 = v53;
          v140 = v49 == 0;
          for ( i = v54; ; i = v55 & (i + 1) )
          {
            v59 = v56 + 16LL * i;
            v60 = *(const char **)v59;
            if ( !*(_QWORD *)v59 || v140 )
            {
              if ( v52 == v60 )
              {
LABEL_661:
                v61 = (__int64)v52;
                a1 = v137;
                if ( *(_QWORD *)(v59 + 8) )
                  sub_684920(3141);
                goto LABEL_298;
              }
              if ( !v60 )
              {
                v61 = (__int64)v52;
                a1 = v137;
LABEL_298:
                sub_617650(qword_4D048F0, v61, v159);
                goto LABEL_2;
              }
            }
            else
            {
              nb = v55;
              v58 = strcmp(v60, v52);
              v55 = nb;
              if ( !v58 )
                goto LABEL_661;
            }
          }
        }
        na = sub_721290(s2);
        v125 = qword_4D048E8;
        v126 = qword_4D048E8[2];
        if ( v126 == qword_4D048E8[1] )
        {
          if ( v126 <= 1 )
          {
            v130 = 16;
            v129 = 2;
          }
          else
          {
            v129 = v126 + (v126 >> 1) + 1;
            v130 = 8 * v129;
          }
          v139 = qword_4D048E8[1];
          v143 = (_QWORD *)*qword_4D048E8;
          v131 = sub_822B10(v130);
          v133 = v143;
          if ( v126 > 0 )
          {
            v132 = (_QWORD *)v131;
            v134 = v143;
            do
            {
              if ( v132 )
                *v132 = *v134;
              ++v132;
              ++v134;
            }
            while ( (_QWORD *)(v131 + 8 * v126) != v132 );
          }
          v144 = v131;
          sub_822B90(v133, 8 * v139, v132, v139);
          v127 = v144;
          v125[1] = v129;
          *v125 = v144;
        }
        else
        {
          v127 = *qword_4D048E8;
        }
        v128 = (size_t *)(v127 + 8 * v126);
        if ( v128 )
          *v128 = na;
        v125[2] = v126 + 1;
        continue;
      case 0x115u:
        s2 = 0;
        sub_6145B0(src, (unsigned __int64 *)&s2, &v158);
        s2 = (char *)sub_721290(s2);
        v65 = s2;
        if ( !v158 )
          goto LABEL_756;
        v66 = (__int64 *)unk_4D048E0;
        v67 = sub_722DF0(s2);
        v68 = sub_887620(v67);
        v69 = *((_DWORD *)v66 + 2);
        v70 = v65;
        v138 = a1;
        v14 = v65 == 0;
        v71 = *v66;
        v72 = v69 & v68;
        v73 = v14;
        while ( 2 )
        {
          v75 = v71 + 16LL * v72;
          v76 = *(char **)v75;
          if ( !*(_QWORD *)v75 || v73 )
          {
            if ( v70 != v76 )
            {
LABEL_377:
              if ( *(_QWORD *)v75 )
              {
                v72 = v69 & (v72 + 1);
                continue;
              }
              a1 = v138;
              goto LABEL_382;
            }
          }
          else
          {
            v141 = v69;
            v74 = sub_722E50(v76, v70);
            v69 = v141;
            if ( v74 )
              goto LABEL_377;
          }
          break;
        }
        a1 = v138;
        if ( *(_QWORD *)(v75 + 8) )
          goto LABEL_768;
LABEL_382:
        v77 = unk_4D048E0;
        v78 = sub_721290(v158);
        v79 = s2;
        v80 = v78;
        v81 = sub_722DF0(s2);
        v82 = sub_887620(v81);
        v83 = *(_DWORD *)(v77 + 8);
        v84 = v83 & v82;
        v85 = (__int64 *)(*(_QWORD *)v77 + 16 * v84);
        if ( *v85 )
        {
          v117 = v83 & v82;
          do
          {
            v117 = v83 & (v117 + 1);
            v118 = (__int64 *)(*(_QWORD *)v77 + 16LL * v117);
          }
          while ( *v118 );
          v159[0] = 0;
          v119 = *v85;
          *v118 = *v85;
          if ( v119 )
            v118[1] = v85[1];
          v120 = v159[0];
          *v85 = v159[0];
          if ( v120 )
            v85[1] = 0;
        }
        sub_615770((__int64 *)v77, v84, (__int64)v79, v80);
        continue;
      case 0x116u:
      case 0x117u:
        if ( v6 == 279 )
          v135 = qword_4D048D0;
        else
          v135 = unk_4D048D8;
        s2 = 0;
        sub_6145B0(src, (unsigned __int64 *)&s2, &v158);
        v27 = (char *)sub_721290(s2);
        s2 = v27;
        v28 = v27;
        if ( !v158 )
LABEL_756:
          sub_684920(3143);
        v29 = sub_722DF0(v27);
        v30 = sub_887620(v29);
        v31 = v28;
        v32 = *(_DWORD *)(v135 + 8);
        v14 = v28 == 0;
        v33 = *(_QWORD *)v135;
        v34 = v32 & v30;
        v35 = v14;
        while ( 2 )
        {
          v38 = v33 + 16LL * v34;
          v39 = *(char **)v38;
          if ( !*(_QWORD *)v38 || v35 )
          {
            v37 = v31 == v39;
          }
          else
          {
            v136 = v32;
            v36 = sub_722E50(v39, v31);
            v35 = 0;
            v32 = v136;
            v37 = v36 == 0;
          }
          if ( v37 )
          {
            if ( *(_QWORD *)(v38 + 8) )
LABEL_768:
              sub_684920(3142);
          }
          else if ( *(_QWORD *)v38 )
          {
            v34 = v32 & (v34 + 1);
            continue;
          }
          break;
        }
        v105 = sub_721290(v158);
        v106 = s2;
        v107 = v105;
        v108 = sub_722DF0(s2);
        v109 = sub_887620(v108);
        v110 = *(_DWORD *)(v135 + 8);
        v111 = v110 & v109;
        v112 = (__int64 *)(*(_QWORD *)v135 + 16LL * (v110 & v109));
        if ( *v112 )
        {
          v113 = v110 & v109;
          do
          {
            v113 = v110 & (v113 + 1);
            v114 = (__int64 *)(*(_QWORD *)v135 + 16LL * v113);
          }
          while ( *v114 );
          v159[0] = 0;
          v115 = *v112;
          *v114 = *v112;
          if ( v115 )
            v114[1] = v112[1];
          v116 = v159[0];
          *v112 = v159[0];
          if ( v116 )
            v112[1] = 0;
        }
        sub_615770((__int64 *)v135, v111, (__int64)v106, v107);
        continue;
      case 0x118u:
        unk_4D04450 = v4;
        continue;
      case 0x119u:
        unk_4D0444C = v4 == 0;
        continue;
      case 0x11Au:
        unk_4D04448 = v4;
        continue;
      case 0x11Bu:
        unk_4D0445C = v4;
        if ( v4 )
        {
          if ( !byte_4CF8128 )
            dword_4D04278 = 0;
          unk_4D04274 = 0;
        }
        continue;
      case 0x11Cu:
        unk_4D04454 = v4;
        continue;
      case 0x11Du:
        unk_4D04494 = v4;
        continue;
      case 0x11Eu:
        unk_4F073C8 = v4;
        continue;
      case 0x11Fu:
        LODWORD(qword_4D045BC) = 1;
        continue;
      case 0x120u:
        unk_4F06908 = v4;
        continue;
      case 0x121u:
        unk_4D041A0 = v4;
        continue;
      case 0x122u:
        dword_4D0419C = v4;
        continue;
      case 0x123u:
        unk_4F07518 = v4;
        continue;
      case 0x124u:
        if ( dword_4CF8188 <= 0 )
        {
          v150 = 1;
        }
        else
        {
          v62 = (const char **)&unk_4CF81A8;
          v63 = 0;
          do
          {
            if ( *v62 )
              printf("--%s ", *v62);
            ++v63;
            v62 += 5;
          }
          while ( dword_4CF8188 > v63 );
          v150 = 1;
        }
        continue;
      case 0x125u:
        if ( !strcmp(src, "text") )
        {
          unk_4D04198 = 0;
        }
        else
        {
          if ( strcmp(src, "sarif") )
            sub_684920(3299);
          unk_4D04198 = 1;
        }
        continue;
      case 0x126u:
        unk_4D04194 = v4;
        continue;
      default:
        goto LABEL_6;
    }
  }
  if ( !unk_4D04720 )
    unk_4D04720 = unk_4D0455C != 0;
  if ( unk_4D04258 )
    unk_4D0425C = 1;
  sub_6158E0();
  if ( dword_4F077C4 != 2 )
  {
    sub_614820();
    if ( !dword_4D04964 )
      goto LABEL_15;
    goto LABEL_108;
  }
  if ( qword_4D0495C )
    sub_60D3A0();
  sub_615370();
  if ( dword_4D04964 )
LABEL_108:
    sub_60F990();
LABEL_15:
  if ( unk_4D04794 )
    unk_4D04798 = 1;
  if ( unk_4D04780 )
    unk_4D04788 = 1;
  v8 = dword_4D04278;
  if ( unk_4D047C8 )
  {
    if ( unk_4F0697C )
    {
      if ( !unk_4D0445C )
      {
        if ( !dword_4D04278 )
          goto LABEL_31;
        goto LABEL_28;
      }
      goto LABEL_24;
    }
    if ( dword_4D04278 )
    {
      if ( byte_4CF8128 )
      {
        if ( !unk_4D0445C )
          goto LABEL_664;
        goto LABEL_24;
      }
      dword_4D04278 = 0;
    }
    goto LABEL_113;
  }
  if ( !dword_4D04278 )
  {
    if ( unk_4F0697C )
      goto LABEL_163;
LABEL_113:
    if ( !unk_4D0445C )
      goto LABEL_31;
    v8 = dword_4D04278;
    goto LABEL_24;
  }
  if ( byte_4CF8128 )
  {
    if ( unk_4F0697C )
    {
      if ( !unk_4D0445C )
      {
        if ( !byte_4CF8123 )
        {
LABEL_28:
          if ( byte_4CF8071 && unk_4D0472C )
            sub_6849E0(1060);
          unk_4D0472C = 0;
          unk_4D047C8 = 1;
          if ( unk_4D04458 )
            sub_6849E0(3062);
          goto LABEL_31;
        }
        goto LABEL_678;
      }
    }
    else if ( !unk_4D0445C )
    {
      goto LABEL_25;
    }
LABEL_24:
    unk_4D04458 = 1;
    if ( !v8 )
      goto LABEL_31;
LABEL_25:
    if ( !byte_4CF8123 || unk_4D047C8 )
    {
      if ( unk_4F0697C )
        goto LABEL_28;
LABEL_664:
      sub_6849E0(1281);
    }
LABEL_678:
    sub_6849E0(1059);
  }
  dword_4D04278 = 0;
  if ( !unk_4F0697C )
    goto LABEL_113;
LABEL_163:
  if ( unk_4D0445C )
    unk_4D04458 = 1;
LABEL_31:
  if ( unk_4D04958 )
  {
    dword_4D04278 = 0;
    sub_67D850(1073, 5, 1);
  }
  if ( dword_4D047B0 )
  {
    if ( !unk_4D047C8 )
      goto LABEL_37;
  }
  else
  {
    if ( !unk_4D047C8 )
      goto LABEL_37;
    if ( !byte_4CF8123 )
    {
      unk_4D047C8 = 0;
      goto LABEL_37;
    }
    if ( byte_4CF8125 )
      sub_6849E0(1026);
  }
  dword_4D047B0 = 1;
  unk_4D047D8 = 1;
LABEL_37:
  if ( unk_4D0448C )
  {
    unk_4D04440 = 1;
    unk_4D04438 = 1;
  }
  if ( unk_4D04408 | unk_4F07754 | unk_4D04430 && !word_4CF8154 )
  {
    unk_4F07734 = 1;
    unk_4F07730 = 0;
  }
  if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 && !byte_4CF8153 )
    unk_4F07738 = 1;
  if ( dword_4D04434 )
    dword_4D04464 = 1;
  if ( unk_4D044B4 && unk_4F07734 | unk_4D0448C )
    sub_6849E0(1908);
  if ( unk_4F0775C )
  {
    if ( byte_4CF814B && !byte_4CF814C )
      unk_4F07758 = 0;
  }
  else if ( !unk_4F07758 )
  {
    if ( byte_4CF814B )
    {
      if ( byte_4CF814C )
        sub_6849E0(1786);
      unk_4F07758 = 1;
    }
    else
    {
      unk_4F0775C = 1;
    }
  }
  if ( unk_4D041F0 && !unk_4D043A8 )
  {
    if ( byte_4CF813D )
    {
      if ( byte_4CF816A )
        sub_6849E0(3140);
      unk_4D041F0 = 0;
    }
    else
    {
      unk_4D043A8 = 1;
    }
  }
  if ( dword_4F077C0 )
  {
    sub_60E530();
  }
  else if ( dword_4F077BC )
  {
    sub_60E7C0();
  }
  else
  {
    if ( unk_4F072F0 )
      sub_6849E0(1164);
    if ( unk_4D04320 )
    {
      if ( byte_4CF812D )
        sub_6849E0(1603);
      unk_4D04320 = 0;
    }
    if ( unk_4D044BC && byte_4CF8150 )
      sub_6849E0(1900);
  }
  if ( !(unk_4D045E8 | unk_4D04630) )
    sub_6849E0(3460);
  unk_4D04524 = unk_4D041DC;
  if ( unk_4D04528 )
    unk_4D041DC = 1;
  if ( unk_4D043A8 && dword_4F077C4 == 2 )
    unk_4D043A0 = 1;
  if ( dword_4D047B0 && !byte_4CF8100 )
    unk_4D0482C = 0;
  unk_4D044CC = (unk_4D044C0 | unk_4D044C8) != 0;
  sub_610000();
  if ( dword_4D0488C && !unk_4D0439C )
  {
    if ( byte_4CF80F9 )
      sub_6849E0(2682);
    unk_4D0439C = 1;
  }
  if ( !dword_4D048B4 && dword_4D048B0 )
  {
    if ( byte_4CF8159 )
      sub_6849E0(2805);
    dword_4D048B0 = 0;
  }
  if ( unk_4D042AC )
    unk_4F068F8 = 0;
  v9 = 1;
  if ( dword_4F077C4 == 2 )
    v9 = qword_4D0495C != 0;
  unk_4D04000 = v9;
  if ( unk_4D04950 )
  {
    unk_4F07470 = 5;
    unk_4D0435C = 1;
  }
  else
  {
    unk_4F07470 = 8;
  }
  if ( word_4D04898 && !unk_4D042E0 )
    unk_4D042E0 = 2000000;
  if ( unk_4D04350 )
    unk_4D04348 = 0;
  v10 = 1;
  if ( dword_4F077C4 != 1 )
    v10 = dword_4CF8040 != 0;
  unk_4D04954 = v10;
  if ( !byte_4CF8152 )
    unk_4F04D9C = v10;
  if ( !unk_4D04358 )
  {
    if ( !unk_4D04948 || (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) && !dword_4D048B8 )
      goto LABEL_99;
LABEL_118:
    unk_4D04948 = 0;
    goto LABEL_99;
  }
  unk_4D0428C = 1;
  unk_4D04288 = 1;
  dword_4D04284 = 1;
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201401 )
    sub_6849E0(2655);
  if ( unk_4D04948 )
    goto LABEL_118;
LABEL_99:
  sub_720A60(&qword_4F076A8, &qword_4F076A0);
  if ( HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x765Bu )
  {
    sub_722C80(&v156, 1);
    sub_722C80(&v156, 0);
  }
  if ( v156 )
    v11 = *(_QWORD *)(v156 + 16);
  else
    v11 = qword_4F076A8;
  unk_4F07688 = v11;
  v12 = dword_4CF8030;
  if ( dword_4CF8030 >= a1 )
  {
    if ( v150 )
      sub_720FF0(3);
    sub_6849E0(593);
  }
  ++dword_4CF8030;
  v13 = *(char **)(a2 + 8 * v12);
  v14 = *v13 == 45;
  src = v13;
  if ( v14 && !v13[1] )
    src = "-";
  v15 = (char *)((__int64 (*)(void))sub_721290)();
  qword_4F076F0 = v15;
  if ( unk_4F07680 )
  {
    unk_4F076E8 = sub_722430(v15);
    sub_7209D0(unk_4F076E8, &qword_4F076A8, &qword_4F076A0);
  }
  if ( dword_4CF8030 < a1 )
    sub_6849E0(595);
  if ( unk_4D04944 )
  {
    unk_4D04920 = v145;
    unk_4D04940 = 1;
    if ( unk_4D04910 )
    {
      unk_4D0493C = 0;
      if ( v149 )
      {
        unk_4D04944 = 0;
LABEL_130:
        unk_4F07570 = 1;
        goto LABEL_131;
      }
      unk_4D048FC = 1;
      unk_4D048F8 = 1;
LABEL_604:
      if ( unk_4F07481 == 5 )
        unk_4F07481 = 7;
      goto LABEL_130;
    }
    if ( !v149 )
    {
      unk_4D048FC = 1;
      unk_4D048F8 = 1;
      if ( !qword_4D04914 )
        goto LABEL_130;
      goto LABEL_604;
    }
    unk_4D04944 = 0;
    unk_4F07570 = 1;
  }
  else
  {
    if ( unk_4D04910 )
      unk_4D0493C = 0;
    unk_4F07570 = 1;
    if ( v145 )
      sub_6849E0(596);
  }
LABEL_131:
  if ( v146 )
    unk_4D04908 = sub_685E40(v146, 0, 0, 16, 1514);
  if ( v147 )
    unk_4D04900 = sub_685E40(v147, 0, 0, 16, 1515);
  if ( v148 )
    qword_4F07510 = (FILE *)sub_685E40(v148, 0, 0, 16, 1498);
  if ( byte_4CF8081 )
  {
    unk_4F073C8 = 0;
    unk_4F073CC = 0;
  }
  else if ( unk_4F073C8 )
  {
    sub_67C750();
  }
  sub_822260(1);
  sub_822260(2);
  sub_822260(3);
  sub_822260(4);
  sub_822260(5);
  sub_822260(6);
  sub_822260(7);
  sub_822260(8);
  sub_822260(10);
  sub_822260(9);
  sub_822260(11);
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( byte_4CF8131 )
    {
      v16 = unk_4F077B0;
    }
    else if ( (unsigned __int8)byte_4CF816F
            | (unsigned __int8)(byte_4CF816E
                              | byte_4CF8169
                              | byte_4CF8163
                              | byte_4CF813F
                              | byte_4CF8157
                              | byte_4CF8166
                              | byte_4CF8165
                              | byte_4CF8164
                              | byte_4CF8126
                              | byte_4CF8127) )
    {
      unk_4F077B0 = 1;
      v16 = 1;
    }
    else
    {
      unk_4F077B0 = 0;
      v16 = 0;
    }
    if ( !(_DWORD)qword_4F077B4
      && dword_4F077C4 == 2
      && (!dword_4F077BC || qword_4F077A8 > 0x9F5Fu)
      && v16
      && unk_4F07718 )
    {
      goto LABEL_145;
    }
    unk_4D041A4 = 1;
  }
  else
  {
    v16 = unk_4F077B0;
  }
  if ( dword_4F077C0 && !((unsigned int)qword_4F077B4 | v16) && qword_4F077A8 > 0xC34Fu )
    unk_4F07710 = 1;
LABEL_145:
  result = &qword_4D046F0;
  if ( qword_4D046F0 )
  {
    result = (__int64 *)&dword_4D046EC;
    dword_4D046EC = 1;
  }
  return result;
}
