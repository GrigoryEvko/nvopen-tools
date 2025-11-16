// Function: sub_221D830
// Address: 0x221d830
//
char *__fastcall sub_221D830(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        char *a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        char *s)
{
  int v9; // r15d
  unsigned __int64 v10; // r13
  unsigned int v11; // r12d
  size_t v14; // rax
  __int64 v15; // r8
  char *v16; // r14
  char *v17; // r10
  char *v18; // r12
  size_t i; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  size_t v22; // r9
  char v23; // al
  __int64 v24; // rsi
  __int64 (__fastcall *v25)(__int64, unsigned int); // rax
  int v26; // eax
  unsigned __int8 v27; // al
  __int64 (__fastcall *v28)(__int64, unsigned int); // rax
  int v29; // eax
  unsigned int v31; // eax
  char v32; // al
  __int64 (__fastcall *v33)(__int64, unsigned int); // rax
  _BYTE *v34; // rax
  __int64 (__fastcall *v35)(__int64, unsigned int); // rax
  unsigned __int64 v36; // r13
  unsigned int v37; // edx
  int v38; // eax
  _QWORD *v39; // rax
  unsigned __int64 v40; // r13
  unsigned int v41; // edx
  _QWORD *v42; // rax
  unsigned __int64 v43; // r13
  unsigned int v44; // edx
  char v45; // al
  _BYTE *(__fastcall *v46)(__int64, _BYTE *, _BYTE *, void *); // rax
  unsigned __int64 v47; // r13
  unsigned int v48; // edx
  char v49; // al
  void (*v50)(char *, char *, const char *, ...); // rax
  unsigned __int8 *v51; // rax
  char v52; // al
  __int64 (__fastcall *v53)(__int64, unsigned int); // rax
  unsigned __int64 v54; // r13
  unsigned int v55; // edx
  _QWORD *v56; // rax
  unsigned __int64 v57; // r13
  unsigned int v58; // edx
  unsigned __int64 v59; // r13
  unsigned int v60; // edx
  unsigned __int64 v61; // r13
  unsigned int v62; // edx
  unsigned __int64 v63; // r13
  unsigned int v64; // edx
  unsigned __int8 *v65; // rax
  __int64 v66; // rdx
  int v67; // eax
  unsigned __int64 v68; // r13
  unsigned int v69; // edx
  _BYTE *v70; // rax
  unsigned __int64 v71; // r13
  char *v72; // rax
  char *v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // rsi
  char v76; // r11
  unsigned __int8 v77; // cl
  char v78; // cl
  char v79; // al
  _QWORD *v80; // rsi
  unsigned int v81; // edx
  unsigned __int64 v82; // r13
  __int64 v83; // rdx
  _QWORD *v84; // rax
  char v85; // al
  _BYTE *(__fastcall *v86)(__int64, char *, const char *, void *); // rax
  unsigned int v87; // eax
  int v88; // eax
  unsigned int v89; // eax
  unsigned __int8 *v90; // rax
  unsigned __int64 v91; // r13
  __int64 v92; // rdx
  unsigned int v93; // eax
  unsigned int v94; // eax
  int v95; // eax
  int v96; // eax
  bool v97; // zf
  __int64 (__fastcall *v98)(__int64, unsigned int); // rax
  char v99; // al
  unsigned int v100; // eax
  unsigned int v101; // eax
  __int64 v102; // rcx
  __int64 (__fastcall *v103)(__int64, unsigned int); // rdx
  unsigned __int8 *v104; // rax
  unsigned __int8 *v105; // rax
  int v106; // eax
  int v107; // eax
  size_t v108; // [rsp+0h] [rbp-268h]
  unsigned __int64 v109; // [rsp+8h] [rbp-260h]
  char *v112; // [rsp+28h] [rbp-240h]
  unsigned __int8 v113; // [rsp+28h] [rbp-240h]
  char *v114; // [rsp+28h] [rbp-240h]
  __int64 v115; // [rsp+28h] [rbp-240h]
  unsigned int v116; // [rsp+28h] [rbp-240h]
  int v117; // [rsp+28h] [rbp-240h]
  int v118; // [rsp+28h] [rbp-240h]
  int v119; // [rsp+28h] [rbp-240h]
  unsigned int v120; // [rsp+28h] [rbp-240h]
  unsigned int v121; // [rsp+28h] [rbp-240h]
  unsigned int v122; // [rsp+28h] [rbp-240h]
  int v123; // [rsp+28h] [rbp-240h]
  char *v124; // [rsp+28h] [rbp-240h]
  unsigned int v125; // [rsp+28h] [rbp-240h]
  unsigned int v126; // [rsp+28h] [rbp-240h]
  char *v127; // [rsp+28h] [rbp-240h]
  unsigned int v129; // [rsp+30h] [rbp-238h]
  int v130[2]; // [rsp+30h] [rbp-238h]
  int v131; // [rsp+30h] [rbp-238h]
  int v132; // [rsp+30h] [rbp-238h]
  int v133; // [rsp+30h] [rbp-238h]
  int v134[2]; // [rsp+30h] [rbp-238h]
  unsigned int v135; // [rsp+30h] [rbp-238h]
  unsigned int v136; // [rsp+30h] [rbp-238h]
  __int64 v137; // [rsp+38h] [rbp-230h]
  unsigned int v138; // [rsp+40h] [rbp-228h]
  unsigned __int64 v139; // [rsp+40h] [rbp-228h]
  unsigned int v140; // [rsp+40h] [rbp-228h]
  char v141; // [rsp+48h] [rbp-220h]
  unsigned __int8 v142; // [rsp+48h] [rbp-220h]
  unsigned int v143; // [rsp+48h] [rbp-220h]
  unsigned int v144; // [rsp+48h] [rbp-220h]
  unsigned __int8 v145; // [rsp+50h] [rbp-218h]
  char *v146; // [rsp+50h] [rbp-218h]
  unsigned int v147; // [rsp+50h] [rbp-218h]
  unsigned int v148; // [rsp+50h] [rbp-218h]
  char *v149; // [rsp+58h] [rbp-210h]
  unsigned int v150; // [rsp+60h] [rbp-208h]
  int v151; // [rsp+68h] [rbp-200h]
  unsigned int v152; // [rsp+6Ch] [rbp-1FCh]
  int v153; // [rsp+1C8h] [rbp-A0h] BYREF
  int v154; // [rsp+1CCh] [rbp-9Ch] BYREF
  _QWORD v155[2]; // [rsp+1D0h] [rbp-98h] BYREF
  __int64 v156; // [rsp+1E0h] [rbp-88h]
  __int64 v157; // [rsp+1E8h] [rbp-80h]
  __int64 v158; // [rsp+1F0h] [rbp-78h]
  __int64 v159; // [rsp+1F8h] [rbp-70h]
  __int64 v160; // [rsp+200h] [rbp-68h]
  __int64 v161; // [rsp+208h] [rbp-60h]
  __int64 v162; // [rsp+210h] [rbp-58h]
  __int64 v163; // [rsp+218h] [rbp-50h]
  __int64 v164; // [rsp+220h] [rbp-48h]
  __int64 v165; // [rsp+228h] [rbp-40h]
  char *si; // [rsp+280h] [rbp+18h]
  char *sg; // [rsp+280h] [rbp+18h]
  char *sl; // [rsp+280h] [rbp+18h]
  char *sa; // [rsp+280h] [rbp+18h]
  char *sm; // [rsp+280h] [rbp+18h]
  char *so; // [rsp+280h] [rbp+18h]
  char *sp; // [rsp+280h] [rbp+18h]
  char *sr; // [rsp+280h] [rbp+18h]
  char *sb; // [rsp+280h] [rbp+18h]
  char *sc; // [rsp+280h] [rbp+18h]
  char *st; // [rsp+280h] [rbp+18h]
  char *su; // [rsp+280h] [rbp+18h]
  char *sv; // [rsp+280h] [rbp+18h]
  char *sw; // [rsp+280h] [rbp+18h]
  char *sx; // [rsp+280h] [rbp+18h]
  char *sy; // [rsp+280h] [rbp+18h]
  char *sd; // [rsp+280h] [rbp+18h]
  char *sz; // [rsp+280h] [rbp+18h]
  char *sbb; // [rsp+280h] [rbp+18h]
  char *sh; // [rsp+280h] [rbp+18h]
  char *sbc; // [rsp+280h] [rbp+18h]
  char *sk; // [rsp+280h] [rbp+18h]
  char *sbd; // [rsp+280h] [rbp+18h]
  char *sbe; // [rsp+280h] [rbp+18h]
  char *sbf; // [rsp+280h] [rbp+18h]
  char *sn; // [rsp+280h] [rbp+18h]
  char *sq; // [rsp+280h] [rbp+18h]
  char *sba; // [rsp+280h] [rbp+18h]
  char *sj; // [rsp+280h] [rbp+18h]
  char *ss; // [rsp+280h] [rbp+18h]
  char *sbg; // [rsp+280h] [rbp+18h]
  char *sbh; // [rsp+280h] [rbp+18h]
  char *sbi; // [rsp+280h] [rbp+18h]
  char *se; // [rsp+280h] [rbp+18h]
  char *sbj; // [rsp+280h] [rbp+18h]
  char *sbk; // [rsp+280h] [rbp+18h]
  char *sf; // [rsp+280h] [rbp+18h]
  char *sbl; // [rsp+280h] [rbp+18h]
  char *sbm; // [rsp+280h] [rbp+18h]

  v10 = a6 + 208;
  v11 = a3;
  v109 = a3;
  v137 = sub_22311C0(a6 + 208);
  v112 = (char *)sub_222F790(v10);
  v14 = strlen(s);
  v15 = v11;
  v108 = v14;
  v16 = v112;
  v17 = s;
  v151 = a5;
  v18 = a4;
  v153 = 0;
  for ( i = 0; ; ++i )
  {
    LOBYTE(v10) = (_DWORD)v15 == -1;
    if ( ((unsigned __int8)v10 & (a2 != 0)) != 0 )
    {
      if ( *((_QWORD *)a2 + 2) >= *((_QWORD *)a2 + 3) )
      {
        sbc = v17;
        v135 = v15;
        v88 = (*(__int64 (__fastcall **)(char *))(*(_QWORD *)a2 + 72LL))(a2);
        v20 = (unsigned __int8)v10 & (a2 != 0);
        v15 = v135;
        v17 = sbc;
        if ( v88 == -1 )
          a2 = 0;
        else
          v20 = 0;
      }
      else
      {
        v20 = 0;
      }
    }
    else
    {
      v20 = (unsigned int)v10;
    }
    LOBYTE(v9) = v151 == -1;
    if ( v18 && v151 == -1 )
    {
      if ( *((_QWORD *)v18 + 2) >= *((_QWORD *)v18 + 3) )
      {
        si = v17;
        v129 = v15;
        v113 = v20;
        v29 = (*(__int64 (__fastcall **)(char *))(*(_QWORD *)v18 + 72LL))(v18);
        v20 = v113;
        v15 = v129;
        v17 = si;
        if ( v29 == -1 )
        {
          v20 = v113 ^ 1u;
          v18 = 0;
        }
      }
    }
    else
    {
      v20 = v9 ^ (unsigned int)v20;
    }
    if ( i >= v108 || !(_BYTE)v20 )
      break;
    if ( v153 )
      goto LABEL_38;
    v21 = (__int64)&v17[i];
    v22 = (unsigned __int8)v17[i];
    v23 = v16[v22 + 313];
    v24 = v22;
    if ( !v23 )
    {
      v25 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 64LL);
      if ( v25 != sub_2216C50 )
      {
        v138 = v15;
        sg = v17;
        *(_QWORD *)v130 = (unsigned __int8)v17[i];
        v114 = &v17[i];
        v31 = ((__int64 (__fastcall *)(char *, _QWORD, _QWORD))v25)(v16, (unsigned int)(char)v22, 0);
        v17 = sg;
        v15 = v138;
        v22 = *(_QWORD *)v130;
        v21 = (__int64)v114;
        v24 = v31;
      }
      if ( !(_BYTE)v24 )
      {
LABEL_16:
        v9 = *(unsigned __int8 *)v21;
        if ( !a2 || !(_BYTE)v10 )
        {
          LOBYTE(v26) = v15;
          goto LABEL_19;
        }
        v34 = (_BYTE *)*((_QWORD *)a2 + 2);
        if ( (unsigned __int64)v34 < *((_QWORD *)a2 + 3) )
        {
          if ( (_BYTE)v9 != *v34 )
            goto LABEL_20;
          goto LABEL_49;
        }
        sbf = v17;
        v116 = v15;
        v26 = (*(__int64 (__fastcall **)(char *, __int64, __int64, __int64, __int64, size_t))(*(_QWORD *)a2 + 72LL))(
                a2,
                v24,
                v20,
                v21,
                v15,
                v22);
        v20 = 0;
        v15 = v116;
        v17 = sbf;
        if ( v26 == -1 )
          a2 = 0;
LABEL_19:
        if ( (_BYTE)v9 != (_BYTE)v26 )
          goto LABEL_20;
LABEL_48:
        v34 = (_BYTE *)*((_QWORD *)a2 + 2);
        if ( (unsigned __int64)v34 >= *((_QWORD *)a2 + 3) )
        {
          sbd = v17;
          (*(void (__fastcall **)(char *, __int64, __int64, __int64, __int64, size_t))(*(_QWORD *)a2 + 80LL))(
            a2,
            v24,
            v20,
            v21,
            v15,
            v22);
          v17 = sbd;
        }
        else
        {
LABEL_49:
          *((_QWORD *)a2 + 2) = v34 + 1;
        }
        v15 = 0xFFFFFFFFLL;
        continue;
      }
      v16[v22 + 313] = v24;
      v23 = v24;
    }
    if ( v23 != 37 )
      goto LABEL_16;
    v21 = (unsigned __int8)v17[i + 1];
    v22 = i + 1;
    v27 = v16[v21 + 313];
    v24 = v21;
    if ( !v27 )
    {
      v28 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 64LL);
      if ( v28 != sub_2216C50 )
      {
        sh = v17;
        v140 = v15;
        *(_QWORD *)v134 = (unsigned __int8)v17[i + 1];
        v87 = ((__int64 (__fastcall *)(char *, _QWORD, _QWORD))v28)(v16, (unsigned int)(char)v21, 0);
        v17 = sh;
        v15 = v140;
        v21 = *(_QWORD *)v134;
        v22 = i + 1;
        v24 = v87;
      }
      if ( !(_BYTE)v24 )
      {
        i = v22;
LABEL_20:
        v153 |= 4u;
        continue;
      }
      v16[v21 + 313] = v24;
      v27 = v24;
    }
    v154 = 0;
    if ( v27 == 69 || v27 == 79 )
    {
      i += 2LL;
      v21 = (unsigned __int8)v17[i];
      v27 = v16[v21 + 313];
      v20 = v21;
      if ( !v27 )
      {
        v35 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 64LL);
        if ( v35 != sub_2216C50 )
        {
          sk = v17;
          v24 = (unsigned int)(char)v21;
          v136 = v15;
          v115 = (unsigned __int8)v17[i];
          v89 = ((__int64 (__fastcall *)(char *, __int64, _QWORD, __int64, __int64, size_t))v35)(
                  v16,
                  v24,
                  0,
                  v21,
                  v15,
                  v22);
          v17 = sk;
          v15 = v136;
          v21 = v115;
          v20 = v89;
        }
        if ( !(_BYTE)v20 )
          goto LABEL_20;
        v16[v21 + 313] = v20;
        v27 = v20;
      }
    }
    else
    {
      i = v22;
    }
    switch ( v27 )
    {
      case 'A':
        sc = v17;
        v56 = *(_QWORD **)(v137 + 16);
        v57 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        v155[0] = v56[11];
        v155[1] = v56[12];
        v156 = v56[13];
        v157 = v56[14];
        v158 = v56[15];
        v159 = v56[16];
        v160 = v56[17];
        a2 = sub_221C750(a1, a2, v57, v18, a5, &v154, (__int64)v155, 7, a6, &v153);
        goto LABEL_98;
      case 'B':
        sa = v17;
        v42 = *(_QWORD **)(v137 + 16);
        v40 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        v155[0] = v42[25];
        v155[1] = v42[26];
        v156 = v42[27];
        v157 = v42[28];
        v158 = v42[29];
        v159 = v42[30];
        v160 = v42[31];
        v161 = v42[32];
        v162 = v42[33];
        v163 = v42[34];
        v164 = v42[35];
        v165 = v42[36];
        a2 = sub_221C750(a1, a2, v40, v18, a5, &v154, (__int64)v155, 12, a6, &v153);
        goto LABEL_66;
      case 'C':
      case 'Y':
      case 'y':
        sl = v17;
        v36 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 0, 9999, 4u, a6, &v153);
        v10 = v37 | v36 & 0xFFFFFFFF00000000LL;
        v15 = v37;
        v109 = v10;
        v17 = sl;
        if ( !v153 )
        {
          v38 = v154 - 1900;
          if ( v154 < 0 )
            v38 = v154 + 100;
          a8[5] = v38;
        }
        continue;
      case 'D':
        v85 = v16[56];
        if ( v85 == 1 )
          goto LABEL_148;
        if ( !v85 )
        {
          sba = v17;
          v119 = v15;
          sub_2216D60((__int64)v16);
          v17 = sba;
          LODWORD(v15) = v119;
        }
        v86 = *(_BYTE *(__fastcall **)(__int64, char *, const char *, void *))(*(_QWORD *)v16 + 56LL);
        if ( (char *)v86 == (char *)sub_2216D40 )
        {
LABEL_148:
          strcpy((char *)v155, "%m/%d/%y");
        }
        else
        {
          v133 = v15;
          sbb = v17;
          v86((__int64)v16, &aHM[-9], "%H:%M", v155);
          v17 = sbb;
          LODWORD(v15) = v133;
        }
        goto LABEL_151;
      case 'H':
        st = v17;
        v59 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 0, 23, 2u, a6, &v153);
        v10 = v60 | v59 & 0xFFFFFFFF00000000LL;
        v15 = v60;
        v109 = v10;
        v17 = st;
        if ( !v153 )
          a8[2] = v154;
        continue;
      case 'I':
        su = v17;
        v61 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 1, 12, 2u, a6, &v153);
        v10 = v62 | v61 & 0xFFFFFFFF00000000LL;
        v15 = v62;
        v109 = v10;
        v17 = su;
        if ( !v153 )
          a8[2] = v154;
        continue;
      case 'M':
        sm = v17;
        v43 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 0, 59, 2u, a6, &v153);
        v10 = v44 | v43 & 0xFFFFFFFF00000000LL;
        v15 = v44;
        v109 = v10;
        v17 = sm;
        if ( !v153 )
          a8[1] = v154;
        continue;
      case 'R':
        v45 = v16[56];
        if ( v45 == 1 )
          goto LABEL_149;
        if ( !v45 )
        {
          sn = v17;
          v117 = v15;
          sub_2216D60((__int64)v16);
          v17 = sn;
          LODWORD(v15) = v117;
        }
        v46 = *(_BYTE *(__fastcall **)(__int64, _BYTE *, _BYTE *, void *))(*(_QWORD *)v16 + 56LL);
        if ( v46 == sub_2216D40 )
        {
LABEL_149:
          strcpy((char *)v155, "%H:%M");
        }
        else
        {
          v131 = v15;
          so = v17;
          v46((__int64)v16, &byte_4360B49[-6], byte_4360B49, v155);
          v17 = so;
          LODWORD(v15) = v131;
        }
        goto LABEL_151;
      case 'S':
        sp = v17;
        v47 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 0, 60, 2u, a6, &v153);
        v10 = v48 | v47 & 0xFFFFFFFF00000000LL;
        v15 = v48;
        v109 = v10;
        v17 = sp;
        if ( !v153 )
          *a8 = v154;
        continue;
      case 'T':
        v49 = v16[56];
        if ( v49 == 1 )
          goto LABEL_150;
        if ( !v49 )
        {
          sq = v17;
          v118 = v15;
          sub_2216D60((__int64)v16);
          v17 = sq;
          LODWORD(v15) = v118;
        }
        v50 = *(void (**)(char *, char *, const char *, ...))(*(_QWORD *)v16 + 56LL);
        if ( (char *)v50 == (char *)sub_2216D40 )
        {
LABEL_150:
          strcpy((char *)v155, "%H:%M:%S");
        }
        else
        {
          v132 = v15;
          sr = v17;
          v50(v16, &a9lu[-9], "%.9lu", v155);
          v17 = sr;
          LODWORD(v15) = v132;
        }
LABEL_151:
        sb = v17;
        v54 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221D830(a1, (int)a2, v15, (int)v18, a5, a6, (__int64)&v153, (__int64)a8, (char *)v155);
        goto LABEL_96;
      case 'X':
        sb = v17;
        v54 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221D830(
                       a1,
                       (int)a2,
                       v15,
                       (int)v18,
                       a5,
                       a6,
                       (__int64)&v153,
                       (__int64)a8,
                       *(char **)(*(_QWORD *)(v137 + 16) + 32LL));
        goto LABEL_96;
      case 'Z':
        if ( a2 && (_BYTE)v10 )
        {
          v70 = (_BYTE *)*((_QWORD *)a2 + 2);
          if ( (unsigned __int64)v70 >= *((_QWORD *)a2 + 3) )
          {
            sbh = v17;
            v122 = v15;
            LODWORD(v70) = (*(__int64 (__fastcall **)(char *, __int64, __int64))(*(_QWORD *)a2 + 72LL))(a2, v24, v20);
            v17 = sbh;
            v15 = v122;
            v97 = (_DWORD)v70 == -1;
            if ( (_DWORD)v70 == -1 )
              LOBYTE(v70) = -1;
            if ( v97 )
              a2 = 0;
          }
          else
          {
            LOBYTE(v70) = *v70;
          }
        }
        else
        {
          LOBYTE(v70) = v15;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v16 + 6) + 2LL * (unsigned __int8)v70 + 1) & 1) == 0 )
          goto LABEL_20;
        sd = v17;
        v71 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        v72 = sub_221C750(a1, a2, v71, v18, a5, v155, (__int64)off_4CDFB60, 14, a6, &v153);
        v74 = (unsigned int)v73;
        a2 = v72;
        v10 = (unsigned int)v73 | v71 & 0xFFFFFFFF00000000LL;
        v15 = (unsigned int)v73;
        v139 = v10;
        v109 = v10;
        LOBYTE(v10) = (_DWORD)v73 == -1;
        v17 = sd;
        v75 = (unsigned int)v10;
        v76 = v10 & (v72 != 0);
        if ( v76 )
        {
          v75 = 0;
          if ( *((_QWORD *)v72 + 2) >= *((_QWORD *)v72 + 3) )
          {
            v150 = (unsigned int)v73;
            v146 = v73;
            v142 = v10 & (v72 != 0);
            LOBYTE(v74) = v142;
            v96 = (*(__int64 (__fastcall **)(char *, _QWORD, char *, __int64, _QWORD))(*(_QWORD *)v72 + 72LL))(
                    v72,
                    0,
                    v73,
                    v74,
                    (unsigned int)v73);
            v75 = 0;
            v76 = v142;
            v73 = v146;
            v15 = v150;
            v17 = sd;
            if ( v96 == -1 )
            {
              v75 = v142;
              a2 = 0;
              v76 = 0;
            }
          }
        }
        v77 = v9 & (v18 != 0);
        if ( v77 )
        {
          v9 = 0;
          if ( *((_QWORD *)v18 + 2) >= *((_QWORD *)v18 + 3) )
          {
            sbg = v17;
            v152 = v15;
            v149 = v73;
            v145 = v77;
            v141 = v76;
            v95 = (*(__int64 (__fastcall **)(char *))(*(_QWORD *)v18 + 72LL))(v18);
            v17 = sbg;
            v15 = v152;
            if ( v95 == -1 )
            {
              v9 = v145;
              v18 = 0;
            }
            v73 = v149;
            v76 = v141;
            v75 = (unsigned __int8)v75;
          }
        }
        if ( (_BYTE)v75 != (_BYTE)v9 && !(LODWORD(v155[0]) | v153) )
        {
          v9 = (int)v73;
          if ( v76 )
          {
            v104 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
            if ( (unsigned __int64)v104 >= *((_QWORD *)a2 + 3) )
            {
              sbl = v17;
              v148 = v15;
              v127 = v73;
              v106 = (*(__int64 (__fastcall **)(char *))(*(_QWORD *)a2 + 72LL))(a2);
              v17 = sbl;
              v15 = v148;
              v9 = v106;
              if ( v106 == -1 )
                a2 = 0;
              v73 = v127;
            }
            else
            {
              v9 = *v104;
            }
          }
          if ( v16[56] )
          {
            v78 = v16[102];
          }
          else
          {
            se = v17;
            v147 = v15;
            v124 = v73;
            sub_2216D60((__int64)v16);
            v73 = v124;
            v78 = 45;
            v15 = v147;
            v98 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 48LL);
            v17 = se;
            if ( v98 != sub_CE72A0 )
            {
              v75 = 45;
              v99 = ((__int64 (__fastcall *)(char *, __int64, char *, __int64, _QWORD, _QWORD *))v98)(
                      v16,
                      45,
                      v124,
                      45,
                      v147,
                      v155);
              v17 = se;
              v15 = v147;
              v73 = v124;
              v78 = v99;
            }
          }
          if ( (_BYTE)v9 == v78 )
            goto LABEL_141;
          if ( a2 && (_BYTE)v10 )
          {
            v105 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
            if ( (unsigned __int64)v105 >= *((_QWORD *)a2 + 3) )
            {
              sbm = v17;
              v144 = v15;
              v107 = (*(__int64 (__fastcall **)(char *, __int64, char *))(*(_QWORD *)a2 + 72LL))(a2, v75, v73);
              v15 = v144;
              LODWORD(v73) = v107;
              v17 = sbm;
              if ( v107 == -1 )
                a2 = 0;
            }
            else
            {
              LODWORD(v73) = *v105;
            }
          }
          LODWORD(v10) = (_DWORD)v73;
          if ( v16[56] )
          {
            v79 = v16[100];
          }
          else
          {
            sf = v17;
            v143 = v15;
            sub_2216D60((__int64)v16);
            v15 = v143;
            v17 = sf;
            v103 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 48LL);
            v79 = 43;
            if ( v103 != sub_CE72A0 )
            {
              v79 = ((__int64 (__fastcall *)(char *, __int64, __int64 (__fastcall *)(__int64, unsigned int), __int64, _QWORD, _QWORD *))v103)(
                      v16,
                      43,
                      v103,
                      v102,
                      v143,
                      v155);
              v17 = sf;
              v15 = v143;
            }
          }
          if ( (_BYTE)v10 == v79 )
          {
LABEL_141:
            sz = v17;
            v9 = a1;
            v80 = sub_221C220(a1, a2, v139, v18, a5, (int *)v155, 0, 23, 2u, a6, &v153);
            v82 = v81 | v139 & 0xFFFFFFFF00000000LL;
            a2 = (char *)sub_221C220(a1, v80, v81, v18, a5, (int *)v155, 0, 59, 2u, a6, &v153);
            v15 = v83;
            v10 = (unsigned int)v83 | v82 & 0xFFFFFFFF00000000LL;
            v109 = v10;
            v17 = sz;
          }
        }
        continue;
      case 'a':
        sc = v17;
        v84 = *(_QWORD **)(v137 + 16);
        v57 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        v155[0] = v84[18];
        v155[1] = v84[19];
        v156 = v84[20];
        v157 = v84[21];
        v158 = v84[22];
        v159 = v84[23];
        v160 = v84[24];
        a2 = sub_221C750(a1, a2, v57, v18, a5, &v154, (__int64)v155, 7, a6, &v153);
LABEL_98:
        v15 = v58;
        v10 = v58 | v57 & 0xFFFFFFFF00000000LL;
        v109 = v10;
        v17 = sc;
        if ( !v153 )
          a8[6] = v154;
        continue;
      case 'b':
      case 'h':
        sa = v17;
        v39 = *(_QWORD **)(v137 + 16);
        v40 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        v155[0] = v39[37];
        v155[1] = v39[38];
        v156 = v39[39];
        v157 = v39[40];
        v158 = v39[41];
        v159 = v39[42];
        v160 = v39[43];
        v161 = v39[44];
        v162 = v39[45];
        v163 = v39[46];
        v164 = v39[47];
        v165 = v39[48];
        a2 = sub_221C750(a1, a2, v40, v18, a5, &v154, (__int64)v155, 12, a6, &v153);
LABEL_66:
        v15 = v41;
        v10 = v41 | v40 & 0xFFFFFFFF00000000LL;
        v109 = v10;
        v17 = sa;
        if ( !v153 )
          a8[4] = v154;
        continue;
      case 'c':
        sb = v17;
        v54 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221D830(
                       a1,
                       (int)a2,
                       v15,
                       (int)v18,
                       a5,
                       a6,
                       (__int64)&v153,
                       (__int64)a8,
                       *(char **)(*(_QWORD *)(v137 + 16) + 48LL));
        goto LABEL_96;
      case 'd':
        sv = v17;
        v63 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 1, 31, 2u, a6, &v153);
        v10 = v64 | v63 & 0xFFFFFFFF00000000LL;
        v15 = v64;
        v109 = v10;
        v9 = v153;
        v17 = sv;
        if ( !v153 )
          goto LABEL_106;
        continue;
      case 'e':
        if ( a2 && (_BYTE)v10 )
        {
          v65 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
          if ( (unsigned __int64)v65 < *((_QWORD *)a2 + 3) )
          {
            if ( (*(_BYTE *)(*((_QWORD *)v16 + 6) + 2LL * *v65 + 1) & 0x20) != 0 )
              goto LABEL_111;
            goto LABEL_159;
          }
          sbi = v17;
          v123 = v15;
          v67 = (*(__int64 (__fastcall **)(char *, __int64, __int64))(*(_QWORD *)a2 + 72LL))(a2, v24, v20);
          LODWORD(v15) = v123;
          v17 = sbi;
          if ( v67 == -1 )
            a2 = 0;
        }
        else
        {
          LOBYTE(v67) = v15;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v16 + 6) + 2LL * (unsigned __int8)v67 + 1) & 0x20) != 0 )
        {
          v65 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
          if ( (unsigned __int64)v65 < *((_QWORD *)a2 + 3) )
          {
LABEL_111:
            *((_QWORD *)a2 + 2) = v65 + 1;
          }
          else
          {
            sx = v17;
            (*(void (__fastcall **)(char *, __int64))(*(_QWORD *)a2 + 80LL))(a2, v24);
            v17 = sx;
          }
          sw = v17;
          a2 = (char *)sub_221C220(a1, a2, -1, v18, a5, &v154, 1, 9, 1u, a6, &v153);
          v15 = v66;
          v109 = (unsigned int)v66 | v109 & 0xFFFFFFFF00000000LL;
          v17 = sw;
          goto LABEL_113;
        }
LABEL_159:
        sbe = v17;
        v91 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 10, 31, 2u, a6, &v153);
        v15 = v92;
        v109 = (unsigned int)v92 | v91 & 0xFFFFFFFF00000000LL;
        v17 = sbe;
LABEL_113:
        LODWORD(v10) = v153;
        if ( !v153 )
LABEL_106:
          a8[3] = v154;
        break;
      case 'm':
        sy = v17;
        v68 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221C220(a1, a2, v15, v18, a5, &v154, 1, 12, 2u, a6, &v153);
        v10 = v69 | v68 & 0xFFFFFFFF00000000LL;
        v15 = v69;
        v109 = v10;
        v17 = sy;
        if ( !v153 )
          a8[4] = v154 - 1;
        continue;
      case 'n':
        if ( a2 && (_BYTE)v10 )
        {
          v51 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
          if ( (unsigned __int64)v51 >= *((_QWORD *)a2 + 3) )
          {
            sbj = v17;
            v125 = v15;
            v100 = (*(__int64 (__fastcall **)(char *, __int64, __int64))(*(_QWORD *)a2 + 72LL))(a2, v24, v20);
            v15 = v125;
            v17 = sbj;
            if ( v100 == -1 )
            {
              LODWORD(v10) = 255;
              v20 = 0xFFFFFFFFLL;
              a2 = 0;
            }
            else
            {
              v20 = v100;
              LODWORD(v10) = (unsigned __int8)v100;
            }
          }
          else
          {
            LODWORD(v10) = *v51;
            v20 = *v51;
          }
        }
        else
        {
          v20 = (unsigned int)v15;
          LODWORD(v10) = (unsigned __int8)v15;
        }
        v52 = v16[(int)v10 + 313];
        if ( v52 )
          goto LABEL_93;
        v53 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 64LL);
        if ( v53 != sub_2216C50 )
        {
          ss = v17;
          v24 = (unsigned int)(char)v20;
          v121 = v15;
          v94 = ((__int64 (__fastcall *)(char *, __int64, _QWORD))v53)(v16, v24, 0);
          v17 = ss;
          v15 = v121;
          v20 = v94;
        }
        if ( (_BYTE)v20 )
        {
          v16[(int)v10 + 313] = v20;
          v52 = v20;
LABEL_93:
          if ( v52 == 10 )
            goto LABEL_48;
        }
        goto LABEL_20;
      case 't':
        if ( a2 && (_BYTE)v10 )
        {
          v90 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
          if ( (unsigned __int64)v90 >= *((_QWORD *)a2 + 3) )
          {
            sbk = v17;
            v126 = v15;
            v101 = (*(__int64 (__fastcall **)(char *, __int64, __int64))(*(_QWORD *)a2 + 72LL))(a2, v24, v20);
            v15 = v126;
            v17 = sbk;
            if ( v101 == -1 )
            {
              LODWORD(v10) = 255;
              v20 = 0xFFFFFFFFLL;
              a2 = 0;
            }
            else
            {
              v20 = v101;
              LODWORD(v10) = (unsigned __int8)v101;
            }
          }
          else
          {
            LODWORD(v10) = *v90;
            v20 = *v90;
          }
        }
        else
        {
          v20 = (unsigned int)v15;
          LODWORD(v10) = (unsigned __int8)v15;
        }
        v32 = v16[(int)v10 + 313];
        if ( v32 )
          goto LABEL_47;
        v33 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 64LL);
        if ( v33 != sub_2216C50 )
        {
          sj = v17;
          v24 = (unsigned int)(char)v20;
          v120 = v15;
          v93 = ((__int64 (__fastcall *)(char *, __int64, _QWORD))v33)(v16, v24, 0);
          v17 = sj;
          v15 = v120;
          v20 = v93;
        }
        if ( (_BYTE)v20 )
        {
          v16[(int)v10 + 313] = v20;
          v32 = v20;
LABEL_47:
          if ( v32 == 9 )
            goto LABEL_48;
        }
        goto LABEL_20;
      case 'x':
        sb = v17;
        v54 = (unsigned int)v15 | v109 & 0xFFFFFFFF00000000LL;
        a2 = (char *)sub_221D830(
                       a1,
                       (int)a2,
                       v15,
                       (int)v18,
                       a5,
                       a6,
                       (__int64)&v153,
                       (__int64)a8,
                       *(char **)(*(_QWORD *)(v137 + 16) + 16LL));
LABEL_96:
        v15 = v55;
        v10 = v55 | v54 & 0xFFFFFFFF00000000LL;
        v109 = v10;
        v17 = sb;
        continue;
      default:
        goto LABEL_20;
    }
  }
  if ( v153 || i != v108 )
LABEL_38:
    *a7 |= 4u;
  return a2;
}
