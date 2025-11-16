// Function: sub_29CC120
// Address: 0x29cc120
//
__int64 __fastcall sub_29CC120(
        unsigned __int8 *a1,
        char a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8,
        char a9)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r12
  unsigned int v12; // r13d
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // r9
  unsigned __int8 *v18; // r10
  unsigned __int8 v19; // al
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // r9
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r15
  _QWORD *v30; // rdx
  __int64 v31; // r10
  __int64 *v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rsi
  int v36; // edx
  __int64 v37; // r12
  unsigned __int8 v38; // al
  unsigned int v39; // ecx
  char v40; // dl
  unsigned __int8 v41; // cl
  __int64 v42; // rbx
  _QWORD *v43; // rdi
  __int64 *v45; // r14
  unsigned int v46; // r15d
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // r9
  __int64 v50; // rax
  unsigned int v51; // r12d
  __int64 v52; // r13
  unsigned int v53; // esi
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // rdx
  _QWORD *v57; // rax
  __int64 v58; // r9
  __int64 v59; // rbx
  _QWORD *v60; // r8
  int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  int v65; // edx
  unsigned __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rsi
  int v69; // esi
  __int64 v70; // r11
  __int64 v71; // rcx
  _QWORD *v72; // rdx
  __int64 v73; // rdi
  _QWORD *v74; // rdx
  __int64 v75; // rax
  __int16 v76; // dx
  unsigned int v77; // edx
  __int64 v78; // rsi
  int v79; // r11d
  _QWORD *v80; // r10
  int v81; // r11d
  unsigned int v82; // ecx
  __int64 v83; // rsi
  int v84; // eax
  unsigned int i; // ebx
  __int64 v86; // rax
  char v87; // dl
  __int64 v88; // r13
  _QWORD *v89; // rdi
  __int64 v90; // rax
  __int16 v91; // dx
  __int64 v92; // r12
  int v93; // eax
  unsigned int j; // r13d
  unsigned __int8 v95; // al
  char v96; // dl
  __int64 v97; // rbx
  _QWORD *v98; // rdi
  __int64 v99; // rax
  __int16 v100; // dx
  __int64 v101; // r12
  unsigned int v102; // eax
  _QWORD *v103; // r8
  int v104; // edx
  unsigned int v105; // ecx
  __int64 v106; // rdi
  int v107; // r10d
  _QWORD *v108; // rsi
  int v109; // r10d
  unsigned int v110; // ecx
  __int64 v111; // rdi
  __int64 v112; // [rsp+8h] [rbp-E8h]
  __int64 v113; // [rsp+10h] [rbp-E0h]
  __int64 v114; // [rsp+10h] [rbp-E0h]
  __int64 v115; // [rsp+18h] [rbp-D8h]
  _QWORD *v116; // [rsp+18h] [rbp-D8h]
  __int64 v117; // [rsp+20h] [rbp-D0h]
  int v118; // [rsp+20h] [rbp-D0h]
  _QWORD *v119; // [rsp+20h] [rbp-D0h]
  int v120; // [rsp+30h] [rbp-C0h]
  __int64 v121; // [rsp+30h] [rbp-C0h]
  __int64 v122; // [rsp+30h] [rbp-C0h]
  unsigned int v123; // [rsp+38h] [rbp-B8h]
  __int64 v124; // [rsp+38h] [rbp-B8h]
  _QWORD *v125; // [rsp+38h] [rbp-B8h]
  unsigned int v126; // [rsp+38h] [rbp-B8h]
  __int64 v127; // [rsp+38h] [rbp-B8h]
  __int64 v128; // [rsp+40h] [rbp-B0h]
  __int64 v130; // [rsp+48h] [rbp-A8h]
  int v131; // [rsp+48h] [rbp-A8h]
  __int64 v132; // [rsp+50h] [rbp-A0h]
  __int64 v133; // [rsp+58h] [rbp-98h]
  __int64 v134; // [rsp+58h] [rbp-98h]
  __int64 v135; // [rsp+58h] [rbp-98h]
  unsigned __int8 *v136; // [rsp+58h] [rbp-98h]
  int v137; // [rsp+58h] [rbp-98h]
  __int64 v138; // [rsp+58h] [rbp-98h]
  __int64 v139; // [rsp+60h] [rbp-90h] BYREF
  __int64 v140; // [rsp+68h] [rbp-88h]
  __int64 v141; // [rsp+70h] [rbp-80h]
  __int64 v142; // [rsp+78h] [rbp-78h]
  int v143; // [rsp+80h] [rbp-70h]
  char v144; // [rsp+84h] [rbp-6Ch]
  const char *v145; // [rsp+90h] [rbp-60h] BYREF
  __int64 v146; // [rsp+98h] [rbp-58h]
  char *v147; // [rsp+A0h] [rbp-50h]
  __int16 v148; // [rsp+B0h] [rbp-40h]

  v132 = *((_QWORD *)a1 + 2);
  if ( !v132 )
  {
    sub_B43D60(a1);
    return v132;
  }
  v9 = *(_QWORD *)(*((_QWORD *)a1 + 5) + 72LL);
  v10 = sub_B2BEC0(v9);
  if ( a9 )
  {
    v45 = (__int64 *)*((_QWORD *)a1 + 1);
    v46 = *(_DWORD *)(v10 + 4);
    v145 = sub_BD5D20((__int64)a1);
    v147 = ".reg2mem";
    v148 = 773;
    v146 = v47;
    v48 = sub_BD2C40(80, 1u);
    v18 = a1;
    v132 = (__int64)v48;
    if ( v48 )
    {
      sub_B4CDD0((__int64)v48, v45, v46, 0, (__int64)&v145, v49, a7, a8);
      v18 = a1;
      v19 = *a1;
      if ( *a1 == 34 )
      {
LABEL_7:
        v133 = (__int64)v18;
        v20 = sub_AA54C0(*((_QWORD *)v18 - 12));
        v18 = (unsigned __int8 *)v133;
        if ( !v20 )
        {
          v102 = sub_D0E820(*(_QWORD *)(v133 + 40), *(_QWORD *)(v133 - 96));
          v148 = 257;
          v139 = 0;
          v141 = 0;
          v140 = 0;
          v142 = 0;
          v143 = 0;
          v144 = 1;
          sub_F451F0(v133, v102, (__int64)&v139, (void **)&v145);
          v18 = (unsigned __int8 *)v133;
        }
        goto LABEL_9;
      }
      goto LABEL_41;
    }
  }
  else
  {
    v11 = (__int64 *)*((_QWORD *)a1 + 1);
    v12 = *(_DWORD *)(v10 + 4);
    v148 = 773;
    v145 = sub_BD5D20((__int64)a1);
    v146 = v13;
    v147 = ".reg2mem";
    v14 = *(_QWORD *)(v9 + 80);
    if ( !v14 )
      BUG();
    v15 = *(_QWORD *)(v14 + 32);
    v16 = sub_BD2C40(80, 1u);
    v18 = a1;
    v132 = (__int64)v16;
    if ( v16 )
    {
      sub_B4CDD0((__int64)v16, v11, v12, 0, (__int64)&v145, v17, v15, 1);
      v18 = a1;
    }
  }
  v19 = *v18;
  if ( *v18 == 34 )
    goto LABEL_7;
LABEL_41:
  if ( v19 == 40 )
  {
    v50 = *((unsigned int *)v18 + 22);
    if ( (_DWORD)v50 != -1 )
    {
      v51 = 0;
      v52 = (__int64)v18;
      while ( 1 )
      {
        if ( v51 )
        {
          if ( !sub_AA54C0(*(_QWORD *)(v52 + 32 * (v51 - 1 - v50) - 32)) )
            goto LABEL_48;
LABEL_45:
          v50 = *(unsigned int *)(v52 + 88);
          if ( ++v51 >= (int)v50 + 1 )
            goto LABEL_49;
        }
        else
        {
          if ( sub_AA54C0(*(_QWORD *)(v52 + -32 - 32 * v50 - 32)) )
            goto LABEL_45;
LABEL_48:
          v53 = v51;
          v148 = 257;
          ++v51;
          v139 = 0;
          v140 = 0;
          v141 = 0;
          v142 = 0;
          v143 = 0;
          v144 = 1;
          sub_F451F0(v52, v53, (__int64)&v139, (void **)&v145);
          v50 = *(unsigned int *)(v52 + 88);
          if ( v51 >= (int)v50 + 1 )
          {
LABEL_49:
            v18 = (unsigned __int8 *)v52;
            goto LABEL_9;
          }
        }
      }
    }
    v21 = *((_QWORD *)v18 + 2);
    if ( !v21 )
    {
LABEL_101:
      v130 = (__int64)v18;
      v84 = sub_B46E30((__int64)v18);
      if ( v84 )
      {
        v137 = v84;
        for ( i = 0; i != v137; ++i )
        {
          v90 = sub_B46EC0(v130, i);
          v92 = sub_AA5190(v90);
          if ( v92 )
          {
            LOBYTE(v86) = v91;
            v87 = HIBYTE(v91);
          }
          else
          {
            v87 = 0;
            LOBYTE(v86) = 0;
          }
          v86 = (unsigned __int8)v86;
          BYTE1(v86) = v87;
          v88 = v86;
          v89 = sub_BD2C40(80, unk_3F10A10);
          if ( v89 )
            sub_B4D460((__int64)v89, v130, v132, v92, v88);
        }
      }
      return v132;
    }
  }
  else
  {
LABEL_9:
    v21 = *((_QWORD *)v18 + 2);
    if ( !v21 )
      goto LABEL_28;
  }
  v134 = (__int64)v18;
  do
  {
    v22 = *(_QWORD *)(v21 + 24);
    if ( *(_BYTE *)v22 == 84 )
    {
      v139 = 0;
      v140 = 0;
      v141 = 0;
      LODWORD(v142) = 0;
      if ( (*(_DWORD *)(v22 + 4) & 0x7FFFFFF) != 0 )
      {
        v23 = 0;
        v24 = 0;
        v25 = 0;
        v26 = 8LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF);
        while ( 1 )
        {
          while ( 1 )
          {
            v27 = *(_QWORD *)(v22 - 8);
            v28 = v27 + 4 * v23;
            if ( *(_QWORD *)v28 )
            {
              if ( *(_QWORD *)v28 == v134 )
                break;
            }
            v23 += 8;
            if ( v26 == v23 )
              goto LABEL_24;
          }
          v29 = *(_QWORD *)(32LL * *(unsigned int *)(v22 + 72) + v27 + v23);
          if ( !(_DWORD)v24 )
            break;
          v123 = (v24 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v30 = (_QWORD *)(v25 + 16LL * v123);
          v31 = *v30;
          if ( v29 != *v30 )
          {
            v120 = 1;
            v60 = 0;
            while ( v31 != -4096 )
            {
              if ( !v60 && v31 == -8192 )
                v60 = v30;
              v123 = (v24 - 1) & (v120 + v123);
              v30 = (_QWORD *)(v25 + 16LL * v123);
              v31 = *v30;
              if ( v29 == *v30 )
                goto LABEL_19;
              ++v120;
            }
            if ( !v60 )
              v60 = v30;
            ++v139;
            v61 = v141 + 1;
            if ( 4 * ((int)v141 + 1) < (unsigned int)(3 * v24) )
            {
              if ( (int)v24 - (v61 + HIDWORD(v141)) <= (unsigned int)v24 >> 3 )
              {
                v122 = v26;
                sub_116E750((__int64)&v139, v24);
                if ( !(_DWORD)v142 )
                {
LABEL_167:
                  LODWORD(v141) = v141 + 1;
                  BUG();
                }
                v80 = 0;
                v26 = v122;
                v81 = 1;
                v82 = (v142 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v61 = v141 + 1;
                v60 = (_QWORD *)(v140 + 16LL * v82);
                v83 = *v60;
                if ( v29 != *v60 )
                {
                  while ( v83 != -4096 )
                  {
                    if ( !v80 && v83 == -8192 )
                      v80 = v60;
                    v82 = (v142 - 1) & (v81 + v82);
                    v60 = (_QWORD *)(v140 + 16LL * v82);
                    v83 = *v60;
                    if ( v29 == *v60 )
                      goto LABEL_65;
                    ++v81;
                  }
                  goto LABEL_89;
                }
              }
              goto LABEL_65;
            }
LABEL_85:
            v127 = v26;
            sub_116E750((__int64)&v139, 2 * v24);
            if ( !(_DWORD)v142 )
              goto LABEL_167;
            v26 = v127;
            v77 = (v142 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v61 = v141 + 1;
            v60 = (_QWORD *)(v140 + 16LL * v77);
            v78 = *v60;
            if ( v29 != *v60 )
            {
              v79 = 1;
              v80 = 0;
              while ( v78 != -4096 )
              {
                if ( !v80 && v78 == -8192 )
                  v80 = v60;
                v77 = (v142 - 1) & (v79 + v77);
                v60 = (_QWORD *)(v140 + 16LL * v77);
                v78 = *v60;
                if ( v29 == *v60 )
                  goto LABEL_65;
                ++v79;
              }
LABEL_89:
              if ( v80 )
                v60 = v80;
            }
LABEL_65:
            LODWORD(v141) = v61;
            if ( *v60 != -4096 )
              --HIDWORD(v141);
            v60[1] = 0;
            *v60 = v29;
            v32 = v60 + 1;
            goto LABEL_68;
          }
LABEL_19:
          v32 = v30 + 1;
          v33 = v30[1];
          if ( v33 )
          {
            v34 = *(_QWORD *)(v28 + 8);
            **(_QWORD **)(v28 + 16) = v34;
            if ( !v34 )
            {
              *(_QWORD *)v28 = v33;
LABEL_51:
              v54 = *(_QWORD *)(v33 + 16);
              *(_QWORD *)(v28 + 8) = v54;
              if ( v54 )
                *(_QWORD *)(v54 + 16) = v28 + 8;
              *(_QWORD *)(v28 + 16) = v33 + 16;
              *(_QWORD *)(v33 + 16) = v28;
              goto LABEL_23;
            }
LABEL_21:
            *(_QWORD *)(v34 + 16) = *(_QWORD *)(v28 + 16);
            goto LABEL_22;
          }
LABEL_68:
          v121 = v26;
          v124 = *(_QWORD *)(v134 + 8);
          v145 = sub_BD5D20(v134);
          v148 = 773;
          v146 = v62;
          v147 = ".reload";
          v63 = *(_QWORD *)(*(_QWORD *)(v22 - 8) + 32LL * *(unsigned int *)(v22 + 72) + v23);
          v64 = *(_QWORD *)(v63 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v64 == v63 + 48 )
          {
            v66 = 0;
          }
          else
          {
            if ( !v64 )
              BUG();
            v65 = *(unsigned __int8 *)(v64 - 24);
            v66 = v64 - 24;
            if ( (unsigned int)(v65 - 30) >= 0xB )
              v66 = 0;
          }
          v117 = v66 + 24;
          v67 = sub_BD2C40(80, 1u);
          v26 = v121;
          if ( v67 )
          {
            v68 = v124;
            v125 = v67;
            sub_B4D1B0((__int64)v67, v68, v132, (__int64)&v145, a2, v121, v117, 0);
            v26 = v121;
            v67 = v125;
          }
          *v32 = (__int64)v67;
          v69 = v142;
          v70 = *(_QWORD *)(*(_QWORD *)(v22 - 8) + 32LL * *(unsigned int *)(v22 + 72) + v23);
          if ( (_DWORD)v142 )
          {
            v126 = ((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4);
            v71 = ((_DWORD)v142 - 1) & v126;
            v72 = (_QWORD *)(v140 + 16 * v71);
            v73 = *v72;
            if ( v70 == *v72 )
            {
LABEL_76:
              v74 = v72 + 1;
              goto LABEL_77;
            }
            v118 = 1;
            v103 = 0;
            while ( v73 != -4096 )
            {
              if ( v73 == -8192 && !v103 )
                v103 = v72;
              LODWORD(v71) = (v142 - 1) & (v118 + v71);
              v72 = (_QWORD *)(v140 + 16LL * (unsigned int)v71);
              v73 = *v72;
              if ( v70 == *v72 )
                goto LABEL_76;
              ++v118;
            }
            v69 = v142;
            if ( !v103 )
              v103 = v72;
            ++v139;
            v104 = v141 + 1;
            if ( 4 * ((int)v141 + 1) < (unsigned int)(3 * v142) )
            {
              if ( (int)v142 - HIDWORD(v141) - v104 > (unsigned int)v142 >> 3 )
                goto LABEL_125;
              v112 = v26;
              v114 = v70;
              v116 = v67;
              sub_116E750((__int64)&v139, v142);
              if ( !(_DWORD)v142 )
              {
LABEL_168:
                LODWORD(v141) = v141 + 1;
                BUG();
              }
              v109 = 1;
              v70 = v114;
              v110 = (v142 - 1) & v126;
              v108 = 0;
              v26 = v112;
              v103 = (_QWORD *)(v140 + 16LL * v110);
              v111 = *v103;
              v104 = v141 + 1;
              v67 = v116;
              if ( v114 == *v103 )
                goto LABEL_125;
              while ( v111 != -4096 )
              {
                if ( !v108 && v111 == -8192 )
                  v108 = v103;
                v110 = (v142 - 1) & (v109 + v110);
                v103 = (_QWORD *)(v140 + 16LL * v110);
                v111 = *v103;
                if ( v114 == *v103 )
                  goto LABEL_125;
                ++v109;
              }
              goto LABEL_133;
            }
          }
          else
          {
            ++v139;
          }
          v113 = v26;
          v115 = v70;
          v119 = v67;
          sub_116E750((__int64)&v139, 2 * v69);
          if ( !(_DWORD)v142 )
            goto LABEL_168;
          v70 = v115;
          v26 = v113;
          v105 = (v142 - 1) & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
          v103 = (_QWORD *)(v140 + 16LL * v105);
          v106 = *v103;
          v104 = v141 + 1;
          v67 = v119;
          if ( v115 == *v103 )
            goto LABEL_125;
          v107 = 1;
          v108 = 0;
          while ( v106 != -4096 )
          {
            if ( v106 == -8192 && !v108 )
              v108 = v103;
            v105 = (v142 - 1) & (v107 + v105);
            v103 = (_QWORD *)(v140 + 16LL * v105);
            v106 = *v103;
            if ( v115 == *v103 )
              goto LABEL_125;
            ++v107;
          }
LABEL_133:
          if ( v108 )
            v103 = v108;
LABEL_125:
          LODWORD(v141) = v104;
          if ( *v103 != -4096 )
            --HIDWORD(v141);
          *v103 = v70;
          v74 = v103 + 1;
          v103[1] = 0;
LABEL_77:
          *v74 = v67;
          v33 = *v32;
          v28 = 4 * v23 + *(_QWORD *)(v22 - 8);
          if ( *(_QWORD *)v28 )
          {
            v34 = *(_QWORD *)(v28 + 8);
            **(_QWORD **)(v28 + 16) = v34;
            if ( v34 )
              goto LABEL_21;
          }
LABEL_22:
          *(_QWORD *)v28 = v33;
          if ( v33 )
            goto LABEL_51;
LABEL_23:
          v23 += 8;
          v24 = (unsigned int)v142;
          v25 = v140;
          if ( v26 == v23 )
          {
LABEL_24:
            v35 = 16 * v24;
            goto LABEL_25;
          }
        }
        ++v139;
        goto LABEL_85;
      }
      v25 = 0;
      v35 = 0;
LABEL_25:
      sub_C7D6A0(v25, v35, 8);
    }
    else
    {
      v55 = *(_QWORD *)(v134 + 8);
      v145 = sub_BD5D20(v134);
      v147 = ".reload";
      v148 = 773;
      v146 = v56;
      v57 = sub_BD2C40(80, 1u);
      v59 = (__int64)v57;
      if ( v57 )
        sub_B4D1B0((__int64)v57, v55, v132, (__int64)&v145, a2, v58, v22 + 24, 0);
      sub_BD2ED0(v22, v134, v59);
    }
    v21 = *(_QWORD *)(v134 + 16);
  }
  while ( v21 );
  v18 = (unsigned __int8 *)v134;
LABEL_28:
  v36 = *v18;
  if ( (unsigned int)(v36 - 30) <= 0xA )
  {
    if ( (_BYTE)v36 == 34 )
    {
      v136 = v18;
      v75 = sub_AA5190(*((_QWORD *)v18 - 12));
      v18 = v136;
      v37 = v75;
      if ( v75 )
      {
        v41 = v76;
        v40 = HIBYTE(v76);
      }
      else
      {
LABEL_35:
        v40 = 0;
        v41 = 0;
      }
      v135 = (__int64)v18;
      v42 = v41;
      BYTE1(v42) = v40;
      v43 = sub_BD2C40(80, unk_3F10A10);
      if ( v43 )
        sub_B4D460((__int64)v43, v135, v132, v37, v42);
      return v132;
    }
    if ( (_BYTE)v36 != 40 )
      BUG();
    goto LABEL_101;
  }
  v37 = *((_QWORD *)v18 + 4);
  while ( 2 )
  {
    if ( !v37 )
      BUG();
    v38 = *(_BYTE *)(v37 - 24);
    if ( v38 == 84 )
    {
LABEL_55:
      v37 = *(_QWORD *)(v37 + 8);
      continue;
    }
    break;
  }
  v39 = v38 - 39;
  if ( v39 <= 0x38 && ((1LL << v39) & 0x100060000000001LL) != 0 )
  {
    if ( v38 == 39 )
      goto LABEL_109;
    goto LABEL_55;
  }
  if ( v38 != 39 )
    goto LABEL_35;
LABEL_109:
  v128 = (__int64)v18;
  v138 = v37 - 24;
  v93 = sub_B46E30(v37 - 24);
  if ( v93 )
  {
    v131 = v93;
    for ( j = 0; j != v131; ++j )
    {
      v99 = sub_B46EC0(v138, j);
      v101 = sub_AA5190(v99);
      if ( v101 )
      {
        v95 = v100;
        v96 = HIBYTE(v100);
      }
      else
      {
        v96 = 0;
        v95 = 0;
      }
      v97 = v95;
      BYTE1(v97) = v96;
      v98 = sub_BD2C40(80, unk_3F10A10);
      if ( v98 )
        sub_B4D460((__int64)v98, v128, v132, v101, v97);
    }
  }
  return v132;
}
