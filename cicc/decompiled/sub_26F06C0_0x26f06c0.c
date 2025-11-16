// Function: sub_26F06C0
// Address: 0x26f06c0
//
void __fastcall sub_26F06C0(__int64 **a1)
{
  _QWORD *v1; // r15
  __int64 *v2; // rax
  __int64 *v3; // rdi
  __int64 *v4; // r14
  __int64 v5; // rdi
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // r15
  unsigned __int64 v8; // rax
  _QWORD *v9; // rdi
  unsigned __int64 **v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 i; // r14
  unsigned __int64 **v17; // rax
  __int64 v18; // r12
  unsigned __int64 **v19; // rbx
  unsigned __int64 **v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 *v23; // r15
  char v24; // r12
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 *v27; // rdx
  __int64 *v28; // r14
  unsigned __int64 **v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int8 v31; // dl
  __int64 v32; // rax
  __int64 v33; // rax
  char v34; // r14
  unsigned __int8 v35; // dl
  unsigned __int64 *v36; // rbx
  __int64 v37; // rax
  unsigned __int64 *v38; // r12
  __int64 v39; // rdi
  unsigned int v40; // esi
  unsigned __int64 v41; // rdx
  __int64 v42; // r9
  unsigned __int64 *v43; // r8
  int v44; // r11d
  unsigned int v45; // eax
  unsigned __int64 *v46; // rdi
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rdx
  unsigned __int8 v49; // al
  unsigned __int64 v50; // rdi
  int v51; // ecx
  __int64 v52; // rax
  int v53; // r11d
  unsigned __int64 *v54; // r10
  unsigned __int64 *v55; // r15
  int *v56; // rax
  __int64 v57; // r15
  int *v58; // rsi
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 *v61; // rax
  __int64 *v62; // rdi
  __int64 v63; // rcx
  __int64 v64; // rdx
  char v65; // al
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __m128i *v68; // r12
  unsigned __int8 *v69; // rax
  _QWORD *v70; // rax
  __int64 *v71; // rdx
  __int64 *v72; // r12
  char v73; // r15
  __int64 v74; // rax
  int v75; // r11d
  __int64 v76; // rax
  __int64 v77; // rbx
  __int64 *j; // r12
  __int64 v79; // [rsp+8h] [rbp-8D8h]
  __int64 *v80; // [rsp+60h] [rbp-880h]
  _QWORD *v81; // [rsp+80h] [rbp-860h]
  _QWORD *v82; // [rsp+80h] [rbp-860h]
  unsigned __int64 *v83; // [rsp+80h] [rbp-860h]
  _QWORD *v84; // [rsp+90h] [rbp-850h]
  _QWORD *v85; // [rsp+90h] [rbp-850h]
  char v86; // [rsp+90h] [rbp-850h]
  __int64 *v87; // [rsp+98h] [rbp-848h]
  __int64 v88; // [rsp+98h] [rbp-848h]
  unsigned __int64 *v89; // [rsp+98h] [rbp-848h]
  unsigned __int64 *v90; // [rsp+98h] [rbp-848h]
  __m128i *v91; // [rsp+A8h] [rbp-838h] BYREF
  unsigned __int64 v92; // [rsp+B0h] [rbp-830h] BYREF
  __int64 v93; // [rsp+B8h] [rbp-828h]
  __int64 v94; // [rsp+C0h] [rbp-820h] BYREF
  __int64 v95; // [rsp+C8h] [rbp-818h]
  __int64 v96; // [rsp+D0h] [rbp-810h]
  __int64 v97; // [rsp+D8h] [rbp-808h]
  __int64 v98; // [rsp+E0h] [rbp-800h] BYREF
  int v99; // [rsp+E8h] [rbp-7F8h] BYREF
  int *v100; // [rsp+F0h] [rbp-7F0h]
  int *v101; // [rsp+F8h] [rbp-7E8h]
  int *v102; // [rsp+100h] [rbp-7E0h]
  __int64 v103; // [rsp+108h] [rbp-7D8h]
  unsigned __int64 v104; // [rsp+110h] [rbp-7D0h] BYREF
  __int64 v105; // [rsp+118h] [rbp-7C8h] BYREF
  __int64 *v106; // [rsp+120h] [rbp-7C0h]
  __int64 *v107; // [rsp+128h] [rbp-7B8h]
  __int64 *v108; // [rsp+130h] [rbp-7B0h]
  __int64 v109; // [rsp+138h] [rbp-7A8h]
  __int64 *v110; // [rsp+140h] [rbp-7A0h] BYREF
  __int64 v111; // [rsp+148h] [rbp-798h]
  _BYTE v112[512]; // [rsp+150h] [rbp-790h] BYREF
  unsigned __int64 *v113; // [rsp+350h] [rbp-590h] BYREF
  __int64 v114; // [rsp+358h] [rbp-588h]
  _BYTE v115[64]; // [rsp+360h] [rbp-580h] BYREF
  _BYTE *v116; // [rsp+3A0h] [rbp-540h]
  __int64 v117; // [rsp+3A8h] [rbp-538h]
  _BYTE v118[64]; // [rsp+3B0h] [rbp-530h] BYREF
  _BYTE *v119; // [rsp+3F0h] [rbp-4F0h]
  __int64 v120; // [rsp+3F8h] [rbp-4E8h]
  _BYTE v121[64]; // [rsp+400h] [rbp-4E0h] BYREF
  _BYTE *v122; // [rsp+440h] [rbp-4A0h]
  __int64 v123; // [rsp+448h] [rbp-498h]
  _BYTE v124[64]; // [rsp+450h] [rbp-490h] BYREF
  _BYTE *v125; // [rsp+490h] [rbp-450h]
  __int64 v126; // [rsp+498h] [rbp-448h]
  _BYTE v127[64]; // [rsp+4A0h] [rbp-440h] BYREF
  __int64 v128; // [rsp+4E0h] [rbp-400h]
  char *v129; // [rsp+4E8h] [rbp-3F8h]
  __int64 v130; // [rsp+4F0h] [rbp-3F0h]
  int v131; // [rsp+4F8h] [rbp-3E8h]
  char v132; // [rsp+4FCh] [rbp-3E4h]
  char v133; // [rsp+500h] [rbp-3E0h] BYREF
  unsigned __int64 *v134; // [rsp+600h] [rbp-2E0h] BYREF
  __int64 v135; // [rsp+608h] [rbp-2D8h]
  _BYTE v136[64]; // [rsp+610h] [rbp-2D0h] BYREF
  _BYTE *v137; // [rsp+650h] [rbp-290h]
  __int64 v138; // [rsp+658h] [rbp-288h]
  _BYTE v139[64]; // [rsp+660h] [rbp-280h] BYREF
  _BYTE *v140; // [rsp+6A0h] [rbp-240h]
  __int64 v141; // [rsp+6A8h] [rbp-238h]
  _BYTE v142[64]; // [rsp+6B0h] [rbp-230h] BYREF
  _BYTE *v143; // [rsp+6F0h] [rbp-1F0h]
  __int64 v144; // [rsp+6F8h] [rbp-1E8h]
  _BYTE v145[64]; // [rsp+700h] [rbp-1E0h] BYREF
  _BYTE *v146; // [rsp+740h] [rbp-1A0h]
  __int64 v147; // [rsp+748h] [rbp-198h]
  _BYTE v148[64]; // [rsp+750h] [rbp-190h] BYREF
  __int64 v149; // [rsp+790h] [rbp-150h]
  char *v150; // [rsp+798h] [rbp-148h]
  __int64 v151; // [rsp+7A0h] [rbp-140h]
  int v152; // [rsp+7A8h] [rbp-138h]
  char v153; // [rsp+7ACh] [rbp-134h]
  char v154; // [rsp+7B0h] [rbp-130h] BYREF

  v1 = a1;
  v2 = *a1;
  v116 = v118;
  v80 = v2;
  v119 = v121;
  v113 = (unsigned __int64 *)v115;
  v122 = v124;
  v114 = 0x800000000LL;
  v117 = 0x800000000LL;
  v120 = 0x800000000LL;
  v123 = 0x800000000LL;
  v125 = v127;
  v126 = 0x800000000LL;
  v129 = &v133;
  v128 = 0;
  v130 = 32;
  v131 = 0;
  v132 = 1;
  sub_AE8D20((__int64)&v113, (__int64)a1);
  v3 = a1[2];
  v110 = (__int64 *)v112;
  v111 = 0x4000000000LL;
  v101 = &v99;
  v102 = &v99;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v99 = 0;
  v100 = 0;
  v103 = 0;
  v87 = v1 + 1;
  if ( v1 + 1 != v3 )
  {
    v81 = v1;
    v4 = v3;
    do
    {
      v5 = (__int64)(v4 - 7);
      if ( !v4 )
        v5 = 0;
      v134 = (unsigned __int64 *)v136;
      v135 = 0x100000000LL;
      sub_B92230(v5, (__int64)&v134);
      v6 = &v134[(unsigned int)v135];
      if ( v134 != v6 )
      {
        v7 = v134;
        do
        {
          v8 = *v7++;
          v104 = v8;
          sub_26F0230(&v98, &v104);
        }
        while ( v6 != v7 );
        v6 = v134;
      }
      if ( v6 != (unsigned __int64 *)v136 )
        _libc_free((unsigned __int64)v6);
      v4 = (__int64 *)v4[1];
    }
    while ( v87 != v4 );
    v1 = v81;
  }
  LODWORD(v105) = 0;
  v107 = &v105;
  v108 = &v105;
  v137 = v139;
  v140 = v142;
  v134 = (unsigned __int64 *)v136;
  v143 = v145;
  v135 = 0x800000000LL;
  v138 = 0x800000000LL;
  v141 = 0x800000000LL;
  v144 = 0x800000000LL;
  v146 = v148;
  v9 = (_QWORD *)v1[4];
  v147 = 0x800000000LL;
  v150 = &v154;
  v106 = 0;
  v109 = 0;
  v149 = 0;
  v151 = 32;
  v152 = 0;
  v153 = 1;
  v82 = v1 + 3;
  v84 = v9;
  if ( v1 + 3 != v9 )
  {
    v88 = (__int64)v1;
    v10 = &v134;
    do
    {
      v11 = (__int64)(v84 - 7);
      if ( !v84 )
        v11 = 0;
      v12 = v11;
      v13 = sub_B92180(v11);
      if ( v13 )
        sub_AE8440((__int64)v10, v13);
      v14 = *(_QWORD *)(v12 + 80);
      v15 = v12 + 72;
      if ( v12 + 72 == v14 )
      {
        v29 = v10;
        i = 0;
        v18 = v15;
        v19 = v29;
      }
      else
      {
        if ( !v14 )
          BUG();
        while ( 1 )
        {
          i = *(_QWORD *)(v14 + 32);
          if ( i != v14 + 24 )
            break;
          v14 = *(_QWORD *)(v14 + 8);
          if ( v15 == v14 )
            break;
          if ( !v14 )
            BUG();
        }
LABEL_25:
        v17 = v10;
        v18 = v15;
        v19 = v17;
      }
      if ( v14 == v18 )
      {
        v10 = v19;
      }
      else
      {
        v20 = v19;
        v15 = v18;
        v10 = v20;
        do
        {
          v21 = i - 24;
          if ( !i )
            v21 = 0;
          sub_AE8BE0((__int64)v10, v88, v21);
          for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v14 + 32) )
          {
            v22 = v14 - 24;
            if ( !v14 )
              v22 = 0;
            if ( i != v22 + 48 )
              break;
            v14 = *(_QWORD *)(v14 + 8);
            if ( v15 == v14 )
              goto LABEL_25;
            if ( !v14 )
              BUG();
          }
        }
        while ( v15 != v14 );
      }
      v84 = (_QWORD *)v84[1];
    }
    while ( v82 != v84 );
    v1 = (_QWORD *)v88;
    if ( v134 != &v134[(unsigned int)v135] )
    {
      v89 = &v134[(unsigned int)v135];
      v85 = v1;
      v23 = v134;
      do
      {
        v26 = sub_26F03F0(&v104, (__int64)&v105, v23);
        v28 = v27;
        if ( v27 )
        {
          v24 = v26 || v27 == &v105 || *v23 < v27[4];
          v25 = sub_22077B0(0x28u);
          *(_QWORD *)(v25 + 32) = *v23;
          sub_220F040(v24, v25, v28, &v105);
          ++v109;
        }
        ++v23;
      }
      while ( v89 != v23 );
      v1 = v85;
    }
  }
  v83 = &v113[(unsigned int)v114];
  if ( v113 != v83 )
  {
    v90 = v113;
    v86 = 0;
    v79 = (__int64)v1;
    while ( 1 )
    {
      v30 = *v90;
      v91 = (__m128i *)v30;
      v31 = *(_BYTE *)(v30 - 16);
      v32 = (v31 & 2) != 0 ? *(_QWORD *)(v30 - 32) : v30 - 16 - 8LL * ((v31 >> 2) & 0xF);
      v33 = *(_QWORD *)(v32 + 48);
      v34 = 0;
      if ( v33 )
      {
        v35 = *(_BYTE *)(v33 - 16);
        if ( (v35 & 2) != 0 )
        {
          v36 = *(unsigned __int64 **)(v33 - 32);
          v37 = *(unsigned int *)(v33 - 24);
        }
        else
        {
          v36 = (unsigned __int64 *)(v33 - 16 - 8LL * ((v35 >> 2) & 0xF));
          v37 = (*(_WORD *)(v33 - 16) >> 6) & 0xF;
        }
        v38 = &v36[v37];
        v34 = 0;
        if ( v38 != v36 )
          break;
      }
LABEL_97:
      if ( (_DWORD)v111 )
      {
        v70 = sub_26F0350((__int64)&v104, (unsigned __int64 *)&v91);
        v72 = v71;
        if ( v71 )
        {
          v73 = v70 || v71 == &v105 || (unsigned __int64)v91 < v71[4];
          v74 = sub_22077B0(0x28u);
          *(_QWORD *)(v74 + 32) = v91;
          sub_220F040(v73, v74, v72, &v105);
          ++v109;
        }
      }
      else
      {
        v61 = v106;
        if ( !v106 )
          goto LABEL_156;
        v62 = &v105;
        do
        {
          while ( 1 )
          {
            v63 = v61[2];
            v64 = v61[3];
            if ( v61[4] >= (unsigned __int64)v91 )
              break;
            v61 = (__int64 *)v61[3];
            if ( !v64 )
              goto LABEL_103;
          }
          v62 = v61;
          v61 = (__int64 *)v61[2];
        }
        while ( v63 );
LABEL_103:
        if ( v62 == &v105 )
        {
LABEL_156:
          v86 = 1;
        }
        else
        {
          v65 = v86;
          if ( v62[4] > (unsigned __int64)v91 )
            v65 = 1;
          v86 = v65;
        }
      }
      if ( v34 )
      {
        v68 = v91;
        v69 = (unsigned __int8 *)sub_B9C770(v80, v110, (__int64 *)(unsigned int)v111, 0, 1);
        sub_BA6610(v68, 6u, v69);
      }
      ++v90;
      LODWORD(v111) = 0;
      if ( v83 == v90 )
      {
        if ( v86 )
        {
          v77 = sub_BA8E40(v79, "llvm.dbg.cu", 0xBu);
          sub_B91A30(v77);
          if ( v109 )
          {
            for ( j = v107; j != &v105; j = (__int64 *)sub_220EF30((__int64)j) )
              sub_B979A0(v77, j[4]);
          }
        }
        goto LABEL_111;
      }
    }
    while ( 1 )
    {
      v48 = *v36;
      v92 = v48;
      v49 = *(_BYTE *)(v48 - 16);
      if ( (v49 & 2) != 0 )
        break;
      v39 = *(_QWORD *)(v48 - 8LL * ((v49 >> 2) & 0xF) - 8);
      if ( v39 )
        goto LABEL_61;
      v40 = v97;
      if ( !(_DWORD)v97 )
      {
LABEL_70:
        ++v94;
LABEL_71:
        sub_26F04F0((__int64)&v94, 2 * v40);
        if ( !(_DWORD)v97 )
          goto LABEL_168;
        v50 = v92;
        v42 = v95;
        v51 = v96 + 1;
        LODWORD(v52) = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
        v43 = (unsigned __int64 *)(v95 + 8LL * (unsigned int)v52);
        v41 = *v43;
        if ( v92 != *v43 )
        {
          v53 = 1;
          v54 = 0;
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v54 )
              v54 = v43;
            v52 = ((_DWORD)v97 - 1) & (unsigned int)(v52 + v53);
            v43 = (unsigned __int64 *)(v95 + 8 * v52);
            v41 = *v43;
            if ( v92 == *v43 )
              goto LABEL_87;
            ++v53;
          }
LABEL_75:
          v41 = v50;
          if ( v54 )
            v43 = v54;
          goto LABEL_87;
        }
        goto LABEL_87;
      }
LABEL_65:
      v41 = v92;
      v42 = v95;
      v43 = 0;
      v44 = 1;
      v45 = (v40 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
      v46 = (unsigned __int64 *)(v95 + 8LL * v45);
      v47 = *v46;
      if ( v92 == *v46 )
      {
LABEL_66:
        if ( v38 == ++v36 )
          goto LABEL_97;
      }
      else
      {
        while ( v47 != -4096 )
        {
          if ( v43 || v47 != -8192 )
            v46 = v43;
          v45 = (v40 - 1) & (v44 + v45);
          v55 = (unsigned __int64 *)(v95 + 8LL * v45);
          v47 = *v55;
          if ( v92 == *v55 )
            goto LABEL_66;
          ++v44;
          v43 = v46;
          v46 = (unsigned __int64 *)(v95 + 8LL * v45);
        }
        if ( !v43 )
          v43 = v46;
        ++v94;
        v51 = v96 + 1;
        if ( 4 * ((int)v96 + 1) >= 3 * v40 )
          goto LABEL_71;
        if ( v40 - HIDWORD(v96) - v51 <= v40 >> 3 )
        {
          sub_26F04F0((__int64)&v94, v40);
          if ( !(_DWORD)v97 )
          {
LABEL_168:
            LODWORD(v96) = v96 + 1;
            BUG();
          }
          v50 = v92;
          v54 = 0;
          v42 = v95;
          v75 = 1;
          v51 = v96 + 1;
          LODWORD(v76) = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
          v43 = (unsigned __int64 *)(v95 + 8LL * (unsigned int)v76);
          v41 = *v43;
          if ( v92 != *v43 )
          {
            while ( v41 != -4096 )
            {
              if ( v41 == -8192 && !v54 )
                v54 = v43;
              v76 = ((_DWORD)v97 - 1) & (unsigned int)(v76 + v75);
              v43 = (unsigned __int64 *)(v95 + 8 * v76);
              v41 = *v43;
              if ( v92 == *v43 )
                goto LABEL_87;
              ++v75;
            }
            goto LABEL_75;
          }
        }
LABEL_87:
        LODWORD(v96) = v51;
        if ( *v43 != -4096 )
          --HIDWORD(v96);
        *v43 = v41;
        v56 = v100;
        if ( v100 )
        {
          v57 = v92;
          v58 = &v99;
          do
          {
            while ( 1 )
            {
              v59 = *((_QWORD *)v56 + 2);
              v60 = *((_QWORD *)v56 + 3);
              if ( *((_QWORD *)v56 + 4) >= v92 )
                break;
              v56 = (int *)*((_QWORD *)v56 + 3);
              if ( !v60 )
                goto LABEL_94;
            }
            v58 = v56;
            v56 = (int *)*((_QWORD *)v56 + 2);
          }
          while ( v59 );
LABEL_94:
          if ( v58 != &v99 && *((_QWORD *)v58 + 4) <= v92 )
          {
            v66 = (unsigned int)v111;
            v67 = (unsigned int)v111 + 1LL;
            if ( v67 > HIDWORD(v111) )
            {
              sub_C8D5F0((__int64)&v110, v112, v67, 8u, (__int64)v43, v42);
              v66 = (unsigned int)v111;
            }
            v110[v66] = v57;
            LODWORD(v111) = v111 + 1;
            goto LABEL_66;
          }
        }
        ++v36;
        v34 = 1;
        if ( v38 == v36 )
          goto LABEL_97;
      }
    }
    v39 = *(_QWORD *)(*(_QWORD *)(v48 - 32) + 8LL);
    if ( v39 )
    {
LABEL_61:
      v93 = sub_AF4F20(v39);
      if ( BYTE4(v93) && !(_BYTE)qword_4FF8CC8 )
        sub_26F0230(&v98, &v92);
    }
    v40 = v97;
    if ( !(_DWORD)v97 )
      goto LABEL_70;
    goto LABEL_65;
  }
LABEL_111:
  if ( !v153 )
    _libc_free((unsigned __int64)v150);
  if ( v146 != v148 )
    _libc_free((unsigned __int64)v146);
  if ( v143 != v145 )
    _libc_free((unsigned __int64)v143);
  if ( v140 != v142 )
    _libc_free((unsigned __int64)v140);
  if ( v137 != v139 )
    _libc_free((unsigned __int64)v137);
  if ( v134 != (unsigned __int64 *)v136 )
    _libc_free((unsigned __int64)v134);
  sub_26EF1F0((unsigned __int64)v106);
  sub_26EF3C0((unsigned __int64)v100);
  sub_C7D6A0(v95, 8LL * (unsigned int)v97, 8);
  if ( v110 != (__int64 *)v112 )
    _libc_free((unsigned __int64)v110);
  if ( !v132 )
    _libc_free((unsigned __int64)v129);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
  if ( v116 != v118 )
    _libc_free((unsigned __int64)v116);
  if ( v113 != (unsigned __int64 *)v115 )
    _libc_free((unsigned __int64)v113);
}
