// Function: sub_328B5C0
// Address: 0x328b5c0
//
__int64 __fastcall sub_328B5C0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  int v6; // r15d
  __int16 *v7; // rax
  __int64 v8; // rsi
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r12
  int v16; // eax
  __int64 v17; // rax
  int v18; // edx
  unsigned __int16 v19; // dx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rdx
  int v24; // eax
  int v25; // r14d
  bool v26; // al
  __int64 v27; // rcx
  __int64 v28; // r8
  unsigned __int16 v29; // ax
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 *v32; // rax
  __int64 v33; // r15
  __int64 v34; // rsi
  __int64 v35; // r12
  int v36; // ecx
  __int64 v37; // r8
  int v38; // r9d
  int v39; // esi
  __int64 v40; // rax
  int v41; // edx
  __int128 v42; // rax
  int v43; // r9d
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r15
  __int64 v47; // rdx
  int v48; // eax
  __int64 v49; // rsi
  unsigned __int64 v50; // rax
  __int128 v51; // rax
  int v52; // r9d
  __int64 v53; // rdx
  int v54; // esi
  __int64 *v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r11
  int v58; // edx
  int v59; // ecx
  __int64 v60; // rdx
  int v61; // eax
  int v62; // ecx
  int v63; // eax
  bool v64; // zf
  __int64 v65; // rax
  unsigned int v66; // ecx
  unsigned __int16 v67; // si
  int v68; // eax
  __int64 v69; // rdx
  unsigned int v70; // eax
  __int64 v71; // r8
  unsigned __int64 v72; // rax
  __int64 v73; // rax
  int v74; // ecx
  int v75; // eax
  __int64 v76; // rdi
  char v77; // al
  char *v78; // r10
  __int64 v79; // rcx
  char v80; // al
  __int64 *v81; // rax
  __int64 v82; // r14
  unsigned __int64 v83; // r15
  __int64 v84; // rdx
  char *v85; // rax
  __int64 v86; // rax
  unsigned int v87; // edx
  unsigned __int64 v88; // r15
  __int64 v89; // rax
  unsigned int v90; // edx
  __int128 v91; // rax
  __int64 v92; // rax
  __int128 v93; // rax
  __int64 v94; // rbx
  __int128 v95; // rax
  int v96; // r9d
  __int64 v97; // rax
  unsigned int v98; // edx
  __int128 v99; // [rsp-20h] [rbp-120h]
  __int128 v100; // [rsp-10h] [rbp-110h]
  __int128 v101; // [rsp-10h] [rbp-110h]
  __int128 v102; // [rsp-10h] [rbp-110h]
  __int64 v103; // [rsp+8h] [rbp-F8h]
  int v104; // [rsp+10h] [rbp-F0h]
  unsigned int v105; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v106; // [rsp+20h] [rbp-E0h]
  int v107; // [rsp+20h] [rbp-E0h]
  unsigned int v108; // [rsp+28h] [rbp-D8h]
  __int16 v109; // [rsp+2Ah] [rbp-D6h]
  int v110; // [rsp+30h] [rbp-D0h]
  __m128i v111; // [rsp+30h] [rbp-D0h]
  unsigned int v112; // [rsp+30h] [rbp-D0h]
  unsigned int v113; // [rsp+30h] [rbp-D0h]
  char *v114; // [rsp+30h] [rbp-D0h]
  __int64 v115; // [rsp+30h] [rbp-D0h]
  int v116; // [rsp+30h] [rbp-D0h]
  int v117; // [rsp+30h] [rbp-D0h]
  __int64 v118; // [rsp+40h] [rbp-C0h]
  int v119; // [rsp+40h] [rbp-C0h]
  int v120; // [rsp+40h] [rbp-C0h]
  __m128i v121; // [rsp+40h] [rbp-C0h]
  __int64 v122; // [rsp+40h] [rbp-C0h]
  int v123; // [rsp+40h] [rbp-C0h]
  int v124; // [rsp+40h] [rbp-C0h]
  int v125; // [rsp+50h] [rbp-B0h]
  int v126; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v127; // [rsp+50h] [rbp-B0h]
  __int64 v128; // [rsp+50h] [rbp-B0h]
  __int64 v129; // [rsp+50h] [rbp-B0h]
  char *v130; // [rsp+50h] [rbp-B0h]
  __int64 v131; // [rsp+50h] [rbp-B0h]
  int v132; // [rsp+58h] [rbp-A8h]
  __int64 v133; // [rsp+58h] [rbp-A8h]
  unsigned int v134; // [rsp+90h] [rbp-70h] BYREF
  __int64 v135; // [rsp+98h] [rbp-68h]
  __int64 v136; // [rsp+A0h] [rbp-60h] BYREF
  int v137; // [rsp+A8h] [rbp-58h]
  __int64 v138; // [rsp+B0h] [rbp-50h]
  __int64 v139; // [rsp+B8h] [rbp-48h]
  __int64 v140; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v141; // [rsp+C8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v136 = v8;
  LOWORD(v134) = v9;
  v135 = v10;
  if ( v8 )
    sub_B96E90((__int64)&v136, v8, 1);
  v11 = *a1;
  v137 = *(_DWORD *)(a2 + 72);
  v140 = v5;
  LODWORD(v141) = v6;
  v12 = sub_3402EA0(v11, 197, (unsigned int)&v136, v134, v135, 0, (__int64)&v140, 1);
  v13 = v12;
  if ( v12 )
  {
    v14 = v12;
    goto LABEL_5;
  }
  v16 = *(_DWORD *)(v5 + 24);
  if ( v16 == 197 )
  {
    v14 = **(_QWORD **)(v5 + 40);
    goto LABEL_5;
  }
  if ( v16 != 201 || (v17 = *(_QWORD *)(v5 + 56)) == 0 )
  {
LABEL_19:
    v19 = v134;
    if ( (_WORD)v134 )
    {
      if ( (unsigned __int16)(v134 - 17) > 0xD3u )
      {
LABEL_21:
        v20 = v135;
        goto LABEL_22;
      }
      v19 = word_4456580[(unsigned __int16)v134 - 1];
      v20 = 0;
    }
    else
    {
      v26 = sub_30070B0((__int64)&v134);
      v19 = 0;
      v13 = 0;
      if ( !v26 )
        goto LABEL_21;
      v29 = sub_3009970((__int64)&v134, 197, 0, v27, v28);
      v13 = 0;
      v31 = v30;
      v19 = v29;
      v20 = v31;
    }
LABEL_22:
    LOWORD(v140) = v19;
    v141 = v20;
    if ( v19 )
    {
      if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
        BUG();
      v22 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
    }
    else
    {
      v118 = v13;
      v21 = sub_3007260((__int64)&v140);
      v13 = v118;
      v138 = v21;
      LODWORD(v22) = v21;
      v139 = v23;
    }
    v24 = *(_DWORD *)(v5 + 24);
    if ( (unsigned int)v22 <= 0x1F || v24 != 190 )
    {
LABEL_26:
      if ( ((v24 - 190) & 0xFFFFFFFD) == 0 )
      {
        v44 = *(_QWORD *)(v5 + 56);
        if ( v44 )
        {
LABEL_52:
          v45 = 1;
          do
          {
            if ( *(_DWORD *)(v44 + 8) == v6 )
            {
              if ( !v45 )
                goto LABEL_27;
              v44 = *(_QWORD *)(v44 + 32);
              if ( !v44 )
                goto LABEL_60;
              if ( v6 == *(_DWORD *)(v44 + 8) )
                goto LABEL_27;
              v45 = 0;
            }
            v44 = *(_QWORD *)(v44 + 32);
          }
          while ( v44 );
          if ( v45 == 1 )
            goto LABEL_27;
LABEL_60:
          v46 = *(_QWORD *)(v5 + 40);
          v47 = *(_QWORD *)(v46 + 40);
          v48 = *(_DWORD *)(v47 + 24);
          if ( v48 == 35 || v48 == 11 )
          {
            v49 = *(_QWORD *)(v47 + 96);
            if ( *(_DWORD *)(v49 + 32) > 0x40u )
            {
              v127 = (unsigned int)v22;
              v120 = *(_DWORD *)(v49 + 32);
              if ( v120 - (unsigned int)sub_C444A0(v49 + 24) > 0x40 )
                goto LABEL_27;
              v50 = **(_QWORD **)(v49 + 24);
              if ( v127 <= v50 )
                goto LABEL_27;
            }
            else
            {
              v50 = *(_QWORD *)(v49 + 24);
              if ( (unsigned int)v22 <= v50 )
                goto LABEL_27;
            }
            if ( (v50 & 7) == 0 )
            {
              *(_QWORD *)&v51 = sub_33FAF80(*a1, 197, (unsigned int)&v136, v134, v135, v13, *(_OWORD *)v46);
              v14 = sub_3406EB0(
                      *a1,
                      2 * (unsigned int)(*(_DWORD *)(v5 + 24) == 190) + 190,
                      (unsigned int)&v136,
                      v134,
                      v135,
                      v52,
                      v51,
                      *(_OWORD *)(*(_QWORD *)(v5 + 40) + 40LL));
              goto LABEL_5;
            }
          }
        }
      }
LABEL_27:
      v25 = *(_DWORD *)(a2 + 24);
      if ( ((v25 - 197) & 0xFFFFFFFB) != 0 )
        goto LABEL_28;
      v32 = *(__int64 **)(a2 + 40);
      v33 = *a1;
      v34 = *(_QWORD *)(a2 + 80);
      v35 = *v32;
      v36 = *((_DWORD *)v32 + 2);
      v37 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
      v38 = **(unsigned __int16 **)(a2 + 48);
      v140 = v34;
      if ( v34 )
      {
        v125 = v38;
        v110 = v36;
        v119 = v37;
        sub_B96E90((__int64)&v140, v34, 1);
        v38 = v125;
        v36 = v110;
        LODWORD(v37) = v119;
      }
      LODWORD(v141) = *(_DWORD *)(a2 + 72);
      v39 = *(_DWORD *)(v35 + 24);
      if ( (unsigned int)(v39 - 186) <= 2 )
      {
        v40 = *(_QWORD *)(v35 + 56);
        if ( v40 )
        {
          v41 = 1;
          do
          {
            if ( *(_DWORD *)(v40 + 8) == v36 )
            {
              if ( !v41 )
                goto LABEL_43;
              v40 = *(_QWORD *)(v40 + 32);
              if ( !v40 )
                goto LABEL_74;
              if ( *(_DWORD *)(v40 + 8) == v36 )
                goto LABEL_43;
              v41 = 0;
            }
            v40 = *(_QWORD *)(v40 + 32);
          }
          while ( v40 );
          if ( v41 == 1 )
            goto LABEL_43;
LABEL_74:
          v55 = *(__int64 **)(v35 + 40);
          v56 = v55[5];
          v57 = *v55;
          v58 = *((_DWORD *)v55 + 2);
          v121 = _mm_loadu_si128((const __m128i *)v55);
          v59 = *(_DWORD *)(v56 + 24);
          v111 = _mm_loadu_si128((const __m128i *)(v55 + 5));
          if ( v25 == *(_DWORD *)(*v55 + 24) )
          {
            if ( v59 == v25 )
            {
              v14 = sub_3406EB0(
                      v33,
                      v39,
                      (unsigned int)&v140,
                      v38,
                      v37,
                      v38,
                      *(_OWORD *)*(_QWORD *)(v57 + 40),
                      *(_OWORD *)*(_QWORD *)(v56 + 40));
LABEL_127:
              if ( v140 )
                sub_B91220((__int64)&v140, v140);
              if ( v14 )
                goto LABEL_5;
LABEL_28:
              v14 = 0;
              goto LABEL_5;
            }
            v73 = *(_QWORD *)(v57 + 56);
            if ( v73 )
            {
              v74 = 1;
              do
              {
                if ( v58 == *(_DWORD *)(v73 + 8) )
                {
                  if ( !v74 )
                    goto LABEL_43;
                  v73 = *(_QWORD *)(v73 + 32);
                  if ( !v73 )
                    goto LABEL_125;
                  if ( v58 == *(_DWORD *)(v73 + 8) )
                    goto LABEL_43;
                  v74 = 0;
                }
                v73 = *(_QWORD *)(v73 + 32);
              }
              while ( v73 );
              if ( v74 == 1 )
                goto LABEL_43;
LABEL_125:
              v101 = (__int128)v111;
              v131 = v57;
              v116 = v38;
              v123 = v37;
              *(_QWORD *)&v91 = sub_33FAF80(v33, v25, (unsigned int)&v140, v38, v37, v38, v101);
              v92 = sub_3406EB0(
                      v33,
                      *(_DWORD *)(v35 + 24),
                      (unsigned int)&v140,
                      v116,
                      v123,
                      v116,
                      *(_OWORD *)*(_QWORD *)(v131 + 40),
                      v91);
              goto LABEL_126;
            }
          }
          else if ( v59 == v25 )
          {
            v60 = *(_QWORD *)(v56 + 56);
            if ( v60 )
            {
              v61 = *((_DWORD *)v55 + 12);
              v62 = 1;
              do
              {
                if ( *(_DWORD *)(v60 + 8) == v61 )
                {
                  if ( !v62 )
                    goto LABEL_43;
                  v60 = *(_QWORD *)(v60 + 32);
                  if ( !v60 )
                    goto LABEL_132;
                  if ( *(_DWORD *)(v60 + 8) == v61 )
                    goto LABEL_43;
                  v62 = 0;
                }
                v60 = *(_QWORD *)(v60 + 32);
              }
              while ( v60 );
              if ( v62 == 1 )
                goto LABEL_43;
LABEL_132:
              v102 = (__int128)v121;
              v117 = v38;
              v124 = v37;
              *(_QWORD *)&v93 = sub_33FAF80(v33, v25, (unsigned int)&v140, v38, v37, v38, v102);
              v92 = sub_3406EB0(
                      v33,
                      *(_DWORD *)(v35 + 24),
                      (unsigned int)&v140,
                      v117,
                      v124,
                      v117,
                      v93,
                      *(_OWORD *)*(_QWORD *)(v56 + 40));
LABEL_126:
              v14 = v92;
              goto LABEL_127;
            }
          }
        }
      }
LABEL_43:
      if ( v140 )
        sub_B91220((__int64)&v140, v140);
      goto LABEL_28;
    }
    v44 = *(_QWORD *)(v5 + 56);
    if ( !v44 )
      goto LABEL_27;
    v53 = *(_QWORD *)(v5 + 56);
    v54 = 1;
    do
    {
      if ( *(_DWORD *)(v53 + 8) == v6 )
      {
        if ( !v54 )
          goto LABEL_52;
        v53 = *(_QWORD *)(v53 + 32);
        if ( !v53 )
          goto LABEL_85;
        if ( v6 == *(_DWORD *)(v53 + 8) )
          goto LABEL_52;
        v54 = 0;
      }
      v53 = *(_QWORD *)(v53 + 32);
    }
    while ( v53 );
    if ( v54 == 1 )
      goto LABEL_52;
LABEL_85:
    v63 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL) + 24LL);
    v122 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL);
    if ( v63 != 11 )
    {
      v64 = v63 == 35;
      v65 = 0;
      if ( v64 )
        v65 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL);
      v122 = v65;
    }
    v66 = (unsigned int)v22 >> 1;
    v67 = 6;
    if ( (unsigned int)v22 >> 1 != 16 )
    {
      v67 = 7;
      if ( v66 != 32 )
      {
        v67 = 8;
        if ( v66 != 64 )
        {
          if ( v66 == 128 )
          {
            v67 = 9;
          }
          else
          {
            v126 = v22;
            v112 = (unsigned int)v22 >> 1;
            v68 = sub_3007020(*(_QWORD **)(*a1 + 64LL), v66);
            v66 = v112;
            LODWORD(v22) = v126;
            v109 = HIWORD(v68);
            v67 = v68;
            v13 = v69;
          }
        }
      }
    }
    HIWORD(v70) = v109;
    LOWORD(v70) = v67;
    v108 = v70;
    if ( v122 )
    {
      v71 = *(_QWORD *)(v122 + 96);
      v113 = *(_DWORD *)(v71 + 32);
      if ( v113 > 0x40 )
      {
        v105 = v66;
        v103 = v13;
        v104 = v22;
        v106 = (unsigned int)v22;
        v128 = *(_QWORD *)(v122 + 96);
        v75 = sub_C444A0(v71 + 24);
        LODWORD(v22) = v104;
        v13 = v103;
        v66 = v105;
        if ( v113 - v75 > 0x40 )
          goto LABEL_97;
        v72 = **(_QWORD **)(v128 + 24);
        if ( v106 <= v72 )
          goto LABEL_97;
      }
      else
      {
        v72 = *(_QWORD *)(v71 + 24);
        if ( (unsigned int)v22 <= v72 )
          goto LABEL_97;
      }
      v114 = (char *)v66;
      if ( v66 <= v72 && (v72 & 0xF) == 0 )
      {
        if ( v67 )
        {
          v76 = a1[1];
          if ( *(_QWORD *)(v76 + 8LL * v67 + 112) )
          {
            v107 = v22;
            v129 = v13;
            v77 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64))(*(_QWORD *)v76 + 1392LL))(
                    v76,
                    v134,
                    v135,
                    v108,
                    v13);
            LODWORD(v22) = v107;
            if ( v77 )
            {
              v13 = v129;
              v78 = v114;
              if ( !*((_BYTE *)a1 + 33)
                || (v79 = v129,
                    v130 = v114,
                    v115 = v13,
                    v80 = sub_328A020(a1[1], 0xC5u, v108, v79, 1u),
                    v13 = v115,
                    v78 = v130,
                    LODWORD(v22) = v107,
                    v80) )
              {
                v81 = *(__int64 **)(v5 + 40);
                v82 = *v81;
                v83 = v81[1];
                v84 = *(_QWORD *)(v122 + 96);
                v85 = *(char **)(v84 + 24);
                if ( *(_DWORD *)(v84 + 32) > 0x40u )
                  v85 = *(char **)v85;
                if ( v85 != v78 )
                {
                  v94 = *a1;
                  v133 = v13;
                  *(_QWORD *)&v95 = sub_3400E40(*a1, v85 - v78, v134, v135, &v136);
                  *((_QWORD *)&v99 + 1) = v83;
                  *(_QWORD *)&v99 = v82;
                  v97 = sub_3406EB0(v94, 190, (unsigned int)&v136, v134, v135, v96, v99, v95);
                  v13 = v133;
                  v82 = v97;
                  v83 = v98 | v83 & 0xFFFFFFFF00000000LL;
                }
                v132 = v13;
                v86 = sub_33FB310(*a1, v82, v83, &v136, v108, v13);
                v88 = v87 | v83 & 0xFFFFFFFF00000000LL;
                *((_QWORD *)&v100 + 1) = v88;
                *(_QWORD *)&v100 = v86;
                v89 = sub_33FAF80(*a1, 197, (unsigned int)&v136, v108, v132, v132, v100);
                v14 = sub_33FB310(*a1, v89, v90 | v88 & 0xFFFFFFFF00000000LL, &v136, v134, v135);
                goto LABEL_5;
              }
            }
          }
        }
      }
    }
LABEL_97:
    v24 = *(_DWORD *)(v5 + 24);
    goto LABEL_26;
  }
  v18 = 1;
  do
  {
    if ( *(_DWORD *)(v17 + 8) == v6 )
    {
      if ( !v18 )
        goto LABEL_19;
      v17 = *(_QWORD *)(v17 + 32);
      if ( !v17 )
        goto LABEL_50;
      if ( v6 == *(_DWORD *)(v17 + 8) )
        goto LABEL_19;
      v18 = 0;
    }
    v17 = *(_QWORD *)(v17 + 32);
  }
  while ( v17 );
  if ( v18 == 1 )
    goto LABEL_19;
LABEL_50:
  *(_QWORD *)&v42 = sub_33FAF80(*a1, 197, (unsigned int)&v136, v134, v135, 0, *(_OWORD *)*(_QWORD *)(v5 + 40));
  v14 = sub_33FAF80(*a1, 201, (unsigned int)&v136, v134, v135, v43, v42);
LABEL_5:
  if ( v136 )
    sub_B91220((__int64)&v136, v136);
  return v14;
}
