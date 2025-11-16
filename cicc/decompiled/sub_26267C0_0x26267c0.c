// Function: sub_26267C0
// Address: 0x26267c0
//
__int64 __fastcall sub_26267C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v5; // r14
  __int64 v6; // r13
  __int64 **v8; // rsi
  __int64 **v9; // r15
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v11; // rbx
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // r14
  __int64 (__fastcall *v14)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v15; // r15
  unsigned __int8 *v16; // rbx
  __int64 (__fastcall *v17)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8); // rax
  unsigned __int8 *v18; // r14
  unsigned __int8 *v19; // r8
  __int64 v20; // rbx
  _DWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // rax
  unsigned __int8 *v25; // r15
  __int64 (__fastcall *v26)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 *v30; // rsi
  unsigned __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // r12
  unsigned int *v37; // rbx
  unsigned int *v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // r12
  int v42; // eax
  int v43; // eax
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // r12
  int v49; // eax
  int v50; // eax
  unsigned int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  unsigned __int64 v55; // rdi
  __int64 v56; // rbx
  unsigned int *v57; // rbx
  unsigned int *v58; // r12
  __int64 v59; // rdx
  unsigned int v60; // esi
  unsigned int *v61; // r14
  unsigned int *v62; // rbx
  __int64 v63; // rdx
  unsigned int v64; // esi
  unsigned int *v65; // r14
  unsigned int *v66; // rbx
  __int64 v67; // rdx
  unsigned int v68; // esi
  int v69; // r14d
  _BYTE *v70; // r14
  __int64 v71; // rax
  __int64 v72; // rbx
  _QWORD *v73; // rax
  __int64 v74; // r10
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rdx
  __int64 v80; // r14
  _QWORD *v81; // rsi
  __int64 v82; // r12
  __int64 v83; // rdi
  __int64 v84; // rax
  unsigned int v85; // r8d
  int v86; // edx
  __int64 v87; // rax
  __int64 v88; // rcx
  int v89; // edx
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rax
  __int64 v94; // rax
  int v95; // r12d
  unsigned __int8 *v96; // [rsp+18h] [rbp-208h]
  _QWORD *v97; // [rsp+18h] [rbp-208h]
  unsigned __int8 *v98; // [rsp+18h] [rbp-208h]
  _QWORD *v99; // [rsp+20h] [rbp-200h]
  _QWORD *v100; // [rsp+40h] [rbp-1E0h]
  __int64 v101; // [rsp+40h] [rbp-1E0h]
  __int64 v102; // [rsp+58h] [rbp-1C8h]
  __int64 v103; // [rsp+58h] [rbp-1C8h]
  _BYTE v106[32]; // [rsp+70h] [rbp-1B0h] BYREF
  __int16 v107; // [rsp+90h] [rbp-190h]
  const char *v108[4]; // [rsp+A0h] [rbp-180h] BYREF
  __int16 v109; // [rsp+C0h] [rbp-160h]
  unsigned int *v110; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v111; // [rsp+D8h] [rbp-148h]
  _BYTE v112[32]; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v113; // [rsp+100h] [rbp-120h]
  __int64 v114; // [rsp+108h] [rbp-118h]
  __int64 v115; // [rsp+110h] [rbp-110h]
  __int64 v116; // [rsp+118h] [rbp-108h]
  void **v117; // [rsp+120h] [rbp-100h]
  void **v118; // [rsp+128h] [rbp-F8h]
  __int64 v119; // [rsp+130h] [rbp-F0h]
  int v120; // [rsp+138h] [rbp-E8h]
  __int16 v121; // [rsp+13Ch] [rbp-E4h]
  char v122; // [rsp+13Eh] [rbp-E2h]
  __int64 v123; // [rsp+140h] [rbp-E0h]
  __int64 v124; // [rsp+148h] [rbp-D8h]
  void *v125; // [rsp+150h] [rbp-D0h] BYREF
  void *v126; // [rsp+158h] [rbp-C8h] BYREF
  _BYTE *v127; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+168h] [rbp-B8h]
  _BYTE v129[16]; // [rsp+170h] [rbp-B0h] BYREF
  __int16 v130; // [rsp+180h] [rbp-A0h]
  __int64 v131; // [rsp+190h] [rbp-90h]
  __int64 v132; // [rsp+198h] [rbp-88h]
  __int16 v133; // [rsp+1A0h] [rbp-80h]
  __int64 v134; // [rsp+1A8h] [rbp-78h]
  void **v135; // [rsp+1B0h] [rbp-70h]
  _QWORD *v136; // [rsp+1B8h] [rbp-68h]
  __int64 v137; // [rsp+1C0h] [rbp-60h]
  int v138; // [rsp+1C8h] [rbp-58h]
  __int16 v139; // [rsp+1CCh] [rbp-54h]
  char v140; // [rsp+1CEh] [rbp-52h]
  __int64 v141; // [rsp+1D0h] [rbp-50h]
  __int64 v142; // [rsp+1D8h] [rbp-48h]
  void *v143; // [rsp+1E0h] [rbp-40h] BYREF
  _QWORD v144[7]; // [rsp+1E8h] [rbp-38h] BYREF

  v4 = (_QWORD *)a3;
  v5 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  v6 = *(_QWORD *)a1 + 312LL;
  if ( (unsigned __int8)sub_2624C30(a2, v6, (unsigned __int8 *)v5, 0) )
    return sub_ACD6D0(**(__int64 ***)a1);
  v99 = (_QWORD *)v4[5];
  v116 = sub_BD5C60((__int64)v4);
  v117 = &v125;
  v118 = &v126;
  v110 = (unsigned int *)v112;
  v125 = &unk_49DA100;
  v111 = 0x200000000LL;
  v126 = &unk_49DA0B0;
  v119 = 0;
  v120 = 0;
  v121 = 512;
  v122 = 7;
  v123 = 0;
  v124 = 0;
  v113 = 0;
  v114 = 0;
  LOWORD(v115) = 0;
  sub_D5F1F0((__int64)&v110, (__int64)v4);
  v109 = 257;
  v8 = *(__int64 ***)(v5 + 8);
  v9 = *(__int64 ***)(a1 + 112);
  if ( v9 == v8 )
  {
    v11 = (unsigned __int8 *)v5;
    goto LABEL_10;
  }
  v10 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v117 + 15);
  if ( v10 != sub_920130 )
  {
    v11 = (unsigned __int8 *)v10((__int64)v117, 47u, (_BYTE *)v5, *(_QWORD *)(a1 + 112));
    goto LABEL_8;
  }
  if ( *(_BYTE *)v5 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v11 = (unsigned __int8 *)sub_ADAB70(47, v5, v9, 0);
    else
      v11 = (unsigned __int8 *)sub_AA93C0(0x2Fu, v5, (__int64)v9);
LABEL_8:
    if ( v11 )
    {
      v8 = *(__int64 ***)(a1 + 112);
      goto LABEL_10;
    }
  }
  v130 = 257;
  v11 = (unsigned __int8 *)sub_B51D30(47, v5, (__int64)v9, (__int64)&v127, 0, 0);
  if ( (unsigned __int8)sub_920620((__int64)v11) )
  {
    v69 = v120;
    if ( v119 )
      sub_B99FD0((__int64)v11, 3u, v119);
    sub_B45150((__int64)v11, v69);
  }
  (*((void (__fastcall **)(void **, unsigned __int8 *, const char **, __int64, __int64))*v118 + 2))(
    v118,
    v11,
    v108,
    v114,
    v115);
  sub_94AAF0(&v110, (__int64)v11);
  v8 = *(__int64 ***)(a1 + 112);
LABEL_10:
  v12 = (unsigned __int8 *)sub_AD4C50(*(_QWORD *)(a4 + 8), v8, 0);
  v13 = v12;
  if ( *(_DWORD *)a4 == 3 )
  {
    v130 = 257;
    v28 = sub_92B530(&v110, 0x20u, (__int64)v11, v12, (__int64)&v127);
    goto LABEL_56;
  }
  v109 = 257;
  v14 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v117 + 4);
  if ( v14 != sub_9201A0 )
  {
    v15 = (unsigned __int8 *)v14((__int64)v117, 15u, v11, v13, 0, 0);
    goto LABEL_16;
  }
  if ( *v11 <= 0x15u && *v13 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(15) )
      v15 = (unsigned __int8 *)sub_AD5570(15, (__int64)v11, v13, 0, 0);
    else
      v15 = (unsigned __int8 *)sub_AABE40(0xFu, v11, v13);
LABEL_16:
    if ( v15 )
      goto LABEL_17;
  }
  v130 = 257;
  v15 = (unsigned __int8 *)sub_B504D0(15, (__int64)v11, (__int64)v13, (__int64)&v127, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, const char **, __int64, __int64))*v118 + 2))(
    v118,
    v15,
    v108,
    v114,
    v115);
  v61 = v110;
  v62 = &v110[4 * (unsigned int)v111];
  if ( v110 != v62 )
  {
    do
    {
      v63 = *((_QWORD *)v61 + 1);
      v64 = *v61;
      v61 += 4;
      sub_B99FD0((__int64)v15, v64, v63);
    }
    while ( v62 != v61 );
  }
LABEL_17:
  v109 = 257;
  v107 = 257;
  v16 = (unsigned __int8 *)sub_A82F30(&v110, *(_QWORD *)(a4 + 16), *(_QWORD *)(a1 + 112), (__int64)v106, 0);
  v17 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8))*((_QWORD *)*v117 + 3);
  if ( v17 != sub_920250 )
  {
    v18 = (unsigned __int8 *)v17((__int64)v117, 26u, v15, v16, 0);
    goto LABEL_22;
  }
  if ( *v15 <= 0x15u && *v16 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(26) )
      v18 = (unsigned __int8 *)sub_AD5570(26, (__int64)v15, v16, 0, 0);
    else
      v18 = (unsigned __int8 *)sub_AABE40(0x1Au, v15, v16);
LABEL_22:
    if ( v18 )
      goto LABEL_23;
  }
  v130 = 257;
  v18 = (unsigned __int8 *)sub_B504D0(26, (__int64)v15, (__int64)v16, (__int64)&v127, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, const char **, __int64, __int64))*v118 + 2))(
    v118,
    v18,
    v108,
    v114,
    v115);
  v56 = 4LL * (unsigned int)v111;
  if ( v110 != &v110[v56] )
  {
    v97 = v4;
    v57 = &v110[v56];
    v58 = v110;
    do
    {
      v59 = *((_QWORD *)v58 + 1);
      v60 = *v58;
      v58 += 4;
      sub_B99FD0((__int64)v18, v60, v59);
    }
    while ( v57 != v58 );
    v4 = v97;
  }
LABEL_23:
  v130 = 257;
  v19 = *(unsigned __int8 **)(a4 + 16);
  v109 = 257;
  v96 = v19;
  v20 = *(_QWORD *)(a1 + 112);
  v21 = sub_AE2980(v6, 0);
  v22 = sub_ACD640(*(_QWORD *)(a1 + 64), (unsigned int)v21[1], 0);
  v23 = sub_AD57F0(v22, v96, 0, 0);
  v24 = (_BYTE *)sub_A82F30(&v110, v23, v20, (__int64)v108, 0);
  v25 = (unsigned __int8 *)sub_920A70(&v110, v15, v24, (__int64)&v127, 0, 0);
  v109 = 257;
  v26 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v117 + 2);
  if ( v26 != sub_9202E0 )
  {
    v27 = v26((__int64)v117, 29u, v18, v25);
    goto LABEL_28;
  }
  if ( *v18 <= 0x15u && *v25 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(29) )
      v27 = sub_AD5570(29, (__int64)v18, v25, 0, 0);
    else
      v27 = sub_AABE40(0x1Du, v18, v25);
LABEL_28:
    if ( v27 )
      goto LABEL_29;
  }
  v130 = 257;
  v27 = sub_B504D0(29, (__int64)v18, (__int64)v25, (__int64)&v127, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v118 + 2))(v118, v27, v108, v114, v115);
  v65 = v110;
  v66 = &v110[4 * (unsigned int)v111];
  if ( v110 != v66 )
  {
    do
    {
      v67 = *((_QWORD *)v65 + 1);
      v68 = *v65;
      v65 += 4;
      sub_B99FD0(v27, v68, v67);
    }
    while ( v66 != v65 );
  }
LABEL_29:
  v130 = 257;
  v28 = sub_92B530(&v110, 0x25u, v27, *(_BYTE **)(a4 + 24), (__int64)&v127);
  if ( *(_DWORD *)a4 != 4 )
  {
    v29 = v4[2];
    v30 = v4 + 3;
    if ( v29
      && !*(_QWORD *)(v29 + 8)
      && (v70 = *(_BYTE **)(v29 + 24), *v70 == 31)
      && (v71 = v4[4], v71 != v4[5] + 48LL)
      && v71
      && v70 == (_BYTE *)(v71 - 24) )
    {
      v130 = 257;
      v72 = sub_AA8550(v99, v30, 0, (__int64)&v127, 0);
      v102 = *((_QWORD *)v70 - 8);
      v73 = sub_BD2C40(72, 3u);
      v74 = (__int64)v73;
      if ( v73 )
      {
        v100 = v73;
        sub_B4C9A0((__int64)v73, v72, v102, v28, 3u, 0, 0, 0);
        v74 = (__int64)v100;
      }
      if ( (v70[7] & 0x20) != 0 )
      {
        v101 = v74;
        v75 = sub_B91C10((__int64)v70, 2);
        v74 = v101;
        v76 = v75;
      }
      else
      {
        v76 = 0;
      }
      v98 = (unsigned __int8 *)v74;
      sub_B99FD0(v74, 2u, v76);
      v77 = v99[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v77 == v99 + 6 )
      {
        v78 = 0;
      }
      else
      {
        if ( !v77 )
LABEL_127:
          BUG();
        v78 = v77 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v77 - 24) - 30 >= 0xB )
          v78 = 0;
      }
      sub_F34910(v78, v98);
      v80 = sub_AA5930(v102);
      if ( v80 != v79 )
      {
        v81 = v4;
        v82 = v79;
        do
        {
          v83 = *(_QWORD *)(v80 - 8);
          v84 = 0x1FFFFFFFE0LL;
          v85 = *(_DWORD *)(v80 + 72);
          v86 = *(_DWORD *)(v80 + 4) & 0x7FFFFFF;
          if ( v86 )
          {
            v87 = 0;
            do
            {
              if ( v72 == *(_QWORD *)(v83 + 32LL * v85 + 8 * v87) )
              {
                v84 = 32 * v87;
                goto LABEL_97;
              }
              ++v87;
            }
            while ( v86 != (_DWORD)v87 );
            v84 = 0x1FFFFFFFE0LL;
          }
LABEL_97:
          v88 = *(_QWORD *)(v83 + v84);
          if ( v86 == v85 )
          {
            v103 = *(_QWORD *)(v83 + v84);
            sub_B48D90(v80);
            v83 = *(_QWORD *)(v80 - 8);
            v88 = v103;
            v86 = *(_DWORD *)(v80 + 4) & 0x7FFFFFF;
          }
          v89 = (v86 + 1) & 0x7FFFFFF;
          *(_DWORD *)(v80 + 4) = v89 | *(_DWORD *)(v80 + 4) & 0xF8000000;
          v90 = v83 + 32LL * (unsigned int)(v89 - 1);
          if ( *(_QWORD *)v90 )
          {
            v91 = *(_QWORD *)(v90 + 8);
            **(_QWORD **)(v90 + 16) = v91;
            if ( v91 )
              *(_QWORD *)(v91 + 16) = *(_QWORD *)(v90 + 16);
          }
          *(_QWORD *)v90 = v88;
          if ( v88 )
          {
            v92 = *(_QWORD *)(v88 + 16);
            *(_QWORD *)(v90 + 8) = v92;
            if ( v92 )
              *(_QWORD *)(v92 + 16) = v90 + 8;
            *(_QWORD *)(v90 + 16) = v88 + 16;
            *(_QWORD *)(v88 + 16) = v90;
          }
          *(_QWORD *)(*(_QWORD *)(v80 - 8)
                    + 32LL * *(unsigned int *)(v80 + 72)
                    + 8LL * ((*(_DWORD *)(v80 + 4) & 0x7FFFFFFu) - 1)) = v99;
          v93 = *(_QWORD *)(v80 + 32);
          if ( !v93 )
            goto LABEL_127;
          v80 = 0;
          if ( *(_BYTE *)(v93 - 24) == 84 )
            v80 = v93 - 24;
        }
        while ( v82 != v80 );
        v4 = v81;
      }
      v94 = sub_BD5C60((__int64)v4);
      v135 = &v143;
      v134 = v94;
      v136 = v144;
      v143 = &unk_49DA100;
      v128 = 0x200000000LL;
      v144[0] = &unk_49DA0B0;
      v127 = v129;
      v137 = 0;
      v138 = 0;
      v139 = 512;
      v140 = 7;
      v141 = 0;
      v142 = 0;
      v131 = 0;
      v132 = 0;
      v133 = 0;
      sub_D5F1F0((__int64)&v127, (__int64)v4);
      v28 = sub_2624F60((__int64 *)a1, (__int64)&v127, a4, v27);
      nullsub_61();
      v143 = &unk_49DA100;
      nullsub_63();
      v55 = (unsigned __int64)v127;
      if ( v127 == v129 )
        goto LABEL_56;
    }
    else
    {
      v31 = sub_F38250(v28, v30, 0, 0, 0, 0, 0, 0);
      v32 = sub_BD5C60(v31);
      v135 = &v143;
      v134 = v32;
      v136 = v144;
      v139 = 512;
      v127 = v129;
      v143 = &unk_49DA100;
      v128 = 0x200000000LL;
      v133 = 0;
      v144[0] = &unk_49DA0B0;
      v137 = 0;
      v138 = 0;
      v140 = 7;
      v141 = 0;
      v142 = 0;
      v131 = 0;
      v132 = 0;
      sub_D5F1F0((__int64)&v127, v31);
      v33 = sub_2624F60((__int64 *)a1, (__int64)&v127, a4, v27);
      sub_D5F1F0((__int64)&v110, (__int64)v4);
      v107 = 257;
      v34 = *(_QWORD *)(a1 + 56);
      v109 = 257;
      v35 = sub_BD2DA0(80);
      v28 = v35;
      if ( v35 )
      {
        v36 = v35;
        sub_B44260(v35, v34, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v28 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v28, v108);
        sub_BD2A10(v28, *(_DWORD *)(v28 + 72), 1);
      }
      else
      {
        v36 = 0;
      }
      if ( (unsigned __int8)sub_920620(v36) )
      {
        v95 = v120;
        if ( v119 )
          sub_B99FD0(v28, 3u, v119);
        sub_B45150(v28, v95);
      }
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v118 + 2))(v118, v28, v106, v114, v115);
      v37 = v110;
      v38 = &v110[4 * (unsigned int)v111];
      if ( v110 != v38 )
      {
        do
        {
          v39 = *((_QWORD *)v37 + 1);
          v40 = *v37;
          v37 += 4;
          sub_B99FD0(v28, v40, v39);
        }
        while ( v38 != v37 );
      }
      v41 = sub_ACD640(*(_QWORD *)(a1 + 56), 0, 0);
      v42 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
      if ( v42 == *(_DWORD *)(v28 + 72) )
      {
        sub_B48D90(v28);
        v42 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
      }
      v43 = (v42 + 1) & 0x7FFFFFF;
      v44 = v43 | *(_DWORD *)(v28 + 4) & 0xF8000000;
      v45 = *(_QWORD *)(v28 - 8) + 32LL * (unsigned int)(v43 - 1);
      *(_DWORD *)(v28 + 4) = v44;
      if ( *(_QWORD *)v45 )
      {
        v46 = *(_QWORD *)(v45 + 8);
        **(_QWORD **)(v45 + 16) = v46;
        if ( v46 )
          *(_QWORD *)(v46 + 16) = *(_QWORD *)(v45 + 16);
      }
      *(_QWORD *)v45 = v41;
      if ( v41 )
      {
        v47 = *(_QWORD *)(v41 + 16);
        *(_QWORD *)(v45 + 8) = v47;
        if ( v47 )
          *(_QWORD *)(v47 + 16) = v45 + 8;
        *(_QWORD *)(v45 + 16) = v41 + 16;
        *(_QWORD *)(v41 + 16) = v45;
      }
      *(_QWORD *)(*(_QWORD *)(v28 - 8)
                + 32LL * *(unsigned int *)(v28 + 72)
                + 8LL * ((*(_DWORD *)(v28 + 4) & 0x7FFFFFFu) - 1)) = v99;
      v48 = v131;
      v49 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
      if ( v49 == *(_DWORD *)(v28 + 72) )
      {
        sub_B48D90(v28);
        v49 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
      }
      v50 = (v49 + 1) & 0x7FFFFFF;
      v51 = v50 | *(_DWORD *)(v28 + 4) & 0xF8000000;
      v52 = *(_QWORD *)(v28 - 8) + 32LL * (unsigned int)(v50 - 1);
      *(_DWORD *)(v28 + 4) = v51;
      if ( *(_QWORD *)v52 )
      {
        v53 = *(_QWORD *)(v52 + 8);
        **(_QWORD **)(v52 + 16) = v53;
        if ( v53 )
          *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
      }
      *(_QWORD *)v52 = v33;
      if ( v33 )
      {
        v54 = *(_QWORD *)(v33 + 16);
        *(_QWORD *)(v52 + 8) = v54;
        if ( v54 )
          *(_QWORD *)(v54 + 16) = v52 + 8;
        *(_QWORD *)(v52 + 16) = v33 + 16;
        *(_QWORD *)(v33 + 16) = v52;
      }
      *(_QWORD *)(*(_QWORD *)(v28 - 8)
                + 32LL * *(unsigned int *)(v28 + 72)
                + 8LL * ((*(_DWORD *)(v28 + 4) & 0x7FFFFFFu) - 1)) = v48;
      nullsub_61();
      v143 = &unk_49DA100;
      nullsub_63();
      v55 = (unsigned __int64)v127;
      if ( v127 == v129 )
        goto LABEL_56;
    }
    _libc_free(v55);
  }
LABEL_56:
  nullsub_61();
  v125 = &unk_49DA100;
  nullsub_63();
  if ( v110 != (unsigned int *)v112 )
    _libc_free((unsigned __int64)v110);
  return v28;
}
