// Function: sub_2B3E180
// Address: 0x2b3e180
//
void *__fastcall sub_2B3E180(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7)
{
  __int64 *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD *v16; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // r14
  _QWORD *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rcx
  unsigned int v39; // eax
  _QWORD *v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  char v43; // dl
  __int64 v44; // rax
  __int64 v45; // rax
  _DWORD *v46; // rax
  __int64 i; // rdx
  int *v48; // rdi
  int v49; // ebx
  __m128i v50; // xmm1
  __m128i v51; // xmm0
  __int64 v52; // rdi
  char *v53; // rdi
  void (__fastcall *v54)(_BYTE *, __m128i *, __int64); // rax
  _DWORD *v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // rax
  _QWORD *v58; // rax
  __int64 *v59; // [rsp+8h] [rbp-538h]
  _QWORD *v60; // [rsp+8h] [rbp-538h]
  __int64 *v63; // [rsp+30h] [rbp-510h]
  __int64 v65; // [rsp+40h] [rbp-500h]
  _QWORD *v66; // [rsp+50h] [rbp-4F0h]
  _QWORD *v67; // [rsp+50h] [rbp-4F0h]
  bool v68; // [rsp+50h] [rbp-4F0h]
  __int64 *v69; // [rsp+58h] [rbp-4E8h]
  unsigned int v70; // [rsp+58h] [rbp-4E8h]
  int v71; // [rsp+7Ch] [rbp-4C4h] BYREF
  __m128i v72; // [rsp+80h] [rbp-4C0h] BYREF
  void (__fastcall *v73)(__m128i *, __m128i *, __int64); // [rsp+90h] [rbp-4B0h]
  char (__fastcall *v74)(__int64 *, __int64 *); // [rsp+98h] [rbp-4A8h]
  void *v75; // [rsp+A0h] [rbp-4A0h] BYREF
  __m128i v76; // [rsp+A8h] [rbp-498h] BYREF
  void (__fastcall *v77)(_BYTE *, __m128i *, __int64); // [rsp+B8h] [rbp-488h]
  char (__fastcall *v78)(__int64 *, __int64 *); // [rsp+C0h] [rbp-480h]
  __int64 v79; // [rsp+D0h] [rbp-470h] BYREF
  int v80; // [rsp+D8h] [rbp-468h] BYREF
  unsigned __int64 v81; // [rsp+E0h] [rbp-460h]
  int *v82; // [rsp+E8h] [rbp-458h]
  int *v83; // [rsp+F0h] [rbp-450h]
  __int64 v84; // [rsp+F8h] [rbp-448h]
  __int64 *v85; // [rsp+100h] [rbp-440h] BYREF
  __int64 v86; // [rsp+108h] [rbp-438h]
  _BYTE v87[48]; // [rsp+110h] [rbp-430h] BYREF
  void *v88; // [rsp+140h] [rbp-400h]
  void *v89; // [rsp+148h] [rbp-3F8h]
  __int64 v90; // [rsp+150h] [rbp-3F0h]
  _QWORD v91[8]; // [rsp+158h] [rbp-3E8h] BYREF
  __int16 v92; // [rsp+198h] [rbp-3A8h]
  const char **v93; // [rsp+1A0h] [rbp-3A0h] BYREF
  __int64 v94; // [rsp+1A8h] [rbp-398h]
  const char *v95; // [rsp+1B0h] [rbp-390h] BYREF
  __int64 v96; // [rsp+1B8h] [rbp-388h]
  __int64 v97; // [rsp+1C0h] [rbp-380h]
  __int64 v98; // [rsp+1C8h] [rbp-378h]
  __int64 v99; // [rsp+1D0h] [rbp-370h]
  int v100; // [rsp+1D8h] [rbp-368h]
  __int64 v101; // [rsp+1E0h] [rbp-360h]
  __int64 v102; // [rsp+1E8h] [rbp-358h]
  __int64 v103; // [rsp+1F0h] [rbp-350h]
  __int64 v104; // [rsp+1F8h] [rbp-348h]
  __int64 v105; // [rsp+200h] [rbp-340h]
  __int64 v106; // [rsp+208h] [rbp-338h]
  __int64 v107; // [rsp+210h] [rbp-330h]
  __int64 v108; // [rsp+218h] [rbp-328h]
  __int64 v109; // [rsp+220h] [rbp-320h]
  char *v110; // [rsp+228h] [rbp-318h]
  __int64 v111; // [rsp+230h] [rbp-310h]
  int v112; // [rsp+238h] [rbp-308h]
  char v113; // [rsp+23Ch] [rbp-304h]
  char v114; // [rsp+240h] [rbp-300h] BYREF
  __int64 v115; // [rsp+2C0h] [rbp-280h]
  __int64 v116; // [rsp+2C8h] [rbp-278h]
  __int64 v117; // [rsp+2D0h] [rbp-270h]
  int v118; // [rsp+2D8h] [rbp-268h]
  char *v119; // [rsp+2E0h] [rbp-260h]
  __int64 v120; // [rsp+2E8h] [rbp-258h]
  char v121; // [rsp+2F0h] [rbp-250h] BYREF
  __int64 v122; // [rsp+320h] [rbp-220h]
  __int64 v123; // [rsp+328h] [rbp-218h]
  __int64 v124; // [rsp+330h] [rbp-210h]
  int v125; // [rsp+338h] [rbp-208h]
  __int64 v126; // [rsp+340h] [rbp-200h]
  char *v127; // [rsp+348h] [rbp-1F8h]
  __int64 v128; // [rsp+350h] [rbp-1F0h]
  int v129; // [rsp+358h] [rbp-1E8h]
  char v130; // [rsp+35Ch] [rbp-1E4h]
  char v131; // [rsp+360h] [rbp-1E0h] BYREF
  __int64 v132; // [rsp+370h] [rbp-1D0h]
  __int64 v133; // [rsp+378h] [rbp-1C8h]
  __int64 v134; // [rsp+380h] [rbp-1C0h]
  __int64 v135; // [rsp+388h] [rbp-1B8h]
  __int64 v136; // [rsp+390h] [rbp-1B0h]
  __int64 v137; // [rsp+398h] [rbp-1A8h]
  __int16 v138; // [rsp+3A0h] [rbp-1A0h]
  char v139; // [rsp+3A2h] [rbp-19Eh]
  char *v140; // [rsp+3A8h] [rbp-198h]
  __int64 v141; // [rsp+3B0h] [rbp-190h]
  char v142; // [rsp+3B8h] [rbp-188h] BYREF
  __int64 v143; // [rsp+3D8h] [rbp-168h]
  __int64 v144; // [rsp+3E0h] [rbp-160h]
  __int16 v145; // [rsp+3E8h] [rbp-158h]
  __int64 v146; // [rsp+3F0h] [rbp-150h]
  _QWORD *v147; // [rsp+3F8h] [rbp-148h]
  void **v148; // [rsp+400h] [rbp-140h]
  __int64 v149; // [rsp+408h] [rbp-138h]
  int v150; // [rsp+410h] [rbp-130h]
  __int16 v151; // [rsp+414h] [rbp-12Ch]
  char v152; // [rsp+416h] [rbp-12Ah]
  __int64 v153; // [rsp+418h] [rbp-128h]
  __int64 v154; // [rsp+420h] [rbp-120h]
  _QWORD v155[3]; // [rsp+428h] [rbp-118h] BYREF
  char v156; // [rsp+440h] [rbp-100h] BYREF
  void *v157; // [rsp+488h] [rbp-B8h] BYREF
  _BYTE v158[16]; // [rsp+490h] [rbp-B0h] BYREF
  __int64 (__fastcall *v159)(_QWORD *, _QWORD *, int); // [rsp+4A0h] [rbp-A0h]
  char (__fastcall *v160)(__int64 *, __int64 *); // [rsp+4A8h] [rbp-98h]
  char *v161; // [rsp+4B0h] [rbp-90h]
  __int64 v162; // [rsp+4B8h] [rbp-88h]
  char v163; // [rsp+4C0h] [rbp-80h] BYREF
  const char *v164; // [rsp+500h] [rbp-40h]

  v85 = (__int64 *)v87;
  v86 = 0x600000000LL;
  v69 = &a1[a2];
  if ( a1 != v69 )
  {
    v8 = a1;
    v9 = 0;
    v10 = 0;
    while ( 1 )
    {
      v13 = sub_DD8400((__int64)a5, *v8);
      if ( !v13 )
        goto LABEL_15;
      v14 = (unsigned int)v86;
      v15 = (unsigned int)v86 + 1LL;
      if ( v15 > HIDWORD(v86) )
      {
        sub_C8D5F0((__int64)&v85, v87, v15, 8u, v11, v12);
        v14 = (unsigned int)v86;
      }
      v85[v14] = (__int64)v13;
      LODWORD(v86) = v86 + 1;
      if ( v9 | v10 )
      {
        v66 = sub_DCC810(a5, (__int64)v13, v10, 0, 0);
        if ( sub_D96A50((__int64)v66) )
          goto LABEL_15;
        if ( sub_D969D0((__int64)v66) )
        {
          v10 = (__int64)v13;
        }
        else
        {
          v67 = sub_DCC810(a5, v9, (__int64)v13, 0, 0);
          if ( sub_D96A50((__int64)v67) )
            goto LABEL_15;
          if ( sub_D969D0((__int64)v67) )
            v9 = (__int64)v13;
        }
        if ( v69 == ++v8 )
          goto LABEL_14;
      }
      else
      {
        v9 = (__int64)v13;
        v10 = (__int64)v13;
        if ( v69 == ++v8 )
          goto LABEL_14;
      }
    }
  }
  v9 = 0;
  v10 = 0;
LABEL_14:
  v16 = sub_DCC810(a5, v9, v10, 0, 0);
  if ( sub_D96A50((__int64)v16) )
    goto LABEL_15;
  v18 = sub_9208B0(a4, a3);
  v94 = v19;
  v93 = (const char **)((unsigned __int64)(v18 + 7) >> 3);
  v70 = sub_CA1930(&v93);
  if ( v70 == 1 && (unsigned int)v86 <= 2 )
    goto LABEL_15;
  v20 = (int)v70 * ((unsigned int)v86 - 1LL);
  v21 = sub_D95540((__int64)v16);
  v22 = sub_DA2C50((__int64)a5, v21, v20, 0);
  v23 = v22;
  if ( *((_WORD *)v16 + 12) == 6 )
  {
    v24 = (__int64 *)v16[4];
    v25 = *v24;
    v26 = v24[1];
    if ( v23 == (_QWORD *)v25 )
    {
      v25 = v26;
    }
    else if ( v23 != (_QWORD *)v26 )
    {
LABEL_15:
      LOBYTE(v89) = 0;
      goto LABEL_16;
    }
  }
  else if ( v16 == v22 )
  {
    v44 = sub_D95540((__int64)v16);
    v25 = (__int64)sub_DA2C50((__int64)a5, v44, 1, 0);
  }
  else
  {
    v25 = sub_DCC290(a5, (__int64)v16, (__int64)v22);
  }
  if ( !v25 || !*(_WORD *)(v25 + 24) )
    goto LABEL_15;
  v80 = 0;
  v82 = &v80;
  v83 = &v80;
  v81 = 0;
  v84 = 0;
  v71 = 0;
  v63 = &v85[(unsigned int)v86];
  if ( v85 != v63 )
  {
    v68 = 1;
    v27 = v85;
    v65 = v25;
    do
    {
      v28 = *v27;
      LODWORD(v75) = 0;
      if ( v28 == v10 )
      {
        v39 = 0;
      }
      else
      {
        v29 = sub_DCC810(a5, v28, v10, 0, 0);
        if ( *((_WORD *)v29 + 12) == 6 )
        {
          v30 = (__int64 *)v29[4];
          v31 = *v30;
          v32 = v30[1];
          if ( v65 == v31 )
          {
            v31 = v32;
          }
          else if ( v65 != v32 )
          {
            goto LABEL_50;
          }
        }
        else if ( v29 == (_QWORD *)v65 )
        {
          v45 = sub_D95540(v65);
          v31 = (__int64)sub_DA2C50((__int64)a5, v45, 1, 0);
        }
        else
        {
          v31 = sub_DCC290(a5, (__int64)v29, v65);
        }
        if ( !v31 || *(_WORD *)(v31 + 24) || sub_D96A50(v31) )
          goto LABEL_50;
        v96 = v31;
        v95 = (const char *)v65;
        v93 = &v95;
        v94 = 0x200000002LL;
        v33 = sub_DC8BD0(a5, (__int64)&v93, 0, 0);
        if ( v93 != &v95 )
        {
          v59 = v33;
          _libc_free((unsigned __int64)v93);
          v33 = v59;
        }
        v96 = (__int64)v33;
        v93 = &v95;
        v95 = (const char *)v10;
        v94 = 0x200000002LL;
        v34 = sub_DC7EB0(a5, (__int64)&v93, 0, 0);
        v35 = (__int64)v34;
        if ( v93 != &v95 )
        {
          v60 = v34;
          _libc_free((unsigned __int64)v93);
          v35 = (__int64)v60;
        }
        v36 = sub_DCC810(a5, v28, v35, 0, 0);
        if ( !sub_D968A0((__int64)v36) )
          goto LABEL_50;
        v37 = *(_QWORD *)(v31 + 32);
        v38 = *(_QWORD **)(v37 + 24);
        if ( *(_DWORD *)(v37 + 32) > 0x40u )
          v38 = (_QWORD *)*v38;
        LODWORD(v75) = (_DWORD)v38;
        v39 = (unsigned int)v38 / v70;
        if ( (_DWORD)v38 != (unsigned int)v38 / v70 * v70 )
          goto LABEL_50;
      }
      if ( (unsigned int)v86 <= v39 )
        goto LABEL_50;
      v40 = sub_2B3E050(&v79, (unsigned int *)&v75, &v71);
      if ( !v43 )
        goto LABEL_50;
      if ( v68 )
        v68 = &v80 == (int *)sub_220EF30((__int64)v40);
      ++v71;
      ++v27;
    }
    while ( v63 != v27 );
    v25 = v65;
    if ( (unsigned int)v86 != v84 )
      goto LABEL_50;
    *(_DWORD *)(a6 + 8) = 0;
    if ( !v68 )
    {
      if ( a2 )
      {
        if ( a2 > *(unsigned int *)(a6 + 12) )
          sub_C8D5F0(a6, (const void *)(a6 + 16), a2, 4u, v41, v42);
        v46 = (_DWORD *)(*(_QWORD *)a6 + 4LL * *(unsigned int *)(a6 + 8));
        for ( i = *(_QWORD *)a6 + 4 * a2; (_DWORD *)i != v46; ++v46 )
        {
          if ( v46 )
            *v46 = 0;
        }
        *(_DWORD *)(a6 + 8) = a2;
      }
      v71 = 0;
      v48 = v82;
      if ( v82 != &v80 )
      {
        v49 = 0;
        do
        {
          *(_DWORD *)(*(_QWORD *)a6 + 4LL * v49) = v48[10];
          v49 = ++v71;
          v48 = (int *)sub_220EF30((__int64)v48);
        }
        while ( v48 != &v80 );
      }
    }
LABEL_74:
    if ( a7 )
    {
      v93 = (const char **)a5;
      v95 = "strided-load-vec";
      v110 = &v114;
      v119 = &v121;
      v120 = 0x200000000LL;
      v94 = a4;
      LOBYTE(v96) = 1;
      v97 = 0;
      v98 = 0;
      v99 = 0;
      v100 = 0;
      v101 = 0;
      v102 = 0;
      v103 = 0;
      v104 = 0;
      v105 = 0;
      v106 = 0;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      v111 = 16;
      v112 = 0;
      v113 = 1;
      v115 = 0;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v122 = 0;
      v123 = 0;
      v124 = 0;
      v125 = 0;
      v126 = 0;
      v127 = &v131;
      v50 = _mm_loadu_si128(&v76);
      v72.m128i_i64[0] = (__int64)&v93;
      v51 = _mm_load_si128(&v72);
      v89 = &unk_49D94D0;
      v77 = (void (__fastcall *)(_BYTE *, __m128i *, __int64))sub_27BFDD0;
      v72 = v50;
      v74 = v78;
      v76 = v51;
      v78 = sub_27BFD20;
      v128 = 2;
      v129 = 0;
      v88 = &unk_49E5698;
      v130 = 1;
      v132 = 0;
      v133 = 0;
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v137 = 0;
      v138 = 1;
      v139 = 0;
      v75 = &unk_49DA0D8;
      v73 = 0;
      v90 = a4;
      v52 = *a5;
      v91[0] = a4;
      memset(&v91[1], 0, 56);
      v92 = 257;
      v146 = sub_B2BE50(v52);
      v147 = v155;
      v148 = &v157;
      v140 = &v142;
      v53 = &v156;
      v155[2] = v90;
      v54 = v77;
      v155[0] = &unk_49E5698;
      v55 = v91;
      v155[1] = &unk_49D94D0;
      v56 = 18;
      v141 = 0x200000000LL;
      v149 = 0;
      v150 = 0;
      v151 = 512;
      v152 = 7;
      v153 = 0;
      v154 = 0;
      v143 = 0;
      v144 = 0;
      v145 = 0;
      while ( v56 )
      {
        *(_DWORD *)v53 = *v55++;
        v53 += 4;
        --v56;
      }
      v157 = &unk_49DA0D8;
      v159 = 0;
      if ( v54 )
      {
        v54(v158, &v76, 2);
        v160 = v78;
        v159 = (__int64 (__fastcall *)(_QWORD *, _QWORD *, int))v77;
      }
      v88 = &unk_49E5698;
      v89 = &unk_49D94D0;
      nullsub_63();
      nullsub_63();
      sub_B32BF0(&v75);
      if ( v73 )
        v73(&v72, &v72, 3);
      v161 = &v163;
      v162 = 0x800000000LL;
      v164 = byte_3F871B3;
      v57 = sub_D95540(v25);
      v58 = sub_F8DB90((__int64)&v93, v25, v57, a7 + 24, 0);
      LOBYTE(v89) = 1;
      v88 = v58;
      sub_27C20B0((__int64)&v93);
    }
    else
    {
      v88 = 0;
      LOBYTE(v89) = 1;
    }
    goto LABEL_51;
  }
  if ( !(_DWORD)v86 )
  {
    *(_DWORD *)(a6 + 8) = 0;
    goto LABEL_74;
  }
LABEL_50:
  LOBYTE(v89) = 0;
LABEL_51:
  sub_2B10510(v81);
LABEL_16:
  if ( v85 != (__int64 *)v87 )
    _libc_free((unsigned __int64)v85);
  return v88;
}
