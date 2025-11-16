// Function: sub_2701220
// Address: 0x2701220
//
void __fastcall sub_2701220(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdi
  bool v12; // zf
  _QWORD *v13; // r13
  __int64 v14; // r14
  __int64 v15; // rax
  char v16; // r15
  _QWORD *v17; // rax
  _BYTE *v18; // r12
  __int64 v19; // r13
  _BYTE *v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // esi
  _QWORD **v23; // r13
  _QWORD **v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // r13
  __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  _QWORD **v30; // r15
  __int64 v31; // r8
  _BYTE *v32; // r14
  __int64 v33; // r9
  _QWORD **v34; // rbx
  _QWORD *v35; // r12
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // r15
  _QWORD *v40; // rax
  __int64 v41; // r12
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rsi
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  char v48; // di
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 *v51; // r15
  __int64 v52; // rax
  unsigned __int8 *v53; // rbx
  _QWORD *v54; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  __int64 v57; // r14
  __int64 v58; // rax
  char v59; // al
  char v60; // r15
  _QWORD *v61; // rax
  __int64 v62; // r9
  unsigned __int64 v63; // r13
  __int64 v64; // r15
  _BYTE *v65; // rbx
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 **v68; // rcx
  unsigned __int64 v69; // rax
  __int64 **v70; // rcx
  unsigned __int8 *v71; // rbx
  unsigned __int64 v72; // rax
  unsigned __int8 *v73; // r13
  __int64 (__fastcall *v74)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v75; // r15
  unsigned __int64 v76; // rdi
  __int64 v77; // r13
  _BYTE *v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  __int64 v81; // [rsp-10h] [rbp-340h]
  __int64 v82; // [rsp-10h] [rbp-340h]
  __int64 v83; // [rsp-8h] [rbp-338h]
  __int64 v85; // [rsp+68h] [rbp-2C8h]
  unsigned __int64 v87; // [rsp+80h] [rbp-2B0h]
  __int64 v88; // [rsp+88h] [rbp-2A8h]
  _BYTE *v89; // [rsp+88h] [rbp-2A8h]
  __int64 v90; // [rsp+88h] [rbp-2A8h]
  __int64 *v91; // [rsp+88h] [rbp-2A8h]
  __int64 v92; // [rsp+90h] [rbp-2A0h]
  __int64 v93; // [rsp+98h] [rbp-298h]
  __int64 v94; // [rsp+A0h] [rbp-290h]
  __int64 v95; // [rsp+A8h] [rbp-288h]
  char v96; // [rsp+BBh] [rbp-275h] BYREF
  int v97; // [rsp+BCh] [rbp-274h] BYREF
  _BYTE *v98; // [rsp+C0h] [rbp-270h] BYREF
  __int64 v99; // [rsp+C8h] [rbp-268h]
  _BYTE v100[16]; // [rsp+D0h] [rbp-260h] BYREF
  _BYTE *v101; // [rsp+E0h] [rbp-250h] BYREF
  __int64 v102; // [rsp+E8h] [rbp-248h]
  _BYTE v103[16]; // [rsp+F0h] [rbp-240h] BYREF
  __int64 *v104; // [rsp+100h] [rbp-230h] BYREF
  __int64 v105; // [rsp+108h] [rbp-228h]
  _BYTE v106[16]; // [rsp+110h] [rbp-220h] BYREF
  __m128i v107[2]; // [rsp+120h] [rbp-210h] BYREF
  __int16 v108; // [rsp+140h] [rbp-1F0h]
  _BYTE *v109; // [rsp+150h] [rbp-1E0h] BYREF
  __int64 v110; // [rsp+158h] [rbp-1D8h]
  _BYTE v111[32]; // [rsp+160h] [rbp-1D0h] BYREF
  __int64 v112; // [rsp+180h] [rbp-1B0h]
  __int64 v113; // [rsp+188h] [rbp-1A8h]
  __int64 v114; // [rsp+190h] [rbp-1A0h]
  __int64 v115; // [rsp+198h] [rbp-198h]
  void **v116; // [rsp+1A0h] [rbp-190h]
  void **v117; // [rsp+1A8h] [rbp-188h]
  __int64 v118; // [rsp+1B0h] [rbp-180h]
  int v119; // [rsp+1B8h] [rbp-178h]
  __int16 v120; // [rsp+1BCh] [rbp-174h]
  char v121; // [rsp+1BEh] [rbp-172h]
  __int64 v122; // [rsp+1C0h] [rbp-170h]
  __int64 v123; // [rsp+1C8h] [rbp-168h]
  void *v124; // [rsp+1D0h] [rbp-160h] BYREF
  void *v125; // [rsp+1D8h] [rbp-158h] BYREF
  unsigned int *v126[2]; // [rsp+1E0h] [rbp-150h] BYREF
  _BYTE v127[16]; // [rsp+1F0h] [rbp-140h] BYREF
  __int16 v128; // [rsp+200h] [rbp-130h]
  __int64 v129; // [rsp+210h] [rbp-120h]
  __int64 v130; // [rsp+218h] [rbp-118h]
  __int16 v131; // [rsp+220h] [rbp-110h]
  __int64 v132; // [rsp+228h] [rbp-108h]
  void **v133; // [rsp+230h] [rbp-100h]
  void **v134; // [rsp+238h] [rbp-F8h]
  __int64 v135; // [rsp+240h] [rbp-F0h]
  int v136; // [rsp+248h] [rbp-E8h]
  __int16 v137; // [rsp+24Ch] [rbp-E4h]
  char v138; // [rsp+24Eh] [rbp-E2h]
  __int64 v139; // [rsp+250h] [rbp-E0h]
  __int64 v140; // [rsp+258h] [rbp-D8h]
  void *v141; // [rsp+260h] [rbp-D0h] BYREF
  void *v142; // [rsp+268h] [rbp-C8h] BYREF
  __m128i v143; // [rsp+270h] [rbp-C0h] BYREF
  _QWORD v144[2]; // [rsp+280h] [rbp-B0h] BYREF
  __int16 v145; // [rsp+290h] [rbp-A0h]
  __int64 v146; // [rsp+2A0h] [rbp-90h]
  __int64 v147; // [rsp+2A8h] [rbp-88h]
  __int16 v148; // [rsp+2B0h] [rbp-80h]
  __int64 v149; // [rsp+2B8h] [rbp-78h]
  void **v150; // [rsp+2C0h] [rbp-70h]
  void **v151; // [rsp+2C8h] [rbp-68h]
  __int64 v152; // [rsp+2D0h] [rbp-60h]
  int v153; // [rsp+2D8h] [rbp-58h]
  __int16 v154; // [rsp+2DCh] [rbp-54h]
  char v155; // [rsp+2DEh] [rbp-52h]
  __int64 v156; // [rsp+2E0h] [rbp-50h]
  __int64 v157; // [rsp+2E8h] [rbp-48h]
  void *v158; // [rsp+2F0h] [rbp-40h] BYREF
  void *v159; // [rsp+2F8h] [rbp-38h] BYREF

  v85 = sub_B6E160(*(__int64 **)a1, 0x166u, 0, 0);
  v95 = *(_QWORD *)(a2 + 16);
  while ( v95 )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(v95 + 24);
      v95 = *(_QWORD *)(v95 + 8);
      v94 = v2;
      if ( *(_BYTE *)v2 == 85 )
        break;
LABEL_3:
      if ( !v95 )
        return;
    }
    v3 = v2;
    v4 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
    v93 = *(_QWORD *)(v3 - 32 * v4);
    v5 = *(_QWORD *)(v3 + 32 * (1 - v4));
    v6 = *(_QWORD *)(v3 + 32 * (2 - v4));
    v98 = v100;
    v101 = v103;
    v88 = v6;
    v7 = *(_QWORD *)(v6 + 24);
    v96 = 0;
    v92 = v7;
    v104 = (__int64 *)v106;
    v105 = 0x100000000LL;
    v99 = 0x100000000LL;
    v102 = 0x100000000LL;
    v8 = sub_B43CB0(v3);
    v9 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 24))(*(_QWORD *)(a1 + 32), v8);
    v10 = v3;
    sub_E025C0((__int64)&v104, (__int64)&v98, (__int64)&v101, &v96, v3, v9);
    v11 = v3;
    if ( (_DWORD)v99 == 1 && !v96 )
    {
      v10 = *(_QWORD *)v98;
      v11 = *(_QWORD *)v98;
    }
    v115 = sub_BD5C60(v11);
    v116 = &v124;
    v117 = &v125;
    LOWORD(v114) = 0;
    v109 = v111;
    v124 = &unk_49DA100;
    v110 = 0x200000000LL;
    v120 = 512;
    v125 = &unk_49DA0B0;
    v118 = 0;
    v119 = 0;
    v121 = 7;
    v122 = 0;
    v123 = 0;
    v112 = 0;
    v113 = 0;
    sub_D5F1F0((__int64)&v109, v10);
    v12 = *(_DWORD *)(a2 + 36) == 357;
    v145 = 257;
    if ( !v12 )
    {
      v13 = sub_F7CA10((__int64 *)&v109, v93, v5, (__int64)&v143, 0);
      v128 = 257;
      v14 = *(_QWORD *)(a1 + 64);
      v15 = sub_AA4E30(v112);
      v16 = sub_AE5020(v15, v14);
      v145 = 257;
      v17 = sub_BD2C40(80, unk_3F10A14);
      v18 = v17;
      if ( v17 )
        sub_B4D190((__int64)v17, v14, (__int64)v13, (__int64)&v143, 0, v16, 0, 0);
      (*((void (__fastcall **)(void **, _BYTE *, unsigned int **, __int64, __int64))*v117 + 2))(
        v117,
        v18,
        v126,
        v113,
        v114);
      v19 = (__int64)v109;
      v20 = &v109[16 * (unsigned int)v110];
      if ( v109 != v20 )
      {
        do
        {
          v21 = *(_QWORD *)(v19 + 8);
          v22 = *(_DWORD *)v19;
          v19 += 16;
          sub_B99FD0((__int64)v18, v22, v21);
        }
        while ( v20 != (_BYTE *)v19 );
      }
      goto LABEL_11;
    }
    v56 = sub_F7CA10((__int64 *)&v109, v93, v5, (__int64)&v143, 0);
    v128 = 257;
    v57 = *(_QWORD *)(a1 + 72);
    v87 = (unsigned __int64)v56;
    v58 = sub_AA4E30(v112);
    v59 = sub_AE5020(v58, v57);
    v145 = 257;
    v60 = v59;
    v61 = sub_BD2C40(80, unk_3F10A14);
    v63 = (unsigned __int64)v61;
    if ( v61 )
    {
      sub_B4D190((__int64)v61, v57, v87, (__int64)&v143, 0, v60, 0, 0);
      v62 = v82;
    }
    (*((void (__fastcall **)(void **, unsigned __int64, unsigned int **, __int64, __int64, __int64))*v117 + 2))(
      v117,
      v63,
      v126,
      v113,
      v114,
      v62);
    v64 = (__int64)v109;
    v65 = &v109[16 * (unsigned int)v110];
    if ( v109 != v65 )
    {
      do
      {
        v66 = *(_QWORD *)(v64 + 8);
        v67 = *(_DWORD *)v64;
        v64 += 16;
        sub_B99FD0(v63, v67, v66);
      }
      while ( v65 != (_BYTE *)v64 );
    }
    v68 = *(__int64 ***)(a1 + 88);
    v145 = 257;
    v69 = sub_26FAB50((__int64 *)&v109, 0x28u, v63, v68, (__int64)&v143, 0, (int)v126[0], 0);
    v70 = *(__int64 ***)(a1 + 88);
    v71 = (unsigned __int8 *)v69;
    v145 = 257;
    v72 = sub_26FAB50((__int64 *)&v109, 0x2Fu, v87, v70, (__int64)&v143, 0, (int)v126[0], 0);
    v128 = 257;
    v73 = (unsigned __int8 *)v72;
    v74 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v116 + 4);
    if ( v74 == sub_9201A0 )
    {
      if ( *v73 > 0x15u || *v71 > 0x15u )
      {
LABEL_68:
        v145 = 257;
        v75 = sub_B504D0(13, (__int64)v73, (__int64)v71, (__int64)&v143, 0, 0);
        (*((void (__fastcall **)(void **, __int64, unsigned int **, __int64, __int64))*v117 + 2))(
          v117,
          v75,
          v126,
          v113,
          v114);
        v77 = (__int64)v109;
        v78 = &v109[16 * (unsigned int)v110];
        if ( v109 != v78 )
        {
          do
          {
            v79 = *(_QWORD *)(v77 + 8);
            v80 = *(_DWORD *)v77;
            v77 += 16;
            sub_B99FD0(v75, v80, v79);
          }
          while ( v78 != (_BYTE *)v77 );
        }
        goto LABEL_66;
      }
      if ( (unsigned __int8)sub_AC47B0(13) )
        v75 = sub_AD5570(13, (__int64)v73, v71, 0, 0);
      else
        v75 = sub_AABE40(0xDu, v73, v71);
    }
    else
    {
      v75 = v74((__int64)v116, 13u, v73, v71, 0, 0);
    }
    if ( !v75 )
      goto LABEL_68;
LABEL_66:
    v145 = 257;
    v18 = (_BYTE *)sub_26FAB50(
                     (__int64 *)&v109,
                     0x30u,
                     v75,
                     *(__int64 ***)(a1 + 64),
                     (__int64)&v143,
                     0,
                     (int)v126[0],
                     0);
LABEL_11:
    v23 = (_QWORD **)v98;
    v24 = (_QWORD **)&v98[8 * (unsigned int)v99];
    if ( v24 != (_QWORD **)v98 )
    {
      do
      {
        v25 = *v23++;
        sub_BD84D0((__int64)v25, (__int64)v18);
        sub_B43D60(v25);
      }
      while ( v24 != v23 );
    }
    if ( (_DWORD)v102 != 1 || v96 )
    {
      v26 = v94;
      v27 = v94;
    }
    else
    {
      v26 = *(_QWORD *)v101;
      v27 = *(_QWORD *)v101;
    }
    v132 = sub_BD5C60(v27);
    v133 = &v141;
    v134 = &v142;
    v126[0] = (unsigned int *)v127;
    v141 = &unk_49DA100;
    v126[1] = (unsigned int *)0x200000000LL;
    v137 = 512;
    v142 = &unk_49DA0B0;
    v135 = 0;
    v136 = 0;
    v138 = 7;
    v139 = 0;
    v140 = 0;
    v129 = 0;
    v130 = 0;
    v131 = 0;
    sub_D5F1F0((__int64)v126, v26);
    v145 = 257;
    v28 = 0;
    v107[0].m128i_i64[0] = v93;
    v107[0].m128i_i64[1] = v88;
    if ( v85 )
      v28 = *(_QWORD *)(v85 + 24);
    v29 = sub_921880(v126, v28, v85, (int)v107, 2, (__int64)&v143, 0);
    v30 = (_QWORD **)v101;
    v31 = v81;
    v32 = (_BYTE *)v29;
    v33 = v83;
    v34 = (_QWORD **)&v101[8 * (unsigned int)v102];
    if ( v34 != (_QWORD **)v101 )
    {
      v89 = v18;
      do
      {
        v35 = *v30++;
        sub_BD84D0((__int64)v35, (__int64)v32);
        sub_B43D60(v35);
      }
      while ( v34 != v30 );
      v18 = v89;
    }
    if ( *(_QWORD *)(v94 + 16) )
    {
      v90 = sub_ACADE0(*(__int64 ***)(v94 + 8));
      v36 = sub_BD5C60(v94);
      v143.m128i_i64[1] = 0x200000000LL;
      v148 = 0;
      v149 = v36;
      v154 = 512;
      v143.m128i_i64[0] = (__int64)v144;
      v150 = &v158;
      v159 = &unk_49DA0B0;
      v151 = &v159;
      v152 = 0;
      v153 = 0;
      v155 = 7;
      v156 = 0;
      v157 = 0;
      v146 = 0;
      v147 = 0;
      v158 = &unk_49DA100;
      sub_D5F1F0((__int64)&v143, v94);
      v108 = 257;
      v97 = 0;
      v37 = sub_2466140(v143.m128i_i64, v90, v18, &v97, 1, (__int64)v107);
      v108 = 257;
      v97 = 1;
      v38 = sub_2466140(v143.m128i_i64, v37, v32, &v97, 1, (__int64)v107);
      sub_BD84D0(v94, v38);
      nullsub_61();
      v158 = &unk_49DA100;
      nullsub_63();
      if ( (_QWORD *)v143.m128i_i64[0] != v144 )
        _libc_free(v143.m128i_u64[0]);
    }
    v39 = (_QWORD *)(a1 + 360);
    v40 = *(_QWORD **)(a1 + 368);
    v41 = a1 + 360;
    if ( !v40 )
      goto LABEL_31;
    do
    {
      while ( 1 )
      {
        v42 = v40[2];
        v43 = v40[3];
        if ( v40[4] >= (unsigned __int64)v32 )
          break;
        v40 = (_QWORD *)v40[3];
        if ( !v43 )
          goto LABEL_29;
      }
      v41 = (__int64)v40;
      v40 = (_QWORD *)v40[2];
    }
    while ( v42 );
LABEL_29:
    if ( v39 == (_QWORD *)v41 || *(_QWORD *)(v41 + 32) > (unsigned __int64)v32 )
    {
LABEL_31:
      v44 = sub_22077B0(0x30u);
      v45 = v41;
      *(_QWORD *)(v44 + 32) = v32;
      v41 = v44;
      *(_DWORD *)(v44 + 40) = 0;
      v46 = sub_2701120((_QWORD *)(a1 + 352), v45, (unsigned __int64 *)(v44 + 32));
      if ( v47 )
      {
        v48 = v46 || v39 == v47 || (unsigned __int64)v32 < v47[4];
        sub_220F040(v48, v41, v47, v39);
        ++*(_QWORD *)(a1 + 392);
      }
      else
      {
        v76 = v41;
        v41 = (__int64)v46;
        j_j___libc_free_0(v76);
      }
    }
    v49 = (unsigned int)v105;
    v12 = v96 == 0;
    *(_DWORD *)(v41 + 40) = v105;
    if ( !v12 )
      *(_DWORD *)(v41 + 40) = v49 + 1;
    v50 = (__int64)v104;
    v91 = &v104[2 * v49];
    if ( v91 != v104 )
    {
      v51 = v104;
      do
      {
        v52 = *v51;
        v53 = (unsigned __int8 *)v51[1];
        v51 += 2;
        v107[0].m128i_i64[0] = v92;
        v107[0].m128i_i64[1] = v52;
        v54 = (_QWORD *)sub_26FA230(a1 + 128, v107, v50, v92, v31, v33);
        v55 = sub_26FE470(v54, v53);
        *((_BYTE *)v55 + 24) = 0;
        v143.m128i_i64[1] = (__int64)v53;
        v143.m128i_i64[0] = v93;
        v144[0] = v41 + 40;
        sub_26F6620(v55, &v143);
      }
      while ( v91 != v51 );
    }
    sub_B43D60((_QWORD *)v94);
    nullsub_61();
    v141 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v126[0] != v127 )
      _libc_free((unsigned __int64)v126[0]);
    nullsub_61();
    v124 = &unk_49DA100;
    nullsub_63();
    if ( v109 != v111 )
      _libc_free((unsigned __int64)v109);
    if ( v101 != v103 )
      _libc_free((unsigned __int64)v101);
    if ( v98 != v100 )
      _libc_free((unsigned __int64)v98);
    if ( v104 == (__int64 *)v106 )
      goto LABEL_3;
    _libc_free((unsigned __int64)v104);
  }
}
