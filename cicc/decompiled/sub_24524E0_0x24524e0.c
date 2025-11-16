// Function: sub_24524E0
// Address: 0x24524e0
//
void __fastcall sub_24524E0(__int64 a1, _QWORD *a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r12
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD **v15; // rax
  int v16; // edx
  _QWORD *v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rax
  char v20; // bl
  _QWORD *v21; // rax
  _BYTE *v22; // r12
  unsigned int *v23; // r15
  unsigned int *v24; // rbx
  __int64 v25; // rdx
  unsigned int v26; // esi
  _QWORD *v27; // rax
  __int64 v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  char v32; // bl
  _QWORD *v33; // rax
  __int64 v34; // r9
  _QWORD *v35; // r12
  __int64 v36; // rbx
  unsigned int *v37; // rbx
  unsigned int *v38; // r13
  __int64 v39; // rdx
  unsigned int v40; // esi
  unsigned int v41; // r15d
  __int64 v42; // rax
  _BYTE *v43; // rcx
  __int64 v44; // r15
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rax
  _BYTE *v48; // rax
  __int64 v49; // rax
  char v50; // bl
  _QWORD *v51; // rax
  __int64 v52; // r9
  __int64 v53; // rbx
  unsigned int *v54; // rbx
  unsigned int *v55; // r13
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rax
  _BYTE *v59; // rcx
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // rax
  char v66; // bl
  _QWORD *v67; // rax
  __int64 v68; // r9
  __int64 v69; // r13
  unsigned int *v70; // r14
  unsigned int *v71; // rbx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // [rsp+8h] [rbp-2A8h]
  _QWORD *v77; // [rsp+10h] [rbp-2A0h]
  unsigned __int64 v78; // [rsp+10h] [rbp-2A0h]
  unsigned __int64 v79; // [rsp+38h] [rbp-278h]
  _BYTE *v80; // [rsp+58h] [rbp-258h]
  char v81; // [rsp+66h] [rbp-24Ah]
  char v82; // [rsp+67h] [rbp-249h]
  __int64 v83; // [rsp+78h] [rbp-238h] BYREF
  __int64 v84; // [rsp+80h] [rbp-230h] BYREF
  unsigned __int64 v85; // [rsp+88h] [rbp-228h] BYREF
  unsigned __int64 v86; // [rsp+94h] [rbp-21Ch]
  int v87; // [rsp+9Ch] [rbp-214h]
  _BYTE v88[32]; // [rsp+A0h] [rbp-210h] BYREF
  __int16 v89; // [rsp+C0h] [rbp-1F0h]
  unsigned int *v90; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-1D8h]
  _BYTE v92[32]; // [rsp+E0h] [rbp-1D0h] BYREF
  __int64 v93; // [rsp+100h] [rbp-1B0h]
  __int64 v94; // [rsp+108h] [rbp-1A8h]
  __int64 v95; // [rsp+110h] [rbp-1A0h]
  _QWORD *v96; // [rsp+118h] [rbp-198h]
  void **v97; // [rsp+120h] [rbp-190h]
  void **v98; // [rsp+128h] [rbp-188h]
  __int64 v99; // [rsp+130h] [rbp-180h]
  int v100; // [rsp+138h] [rbp-178h]
  __int16 v101; // [rsp+13Ch] [rbp-174h]
  char v102; // [rsp+13Eh] [rbp-172h]
  __int64 v103; // [rsp+140h] [rbp-170h]
  __int64 v104; // [rsp+148h] [rbp-168h]
  void *v105; // [rsp+150h] [rbp-160h] BYREF
  void *v106; // [rsp+158h] [rbp-158h] BYREF
  unsigned int *v107[2]; // [rsp+160h] [rbp-150h] BYREF
  _BYTE v108[16]; // [rsp+170h] [rbp-140h] BYREF
  __int16 v109; // [rsp+180h] [rbp-130h]
  __int64 v110; // [rsp+190h] [rbp-120h]
  __int64 v111; // [rsp+198h] [rbp-118h]
  __int16 v112; // [rsp+1A0h] [rbp-110h]
  _QWORD *v113; // [rsp+1A8h] [rbp-108h]
  void **v114; // [rsp+1B0h] [rbp-100h]
  void **v115; // [rsp+1B8h] [rbp-F8h]
  __int64 v116; // [rsp+1C0h] [rbp-F0h]
  int v117; // [rsp+1C8h] [rbp-E8h]
  __int16 v118; // [rsp+1CCh] [rbp-E4h]
  char v119; // [rsp+1CEh] [rbp-E2h]
  __int64 v120; // [rsp+1D0h] [rbp-E0h]
  __int64 v121; // [rsp+1D8h] [rbp-D8h]
  void *v122; // [rsp+1E0h] [rbp-D0h] BYREF
  void *v123; // [rsp+1E8h] [rbp-C8h] BYREF
  unsigned int *v124; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+1F8h] [rbp-B8h]
  _BYTE v126[16]; // [rsp+200h] [rbp-B0h] BYREF
  __int16 v127; // [rsp+210h] [rbp-A0h]
  __int64 v128; // [rsp+220h] [rbp-90h]
  __int64 v129; // [rsp+228h] [rbp-88h]
  __int64 v130; // [rsp+230h] [rbp-80h]
  _QWORD *v131; // [rsp+238h] [rbp-78h]
  void **v132; // [rsp+240h] [rbp-70h]
  _QWORD *v133; // [rsp+248h] [rbp-68h]
  __int64 v134; // [rsp+250h] [rbp-60h]
  int v135; // [rsp+258h] [rbp-58h]
  __int16 v136; // [rsp+25Ch] [rbp-54h]
  char v137; // [rsp+25Eh] [rbp-52h]
  __int64 v138; // [rsp+260h] [rbp-50h]
  __int64 v139; // [rsp+268h] [rbp-48h]
  void *v140; // [rsp+270h] [rbp-40h] BYREF
  _QWORD v141[7]; // [rsp+278h] [rbp-38h] BYREF

  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_20;
  v11 = v5[7];
  if ( !v11 )
    goto LABEL_20;
  v12 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_4FE65E8 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_15:
  if ( v5 + 6 == v12 || dword_4FE65E8 < *((_DWORD *)v12 + 8) || *((int *)v12 + 9) <= 0 )
  {
LABEL_20:
    if ( !*(_BYTE *)(a1 + 12) )
      return;
  }
  else if ( !(_BYTE)qword_4FE6668 )
  {
    return;
  }
  v86 = sub_2450190();
  v15 = *(_QWORD ***)a1;
  v82 = v16;
  v79 = HIDWORD(v86);
  v87 = v16;
  v17 = *v15;
  if ( (_BYTE)v16 )
    v18 = sub_BCB2C0(v17);
  else
    v18 = sub_BCB2D0(v17);
  v80 = sub_BA8CD0(*(_QWORD *)a1, (__int64)"__llvm_profile_sampling", 0x17u, 0);
  v83 = sub_BD5C60((__int64)a2);
  v96 = (_QWORD *)sub_BD5C60((__int64)a2);
  v97 = &v105;
  v98 = &v106;
  v90 = (unsigned int *)v92;
  v105 = &unk_49DA100;
  v101 = 512;
  v91 = 0x200000000LL;
  LOWORD(v95) = 0;
  v106 = &unk_49DA0B0;
  v99 = 0;
  v100 = 0;
  v102 = 7;
  v103 = 0;
  v104 = 0;
  v93 = 0;
  v94 = 0;
  sub_D5F1F0((__int64)&v90, (__int64)a2);
  v109 = 257;
  v19 = sub_AA4E30(v93);
  v20 = sub_AE5020(v19, v18);
  v127 = 257;
  v21 = sub_BD2C40(80, unk_3F10A14);
  v22 = v21;
  if ( v21 )
    sub_B4D190((__int64)v21, v18, (__int64)v80, (__int64)&v124, 0, v20, 0, 0);
  (*((void (__fastcall **)(void **, _BYTE *, unsigned int **, __int64, __int64))*v98 + 2))(v98, v22, v107, v94, v95);
  v23 = v90;
  v24 = &v90[4 * (unsigned int)v91];
  if ( v90 != v24 )
  {
    do
    {
      v25 = *((_QWORD *)v23 + 1);
      v26 = *v23;
      v23 += 4;
      sub_B99FD0((__int64)v22, v26, v25);
    }
    while ( v24 != v23 );
  }
  v81 = BYTE1(v87);
  if ( BYTE1(v87) )
  {
    v27 = (_QWORD *)sub_BD5C60((__int64)a2);
    v137 = 7;
    v131 = v27;
    v132 = &v140;
    v133 = v141;
    v136 = 512;
    LOWORD(v130) = 0;
    v124 = (unsigned int *)v126;
    v140 = &unk_49DA100;
    v125 = 0x200000000LL;
    v134 = 0;
    v141[0] = &unk_49DA0B0;
    v135 = 0;
    v138 = 0;
    v139 = 0;
    v128 = 0;
    v129 = 0;
    sub_D5F1F0((__int64)&v124, (__int64)a2);
    v109 = 257;
    if ( v82 )
      v28 = sub_BCB2C0(v131);
    else
      v28 = sub_BCB2D0(v131);
    v29 = (_BYTE *)sub_ACD640(v28, 1, 0);
    v30 = sub_929C50(&v124, v22, v29, (__int64)v107, 0, 0);
    v31 = sub_AA4E30(v128);
    v32 = sub_AE5020(v31, *(_QWORD *)(v30 + 8));
    v109 = 257;
    v33 = sub_BD2C40(80, unk_3F10A10);
    v35 = v33;
    if ( v33 )
      sub_B4D3C0((__int64)v33, v30, (__int64)v80, 0, v32, v34, 0, 0);
    (*(void (__fastcall **)(_QWORD *, _QWORD *, unsigned int **, __int64, __int64))(*v133 + 16LL))(
      v133,
      v35,
      v107,
      v129,
      v130);
    v36 = 4LL * (unsigned int)v125;
    if ( v124 != &v124[v36] )
    {
      v77 = a2;
      v37 = &v124[v36];
      v38 = v124;
      do
      {
        v39 = *((_QWORD *)v38 + 1);
        v40 = *v38;
        v38 += 4;
        sub_B99FD0((__int64)v35, v40, v39);
      }
      while ( v37 != v38 );
      a2 = v77;
    }
  }
  else
  {
    v127 = 257;
    v41 = v86 - 1;
    if ( v82 )
    {
      v74 = sub_BCB2C0(v96);
      v43 = (_BYTE *)sub_ACD640(v74, (unsigned __int16)v41, 0);
    }
    else
    {
      v42 = sub_BCB2D0(v96);
      v43 = (_BYTE *)sub_ACD640(v42, v41, 0);
    }
    v44 = sub_92B530(&v90, 0x25u, (__int64)v22, v43, (__int64)&v124);
    v45 = sub_B8C2F0(&v83, v86, HIDWORD(v86) - (int)v86, 0);
    v78 = sub_F38250(v44, a2 + 3, 0, 0, v45, 0, 0, 0);
    v46 = (_QWORD *)sub_BD5C60((__int64)a2);
    v137 = 7;
    v131 = v46;
    v132 = &v140;
    v133 = v141;
    v124 = (unsigned int *)v126;
    v140 = &unk_49DA100;
    v136 = 512;
    LOWORD(v130) = 0;
    v125 = 0x200000000LL;
    v134 = 0;
    v135 = 0;
    v138 = 0;
    v139 = 0;
    v128 = 0;
    v129 = 0;
    v141[0] = &unk_49DA0B0;
    sub_D5F1F0((__int64)&v124, (__int64)a2);
    v109 = 257;
    if ( v82 )
      v47 = sub_BCB2C0(v131);
    else
      v47 = sub_BCB2D0(v131);
    v48 = (_BYTE *)sub_ACD640(v47, 1, 0);
    v30 = sub_929C50(&v124, v22, v48, (__int64)v107, 0, 0);
    v49 = sub_AA4E30(v128);
    v50 = sub_AE5020(v49, *(_QWORD *)(v30 + 8));
    v109 = 257;
    v51 = sub_BD2C40(80, unk_3F10A10);
    v35 = v51;
    if ( v51 )
      sub_B4D3C0((__int64)v51, v30, (__int64)v80, 0, v50, v52, 0, 0);
    (*(void (__fastcall **)(_QWORD *, _QWORD *, unsigned int **, __int64, __int64))(*v133 + 16LL))(
      v133,
      v35,
      v107,
      v129,
      v130);
    v53 = 4LL * (unsigned int)v125;
    if ( v124 != &v124[v53] )
    {
      v76 = a2;
      v54 = &v124[v53];
      v55 = v124;
      do
      {
        v56 = *((_QWORD *)v55 + 1);
        v57 = *v55;
        v55 += 4;
        sub_B99FD0((__int64)v35, v57, v56);
      }
      while ( v54 != v55 );
      a2 = v76;
    }
    sub_B444E0(a2, v78 + 24, 0);
  }
  nullsub_61();
  v140 = &unk_49DA100;
  nullsub_63();
  if ( v124 != (unsigned int *)v126 )
    _libc_free((unsigned __int64)v124);
  if ( !BYTE2(v87) )
  {
    v113 = (_QWORD *)sub_BD5C60((__int64)v35);
    v114 = &v122;
    v115 = &v123;
    v112 = 0;
    v107[0] = (unsigned int *)v108;
    v107[1] = (unsigned int *)0x200000000LL;
    v122 = &unk_49DA100;
    v116 = 0;
    v118 = 512;
    v123 = &unk_49DA0B0;
    v117 = 0;
    v119 = 7;
    v120 = 0;
    v121 = 0;
    v110 = 0;
    v111 = 0;
    sub_D5F1F0((__int64)v107, (__int64)v35);
    v127 = 257;
    if ( v82 )
    {
      v58 = sub_BCB2C0(v113);
      v59 = (_BYTE *)sub_ACD640(v58, (unsigned __int16)v79, 0);
    }
    else
    {
      v75 = sub_BCB2D0(v113);
      v59 = (_BYTE *)sub_ACD640(v75, (unsigned int)v79, 0);
    }
    v60 = sub_92B530(v107, 0x23u, v30, v59, (__int64)&v124);
    v61 = sub_B8C2F0(&v83, 1u, (int)v79 - 1, 0);
    sub_F38330(v60, v35 + 3, 0, (unsigned __int64 *)&v84, &v85, v61, 0, 0);
    if ( v81 )
      sub_B444E0(a2, v84 + 24, 0);
    v62 = v84;
    v131 = (_QWORD *)sub_BD5C60(v84);
    v134 = 0;
    v132 = &v140;
    v124 = (unsigned int *)v126;
    v133 = v141;
    v125 = 0x200000000LL;
    v135 = 0;
    v140 = &unk_49DA100;
    v136 = 512;
    v137 = 7;
    v138 = 0;
    v139 = 0;
    v128 = 0;
    v129 = 0;
    LOWORD(v130) = 0;
    v141[0] = &unk_49DA0B0;
    sub_D5F1F0((__int64)&v124, v62);
    if ( v82 )
      v63 = sub_BCB2C0(v131);
    else
      v63 = sub_BCB2D0(v131);
    v64 = sub_ACD640(v63, 0, 0);
    v65 = sub_AA4E30(v128);
    v66 = sub_AE5020(v65, *(_QWORD *)(v64 + 8));
    v89 = 257;
    v67 = sub_BD2C40(80, unk_3F10A10);
    v69 = (__int64)v67;
    if ( v67 )
      sub_B4D3C0((__int64)v67, v64, (__int64)v80, 0, v66, v68, 0, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64))(*v133 + 16LL))(v133, v69, v88, v129, v130);
    v70 = v124;
    v71 = &v124[4 * (unsigned int)v125];
    if ( v124 != v71 )
    {
      do
      {
        v72 = *((_QWORD *)v70 + 1);
        v73 = *v70;
        v70 += 4;
        sub_B99FD0(v69, v73, v72);
      }
      while ( v71 != v70 );
    }
    sub_B444E0(v35, v85 + 24, 0);
    nullsub_61();
    v140 = &unk_49DA100;
    nullsub_63();
    if ( v124 != (unsigned int *)v126 )
      _libc_free((unsigned __int64)v124);
    nullsub_61();
    v122 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v107[0] != v108 )
      _libc_free((unsigned __int64)v107[0]);
  }
  nullsub_61();
  v105 = &unk_49DA100;
  nullsub_63();
  if ( v90 != (unsigned int *)v92 )
    _libc_free((unsigned __int64)v90);
}
