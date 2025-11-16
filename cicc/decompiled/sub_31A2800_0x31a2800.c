// Function: sub_31A2800
// Address: 0x31a2800
//
__int64 __fastcall sub_31A2800(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  int v8; // eax
  unsigned __int8 *v9; // rdi
  unsigned __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned __int8 *v18; // r12
  __int64 v19; // rax
  unsigned __int8 *v20; // rbx
  __int64 (__fastcall *v21)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  _BYTE *v22; // r12
  unsigned __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rbx
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // r12
  int v32; // eax
  int v33; // eax
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // r12
  __int64 v39; // rdx
  int v40; // eax
  int v41; // eax
  unsigned int v42; // ecx
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 (__fastcall *v46)(__int64, _BYTE *, _BYTE *); // rax
  __int64 v47; // r13
  __int64 v48; // rax
  unsigned int v49; // eax
  __int64 v50; // rax
  unsigned __int64 v51; // rsi
  __int64 v52; // rax
  _BYTE *v53; // r14
  __int64 (__fastcall *v54)(__int64, _BYTE *, _BYTE *, _BYTE *); // rax
  __int64 v55; // rdi
  __int64 v56; // r13
  int v57; // eax
  int v58; // eax
  unsigned int v59; // edx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // rax
  unsigned __int8 *v64; // r14
  __int64 (__fastcall *v65)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v66; // r10
  int v67; // eax
  int v68; // eax
  unsigned int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 (__fastcall *v73)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v74; // rax
  _QWORD *v75; // r14
  _QWORD *v76; // rax
  __int64 v77; // rbx
  unsigned int *v78; // r15
  unsigned int *v79; // r14
  __int64 v80; // rdx
  unsigned int v81; // esi
  __int64 v83; // rdx
  int v84; // ecx
  int v85; // eax
  _QWORD *v86; // rdi
  __int64 *v87; // rax
  __int64 v88; // rax
  unsigned int *v89; // r15
  unsigned int *v90; // rbx
  __int64 v91; // rdx
  unsigned int v92; // esi
  unsigned int *v93; // r15
  unsigned int *v94; // r14
  __int64 v95; // rdx
  unsigned int v96; // esi
  _QWORD *v97; // rax
  __int64 v98; // rdx
  unsigned int *v99; // r15
  unsigned int *v100; // rbx
  __int64 v101; // rdx
  unsigned int v102; // esi
  _QWORD *v103; // rax
  __int64 v104; // r9
  unsigned int *v105; // r15
  unsigned int *v106; // r14
  __int64 v107; // rdx
  unsigned int v108; // esi
  __int64 v109; // rax
  __int64 v110; // [rsp+0h] [rbp-250h]
  __int64 v111; // [rsp+8h] [rbp-248h]
  __int64 v112; // [rsp+8h] [rbp-248h]
  _QWORD *v114; // [rsp+58h] [rbp-1F8h]
  __int64 v115; // [rsp+58h] [rbp-1F8h]
  __int64 v116; // [rsp+58h] [rbp-1F8h]
  __int64 v117; // [rsp+58h] [rbp-1F8h]
  __int64 v118; // [rsp+58h] [rbp-1F8h]
  __int64 v119; // [rsp+58h] [rbp-1F8h]
  __int64 v120; // [rsp+58h] [rbp-1F8h]
  unsigned __int8 *v121; // [rsp+60h] [rbp-1F0h]
  __int64 v122; // [rsp+70h] [rbp-1E0h]
  __int64 v124; // [rsp+80h] [rbp-1D0h]
  __int64 v125; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v126; // [rsp+98h] [rbp-1B8h]
  _BYTE v127[32]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v128; // [rsp+C0h] [rbp-190h]
  _QWORD v129[4]; // [rsp+D0h] [rbp-180h] BYREF
  __int16 v130; // [rsp+F0h] [rbp-160h]
  _BYTE *v131; // [rsp+100h] [rbp-150h] BYREF
  __int64 v132; // [rsp+108h] [rbp-148h]
  _BYTE v133[32]; // [rsp+110h] [rbp-140h] BYREF
  __int64 v134; // [rsp+130h] [rbp-120h]
  __int64 v135; // [rsp+138h] [rbp-118h]
  __int64 v136; // [rsp+140h] [rbp-110h]
  _QWORD *v137; // [rsp+148h] [rbp-108h]
  void **v138; // [rsp+150h] [rbp-100h]
  void **v139; // [rsp+158h] [rbp-F8h]
  __int64 v140; // [rsp+160h] [rbp-F0h]
  int v141; // [rsp+168h] [rbp-E8h]
  __int16 v142; // [rsp+16Ch] [rbp-E4h]
  char v143; // [rsp+16Eh] [rbp-E2h]
  __int64 v144; // [rsp+170h] [rbp-E0h]
  __int64 v145; // [rsp+178h] [rbp-D8h]
  void *v146; // [rsp+180h] [rbp-D0h] BYREF
  void *v147; // [rsp+188h] [rbp-C8h] BYREF
  unsigned int *v148; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v149; // [rsp+198h] [rbp-B8h]
  _BYTE v150[16]; // [rsp+1A0h] [rbp-B0h] BYREF
  __int16 v151; // [rsp+1B0h] [rbp-A0h]
  __int64 v152; // [rsp+1C0h] [rbp-90h]
  __int64 v153; // [rsp+1C8h] [rbp-88h]
  __int64 v154; // [rsp+1D0h] [rbp-80h]
  _QWORD *v155; // [rsp+1D8h] [rbp-78h]
  void **v156; // [rsp+1E0h] [rbp-70h]
  void **v157; // [rsp+1E8h] [rbp-68h]
  __int64 v158; // [rsp+1F0h] [rbp-60h]
  int v159; // [rsp+1F8h] [rbp-58h]
  __int16 v160; // [rsp+1FCh] [rbp-54h]
  char v161; // [rsp+1FEh] [rbp-52h]
  __int64 v162; // [rsp+200h] [rbp-50h]
  __int64 v163; // [rsp+208h] [rbp-48h]
  void *v164; // [rsp+210h] [rbp-40h] BYREF
  void *v165; // [rsp+218h] [rbp-38h] BYREF

  v2 = *(_QWORD **)(a2 + 40);
  v3 = v2[9];
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  v5 = sub_AA48A0((__int64)v2);
  v151 = 257;
  v122 = sub_AA8550(v2, (__int64 *)(a2 + 24), 0, (__int64)&v148, 0);
  v151 = 257;
  v6 = sub_22077B0(0x50u);
  v124 = v6;
  if ( v6 )
    sub_AA4D50(v6, v5, (__int64)&v148, v3, v122);
  v7 = v2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 + 6 == (_QWORD *)v7 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = (unsigned __int8 *)(v7 - 24);
    if ( (unsigned int)(v8 - 30) >= 0xB )
      v9 = 0;
  }
  sub_B46F90(v9, 0, v124);
  v10 = v2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v10 == v2 + 6 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = 0;
    v13 = v10 - 24;
    if ( (unsigned int)(v11 - 30) < 0xB )
      v12 = v13;
  }
  v137 = (_QWORD *)sub_BD5C60(v12);
  v138 = &v146;
  v139 = &v147;
  v142 = 512;
  LOWORD(v136) = 0;
  v132 = 0x200000000LL;
  v146 = &unk_49DA100;
  v131 = v133;
  v143 = 7;
  v140 = 0;
  v141 = 0;
  v144 = 0;
  v145 = 0;
  v134 = 0;
  v135 = 0;
  v147 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v131, v12);
  if ( *(_BYTE *)(v4 + 8) != 18 )
  {
    v26 = *(unsigned int *)(v4 + 32);
    v27 = sub_BCB2E0(v137);
    v121 = (unsigned __int8 *)sub_ACD640(v27, v26, 0);
    goto LABEL_23;
  }
  v151 = 257;
  v14 = sub_BCB2E0(v137);
  v15 = sub_ACD640(v14, 1, 0);
  v16 = sub_B33D80((__int64)&v131, v15, (__int64)&v148);
  v17 = *(unsigned int *)(v4 + 32);
  v18 = (unsigned __int8 *)v16;
  v19 = sub_BCB2E0(v137);
  v20 = (unsigned __int8 *)sub_ACD640(v19, v17, 0);
  v130 = 257;
  v21 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v138 + 4);
  if ( v21 != sub_9201A0 )
  {
    v121 = (unsigned __int8 *)v21((__int64)v138, 17u, v18, v20, 0, 0);
    goto LABEL_17;
  }
  if ( *v18 <= 0x15u && *v20 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(17) )
      v121 = (unsigned __int8 *)sub_AD5570(17, (__int64)v18, v20, 0, 0);
    else
      v121 = (unsigned __int8 *)sub_AABE40(0x11u, v18, v20);
LABEL_17:
    if ( v121 )
      goto LABEL_23;
  }
  v151 = 257;
  v121 = (unsigned __int8 *)sub_B504D0(17, (__int64)v18, (__int64)v20, (__int64)&v148, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v139 + 2))(
    v139,
    v121,
    v129,
    v135,
    v136);
  if ( v131 != &v131[16 * (unsigned int)v132] )
  {
    v22 = &v131[16 * (unsigned int)v132];
    v114 = v2;
    v23 = (unsigned __int64)v131;
    do
    {
      v24 = *(_QWORD *)(v23 + 8);
      v25 = *(_DWORD *)v23;
      v23 += 16LL;
      sub_B99FD0((__int64)v121, v25, v24);
    }
    while ( v22 != (_BYTE *)v23 );
    v2 = v114;
  }
LABEL_23:
  v28 = (_QWORD *)sub_AA48A0(v124);
  v152 = v124;
  LOWORD(v154) = 0;
  v148 = (unsigned int *)v150;
  v149 = 0x200000000LL;
  v156 = &v164;
  v157 = &v165;
  v160 = 512;
  v155 = v28;
  v164 = &unk_49DA100;
  v158 = 0;
  v159 = 0;
  v165 = &unk_49DA0B0;
  v161 = 7;
  v162 = 0;
  v163 = 0;
  v153 = v124 + 48;
  v29 = sub_BCB2E0(v28);
  v130 = 257;
  v115 = v29;
  v30 = sub_D5C860((__int64 *)&v148, v29, 2, (__int64)v129);
  v31 = sub_AD64C0(v29, 0, 0);
  v32 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
  if ( v32 == *(_DWORD *)(v30 + 72) )
  {
    sub_B48D90(v30);
    v32 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
  }
  v33 = (v32 + 1) & 0x7FFFFFF;
  v34 = v33 | *(_DWORD *)(v30 + 4) & 0xF8000000;
  v35 = *(_QWORD *)(v30 - 8) + 32LL * (unsigned int)(v33 - 1);
  *(_DWORD *)(v30 + 4) = v34;
  if ( *(_QWORD *)v35 )
  {
    v36 = *(_QWORD *)(v35 + 8);
    **(_QWORD **)(v35 + 16) = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = *(_QWORD *)(v35 + 16);
  }
  *(_QWORD *)v35 = v31;
  if ( v31 )
  {
    v37 = *(_QWORD *)(v31 + 16);
    *(_QWORD *)(v35 + 8) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = v35 + 8;
    *(_QWORD *)(v35 + 16) = v31 + 16;
    *(_QWORD *)(v31 + 16) = v35;
  }
  *(_QWORD *)(*(_QWORD *)(v30 - 8) + 32LL * *(unsigned int *)(v30 + 72)
                                   + 8LL * ((*(_DWORD *)(v30 + 4) & 0x7FFFFFFu) - 1)) = v2;
  v130 = 257;
  v38 = sub_D5C860((__int64 *)&v148, v4, 2, (__int64)v129);
  v39 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v40 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  if ( v40 == *(_DWORD *)(v38 + 72) )
  {
    v112 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_B48D90(v38);
    v39 = v112;
    v40 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  }
  v41 = (v40 + 1) & 0x7FFFFFF;
  v42 = v41 | *(_DWORD *)(v38 + 4) & 0xF8000000;
  v43 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v41 - 1);
  *(_DWORD *)(v38 + 4) = v42;
  if ( *(_QWORD *)v43 )
  {
    v44 = *(_QWORD *)(v43 + 8);
    **(_QWORD **)(v43 + 16) = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 16) = *(_QWORD *)(v43 + 16);
  }
  *(_QWORD *)v43 = v39;
  if ( v39 )
  {
    v45 = *(_QWORD *)(v39 + 16);
    *(_QWORD *)(v43 + 8) = v45;
    if ( v45 )
      *(_QWORD *)(v45 + 16) = v43 + 8;
    *(_QWORD *)(v43 + 16) = v39 + 16;
    *(_QWORD *)(v39 + 16) = v43;
  }
  *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * *(unsigned int *)(v38 + 72)
                                   + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v2;
  v128 = 257;
  v46 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *))*((_QWORD *)*v156 + 12);
  if ( v46 != sub_948070 )
  {
    v47 = v46((__int64)v156, (_BYTE *)v38, (_BYTE *)v30);
LABEL_45:
    if ( v47 )
      goto LABEL_46;
    goto LABEL_104;
  }
  if ( *(_BYTE *)v38 <= 0x15u && *(_BYTE *)v30 <= 0x15u )
  {
    v47 = sub_AD5840(v38, (unsigned __int8 *)v30, 0);
    goto LABEL_45;
  }
LABEL_104:
  v130 = 257;
  v97 = sub_BD2C40(72, 2u);
  v47 = (__int64)v97;
  if ( v97 )
    sub_B4DE80((__int64)v97, v38, v30, (__int64)v129, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v157 + 2))(v157, v47, v127, v153, v154);
  v98 = 4LL * (unsigned int)v149;
  if ( v148 != &v148[v98] )
  {
    v111 = v4;
    v99 = &v148[v98];
    v110 = v30;
    v100 = v148;
    do
    {
      v101 = *((_QWORD *)v100 + 1);
      v102 = *v100;
      v100 += 4;
      sub_B99FD0(v47, v102, v101);
    }
    while ( v99 != v100 );
    v4 = v111;
    v30 = v110;
  }
LABEL_46:
  v48 = *(_QWORD *)(v4 + 24);
  v125 = v47;
  v129[0] = v48;
  v49 = sub_B49240(a2);
  v50 = sub_B6E160(a1, v49, (__int64)v129, 1);
  v130 = 257;
  v51 = 0;
  if ( v50 )
    v51 = *(_QWORD *)(v50 + 24);
  v52 = sub_921880(&v148, v51, v50, (int)&v125, 1, (__int64)v129, 0);
  v128 = 257;
  v53 = (_BYTE *)v52;
  v54 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _BYTE *))*((_QWORD *)*v156 + 13);
  if ( v54 != sub_948040 )
  {
    v56 = v54((__int64)v156, (_BYTE *)v38, v53, (_BYTE *)v30);
LABEL_55:
    if ( v56 )
      goto LABEL_56;
    goto LABEL_110;
  }
  v55 = 0;
  if ( *(_BYTE *)v38 <= 0x15u )
    v55 = v38;
  if ( *v53 <= 0x15u && *(_BYTE *)v30 <= 0x15u && v55 )
  {
    v56 = sub_AD5A90(v55, v53, (unsigned __int8 *)v30, 0);
    goto LABEL_55;
  }
LABEL_110:
  v130 = 257;
  v103 = sub_BD2C40(72, 3u);
  v104 = 0;
  v56 = (__int64)v103;
  if ( v103 )
    sub_B4DFA0((__int64)v103, v38, (__int64)v53, v30, (__int64)v129, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64, __int64))*v157 + 2))(
    v157,
    v56,
    v127,
    v153,
    v154,
    v104);
  v105 = v148;
  v106 = &v148[4 * (unsigned int)v149];
  if ( v148 != v106 )
  {
    do
    {
      v107 = *((_QWORD *)v105 + 1);
      v108 = *v105;
      v105 += 4;
      sub_B99FD0(v56, v108, v107);
    }
    while ( v106 != v105 );
  }
LABEL_56:
  v57 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  if ( v57 == *(_DWORD *)(v38 + 72) )
  {
    sub_B48D90(v38);
    v57 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
  }
  v58 = (v57 + 1) & 0x7FFFFFF;
  v59 = v58 | *(_DWORD *)(v38 + 4) & 0xF8000000;
  v60 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v58 - 1);
  *(_DWORD *)(v38 + 4) = v59;
  if ( *(_QWORD *)v60 )
  {
    v61 = *(_QWORD *)(v60 + 8);
    **(_QWORD **)(v60 + 16) = v61;
    if ( v61 )
      *(_QWORD *)(v61 + 16) = *(_QWORD *)(v60 + 16);
  }
  *(_QWORD *)v60 = v56;
  if ( v56 )
  {
    v62 = *(_QWORD *)(v56 + 16);
    *(_QWORD *)(v60 + 8) = v62;
    if ( v62 )
      *(_QWORD *)(v62 + 16) = v60 + 8;
    *(_QWORD *)(v60 + 16) = v56 + 16;
    *(_QWORD *)(v56 + 16) = v60;
  }
  *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * *(unsigned int *)(v38 + 72)
                                   + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v124;
  v63 = sub_AD64C0(v115, 1, 0);
  v128 = 257;
  v64 = (unsigned __int8 *)v63;
  v65 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v156 + 4);
  if ( v65 == sub_9201A0 )
  {
    if ( *(_BYTE *)v30 > 0x15u || *v64 > 0x15u )
      goto LABEL_100;
    if ( (unsigned __int8)sub_AC47B0(13) )
      v66 = sub_AD5570(13, v30, v64, 0, 0);
    else
      v66 = sub_AABE40(0xDu, (unsigned __int8 *)v30, v64);
  }
  else
  {
    v66 = v65((__int64)v156, 13u, (_BYTE *)v30, v64, 0, 0);
  }
  if ( v66 )
    goto LABEL_71;
LABEL_100:
  v130 = 257;
  v118 = sub_B504D0(13, v30, (__int64)v64, (__int64)v129, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v157 + 2))(v157, v118, v127, v153, v154);
  v93 = v148;
  v66 = v118;
  v94 = &v148[4 * (unsigned int)v149];
  if ( v148 == v94 )
  {
LABEL_71:
    v67 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
    if ( v67 != *(_DWORD *)(v30 + 72) )
      goto LABEL_72;
LABEL_103:
    v119 = v66;
    sub_B48D90(v30);
    v66 = v119;
    v67 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
    goto LABEL_72;
  }
  do
  {
    v95 = *((_QWORD *)v93 + 1);
    v96 = *v93;
    v93 += 4;
    sub_B99FD0(v118, v96, v95);
  }
  while ( v94 != v93 );
  v66 = v118;
  v67 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
  if ( v67 == *(_DWORD *)(v30 + 72) )
    goto LABEL_103;
LABEL_72:
  v68 = (v67 + 1) & 0x7FFFFFF;
  v69 = v68 | *(_DWORD *)(v30 + 4) & 0xF8000000;
  v70 = *(_QWORD *)(v30 - 8) + 32LL * (unsigned int)(v68 - 1);
  *(_DWORD *)(v30 + 4) = v69;
  if ( *(_QWORD *)v70 )
  {
    v71 = *(_QWORD *)(v70 + 8);
    **(_QWORD **)(v70 + 16) = v71;
    if ( v71 )
      *(_QWORD *)(v71 + 16) = *(_QWORD *)(v70 + 16);
  }
  *(_QWORD *)v70 = v66;
  if ( v66 )
  {
    v72 = *(_QWORD *)(v66 + 16);
    *(_QWORD *)(v70 + 8) = v72;
    if ( v72 )
      *(_QWORD *)(v72 + 16) = v70 + 8;
    *(_QWORD *)(v70 + 16) = v66 + 16;
    *(_QWORD *)(v66 + 16) = v70;
  }
  *(_QWORD *)(*(_QWORD *)(v30 - 8) + 32LL * *(unsigned int *)(v30 + 72)
                                   + 8LL * ((*(_DWORD *)(v30 + 4) & 0x7FFFFFFu) - 1)) = v124;
  v128 = 257;
  v73 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v156 + 7);
  if ( v73 != sub_928890 )
  {
    v120 = v66;
    v109 = v73((__int64)v156, 32u, (_BYTE *)v66, v121);
    v66 = v120;
    v75 = (_QWORD *)v109;
LABEL_83:
    if ( v75 )
      goto LABEL_84;
    goto LABEL_93;
  }
  if ( *(_BYTE *)v66 <= 0x15u && *v121 <= 0x15u )
  {
    v116 = v66;
    v74 = sub_AAB310(0x20u, (unsigned __int8 *)v66, v121);
    v66 = v116;
    v75 = (_QWORD *)v74;
    goto LABEL_83;
  }
LABEL_93:
  v117 = v66;
  v130 = 257;
  v75 = sub_BD2C40(72, unk_3F10FD0);
  if ( v75 )
  {
    v83 = *(_QWORD *)(v117 + 8);
    v84 = *(unsigned __int8 *)(v83 + 8);
    if ( (unsigned int)(v84 - 17) > 1 )
    {
      v88 = sub_BCB2A0(*(_QWORD **)v83);
    }
    else
    {
      v85 = *(_DWORD *)(v83 + 32);
      v86 = *(_QWORD **)v83;
      BYTE4(v126) = (_BYTE)v84 == 18;
      LODWORD(v126) = v85;
      v87 = (__int64 *)sub_BCB2A0(v86);
      v88 = sub_BCE1B0(v87, v126);
    }
    sub_B523C0((__int64)v75, v88, 53, 32, v117, (__int64)v121, (__int64)v129, 0, 0, 0);
  }
  (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v157 + 2))(v157, v75, v127, v153, v154);
  v89 = v148;
  v90 = &v148[4 * (unsigned int)v149];
  if ( v148 != v90 )
  {
    do
    {
      v91 = *((_QWORD *)v89 + 1);
      v92 = *v89;
      v89 += 4;
      sub_B99FD0((__int64)v75, v92, v91);
    }
    while ( v90 != v89 );
  }
LABEL_84:
  v130 = 257;
  v76 = sub_BD2C40(72, 3u);
  v77 = (__int64)v76;
  if ( v76 )
    sub_B4C9A0((__int64)v76, v122, v124, (__int64)v75, 3u, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v157 + 2))(v157, v77, v129, v153, v154);
  v78 = v148;
  v79 = &v148[4 * (unsigned int)v149];
  if ( v148 != v79 )
  {
    do
    {
      v80 = *((_QWORD *)v78 + 1);
      v81 = *v78;
      v78 += 4;
      sub_B99FD0(v77, v81, v80);
    }
    while ( v79 != v78 );
  }
  sub_BD84D0(a2, v56);
  sub_B43D60((_QWORD *)a2);
  nullsub_61();
  v164 = &unk_49DA100;
  nullsub_63();
  if ( v148 != (unsigned int *)v150 )
    _libc_free((unsigned __int64)v148);
  nullsub_61();
  v146 = &unk_49DA100;
  nullsub_63();
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  return 1;
}
