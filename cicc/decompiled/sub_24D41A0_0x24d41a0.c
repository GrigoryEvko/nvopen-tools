// Function: sub_24D41A0
// Address: 0x24d41a0
//
__int64 __fastcall sub_24D41A0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int8 a6,
        unsigned __int8 a7,
        _BYTE *a8,
        unsigned __int8 *a9,
        char a10,
        char a11,
        __int64 a12)
{
  unsigned __int64 v14; // r14
  __int64 *v15; // rdi
  __int64 **v16; // rax
  _BYTE *v17; // rax
  __int64 **v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int8 *v21; // rbx
  __int64 (__fastcall *v22)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int8 *v26; // rbx
  __int64 (__fastcall *v27)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  _BYTE *v28; // r15
  _BYTE *v29; // rax
  __int64 *v30; // rdi
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r14
  __int64 v37; // rax
  char v38; // al
  _QWORD *v39; // rax
  __int64 v40; // r15
  __int64 v41; // r14
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rsi
  _BYTE *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  unsigned __int8 *v51; // r14
  __int64 **v52; // r13
  __int64 v53; // rcx
  __int64 v54; // rdi
  _BYTE *v55; // rax
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // r15
  __int64 v60; // rax
  char v61; // al
  char v62; // si
  _QWORD *v63; // rax
  __int64 v64; // r9
  __int64 v65; // r13
  __int64 v66; // rsi
  __int64 v67; // rbx
  unsigned int *v68; // r15
  __int64 v69; // rdx
  _BYTE *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdi
  unsigned __int8 *v73; // rbx
  __int64 (__fastcall *v74)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v75; // rax
  __int64 v76; // r15
  unsigned __int64 v77; // rax
  __int64 *v78; // rdi
  __int64 **v79; // rax
  const char *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rsi
  __int64 *v83; // rdi
  __int64 **v84; // rax
  const char *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rsi
  unsigned __int8 *v88; // r14
  __int64 **v89; // r15
  __int64 v90; // rcx
  __int64 v91; // rdi
  _BYTE *v92; // rax
  unsigned __int64 v93; // rax
  unsigned __int64 v94; // rax
  __int64 *v95; // rdi
  __int64 v96; // rbx
  __int64 v97; // r13
  __int64 v98; // rax
  char v99; // si
  _QWORD *v100; // r15
  __int64 v101; // rbx
  __int64 v102; // r13
  __int64 v103; // rdx
  unsigned int v104; // esi
  unsigned __int64 v105; // r15
  _BYTE *v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdi
  unsigned __int8 *v109; // r15
  __int64 (__fastcall *v110)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v111; // rax
  __int64 v112; // rax
  unsigned __int64 v113; // rax
  __int64 *v114; // rdi
  __int64 **v115; // rax
  const char *v116; // rax
  __int64 v117; // rsi
  __int64 v118; // rdx
  __int64 v120; // r15
  __int64 v121; // rbx
  __int64 v122; // rdx
  unsigned int v123; // esi
  __int64 v124; // r13
  __int64 i; // rbx
  __int64 v126; // rdx
  unsigned int v127; // esi
  __int64 v128; // r15
  __int64 v129; // rbx
  __int64 v130; // rdx
  unsigned int v131; // esi
  __int64 v132; // r14
  __int64 v133; // rbx
  __int64 v134; // rdx
  unsigned int v135; // esi
  __int64 v136; // rax
  __int64 v137; // r15
  __int64 v138; // rax
  __int64 v139; // rdi
  __int64 v140; // r13
  __int64 v141; // rax
  char v142; // bl
  _QWORD *v143; // rax
  __int64 v144; // r9
  __int64 v145; // r14
  __int64 v146; // rsi
  unsigned int *v147; // r13
  __int64 v148; // rbx
  __int64 v149; // rdx
  _BYTE *v150; // rax
  __int64 v151; // rdi
  __int64 v152; // rax
  unsigned __int64 v153; // rbx
  unsigned __int8 *v154; // rdi
  __int64 v155; // [rsp-10h] [rbp-1F0h]
  __int64 v156; // [rsp-8h] [rbp-1E8h]
  __int64 v157; // [rsp+0h] [rbp-1E0h]
  __int64 v158; // [rsp+8h] [rbp-1D8h]
  __int64 v159; // [rsp+10h] [rbp-1D0h]
  _QWORD *v162; // [rsp+28h] [rbp-1B8h]
  __int64 **v163; // [rsp+28h] [rbp-1B8h]
  _QWORD *v164; // [rsp+38h] [rbp-1A8h]
  char v165; // [rsp+48h] [rbp-198h]
  __int64 v166; // [rsp+48h] [rbp-198h]
  __int64 v167; // [rsp+48h] [rbp-198h]
  __int64 v168; // [rsp+58h] [rbp-188h]
  unsigned __int64 v169; // [rsp+60h] [rbp-180h] BYREF
  __int64 v170; // [rsp+68h] [rbp-178h] BYREF
  _BYTE *v171; // [rsp+78h] [rbp-168h] BYREF
  _BYTE *v172; // [rsp+80h] [rbp-160h] BYREF
  __int64 **v173; // [rsp+88h] [rbp-158h] BYREF
  unsigned __int64 v174; // [rsp+90h] [rbp-150h] BYREF
  __int64 v175; // [rsp+98h] [rbp-148h] BYREF
  __int64 v176; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v177; // [rsp+A8h] [rbp-138h] BYREF
  unsigned __int64 v178[4]; // [rsp+B0h] [rbp-130h] BYREF
  char v179; // [rsp+D0h] [rbp-110h]
  char v180; // [rsp+D1h] [rbp-10Fh]
  const char *v181; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v182; // [rsp+E8h] [rbp-F8h]
  _BYTE *v183; // [rsp+F0h] [rbp-F0h]
  __int64 v184; // [rsp+F8h] [rbp-E8h]
  __int16 v185; // [rsp+100h] [rbp-E0h]
  _QWORD v186[4]; // [rsp+110h] [rbp-D0h] BYREF
  __int16 v187; // [rsp+130h] [rbp-B0h]
  const char *v188[4]; // [rsp+140h] [rbp-A0h] BYREF
  __int16 v189; // [rsp+160h] [rbp-80h]
  unsigned int **v190[4]; // [rsp+170h] [rbp-70h] BYREF
  _BYTE **v191; // [rsp+190h] [rbp-50h]
  _QWORD *v192; // [rsp+198h] [rbp-48h]
  __int64 ***v193; // [rsp+1A0h] [rbp-40h]

  v170 = a3;
  v169 = a5;
  if ( a3 )
  {
    v14 = *sub_24D3EB0(a12, &v170);
  }
  else
  {
    v136 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
    v14 = sub_AD6530(v136, 0);
  }
  v15 = *(__int64 **)(a2 + 72);
  LOWORD(v191) = 257;
  v16 = (__int64 **)sub_BCE3C0(v15, 0);
  v17 = (_BYTE *)sub_24D30A0((__int64 *)a2, 0x31u, v14, v16, (__int64)v190, 0, (int)v188[0], 0);
  v18 = (__int64 **)a1[9];
  v171 = v17;
  v168 = a1[10];
  v188[0] = "shadow.ptr.int";
  v186[0] = "app.ptr.shifted";
  v181 = "app.ptr.masked";
  v178[0] = (unsigned __int64)"app.ptr.int";
  v189 = 259;
  v187 = 259;
  v185 = 259;
  v180 = 1;
  v179 = 3;
  v19 = sub_24D30A0((__int64 *)a2, 0x2Fu, a4, v18, (__int64)v178, 0, (int)v190[0], 0);
  v20 = *(_QWORD *)(a2 + 80);
  v21 = (unsigned __int8 *)v19;
  v22 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v20 + 16LL);
  if ( v22 != sub_9202E0 )
  {
    v23 = v22(v20, 28u, v21, a9);
    goto LABEL_8;
  }
  if ( *v21 <= 0x15u && *a9 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v23 = sub_AD5570(28, (__int64)v21, a9, 0, 0);
    else
      v23 = sub_AABE40(0x1Cu, v21, a9);
LABEL_8:
    if ( v23 )
      goto LABEL_9;
  }
  LOWORD(v191) = 257;
  v23 = sub_B504D0(28, (__int64)v21, (__int64)a9, (__int64)v190, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v23,
    &v181,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v120 = *(_QWORD *)a2;
  v121 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v121 )
  {
    do
    {
      v122 = *(_QWORD *)(v120 + 8);
      v123 = *(_DWORD *)v120;
      v120 += 16;
      sub_B99FD0(v23, v123, v122);
    }
    while ( v121 != v120 );
  }
LABEL_9:
  v24 = sub_AD64C0(*(_QWORD *)(v23 + 8), v168, 0);
  v25 = *(_QWORD *)(a2 + 80);
  v26 = (unsigned __int8 *)v24;
  v27 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v25 + 32LL);
  if ( v27 != sub_9201A0 )
  {
    v28 = (_BYTE *)v27(v25, 25u, (_BYTE *)v23, v26, 0, 0);
    goto LABEL_14;
  }
  if ( *(_BYTE *)v23 <= 0x15u && *v26 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(25) )
      v28 = (_BYTE *)sub_AD5570(25, v23, v26, 0, 0);
    else
      v28 = (_BYTE *)sub_AABE40(0x19u, (unsigned __int8 *)v23, v26);
LABEL_14:
    if ( v28 )
      goto LABEL_15;
  }
  LOWORD(v191) = 257;
  v28 = (_BYTE *)sub_B504D0(25, v23, (__int64)v26, (__int64)v190, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v28,
    v186,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v132 = *(_QWORD *)a2;
  v133 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v133 )
  {
    do
    {
      v134 = *(_QWORD *)(v132 + 8);
      v135 = *(_DWORD *)v132;
      v132 += 16;
      sub_B99FD0((__int64)v28, v135, v134);
    }
    while ( v133 != v132 );
  }
LABEL_15:
  v29 = (_BYTE *)sub_929C50((unsigned int **)a2, v28, a8, (__int64)v188, 0, 0);
  v30 = *(__int64 **)(a2 + 72);
  v172 = v29;
  v173 = (__int64 **)sub_BCE3C0(v30, 0);
  v190[0] = (unsigned int **)"shadow.ptr";
  LOWORD(v191) = 259;
  v31 = sub_24D30A0((__int64 *)a2, 0x30u, (unsigned __int64)v172, v173, (__int64)v190, 0, (int)v188[0], 0);
  v190[0] = (unsigned int **)a2;
  v174 = v31;
  v190[1] = (unsigned int **)&v171;
  v190[2] = (unsigned int **)&v174;
  v190[3] = (unsigned int **)&v169;
  v191 = &v172;
  v192 = a1;
  v193 = &v173;
  if ( a10 || (_BYTE)qword_4FEE468 && a7 )
  {
    sub_24D3240(v190);
  }
  else
  {
    v188[0] = *(const char **)(a2 + 72);
    v159 = sub_B8C2F0(v188, 1u, 0x186A0u, 0);
    if ( a11 )
    {
      v32 = sub_ACD640(a1[11], a6 | (2 * a7), 0);
      v33 = v174;
      v157 = v32;
      v34 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
      v35 = *(_QWORD *)(a2 + 48);
      v36 = v34;
      v187 = 259;
      v186[0] = "shadow.desc";
      v37 = sub_AA4E30(v35);
      v38 = sub_AE5020(v37, v36);
      v189 = 257;
      v165 = v38;
      v39 = sub_BD2C40(80, unk_3F10A14);
      v40 = (__int64)v39;
      if ( v39 )
        sub_B4D190((__int64)v39, v36, v33, (__int64)v188, 0, v165, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v40,
        v186,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v41 = *(_QWORD *)a2;
      v42 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v42 )
      {
        do
        {
          v43 = *(_QWORD *)(v41 + 8);
          v44 = *(_DWORD *)v41;
          v41 += 16;
          sub_B99FD0(v40, v44, v43);
        }
        while ( v42 != v41 );
      }
      v189 = 259;
      v188[0] = "bad.desc";
      v45 = sub_92B530((unsigned int **)a2, 0x21u, v40, v171, (__int64)v188);
      v46 = *(_QWORD *)(a2 + 56);
      if ( v46 )
        v46 -= 24;
      sub_F38330(v45, (__int64 *)(v46 + 24), 0, (unsigned __int64 *)&v175, (unsigned __int64 *)&v176, v159, 0, 0);
      v47 = v175;
      sub_D5F1F0(a2, v175);
      v189 = 257;
      v48 = (_BYTE *)sub_AD6530(*(_QWORD *)(v40 + 8), v47);
      v49 = sub_92B530((unsigned int **)a2, 0x20u, v40, v48, (__int64)v188);
      v50 = *(_QWORD *)(a2 + 56);
      if ( v50 )
        v50 -= 24;
      sub_F38330(v49, (__int64 *)(v50 + 24), 0, (unsigned __int64 *)&v177, v178, 0, 0, 0);
      sub_D5F1F0(a2, v177);
      v158 = sub_ACD640(a1[11], v169, 0);
      v51 = (unsigned __int8 *)sub_ACD720(*(__int64 **)(a2 + 72));
      if ( v169 > 1 )
      {
        v162 = a1;
        v166 = 1;
        while ( 1 )
        {
          v189 = 257;
          v52 = v173;
          v53 = v162[10];
          v54 = v162[9];
          v187 = 257;
          v55 = (_BYTE *)sub_AD64C0(v54, v166 << v53, 0);
          v56 = sub_929C50((unsigned int **)a2, v172, v55, (__int64)v186, 0, 0);
          v57 = sub_24D30A0((__int64 *)a2, 0x30u, v56, v52, (__int64)v188, 0, (int)v181, 0);
          v187 = 257;
          v58 = v57;
          v59 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
          v60 = sub_AA4E30(*(_QWORD *)(a2 + 48));
          v61 = sub_AE5020(v60, v59);
          v189 = 257;
          v62 = v61;
          v63 = sub_BD2C40(80, unk_3F10A14);
          v64 = v155;
          v65 = (__int64)v63;
          if ( v63 )
            sub_B4D190((__int64)v63, v59, v58, (__int64)v188, 0, v62, 0, 0);
          v66 = v65;
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v65,
            v186,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64),
            v64);
          v67 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v67 )
          {
            v68 = *(unsigned int **)a2;
            do
            {
              v69 = *((_QWORD *)v68 + 1);
              v66 = *v68;
              v68 += 4;
              sub_B99FD0(v65, v66, v69);
            }
            while ( (unsigned int *)v67 != v68 );
          }
          v187 = 257;
          v185 = 257;
          v70 = (_BYTE *)sub_AD6530(*(_QWORD *)(v65 + 8), v66);
          v71 = sub_92B530((unsigned int **)a2, 0x21u, v65, v70, (__int64)&v181);
          v72 = *(_QWORD *)(a2 + 80);
          v73 = (unsigned __int8 *)v71;
          v74 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v72 + 16LL);
          if ( v74 == sub_9202E0 )
          {
            if ( *v51 > 0x15u || *v73 > 0x15u )
            {
LABEL_68:
              v189 = 257;
              v51 = (unsigned __int8 *)sub_B504D0(29, (__int64)v51, (__int64)v73, (__int64)v188, 0, 0);
              (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
                *(_QWORD *)(a2 + 88),
                v51,
                v186,
                *(_QWORD *)(a2 + 56),
                *(_QWORD *)(a2 + 64));
              v124 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              for ( i = *(_QWORD *)a2; v124 != i; i += 16 )
              {
                v126 = *(_QWORD *)(i + 8);
                v127 = *(_DWORD *)i;
                sub_B99FD0((__int64)v51, v127, v126);
              }
              goto LABEL_41;
            }
            if ( (unsigned __int8)sub_AC47B0(29) )
              v75 = sub_AD5570(29, (__int64)v51, v73, 0, 0);
            else
              v75 = sub_AABE40(0x1Du, v51, v73);
          }
          else
          {
            v75 = v74(v72, 29u, v51, v73);
          }
          if ( !v75 )
            goto LABEL_68;
          v51 = (unsigned __int8 *)v75;
LABEL_41:
          if ( v169 <= ++v166 )
          {
            a1 = v162;
            break;
          }
        }
      }
      v76 = *(_QWORD *)(a2 + 56);
      if ( v76 )
        v76 -= 24;
      v77 = sub_F38250((__int64)v51, (__int64 *)(v76 + 24), 0, 0, v159, 0, 0, 0);
      sub_D5F1F0(a2, v77);
      v78 = *(__int64 **)(a2 + 72);
      v187 = 257;
      v189 = 257;
      v79 = (__int64 **)sub_BCE3C0(v78, 0);
      v80 = (const char *)sub_24D30A0((__int64 *)a2, 0x31u, a4, v79, (__int64)v186, 0, (int)v181, 0);
      v81 = a1[13];
      v82 = a1[12];
      v181 = v80;
      v183 = v171;
      v184 = v157;
      v182 = v158;
      sub_24D2D90((__int64 *)a2, v82, v81, (__int64 *)&v181, 4, (__int64)v188, 0);
      sub_D5F1F0(a2, v76);
      sub_24D3240(v190);
      sub_D5F1F0(a2, v178[0]);
      v189 = 257;
      v83 = *(__int64 **)(a2 + 72);
      v187 = 257;
      v84 = (__int64 **)sub_BCE3C0(v83, 0);
      v85 = (const char *)sub_24D30A0((__int64 *)a2, 0x31u, a4, v84, (__int64)v186, 0, (int)v181, 0);
      v86 = a1[13];
      v87 = a1[12];
      v181 = v85;
      v184 = v157;
      v182 = v158;
      v183 = v171;
      sub_24D2D90((__int64 *)a2, v87, v86, (__int64 *)&v181, 4, (__int64)v188, 0);
      sub_D5F1F0(a2, v176);
      v88 = (unsigned __int8 *)sub_ACD720(*(__int64 **)(a2 + 72));
      if ( v169 > 1 )
      {
        v167 = 1;
        v164 = a1;
        while ( 1 )
        {
          v189 = 257;
          v89 = v173;
          v90 = v164[10];
          v91 = v164[9];
          v187 = 257;
          v92 = (_BYTE *)sub_AD64C0(v91, v167 << v90, 0);
          v93 = sub_929C50((unsigned int **)a2, v172, v92, (__int64)v186, 0, 0);
          v94 = sub_24D30A0((__int64 *)a2, 0x30u, v93, v89, (__int64)v188, 0, (int)v181, 0);
          v95 = *(__int64 **)(a2 + 72);
          v96 = v94;
          v187 = 257;
          v163 = (__int64 **)v164[9];
          v185 = 257;
          v97 = sub_BCE3C0(v95, 0);
          v98 = sub_AA4E30(*(_QWORD *)(a2 + 48));
          v99 = sub_AE5020(v98, v97);
          v189 = 257;
          v100 = sub_BD2C40(80, unk_3F10A14);
          if ( v100 )
            sub_B4D190((__int64)v100, v97, v96, (__int64)v188, 0, v99, 0, 0);
          (*(void (__fastcall **)(_QWORD, _QWORD *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v100,
            &v181,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64));
          v101 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v101 )
          {
            v102 = *(_QWORD *)a2;
            do
            {
              v103 = *(_QWORD *)(v102 + 8);
              v104 = *(_DWORD *)v102;
              v102 += 16;
              sub_B99FD0((__int64)v100, v104, v103);
            }
            while ( v101 != v102 );
          }
          v105 = sub_24D30A0((__int64 *)a2, 0x2Fu, (unsigned __int64)v100, v163, (__int64)v186, 0, (int)v188[0], 0);
          v187 = 257;
          v185 = 257;
          v106 = (_BYTE *)sub_AD64C0(v164[9], 0, 0);
          v107 = sub_92B530((unsigned int **)a2, 0x27u, v105, v106, (__int64)&v181);
          v108 = *(_QWORD *)(a2 + 80);
          v109 = (unsigned __int8 *)v107;
          v110 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v108 + 16LL);
          if ( v110 == sub_9202E0 )
          {
            if ( *v88 > 0x15u || *v109 > 0x15u )
            {
LABEL_71:
              v189 = 257;
              v88 = (unsigned __int8 *)sub_B504D0(29, (__int64)v88, (__int64)v109, (__int64)v188, 0, 0);
              (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
                *(_QWORD *)(a2 + 88),
                v88,
                v186,
                *(_QWORD *)(a2 + 56),
                *(_QWORD *)(a2 + 64));
              v128 = *(_QWORD *)a2;
              v129 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              if ( *(_QWORD *)a2 != v129 )
              {
                do
                {
                  v130 = *(_QWORD *)(v128 + 8);
                  v131 = *(_DWORD *)v128;
                  v128 += 16;
                  sub_B99FD0((__int64)v88, v131, v130);
                }
                while ( v129 != v128 );
              }
              goto LABEL_59;
            }
            if ( (unsigned __int8)sub_AC47B0(29) )
              v111 = sub_AD5570(29, (__int64)v88, v109, 0, 0);
            else
              v111 = sub_AABE40(0x1Du, v88, v109);
          }
          else
          {
            v111 = v110(v108, 29u, v88, v109);
          }
          if ( !v111 )
            goto LABEL_71;
          v88 = (unsigned __int8 *)v111;
LABEL_59:
          if ( v169 <= ++v167 )
          {
            a1 = v164;
            break;
          }
        }
      }
      v112 = *(_QWORD *)(a2 + 56);
      if ( v112 )
        v112 -= 24;
      v113 = sub_F38250((__int64)v88, (__int64 *)(v112 + 24), 0, 0, v159, 0, 0, 0);
      sub_D5F1F0(a2, v113);
      v114 = *(__int64 **)(a2 + 72);
      v187 = 257;
      v189 = 257;
      v115 = (__int64 **)sub_BCE3C0(v114, 0);
      v116 = (const char *)sub_24D30A0((__int64 *)a2, 0x31u, a4, v115, (__int64)v186, 0, (int)v181, 0);
      v117 = a1[12];
      v181 = v116;
      v118 = a1[13];
      v182 = v158;
      v183 = v171;
      v184 = v157;
      sub_24D2D90((__int64 *)a2, v117, v118, (__int64 *)&v181, 4, (__int64)v188, 0);
    }
    else
    {
      v137 = v174;
      v138 = sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
      v139 = *(_QWORD *)(a2 + 48);
      v140 = v138;
      v187 = 259;
      v186[0] = "shadow.desc";
      v141 = sub_AA4E30(v139);
      v142 = sub_AE5020(v141, v140);
      v189 = 257;
      v143 = sub_BD2C40(80, unk_3F10A14);
      v145 = (__int64)v143;
      if ( v143 )
      {
        sub_B4D190((__int64)v143, v140, v137, (__int64)v188, 0, v142, 0, 0);
        v144 = v156;
      }
      v146 = v145;
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v145,
        v186,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64),
        v144);
      v147 = *(unsigned int **)a2;
      v148 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v148 )
      {
        do
        {
          v149 = *((_QWORD *)v147 + 1);
          v146 = *v147;
          v147 += 4;
          sub_B99FD0(v145, v146, v149);
        }
        while ( (unsigned int *)v148 != v147 );
      }
      v188[0] = "desc.set";
      v189 = 259;
      v150 = (_BYTE *)sub_AD6530(*(_QWORD *)(v145 + 8), v146);
      v151 = sub_92B530((unsigned int **)a2, 0x20u, v145, v150, (__int64)v188);
      v152 = *(_QWORD *)(a2 + 56);
      if ( v152 )
        v152 -= 24;
      v153 = sub_F38250(v151, (__int64 *)(v152 + 24), 0, 0, v159, 0, 0, 0);
      sub_D5F1F0(a2, v153);
      v154 = *(unsigned __int8 **)(v153 + 40);
      v189 = 259;
      v188[0] = "set.type";
      sub_BD6B50(v154, v188);
      sub_24D3240(v190);
    }
  }
  return 1;
}
