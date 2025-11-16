// Function: sub_DE9D10
// Address: 0xde9d10
//
__int64 __fastcall sub_DE9D10(_QWORD *a1, __int64 a2, __int64 a3, __int64 i, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  char v8; // di
  int v9; // edi
  _QWORD *v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r15
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // r15
  _QWORD *v25; // r13
  _QWORD *v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 ***v31; // r9
  __int64 ***v32; // r15
  __int64 **v33; // r14
  __int64 **v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 **v40; // rdi
  char v41; // dl
  __int64 ***v42; // r9
  __int64 ***v43; // r15
  __int64 **v44; // r14
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 v50; // rax
  __int64 ***v51; // r9
  __int64 ***v52; // r15
  __int64 **v53; // r14
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  __int64 ***v60; // r9
  __int64 ***v61; // r15
  __int64 **v62; // r14
  __int64 v63; // rax
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  _QWORD *v67; // rax
  __int64 ***v68; // r9
  __int64 ***v69; // r15
  __int64 **v70; // r14
  __int64 v71; // rax
  __int64 v72; // r9
  __int64 v73; // rdx
  _QWORD *v74; // rax
  __int64 ***v75; // r9
  __int64 ***v76; // r15
  __int64 **v77; // r14
  __int64 v78; // rax
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rdx
  __int64 v82; // rcx
  unsigned __int64 v83; // rax
  __int64 ***v84; // r9
  __int64 ***v85; // r15
  __int64 **v86; // r14
  __int64 v87; // rax
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rdx
  _QWORD *v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r12
  __int64 v96; // r14
  __int64 v97; // rax
  __int64 ***v98; // r9
  __int64 ***v99; // r15
  __int64 **v100; // r14
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rdx
  __int64 *v105; // rax
  __int64 v106; // rdx
  __int64 v107; // rcx
  __int64 v108; // r8
  __int64 v109; // r12
  __int64 v110; // rsi
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 *v113; // r15
  __int64 v114; // rax
  __int64 v115; // r14
  __int64 v116; // rdi
  int v117; // r10d
  __int64 v118; // r14
  _QWORD *v119; // rax
  __int64 v120; // r8
  __int64 v121; // r15
  __int64 v122; // r9
  __int64 v123; // rax
  __int64 v124; // rdi
  __int64 v125; // rdi
  __int64 v126; // rdi
  __int64 v127; // [rsp+8h] [rbp-A8h]
  __int64 v128; // [rsp+8h] [rbp-A8h]
  __int64 v129; // [rsp+8h] [rbp-A8h]
  __int64 v130; // [rsp+8h] [rbp-A8h]
  __int64 v131; // [rsp+8h] [rbp-A8h]
  __int64 v132; // [rsp+8h] [rbp-A8h]
  __int64 v133; // [rsp+8h] [rbp-A8h]
  __int64 v134; // [rsp+8h] [rbp-A8h]
  unsigned int v135; // [rsp+10h] [rbp-A0h]
  __int64 ***v136; // [rsp+10h] [rbp-A0h]
  __int64 ***v137; // [rsp+10h] [rbp-A0h]
  __int64 ***v138; // [rsp+10h] [rbp-A0h]
  __int64 ***v139; // [rsp+10h] [rbp-A0h]
  __int64 ***v140; // [rsp+10h] [rbp-A0h]
  __int64 ***v141; // [rsp+10h] [rbp-A0h]
  __int64 ***v142; // [rsp+10h] [rbp-A0h]
  __int64 ***v143; // [rsp+10h] [rbp-A0h]
  __int64 v144; // [rsp+18h] [rbp-98h]
  char v145; // [rsp+18h] [rbp-98h]
  char v146; // [rsp+18h] [rbp-98h]
  char v147; // [rsp+18h] [rbp-98h]
  char v148; // [rsp+18h] [rbp-98h]
  char v149; // [rsp+18h] [rbp-98h]
  char v150; // [rsp+18h] [rbp-98h]
  char v151; // [rsp+18h] [rbp-98h]
  char v152; // [rsp+18h] [rbp-98h]
  __int64 *v153; // [rsp+18h] [rbp-98h]
  __int64 v154; // [rsp+18h] [rbp-98h]
  __int64 v155; // [rsp+18h] [rbp-98h]
  __int64 v156; // [rsp+20h] [rbp-90h]
  __int64 v157; // [rsp+20h] [rbp-90h]
  __int64 v158; // [rsp+20h] [rbp-90h]
  __int64 v159; // [rsp+20h] [rbp-90h]
  __int64 v160; // [rsp+28h] [rbp-88h] BYREF
  __int64 v161; // [rsp+38h] [rbp-78h] BYREF
  __int64 **v162; // [rsp+40h] [rbp-70h] BYREF
  __int64 v163; // [rsp+48h] [rbp-68h]
  __int64 *v164; // [rsp+50h] [rbp-60h] BYREF
  char v165; // [rsp+58h] [rbp-58h] BYREF
  char v166; // [rsp+70h] [rbp-40h]

  v6 = a2;
  v8 = *((_BYTE *)a1 + 16);
  v160 = a2;
  v9 = v8 & 1;
  if ( v9 )
  {
    v10 = a1 + 3;
    a6 = 3;
  }
  else
  {
    v15 = *((unsigned int *)a1 + 8);
    v10 = (_QWORD *)a1[3];
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    a6 = (unsigned int)(v15 - 1);
  }
  i = (unsigned int)a6 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = &v10[2 * i];
  v12 = *v11;
  if ( v6 == *v11 )
    goto LABEL_4;
  v17 = 1;
  while ( v12 != -4096 )
  {
    v117 = v17 + 1;
    i = (unsigned int)a6 & (v17 + (_DWORD)i);
    v11 = &v10[2 * (unsigned int)i];
    v12 = *v11;
    if ( v6 == *v11 )
      goto LABEL_4;
    v17 = v117;
  }
  if ( (_BYTE)v9 )
  {
    v16 = 8;
    goto LABEL_13;
  }
  v15 = *((unsigned int *)a1 + 8);
LABEL_12:
  v16 = 2 * v15;
LABEL_13:
  v11 = &v10[v16];
LABEL_4:
  v13 = 8;
  if ( !(_BYTE)v9 )
    v13 = 2LL * *((unsigned int *)a1 + 8);
  if ( v11 != &v10[v13] )
    return v11[1];
  switch ( *(_WORD *)(v6 + 24) )
  {
    case 0:
    case 1:
    case 0x10:
      goto LABEL_25;
    case 2:
      v110 = sub_DE9D10(a1, *(_QWORD *)(v6 + 32));
      if ( v110 != *(_QWORD *)(v6 + 32) )
        v6 = (__int64)sub_DC5200(*a1, v110, *(_QWORD *)(v6 + 40), 0);
      goto LABEL_25;
    case 3:
      v109 = sub_DE9D10(a1, *(_QWORD *)(v6 + 32));
      if ( *(_WORD *)(v109 + 24) != 8 || *(_QWORD *)(v109 + 48) != a1[13] || *(_QWORD *)(v109 + 40) != 2 )
        goto LABEL_102;
      v158 = sub_D33D80((_QWORD *)v109, *a1, v106, v107, v108);
      v118 = *(_QWORD *)(v6 + 40);
      v119 = sub_DA4270(*a1, v109, 1);
      v121 = a1[11];
      v122 = (__int64)v119;
      if ( v121 )
      {
        v123 = *(unsigned int *)(v121 + 8);
        if ( v123 + 1 > (unsigned __int64)*(unsigned int *)(v121 + 12) )
        {
          v155 = v122;
          sub_C8D5F0(a1[11], (const void *)(v121 + 16), v123 + 1, 8u, v120, v122);
          v123 = *(unsigned int *)(v121 + 8);
          v122 = v155;
        }
        *(_QWORD *)(*(_QWORD *)v121 + 8 * v123) = v122;
        ++*(_DWORD *)(v121 + 8);
      }
      else
      {
        v126 = a1[12];
        if ( !v126
          || !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v126 + 16LL))(v126, v119, *a1) )
        {
LABEL_102:
          v6 = (__int64)sub_DC2B70(*a1, v109, *(_QWORD *)(v6 + 40), 0);
          goto LABEL_25;
        }
      }
      v24 = *a1;
      v135 = *(_WORD *)(v109 + 28) & 7;
      v144 = a1[13];
      v25 = sub_DC5000(*a1, v158, v118, 0);
      v26 = sub_DC2B70(*a1, **(_QWORD **)(v109 + 32), v118, 0);
      goto LABEL_24;
    case 4:
      v95 = sub_DE9D10(a1, *(_QWORD *)(v6 + 32));
      if ( *(_WORD *)(v95 + 24) != 8 || *(_QWORD *)(v95 + 48) != a1[13] || *(_QWORD *)(v95 + 40) != 2 )
        goto LABEL_90;
      v156 = sub_D33D80((_QWORD *)v95, *a1, v92, v93, v94);
      v18 = *(_QWORD *)(v6 + 40);
      v19 = sub_DA4270(*a1, v95, 2);
      v21 = a1[11];
      v22 = (__int64)v19;
      if ( v21 )
      {
        v23 = *(unsigned int *)(v21 + 8);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
        {
          v154 = v22;
          sub_C8D5F0(a1[11], (const void *)(v21 + 16), v23 + 1, 8u, v20, v22);
          v23 = *(unsigned int *)(v21 + 8);
          v22 = v154;
        }
        *(_QWORD *)(*(_QWORD *)v21 + 8 * v23) = v22;
        ++*(_DWORD *)(v21 + 8);
      }
      else
      {
        v125 = a1[12];
        if ( !v125
          || !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v125 + 16LL))(v125, v19, *a1) )
        {
LABEL_90:
          v6 = (__int64)sub_DC5000(*a1, v95, *(_QWORD *)(v6 + 40), 0);
          goto LABEL_25;
        }
      }
      v24 = *a1;
      v135 = *(_WORD *)(v95 + 28) & 7;
      v144 = a1[13];
      v25 = sub_DC5000(*a1, v156, v18, 0);
      v26 = sub_DC5000(*a1, **(_QWORD **)(v95 + 32), v18, 0);
LABEL_24:
      v6 = (__int64)sub_DC1960(v24, (__int64)v26, (__int64)v25, v144, v135);
      goto LABEL_25;
    case 5:
      v162 = &v164;
      v163 = 0x200000000LL;
      v84 = *(__int64 ****)(v6 + 32);
      v142 = &v84[*(_QWORD *)(v6 + 40)];
      if ( v84 == v142 )
        goto LABEL_25;
      v151 = 0;
      v85 = *(__int64 ****)(v6 + 32);
      do
      {
        v86 = *v85;
        v34 = *v85;
        v87 = sub_DE9D10(a1, *v85);
        v90 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v128 = v87;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v88, v89);
          v90 = (unsigned int)v163;
          v87 = v128;
        }
        v162[v90] = (__int64 *)v87;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v85;
        v151 |= v162[(unsigned int)v163 - 1] != (__int64 *)v86;
      }
      while ( v142 != v85 );
      if ( v151 )
      {
        v34 = (__int64 **)&v162;
        v91 = sub_DC7EB0((__int64 *)*a1, (__int64)&v162, 0, 0);
        v40 = v162;
        v6 = (__int64)v91;
      }
      goto LABEL_51;
    case 6:
      v162 = &v164;
      v163 = 0x200000000LL;
      v98 = *(__int64 ****)(v6 + 32);
      v143 = &v98[*(_QWORD *)(v6 + 40)];
      if ( v98 == v143 )
        goto LABEL_25;
      v152 = 0;
      v99 = *(__int64 ****)(v6 + 32);
      do
      {
        v100 = *v99;
        v34 = *v99;
        v101 = sub_DE9D10(a1, *v99);
        v104 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v133 = v101;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v102, v103);
          v104 = (unsigned int)v163;
          v101 = v133;
        }
        v162[v104] = (__int64 *)v101;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v99;
        v152 |= v162[(unsigned int)v163 - 1] != (__int64 *)v100;
      }
      while ( v143 != v99 );
      if ( v152 )
      {
        v34 = (__int64 **)&v162;
        v105 = sub_DC8BD0((__int64 *)*a1, (__int64)&v162, 0, 0);
        v40 = v162;
        v6 = (__int64)v105;
      }
      goto LABEL_51;
    case 7:
      v96 = sub_DE9D10(a1, *(_QWORD *)(v6 + 32));
      v97 = sub_DE9D10(a1, *(_QWORD *)(v6 + 40));
      if ( v96 != *(_QWORD *)(v6 + 32) || v97 != *(_QWORD *)(v6 + 40) )
        v6 = sub_DCB270(*a1, v96, v97);
      goto LABEL_25;
    case 8:
      v162 = &v164;
      v163 = 0x200000000LL;
      v60 = *(__int64 ****)(v6 + 32);
      v139 = &v60[*(_QWORD *)(v6 + 40)];
      if ( v60 == v139 )
        goto LABEL_25;
      v148 = 0;
      v61 = *(__int64 ****)(v6 + 32);
      do
      {
        v62 = *v61;
        v34 = *v61;
        v63 = sub_DE9D10(a1, *v61);
        v66 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v132 = v63;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v64, v65);
          v66 = (unsigned int)v163;
          v63 = v132;
        }
        v162[v66] = (__int64 *)v63;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v61;
        v148 |= v162[(unsigned int)v163 - 1] != (__int64 *)v62;
      }
      while ( v139 != v61 );
      if ( v148 )
      {
        v34 = (__int64 **)&v162;
        v67 = sub_DBFF60(*a1, (unsigned int *)&v162, *(_QWORD *)(v6 + 48), *(_WORD *)(v6 + 28) & 7);
        v40 = v162;
        v6 = (__int64)v67;
      }
      goto LABEL_51;
    case 9:
      v162 = &v164;
      v163 = 0x200000000LL;
      v51 = *(__int64 ****)(v6 + 32);
      v138 = &v51[*(_QWORD *)(v6 + 40)];
      if ( v51 == v138 )
        goto LABEL_25;
      v147 = 0;
      v52 = *(__int64 ****)(v6 + 32);
      do
      {
        v53 = *v52;
        v34 = *v52;
        v54 = sub_DE9D10(a1, *v52);
        v57 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v134 = v54;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v55, v56);
          v57 = (unsigned int)v163;
          v54 = v134;
        }
        v58 = (__int64)v162;
        v162[v57] = (__int64 *)v54;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v52;
        v147 |= v162[(unsigned int)v163 - 1] != (__int64 *)v53;
      }
      while ( v138 != v52 );
      if ( v147 )
      {
        v34 = (__int64 **)&v162;
        v59 = sub_DCE040((__int64 *)*a1, (__int64)&v162, v57, v58, v55);
        v40 = v162;
        v6 = v59;
      }
      goto LABEL_51;
    case 0xA:
      v162 = &v164;
      v163 = 0x200000000LL;
      v75 = *(__int64 ****)(v6 + 32);
      v141 = &v75[*(_QWORD *)(v6 + 40)];
      if ( v75 == v141 )
        goto LABEL_25;
      v150 = 0;
      v76 = *(__int64 ****)(v6 + 32);
      do
      {
        v77 = *v76;
        v34 = *v76;
        v78 = sub_DE9D10(a1, *v76);
        v81 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v130 = v78;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v79, v80);
          v81 = (unsigned int)v163;
          v78 = v130;
        }
        v82 = (__int64)v162;
        v162[v81] = (__int64 *)v78;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v76;
        v150 |= v162[(unsigned int)v163 - 1] != (__int64 *)v77;
      }
      while ( v141 != v76 );
      if ( v150 )
      {
        v34 = (__int64 **)&v162;
        v83 = sub_DCDF90((__int64 *)*a1, (__int64)&v162, v81, v82, v79);
        v40 = v162;
        v6 = v83;
      }
      goto LABEL_51;
    case 0xB:
      v162 = &v164;
      v163 = 0x200000000LL;
      v68 = *(__int64 ****)(v6 + 32);
      v140 = &v68[*(_QWORD *)(v6 + 40)];
      if ( v68 == v140 )
        goto LABEL_25;
      v149 = 0;
      v69 = *(__int64 ****)(v6 + 32);
      do
      {
        v70 = *v69;
        v34 = *v69;
        v71 = sub_DE9D10(a1, *v69);
        v73 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v127 = v71;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v36, v72);
          v73 = (unsigned int)v163;
          v71 = v127;
        }
        v39 = (__int64)v162;
        v162[v73] = (__int64 *)v71;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v69;
        v149 |= v162[(unsigned int)v163 - 1] != (__int64 *)v70;
      }
      while ( v140 != v69 );
      v41 = 0;
      if ( v149 )
        goto LABEL_73;
      goto LABEL_51;
    case 0xC:
      v162 = &v164;
      v163 = 0x200000000LL;
      v42 = *(__int64 ****)(v6 + 32);
      v137 = &v42[*(_QWORD *)(v6 + 40)];
      if ( v42 == v137 )
        goto LABEL_25;
      v146 = 0;
      v43 = *(__int64 ****)(v6 + 32);
      do
      {
        v44 = *v43;
        v34 = *v43;
        v45 = sub_DE9D10(a1, *v43);
        v48 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v129 = v45;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v46, v47);
          v48 = (unsigned int)v163;
          v45 = v129;
        }
        v49 = (__int64)v162;
        v162[v48] = (__int64 *)v45;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v43;
        v146 |= v162[(unsigned int)v163 - 1] != (__int64 *)v44;
      }
      while ( v137 != v43 );
      if ( v146 )
      {
        v34 = (__int64 **)&v162;
        v50 = sub_DCE150((__int64 *)*a1, (__int64)&v162, v48, v49, v46);
        v40 = v162;
        v6 = v50;
      }
      goto LABEL_51;
    case 0xD:
      v162 = &v164;
      v163 = 0x200000000LL;
      v31 = *(__int64 ****)(v6 + 32);
      v136 = &v31[*(_QWORD *)(v6 + 40)];
      if ( v31 == v136 )
        goto LABEL_25;
      v145 = 0;
      v32 = *(__int64 ****)(v6 + 32);
      do
      {
        v33 = *v32;
        v34 = *v32;
        v35 = sub_DE9D10(a1, *v32);
        v38 = (unsigned int)v163;
        if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
        {
          v34 = &v164;
          v131 = v35;
          sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 8u, v36, v37);
          v38 = (unsigned int)v163;
          v35 = v131;
        }
        v39 = (__int64)v162;
        v162[v38] = (__int64 *)v35;
        v40 = v162;
        LODWORD(v163) = v163 + 1;
        ++v32;
        v145 |= v162[(unsigned int)v163 - 1] != (__int64 *)v33;
      }
      while ( v136 != v32 );
      if ( !v145 )
        goto LABEL_51;
      v41 = 1;
LABEL_73:
      v34 = (__int64 **)&v162;
      v74 = sub_DCEE50((__int64 *)*a1, (__int64)&v162, v41, v39, v36);
      v40 = v162;
      v6 = (__int64)v74;
LABEL_51:
      if ( v40 != &v164 )
        goto LABEL_52;
      goto LABEL_25;
    case 0xE:
      v30 = sub_DE9D10(a1, *(_QWORD *)(v6 + 32));
      if ( v30 != *(_QWORD *)(v6 + 32) )
        v6 = (__int64)sub_DD3A70(*a1, v30, *(_QWORD *)(v6 + 40));
      goto LABEL_25;
    case 0xF:
      v27 = a1[12];
      if ( !v27 )
        goto LABEL_108;
      v28 = *(_DWORD *)(v27 + 32);
      if ( v28 )
      {
        if ( v28 != 1 || *(_QWORD *)(v27 + 40) != v6 || *(_DWORD *)(v27 + 36) != 32 )
          goto LABEL_108;
LABEL_34:
        v6 = *(_QWORD *)(v27 + 48);
        goto LABEL_25;
      }
      v29 = *(__int64 **)(v27 + 40);
      for ( i = (__int64)&v29[*(unsigned int *)(v27 + 48)]; (__int64 *)i != v29; ++v29 )
      {
        v27 = *v29;
        if ( *(_DWORD *)(*v29 + 32) == 1 && *(_QWORD *)(v27 + 40) == v6 && *(_DWORD *)(v27 + 36) == 32 )
          goto LABEL_34;
      }
LABEL_108:
      if ( **(_BYTE **)(v6 - 8) == 84 )
      {
        v34 = (__int64 **)*a1;
        sub_DE97B0((__int64)&v162, *a1, v6 - 32, i, a5, a6);
        if ( v166 )
        {
          v113 = (__int64 *)v163;
          v153 = (__int64 *)(v163 + 8LL * (unsigned int)v164);
          if ( (__int64 *)v163 == v153 )
          {
LABEL_134:
            v6 = (__int64)v162;
          }
          else
          {
            while ( 1 )
            {
              v115 = *v113;
              if ( *(_DWORD *)(*v113 + 32) == 2 )
              {
                v157 = a1[13];
                if ( v157 != *(_QWORD *)(sub_D9ABD0(*v113) + 48) )
                  break;
              }
              v124 = a1[11];
              if ( v124 )
              {
                v114 = *(unsigned int *)(v124 + 8);
                if ( v114 + 1 > (unsigned __int64)*(unsigned int *)(v124 + 12) )
                {
                  v34 = (__int64 **)(v124 + 16);
                  v159 = a1[11];
                  sub_C8D5F0(v124, (const void *)(v124 + 16), v114 + 1, 8u, v111, v112);
                  v124 = v159;
                  v114 = *(unsigned int *)(v159 + 8);
                }
                *(_QWORD *)(*(_QWORD *)v124 + 8 * v114) = v115;
                ++*(_DWORD *)(v124 + 8);
              }
              else
              {
                v116 = a1[12];
                if ( !v116 )
                  break;
                v34 = (__int64 **)v115;
                if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v116 + 16LL))(
                        v116,
                        v115,
                        *a1) )
                  break;
              }
              if ( v153 == ++v113 )
                goto LABEL_134;
            }
          }
          if ( v166 )
          {
            v40 = (__int64 **)v163;
            if ( (char *)v163 != &v165 )
LABEL_52:
              _libc_free(v40, v34);
          }
        }
      }
LABEL_25:
      v161 = v6;
      sub_DB11F0((__int64)&v162, (__int64)(a1 + 1), &v160, &v161);
      v11 = v164;
      return v11[1];
    default:
      BUG();
  }
}
