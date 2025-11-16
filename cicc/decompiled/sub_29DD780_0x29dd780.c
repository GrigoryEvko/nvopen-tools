// Function: sub_29DD780
// Address: 0x29dd780
//
void __fastcall sub_29DD780(__int64 a1, __int64 a2, char a3)
{
  unsigned __int8 *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int *v12; // rax
  int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // r9
  int v17; // edx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  int v23; // r13d
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  unsigned __int8 *v27; // r15
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rcx
  unsigned __int8 *v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // r8
  int v34; // eax
  _QWORD *v35; // r13
  unsigned __int64 v36; // r15
  unsigned __int8 *v37; // rdi
  unsigned __int8 *v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 *v41; // rdi
  __int64 v42; // rdx
  unsigned __int64 v43; // r15
  __int64 *v44; // r8
  unsigned __int64 *v45; // rax
  void *v46; // rax
  void *v47; // rsi
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  char *v50; // rcx
  size_t v51; // rdx
  char *v52; // rax
  unsigned __int64 v53; // rsi
  __int64 v54; // r15
  __int64 v55; // r12
  _QWORD *v56; // rax
  __int64 v57; // r13
  unsigned int *v58; // rbx
  unsigned int *v59; // r12
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // r12
  _QWORD *v63; // rax
  __int64 v64; // r13
  unsigned int *v65; // rbx
  unsigned int *v66; // r12
  __int64 v67; // rdx
  unsigned int v68; // esi
  __int64 v69; // rax
  unsigned __int8 *v70; // r12
  unsigned __int8 *v71; // r13
  __int64 (__fastcall *v72)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v73; // r14
  unsigned __int64 v74; // rbx
  __int64 v75; // rax
  __int64 v76; // rax
  const char *v77; // rax
  const char *v78; // r13
  const char *v79; // r12
  __int64 v80; // rdx
  unsigned int v81; // esi
  unsigned __int8 *v82; // [rsp+0h] [rbp-270h]
  void *v83; // [rsp+8h] [rbp-268h]
  __int64 *v84; // [rsp+8h] [rbp-268h]
  __int64 v86; // [rsp+18h] [rbp-258h]
  _QWORD *v87; // [rsp+28h] [rbp-248h]
  __int64 v88; // [rsp+38h] [rbp-238h]
  __int64 v89; // [rsp+38h] [rbp-238h]
  unsigned __int64 v90; // [rsp+38h] [rbp-238h]
  int v91; // [rsp+38h] [rbp-238h]
  __int64 v92; // [rsp+48h] [rbp-228h] BYREF
  unsigned __int64 v93[4]; // [rsp+50h] [rbp-220h] BYREF
  char v94; // [rsp+70h] [rbp-200h]
  char v95; // [rsp+71h] [rbp-1FFh]
  _QWORD v96[4]; // [rsp+80h] [rbp-1F0h] BYREF
  __int16 v97; // [rsp+A0h] [rbp-1D0h]
  _BYTE *v98; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v99; // [rsp+B8h] [rbp-1B8h]
  _BYTE v100[32]; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 *v101; // [rsp+E0h] [rbp-190h] BYREF
  size_t n; // [rsp+E8h] [rbp-188h]
  __int64 v103; // [rsp+F0h] [rbp-180h] BYREF
  void *src; // [rsp+100h] [rbp-170h]
  _BYTE *v105; // [rsp+108h] [rbp-168h]
  unsigned int *v106; // [rsp+120h] [rbp-150h] BYREF
  int v107; // [rsp+128h] [rbp-148h]
  char v108; // [rsp+130h] [rbp-140h] BYREF
  __int64 v109; // [rsp+158h] [rbp-118h]
  __int64 v110; // [rsp+160h] [rbp-110h]
  __int64 v111; // [rsp+168h] [rbp-108h]
  __int64 v112; // [rsp+178h] [rbp-F8h]
  void *v113; // [rsp+1A0h] [rbp-D0h]
  char *v114; // [rsp+1B0h] [rbp-C0h] BYREF
  __int64 v115; // [rsp+1B8h] [rbp-B8h]
  unsigned __int64 v116; // [rsp+1C0h] [rbp-B0h] BYREF
  char v117; // [rsp+1C8h] [rbp-A8h]
  unsigned __int64 v118; // [rsp+1D0h] [rbp-A0h]
  char *v119; // [rsp+1D8h] [rbp-98h]
  char *v120; // [rsp+1E0h] [rbp-90h]
  __int64 v121; // [rsp+1E8h] [rbp-88h]
  __int64 v122; // [rsp+1F0h] [rbp-80h]
  __int64 v123; // [rsp+200h] [rbp-70h]
  __int64 v124; // [rsp+208h] [rbp-68h]
  void *v125; // [rsp+230h] [rbp-40h]

  v4 = (unsigned __int8 *)a2;
  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_11;
  v5 = sub_BD2BC0(a2);
  v7 = v5 + v6;
  if ( *(char *)(a2 + 7) < 0 )
    v7 -= sub_BD2BC0(a2);
  v8 = v7 >> 4;
  if ( (_DWORD)v8 )
  {
    v9 = 0;
    v10 = 16LL * (unsigned int)v8;
    while ( 1 )
    {
      v11 = 0;
      if ( *(char *)(a2 + 7) < 0 )
        v11 = sub_BD2BC0(a2);
      v12 = (unsigned int *)(v9 + v11);
      if ( !*(_DWORD *)(*(_QWORD *)v12 + 8LL) )
        break;
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_11;
    }
    v13 = *(_DWORD *)(a2 + 4);
    v14 = v12[2];
    v15 = v12[3];
    v116 = *(_QWORD *)v12;
    v14 *= 32;
    v117 = 1;
    v114 = (char *)&v4[v14 - 32LL * (v13 & 0x7FFFFFF)];
    v115 = (32LL * v15 - v14) >> 5;
  }
  else
  {
LABEL_11:
    v117 = 0;
  }
  sub_B56460((__int64)&v101, (__int64)&v114);
  v17 = *v4;
  if ( v17 == 40 )
  {
    v18 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v4);
  }
  else
  {
    v18 = -32;
    if ( v17 != 85 )
    {
      v18 = -96;
      if ( v17 != 34 )
        BUG();
    }
  }
  if ( (v4[7] & 0x80u) != 0 )
  {
    v88 = v18;
    v19 = sub_BD2BC0((__int64)v4);
    v18 = v88;
    v21 = v19 + v20;
    v22 = 0;
    if ( (v4[7] & 0x80u) != 0 )
    {
      v22 = sub_BD2BC0((__int64)v4);
      v18 = v88;
    }
    if ( (unsigned int)((v21 - v22) >> 4) )
    {
      v89 = v18;
      if ( (v4[7] & 0x80u) == 0 )
        BUG();
      v23 = *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8);
      if ( (v4[7] & 0x80u) == 0 )
        BUG();
      v24 = sub_BD2BC0((__int64)v4);
      v18 = v89 - 32LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v23);
    }
  }
  v26 = *((_DWORD *)v4 + 1);
  v27 = &v4[v18];
  v98 = v100;
  v28 = 32 * (1LL - (v26 & 0x7FFFFFF));
  v99 = 0x400000000LL;
  v29 = 0;
  v30 = v18 - v28;
  v31 = &v4[v28];
  v32 = v100;
  v33 = v30 >> 5;
  if ( (unsigned __int64)v30 > 0x80 )
  {
    v91 = v30 >> 5;
    sub_C8D5F0((__int64)&v98, v100, v30 >> 5, 8u, v33, v16);
    v29 = v99;
    LODWORD(v33) = v91;
    v32 = &v98[8 * (unsigned int)v99];
  }
  if ( v27 != v31 )
  {
    do
    {
      if ( v32 )
        *v32 = *(_QWORD *)v31;
      v31 += 32;
      ++v32;
    }
    while ( v27 != v31 );
    v29 = v99;
  }
  v34 = *((_DWORD *)v4 + 1);
  v35 = (_QWORD *)*((_QWORD *)v4 + 5);
  LODWORD(v99) = v29 + v33;
  v35 += 6;
  v87 = (_QWORD *)sub_F38250(*(_QWORD *)&v4[-32 * (v34 & 0x7FFFFFF)], (__int64 *)v4 + 3, 0, 1, 0, 0, 0, 0);
  v36 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
  v90 = v36;
  if ( (_QWORD *)v36 == v35 )
    goto LABEL_108;
  if ( !v36 )
    BUG();
  v86 = v36 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 > 0xA )
  {
LABEL_108:
    sub_B4CC70(0);
    BUG();
  }
  sub_B4CC70(v36 - 24);
  v37 = *(unsigned __int8 **)(v36 - 56);
  v114 = "guarded";
  LOWORD(v118) = 259;
  sub_BD6B50(v37, (const char **)&v114);
  v38 = *(unsigned __int8 **)(v36 - 88);
  v114 = "deopt";
  LOWORD(v118) = 259;
  sub_BD6B50(v38, (const char **)&v114);
  if ( (v4[7] & 0x20) != 0 )
  {
    v39 = sub_B91C10((__int64)v4, 14);
    if ( v39 )
      sub_B99FD0(v36 - 24, 0xEu, v39);
  }
  v92 = sub_BD5C60((__int64)v4);
  v40 = sub_B8C2F0(&v92, qword_5009188, 1u, 0);
  sub_B99FD0(v86, 2u, v40);
  v41 = (unsigned __int64 *)&v106;
  sub_23D0AB0((__int64)&v106, (__int64)v87, 0, 0, 0);
  v43 = n;
  v97 = 257;
  v44 = v101;
  v114 = (char *)&v116;
  if ( (__int64 *)((char *)v101 + n) && !v101 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v93[0] = n;
  if ( n > 0xF )
  {
    v84 = v101;
    v77 = (const char *)sub_22409D0((__int64)&v114, v93, 0);
    v44 = v84;
    v114 = (char *)v77;
    v41 = (unsigned __int64 *)v77;
    v116 = v93[0];
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v116) = *(_BYTE *)v101;
      v45 = &v116;
      goto LABEL_39;
    }
    if ( !n )
    {
      v45 = &v116;
      goto LABEL_39;
    }
    v41 = &v116;
  }
  memcpy(v41, v44, v43);
  v43 = v93[0];
  v45 = (unsigned __int64 *)v114;
LABEL_39:
  v115 = v43;
  *((_BYTE *)v45 + v43) = 0;
  v46 = v105;
  v47 = src;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v48 = v105 - (_BYTE *)src;
  if ( v105 == src )
  {
    v51 = 0;
    v48 = 0;
    v50 = 0;
  }
  else
  {
    if ( v48 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v41, src, v42);
    v49 = sub_22077B0(v105 - (_BYTE *)src);
    v47 = src;
    v50 = (char *)v49;
    v46 = v105;
    v51 = v105 - (_BYTE *)src;
  }
  v118 = (unsigned __int64)v50;
  v119 = v50;
  v120 = &v50[v48];
  if ( v46 != v47 )
  {
    v83 = (void *)v51;
    v52 = (char *)memmove(v50, v47, v51);
    v51 = (size_t)v83;
    v50 = v52;
  }
  v53 = 0;
  v119 = &v50[v51];
  if ( a1 )
    v53 = *(_QWORD *)(a1 + 24);
  v54 = sub_B33530(&v106, v53, a1, (int)v98, v99, (__int64)v96, (__int64)&v114, 1, 0);
  if ( v118 )
    j_j___libc_free_0(v118);
  if ( v114 != (char *)&v116 )
    j_j___libc_free_0((unsigned __int64)v114);
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL) == 7 )
  {
    v62 = v111;
    LOWORD(v118) = 257;
    v63 = sub_BD2C40(72, 0);
    v64 = (__int64)v63;
    if ( v63 )
      sub_B4BB80((__int64)v63, v62, 0, 0, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
      v112,
      v64,
      &v114,
      v109,
      v110);
    if ( v106 != &v106[4 * v107] )
    {
      v82 = v4;
      v65 = v106;
      v66 = &v106[4 * v107];
      do
      {
        v67 = *((_QWORD *)v65 + 1);
        v68 = *v65;
        v65 += 4;
        sub_B99FD0(v64, v68, v67);
      }
      while ( v66 != v65 );
      goto LABEL_56;
    }
  }
  else
  {
    v114 = "deoptcall";
    LOWORD(v118) = 259;
    sub_BD6B50((unsigned __int8 *)v54, (const char **)&v114);
    v55 = v111;
    LOWORD(v118) = 257;
    v56 = sub_BD2C40(72, v54 != 0);
    v57 = (__int64)v56;
    if ( v56 )
      sub_B4BB80((__int64)v56, v55, v54, v54 != 0, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
      v112,
      v57,
      &v114,
      v109,
      v110);
    if ( v106 != &v106[4 * v107] )
    {
      v82 = v4;
      v58 = v106;
      v59 = &v106[4 * v107];
      do
      {
        v60 = *((_QWORD *)v58 + 1);
        v61 = *v58;
        v58 += 4;
        sub_B99FD0(v57, v61, v60);
      }
      while ( v59 != v58 );
LABEL_56:
      v4 = v82;
    }
  }
  *(_WORD *)(v54 + 2) = *((_WORD *)v4 + 1) & 0xFFC | *(_WORD *)(v54 + 2) & 0xF003;
  sub_B43D60(v87);
  if ( !a3 )
    goto LABEL_58;
  sub_23D0AB0((__int64)&v114, v86, 0, 0, 0);
  BYTE4(v93[0]) = 0;
  v96[0] = "widenable_cond";
  v97 = 259;
  v69 = sub_B33D10((__int64)&v114, 0xA9u, 0, 0, 0, 0, v93[0], (__int64)v96);
  v95 = 1;
  v70 = (unsigned __int8 *)v69;
  v94 = 3;
  v93[0] = (unsigned __int64)"exiplicit_guard_cond";
  v71 = *(unsigned __int8 **)(v90 - 120);
  v72 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v123 + 16LL);
  if ( v72 == sub_9202E0 )
  {
    if ( *v71 > 0x15u || *v70 > 0x15u )
    {
      v74 = v90 - 120;
      goto LABEL_97;
    }
    if ( (unsigned __int8)sub_AC47B0(28) )
      v73 = sub_AD5570(28, (__int64)v71, v70, 0, 0);
    else
      v73 = sub_AABE40(0x1Cu, v71, v70);
  }
  else
  {
    v73 = v72(v123, 28u, *(_BYTE **)(v90 - 120), v70);
  }
  v74 = v90 - 120;
  if ( v73 )
  {
    if ( !*(_QWORD *)(v90 - 120) || (v75 = *(_QWORD *)(v90 - 112), (**(_QWORD **)(v90 - 104) = v75) == 0) )
    {
      *(_QWORD *)(v90 - 120) = v73;
      goto LABEL_89;
    }
    goto LABEL_87;
  }
LABEL_97:
  v97 = 257;
  v73 = sub_B504D0(28, (__int64)v71, (__int64)v70, (__int64)v96, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v124 + 16LL))(
    v124,
    v73,
    v93,
    v121,
    v122);
  v78 = v114;
  v79 = &v114[16 * (unsigned int)v115];
  if ( v114 != v79 )
  {
    do
    {
      v80 = *((_QWORD *)v78 + 1);
      v81 = *(_DWORD *)v78;
      v78 += 16;
      sub_B99FD0(v73, v81, v80);
    }
    while ( v79 != v78 );
  }
  if ( *(_QWORD *)(v90 - 120) )
  {
    v75 = *(_QWORD *)(v90 - 112);
    **(_QWORD **)(v90 - 104) = v75;
    if ( v75 )
LABEL_87:
      *(_QWORD *)(v75 + 16) = *(_QWORD *)(v90 - 104);
  }
  *(_QWORD *)(v90 - 120) = v73;
  if ( v73 )
  {
LABEL_89:
    v76 = *(_QWORD *)(v73 + 16);
    *(_QWORD *)(v90 - 112) = v76;
    if ( v76 )
      *(_QWORD *)(v76 + 16) = v90 - 112;
    *(_QWORD *)(v90 - 104) = v73 + 16;
    *(_QWORD *)(v73 + 16) = v74;
  }
  nullsub_61();
  v125 = &unk_49DA100;
  nullsub_63();
  if ( v114 != (char *)&v116 )
    _libc_free((unsigned __int64)v114);
LABEL_58:
  nullsub_61();
  v113 = &unk_49DA100;
  nullsub_63();
  if ( v106 != (unsigned int *)&v108 )
    _libc_free((unsigned __int64)v106);
  if ( v98 != v100 )
    _libc_free((unsigned __int64)v98);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  if ( v101 != &v103 )
    j_j___libc_free_0((unsigned __int64)v101);
}
