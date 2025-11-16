// Function: sub_320A500
// Address: 0x320a500
//
void __fastcall sub_320A500(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 *v5; // r14
  __int64 v6; // rbx
  unsigned __int64 *v7; // r15
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 *v10; // rdi
  __int64 v11; // rax
  void (*v12)(); // rax
  const char *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  void (*v21)(); // rax
  __int64 v22; // rdi
  void (*v23)(); // rax
  __int64 v24; // rdi
  void (*v25)(); // rax
  __int64 v26; // rdi
  void (*v27)(); // rax
  __int64 v28; // rdi
  void (*v29)(); // rax
  __int64 v30; // rdi
  void (*v31)(); // rax
  __int64 v32; // rbx
  void (*v33)(); // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  void (*v36)(); // rax
  __int64 v37; // rdi
  void (*v38)(); // rax
  __int64 v39; // rdi
  void (*v40)(); // rax
  unsigned __int8 v41; // bl
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 *v44; // rdi
  void (*v45)(); // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r13
  void (*v52)(); // rax
  __int64 v53; // rdi
  void (*v54)(); // rax
  __int64 v55; // rdi
  void (*v56)(); // rax
  __int64 v57; // rdi
  void (*v58)(); // rax
  __int64 v59; // rdi
  void (*v60)(); // rax
  __int64 v61; // rdi
  void (*v62)(); // rax
  __int64 v63; // rdi
  void (*v64)(); // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 *v74; // r13
  __int64 *v75; // rbx
  __int64 *v76; // rsi
  __int64 *v77; // r13
  __int64 *v78; // rbx
  __int64 v79; // rsi
  __int64 *v80; // r13
  __int64 *i; // rbx
  __int64 v82; // r8
  unsigned __int64 v83; // r9
  _QWORD *v84; // rcx
  _QWORD *v85; // rax
  _QWORD *v86; // rdi
  __int64 v87; // rcx
  __int64 *v88; // r13
  __int64 v89; // r14
  __int64 v90; // rbx
  __int64 v91; // rsi
  unsigned __int8 v92; // al
  __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 *v95; // rbx
  __int64 *v96; // r14
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 *v101; // rbx
  unsigned int v102; // eax
  __int64 v103; // r14
  __int64 v104; // r13
  __int64 v105; // rax
  __int64 v106; // rdi
  void (*v107)(); // rax
  __int64 v108; // rdi
  void (*v109)(); // rax
  __int64 v110; // rdi
  void (*v111)(); // rax
  __int64 v112; // r13
  void (*v113)(); // rax
  __int64 v114; // rdx
  unsigned __int64 v115; // rdi
  void (*v116)(); // rcx
  const char *v117; // rax
  size_t v118; // rdx
  const char *v119; // r8
  size_t v120; // r15
  __int64 v121; // rax
  char *v122; // rdx
  char *v123; // r15
  _BYTE *v124; // rax
  __int64 v125; // r9
  __int64 v126; // rax
  _QWORD *v127; // rdi
  __int64 v128; // [rsp+0h] [rbp-D0h]
  __int64 v129; // [rsp+10h] [rbp-C0h]
  __int64 *v130; // [rsp+20h] [rbp-B0h]
  __int64 *v132; // [rsp+30h] [rbp-A0h]
  __int64 v133; // [rsp+30h] [rbp-A0h]
  unsigned __int64 *src; // [rsp+38h] [rbp-98h]
  void *srca; // [rsp+38h] [rbp-98h]
  void *srcb; // [rsp+38h] [rbp-98h]
  _BYTE *srcc; // [rsp+38h] [rbp-98h]
  const char *srcd; // [rsp+38h] [rbp-98h]
  size_t v139; // [rsp+48h] [rbp-88h] BYREF
  _QWORD *v140; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v141; // [rsp+58h] [rbp-78h]
  _BYTE v142[16]; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v143[2]; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v144[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v145; // [rsp+90h] [rbp-40h]

  v129 = sub_31DB510(a1[1], a2);
  sub_3200CF0((__int64)a1, v129);
  v142[0] = 0;
  v140 = v142;
  v141 = 0;
  v4 = sub_B92180(a2);
  v5 = (unsigned __int64 *)a1[169];
  a1[167] = v4;
  v6 = v4;
  src = (unsigned __int64 *)a1[168];
  if ( src != v5 )
  {
    v7 = (unsigned __int64 *)a1[168];
    do
    {
      if ( (unsigned __int64 *)*v7 != v7 + 2 )
        j_j___libc_free_0(*v7);
      v7 += 5;
    }
    while ( v5 != v7 );
    a1[169] = (__int64)src;
  }
  if ( (*(_BYTE *)(v6 + 35) & 2) == 0 )
  {
    sub_A547D0(v6, 2);
    if ( !v9
      || (v121 = sub_A547D0(v6, 2),
          v123 = v122,
          srcc = (_BYTE *)v121,
          v124 = sub_A17150((_BYTE *)(v6 - 16)),
          sub_3205680((__int64)v143, (__int64)a1, *((_QWORD *)v124 + 1), srcc, v123, v125),
          sub_2240D70((__int64)&v140, v143),
          (_QWORD *)v143[0] == v144) )
    {
      if ( v141 )
        goto LABEL_10;
    }
    else
    {
      j_j___libc_free_0(v143[0]);
      if ( v141 )
        goto LABEL_10;
    }
    v117 = sub_BD5D20(a2);
    v119 = v117;
    v120 = v118;
    if ( !v118 )
    {
      v143[0] = (unsigned __int64)v144;
      goto LABEL_97;
    }
    if ( *v117 == 1 )
    {
      v120 = v118 - 1;
      v119 = v117 + 1;
    }
    v143[0] = (unsigned __int64)v144;
    v139 = v120;
    if ( v120 > 0xF )
    {
      srcd = v119;
      v126 = sub_22409D0((__int64)v143, &v139, 0);
      v119 = srcd;
      v143[0] = v126;
      v127 = (_QWORD *)v126;
      v144[0] = v139;
    }
    else
    {
      if ( v120 == 1 )
      {
        LOBYTE(v144[0]) = *v119;
        goto LABEL_97;
      }
      if ( !v120 )
      {
LABEL_97:
        v143[1] = v120;
        *(_BYTE *)(v143[0] + v120) = 0;
        sub_2240D70((__int64)&v140, v143);
        if ( (_QWORD *)v143[0] != v144 )
          j_j___libc_free_0(v143[0]);
LABEL_10:
        v10 = (__int64 *)a1[66];
        v11 = *v10;
        if ( *(_DWORD *)(*(_QWORD *)(a1[2] + 2488) + 264LL) == 38 )
        {
          v116 = *(void (**)())(v11 + 824);
          if ( v116 != nullsub_112 )
          {
            ((void (__fastcall *)(__int64 *, __int64, _QWORD))v116)(v10, v129, 0);
            v10 = (__int64 *)a1[66];
            v11 = *v10;
          }
        }
        v12 = *(void (**)())(v11 + 120);
        v13 = "Symbol subsection for ";
        v14 = 1027;
        v143[0] = (unsigned __int64)"Symbol subsection for ";
        v144[0] = &v140;
        v145 = 1027;
        if ( v12 != nullsub_98 )
          ((void (__fastcall *)(__int64 *, unsigned __int64 *, __int64))v12)(v10, v143, 1);
        v128 = sub_31F8650((__int64)a1, 241, v14, (__int64)v13, v8);
        v18 = sub_31F8790((__int64)a1, (unsigned int)((*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1) + 4422, v15, v16, v17);
        v19 = a1[66];
        v20 = v18;
        v21 = *(void (**)())(*(_QWORD *)v19 + 120LL);
        v143[0] = (unsigned __int64)"PtrParent";
        v145 = 259;
        if ( v21 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v21)(v19, v143, 1);
          v19 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v19 + 536LL))(v19, 0, 4);
        v22 = a1[66];
        v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
        v143[0] = (unsigned __int64)"PtrEnd";
        v145 = 259;
        if ( v23 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v23)(v22, v143, 1);
          v22 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v22 + 536LL))(v22, 0, 4);
        v24 = a1[66];
        v25 = *(void (**)())(*(_QWORD *)v24 + 120LL);
        v143[0] = (unsigned __int64)"PtrNext";
        v145 = 259;
        if ( v25 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v25)(v24, v143, 1);
          v24 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v24 + 536LL))(v24, 0, 4);
        v26 = a1[66];
        v27 = *(void (**)())(*(_QWORD *)v26 + 120LL);
        v143[0] = (unsigned __int64)"Code size";
        v145 = 259;
        if ( v27 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v27)(v26, v143, 1);
          v26 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v26 + 832LL))(
          v26,
          *(_QWORD *)(a3 + 448),
          v129,
          4);
        v28 = a1[66];
        v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
        v143[0] = (unsigned __int64)"Offset after prologue";
        v145 = 259;
        if ( v29 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v29)(v28, v143, 1);
          v28 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v28 + 536LL))(v28, 0, 4);
        v30 = a1[66];
        v31 = *(void (**)())(*(_QWORD *)v30 + 120LL);
        v143[0] = (unsigned __int64)"Offset before epilogue";
        v145 = 259;
        if ( v31 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v31)(v30, v143, 1);
          v30 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v30 + 536LL))(v30, 0, 4);
        v32 = a1[66];
        v33 = *(void (**)())(*(_QWORD *)v32 + 120LL);
        v143[0] = (unsigned __int64)"Function type index";
        v145 = 259;
        if ( v33 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v33)(v32, v143, 1);
          v32 = a1[66];
        }
        v34 = sub_B92180(a2);
        LODWORD(v143[0]) = sub_3207610((__int64)a1, v34);
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v32 + 536LL))(v32, LODWORD(v143[0]), 4);
        v35 = a1[66];
        v36 = *(void (**)())(*(_QWORD *)v35 + 120LL);
        v143[0] = (unsigned __int64)"Function section relative address";
        v145 = 259;
        if ( v36 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v36)(v35, v143, 1);
          v35 = a1[66];
        }
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v35 + 368LL))(v35, v129, 0);
        v37 = a1[66];
        v38 = *(void (**)())(*(_QWORD *)v37 + 120LL);
        v143[0] = (unsigned __int64)"Function section index";
        v145 = 259;
        if ( v38 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v38)(v37, v143, 1);
          v37 = a1[66];
        }
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v37 + 360LL))(v37, v129);
        v39 = a1[66];
        v40 = *(void (**)())(*(_QWORD *)v39 + 120LL);
        v143[0] = (unsigned __int64)"Flags";
        v145 = 259;
        if ( v40 != nullsub_98 )
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v40)(v39, v143, 1);
        v41 = 0x80 - ((*(_BYTE *)(a3 + 490) == 0) - 1);
        if ( (unsigned __int8)sub_B2D610(a2, 36) )
          v41 |= 8u;
        if ( (unsigned __int8)sub_B2D610(a2, 31) )
          v41 |= 0x40u;
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a1[66] + 536LL))(a1[66], v41, 1);
        v44 = (__int64 *)a1[66];
        v45 = *(void (**)())(*v44 + 120);
        v143[0] = (unsigned __int64)"Function name";
        v145 = 259;
        if ( v45 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64 *, unsigned __int64 *, __int64))v45)(v44, v143, 1);
          v44 = (__int64 *)a1[66];
        }
        sub_31F4F00(v44, v140, v141, 3840, v42, v43);
        sub_31F8930((__int64)a1, v20);
        v49 = sub_31F8790((__int64)a1, 4114, v46, v47, v48);
        v50 = a1[66];
        v51 = v49;
        v52 = *(void (**)())(*(_QWORD *)v50 + 120LL);
        v143[0] = (unsigned __int64)"FrameSize";
        v145 = 259;
        if ( v52 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v52)(v50, v143, 1);
          v50 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v50 + 536LL))(
          v50,
          (unsigned int)(*(_DWORD *)(a3 + 464) - *(_DWORD *)(a3 + 472)),
          4);
        v53 = a1[66];
        v54 = *(void (**)())(*(_QWORD *)v53 + 120LL);
        v143[0] = (unsigned __int64)"Padding";
        v145 = 259;
        if ( v54 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v54)(v53, v143, 1);
          v53 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v53 + 536LL))(v53, 0, 4);
        v55 = a1[66];
        v56 = *(void (**)())(*(_QWORD *)v55 + 120LL);
        v143[0] = (unsigned __int64)"Offset of padding";
        v145 = 259;
        if ( v56 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v56)(v55, v143, 1);
          v55 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v55 + 536LL))(v55, 0, 4);
        v57 = a1[66];
        v58 = *(void (**)())(*(_QWORD *)v57 + 120LL);
        v143[0] = (unsigned __int64)"Bytes of callee saved registers";
        v145 = 259;
        if ( v58 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v58)(v57, v143, 1);
          v57 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v57 + 536LL))(v57, *(unsigned int *)(a3 + 472), 4);
        v59 = a1[66];
        v60 = *(void (**)())(*(_QWORD *)v59 + 120LL);
        v143[0] = (unsigned __int64)"Exception handler offset";
        v145 = 259;
        if ( v60 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v60)(v59, v143, 1);
          v59 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v59 + 536LL))(v59, 0, 4);
        v61 = a1[66];
        v62 = *(void (**)())(*(_QWORD *)v61 + 120LL);
        v143[0] = (unsigned __int64)"Exception handler section";
        v145 = 259;
        if ( v62 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v62)(v61, v143, 1);
          v61 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v61 + 536LL))(v61, 0, 2);
        v63 = a1[66];
        v64 = *(void (**)())(*(_QWORD *)v63 + 120LL);
        v143[0] = (unsigned __int64)"Flags (defines frame register)";
        v145 = 259;
        if ( v64 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v64)(v63, v143, 1);
          v63 = a1[66];
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v63 + 536LL))(v63, *(unsigned int *)(a3 + 484), 4);
        sub_31F8930((__int64)a1, v51);
        sub_31FC2C0((__int64)a1, (__int64 *)(a3 + 80), v65, v66, v67, v68);
        sub_3209920((__int64)a1, a3, *(_QWORD *)(a3 + 152), *(unsigned int *)(a3 + 160), v69, v70);
        v74 = *(__int64 **)(a3 + 256);
        v75 = &v74[2 * *(unsigned int *)(a3 + 264)];
        while ( v75 != v74 )
        {
          v76 = v74;
          v74 += 2;
          sub_3208CF0((__int64)a1, v76);
        }
        v77 = *(__int64 **)(a3 + 344);
        v78 = &v77[*(unsigned int *)(a3 + 352)];
        while ( v78 != v77 )
        {
          v79 = *v77++;
          sub_320A200((__int64)a1, v79, a3, v72, v73);
        }
        v80 = *(__int64 **)(a3 + 56);
        for ( i = &v80[*(unsigned int *)(a3 + 64)]; i != v80; ++v80 )
        {
          v82 = *v80;
          v83 = *(_QWORD *)(a3 + 8);
          v84 = *(_QWORD **)(*(_QWORD *)a3 + 8 * (*v80 % v83));
          if ( v84 )
          {
            v85 = (_QWORD *)*v84;
            if ( v82 == *(_QWORD *)(*v84 + 8LL) )
            {
LABEL_63:
              v84 = (_QWORD *)*v84;
            }
            else
            {
              while ( 1 )
              {
                v86 = (_QWORD *)*v85;
                if ( !*v85 )
                  break;
                v84 = v85;
                if ( *v80 % v83 != v86[1] % v83 )
                  break;
                v85 = (_QWORD *)*v85;
                if ( v82 == v86[1] )
                  goto LABEL_63;
              }
              v84 = 0;
            }
          }
          sub_3209DE0((__int64)a1, (_QWORD *)a3, *v80, (__int64)(v84 + 2));
        }
        v87 = *(_QWORD *)(a3 + 376);
        v88 = *(__int64 **)(a3 + 368);
        v132 = (__int64 *)v87;
        while ( v132 != v88 )
        {
          v89 = *v88;
          v90 = v88[1];
          srca = (void *)sub_31F8790((__int64)a1, 4121, v71, v87, v73);
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1[66] + 368LL))(a1[66], v89, 0);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1[66] + 360LL))(a1[66], v89);
          if ( (*(_BYTE *)(v90 - 16) & 2) != 0 )
            v91 = *(unsigned int *)(v90 - 24);
          else
            v91 = (*(_WORD *)(v90 - 16) >> 6) & 0xF;
          (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1[66] + 536LL))(a1[66], v91, 2);
          v92 = *(_BYTE *)(v90 - 16);
          if ( (v92 & 2) != 0 )
          {
            v93 = *(_QWORD *)(v90 - 32);
            v94 = *(unsigned int *)(v90 - 24);
          }
          else
          {
            v94 = (*(_WORD *)(v90 - 16) >> 6) & 0xF;
            v93 = v90 - 16 - 8LL * ((v92 >> 2) & 0xF);
          }
          v95 = (__int64 *)(v93 + 8 * v94);
          v96 = (__int64 *)v93;
          if ( (__int64 *)v93 != v95 )
          {
            do
            {
              v97 = *v96++;
              v98 = sub_B91420(v97);
              (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1[66] + 512LL))(a1[66], v98, v99 + 1);
            }
            while ( v95 != v96 );
          }
          v88 += 2;
          sub_31F8930((__int64)a1, (__int64)srca);
        }
        v100 = *(_QWORD *)(a3 + 400);
        v101 = *(__int64 **)(a3 + 392);
        v130 = (__int64 *)v100;
        while ( v130 != v101 )
        {
          v103 = *v101;
          v104 = v101[2];
          srcb = (void *)v101[1];
          v105 = sub_31F8790((__int64)a1, 4446, v71, v100, v73);
          v106 = a1[66];
          v133 = v105;
          v107 = *(void (**)())(*(_QWORD *)v106 + 120LL);
          v143[0] = (unsigned __int64)"Call site offset";
          v145 = 259;
          if ( v107 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v107)(v106, v143, 1);
            v106 = a1[66];
          }
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v106 + 368LL))(v106, v104, 0);
          v108 = a1[66];
          v109 = *(void (**)())(*(_QWORD *)v108 + 120LL);
          v143[0] = (unsigned __int64)"Call site section index";
          v145 = 259;
          if ( v109 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v109)(v108, v143, 1);
            v108 = a1[66];
          }
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v108 + 360LL))(v108, v104);
          v110 = a1[66];
          v111 = *(void (**)())(*(_QWORD *)v110 + 120LL);
          v143[0] = (unsigned __int64)"Call instruction length";
          v145 = 259;
          if ( v111 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v111)(v110, v143, 1);
            v110 = a1[66];
          }
          (*(void (__fastcall **)(__int64, void *, __int64, __int64))(*(_QWORD *)v110 + 832LL))(v110, srcb, v104, 2);
          v112 = a1[66];
          v113 = *(void (**)())(*(_QWORD *)v112 + 120LL);
          v143[0] = (unsigned __int64)"Type index";
          v145 = 259;
          if ( v113 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v113)(v112, v143, 1);
            v112 = a1[66];
          }
          if ( v103 )
          {
            v102 = sub_3205010((__int64)a1, v103);
          }
          else
          {
            LODWORD(v143[0]) = 3;
            v102 = 3;
          }
          LODWORD(v139) = v102;
          v101 += 3;
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v112 + 536LL))(v112, v102, 4);
          sub_31F8930((__int64)a1, v133);
        }
        sub_3205D70((__int64)a1, a1 + 168, v71, v100);
        sub_31F9B40((__int64)a1, a3, v114);
        sub_31F93A0((__int64)a1, 0x114Fu);
        sub_31F8740((__int64)a1, v128);
        (*(void (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1[66] + 744LL))(
          a1[66],
          *(unsigned int *)(a3 + 456),
          v129,
          *(_QWORD *)(a3 + 448));
        v115 = (unsigned __int64)v140;
        if ( v140 != (_QWORD *)v142 )
          goto LABEL_91;
        return;
      }
      v127 = v144;
    }
    memcpy(v127, v119, v120);
    v120 = v139;
    goto LABEL_97;
  }
  sub_31F94F0((__int64)a1, a2, a3, v129);
  v115 = (unsigned __int64)v140;
  if ( v140 != (_QWORD *)v142 )
LABEL_91:
    j_j___libc_free_0(v115);
}
