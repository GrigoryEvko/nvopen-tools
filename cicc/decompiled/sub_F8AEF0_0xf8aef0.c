// Function: sub_F8AEF0
// Address: 0xf8aef0
//
__int64 __fastcall sub_F8AEF0(__int64 **a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rbx
  char *v6; // rsi
  __int64 *v7; // rdi
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r12
  _QWORD *v19; // rax
  __int64 *v20; // r14
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r15
  unsigned int v24; // r14d
  unsigned int v25; // eax
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // r15
  __int64 v29; // r13
  __int64 v30; // rdi
  _QWORD *v31; // r9
  __int64 v32; // rax
  __int64 v33; // r13
  unsigned int v34; // esi
  __int64 *v35; // rax
  __int64 v36; // r13
  __int64 v37; // r9
  __int64 v38; // r14
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // r13
  char *v43; // rax
  __int64 *v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // r10
  __int64 v49; // rdi
  __int64 v50; // r14
  _QWORD *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // r15
  __int64 v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  _QWORD **v58; // rdx
  int v59; // ecx
  __int64 *v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // r14
  __int64 v65; // rbx
  __int64 v66; // rdx
  unsigned int v67; // esi
  unsigned int *v68; // r15
  unsigned int *v69; // r14
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // r15
  __int64 v75; // r15
  __int64 v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  __int64 *v79; // rax
  __int64 v80; // r14
  __int64 v81; // rax
  __int64 v82; // r12
  __int64 v83; // rbx
  __int64 v84; // r14
  __int64 v85; // rdx
  unsigned int v86; // esi
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r12
  __int64 v90; // rbx
  __int64 v91; // rdx
  unsigned int v92; // esi
  __int64 v93; // r15
  __int64 v94; // r14
  __int64 v95; // rbx
  __int64 v96; // r15
  __int64 v97; // r14
  __int64 v98; // rdx
  unsigned int v99; // esi
  unsigned int *v100; // rax
  unsigned int *v101; // r12
  unsigned int *v102; // rbx
  __int64 v103; // rdx
  _QWORD *v104; // rax
  __int64 v105; // rax
  __int64 v106; // r13
  __int64 v107; // r13
  __int64 v108; // r12
  __int64 v109; // rbx
  __int64 v110; // rdx
  unsigned int v111; // esi
  __int64 v112; // [rsp+8h] [rbp-198h]
  __int64 v113; // [rsp+8h] [rbp-198h]
  __int64 v114; // [rsp+8h] [rbp-198h]
  char v115; // [rsp+10h] [rbp-190h]
  __int64 v116; // [rsp+10h] [rbp-190h]
  char v117; // [rsp+18h] [rbp-188h]
  __int64 v118; // [rsp+18h] [rbp-188h]
  __int64 v119; // [rsp+28h] [rbp-178h]
  __int64 v120; // [rsp+28h] [rbp-178h]
  char v121; // [rsp+28h] [rbp-178h]
  unsigned int v122; // [rsp+30h] [rbp-170h]
  _BYTE *v123; // [rsp+38h] [rbp-168h]
  _BYTE *v125; // [rsp+48h] [rbp-158h]
  unsigned int v126; // [rsp+50h] [rbp-150h]
  __int64 v127; // [rsp+58h] [rbp-148h]
  _QWORD *v128; // [rsp+60h] [rbp-140h]
  __int64 v129; // [rsp+68h] [rbp-138h]
  char v130; // [rsp+68h] [rbp-138h]
  __int64 v131; // [rsp+70h] [rbp-130h]
  __int64 v132; // [rsp+78h] [rbp-128h]
  _QWORD *v133; // [rsp+78h] [rbp-128h]
  _QWORD *v134; // [rsp+78h] [rbp-128h]
  __int64 v135; // [rsp+78h] [rbp-128h]
  __int64 v136; // [rsp+78h] [rbp-128h]
  __int64 v137; // [rsp+78h] [rbp-128h]
  __int64 v138; // [rsp+78h] [rbp-128h]
  __int64 v139; // [rsp+78h] [rbp-128h]
  __int64 v140; // [rsp+78h] [rbp-128h]
  __int64 v142; // [rsp+88h] [rbp-118h]
  __int64 v143; // [rsp+88h] [rbp-118h]
  __int64 v144; // [rsp+88h] [rbp-118h]
  __int64 v145; // [rsp+98h] [rbp-108h]
  __int64 v146; // [rsp+A0h] [rbp-100h] BYREF
  unsigned int v147; // [rsp+A8h] [rbp-F8h]
  __int64 v148; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v149; // [rsp+D0h] [rbp-D0h]
  _QWORD v150[4]; // [rsp+E0h] [rbp-C0h] BYREF
  __int16 v151; // [rsp+100h] [rbp-A0h]
  char *v152; // [rsp+110h] [rbp-90h] BYREF
  unsigned int v153; // [rsp+118h] [rbp-88h]
  __int16 v154; // [rsp+130h] [rbp-70h]
  _QWORD v155[2]; // [rsp+140h] [rbp-60h] BYREF
  _BYTE v156[80]; // [rsp+150h] [rbp-50h] BYREF

  v5 = (__int64)a1;
  v6 = *(char **)(a2 + 48);
  v7 = *a1;
  v155[0] = v156;
  v155[1] = 0x400000000LL;
  v8 = sub_DEF990(v7, v6, (__int64)v155);
  v12 = sub_D33D80((_QWORD *)a2, *(_QWORD *)v5, v9, v10, v11);
  v13 = **(_QWORD **)(a2 + 32);
  v14 = sub_D95540(v13);
  v15 = *(_QWORD *)v5;
  v132 = v14;
  v16 = sub_D95540(v8);
  v17 = v15;
  v18 = v5 + 520;
  v122 = sub_D97050(v17, v16);
  v126 = sub_D97050(*(_QWORD *)v5, v132);
  sub_D5F1F0(v5 + 520, a3);
  sub_D5F1F0(v5 + 520, a3);
  v127 = sub_F894B0(v5, v8);
  LODWORD(v8) = sub_D97050(*(_QWORD *)v5, v132);
  v19 = (_QWORD *)sub_BD5C60(a3);
  v129 = sub_BCCE00(v19, v8);
  sub_D5F1F0(v5 + 520, a3);
  v131 = sub_F894B0(v5, v12);
  v20 = sub_DCAF50(*(__int64 **)v5, v12, 0);
  sub_D5F1F0(v5 + 520, a3);
  v119 = sub_F894B0(v5, (__int64)v20);
  sub_D5F1F0(v5 + 520, a3);
  v125 = (_BYTE *)sub_F894B0(v5, v13);
  v153 = v126;
  if ( v126 > 0x40 )
    sub_C43690((__int64)&v152, 0, 0);
  else
    v152 = 0;
  v21 = (__int64 *)sub_BD5C60(a3);
  v123 = (_BYTE *)sub_ACCFD0(v21, (__int64)&v152);
  if ( v153 > 0x40 && v152 )
    j_j___libc_free_0_0(v152);
  sub_D5F1F0(v5 + 520, a3);
  v22 = *(_QWORD *)(v5 + 600);
  v151 = 257;
  v128 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, _BYTE *))(*(_QWORD *)v22 + 56LL))(
                     v22,
                     40,
                     v131,
                     v123);
  if ( !v128 )
  {
    v154 = 257;
    v128 = sub_BD2C40(72, unk_3F10FD0);
    if ( v128 )
    {
      v58 = *(_QWORD ***)(v131 + 8);
      v59 = *((unsigned __int8 *)v58 + 8);
      if ( (unsigned int)(v59 - 17) > 1 )
      {
        v61 = sub_BCB2A0(*v58);
      }
      else
      {
        BYTE4(v145) = (_BYTE)v59 == 18;
        LODWORD(v145) = *((_DWORD *)v58 + 8);
        v60 = (__int64 *)sub_BCB2A0(*v58);
        v61 = sub_BCE1B0(v60, v145);
      }
      sub_B523C0((__int64)v128, v61, 53, 40, v131, (__int64)v123, (__int64)&v152, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
      *(_QWORD *)(v5 + 608),
      v128,
      v150,
      *(_QWORD *)(v5 + 576),
      *(_QWORD *)(v5 + 584));
    v62 = *(_QWORD *)(v5 + 520);
    v63 = 16LL * *(unsigned int *)(v5 + 528);
    if ( v62 != v62 + v63 )
    {
      v64 = v62 + v63;
      v116 = v5;
      v65 = *(_QWORD *)(v5 + 520);
      do
      {
        v66 = *(_QWORD *)(v65 + 8);
        v67 = *(_DWORD *)v65;
        v65 += 16;
        sub_B99FD0((__int64)v128, v67, v66);
      }
      while ( v64 != v65 );
      v5 = v116;
    }
  }
  v154 = 257;
  v120 = sub_B36550((unsigned int **)v18, (__int64)v128, v119, v131, (__int64)&v152, 0);
  if ( !a4 && sub_D968A0(v13) )
  {
    v39 = v12;
    if ( (unsigned __int8)sub_DBEDC0(*(_QWORD *)v5, v12) )
    {
      v79 = (__int64 *)sub_BD5C60(a3);
      v41 = sub_ACD720(v79);
      goto LABEL_30;
    }
  }
  v151 = 257;
  v23 = *(_QWORD *)(v127 + 8);
  v24 = sub_BCB060(v23);
  v25 = sub_BCB060(v129);
  if ( v24 >= v25 )
  {
    if ( v129 == v23 || v24 == v25 )
      goto LABEL_12;
    v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v5 + 600) + 120LL))(
            *(_QWORD *)(v5 + 600),
            38,
            v127,
            v129);
    if ( v26 )
      goto LABEL_13;
    v154 = 257;
    v26 = sub_B51D30(38, v127, v129, (__int64)&v152, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
      *(_QWORD *)(v5 + 608),
      v26,
      v150,
      *(_QWORD *)(v18 + 56),
      *(_QWORD *)(v18 + 64));
    v73 = *(_QWORD *)(v5 + 520);
    v74 = 16LL * *(unsigned int *)(v5 + 528);
    if ( v73 == v73 + v74 )
      goto LABEL_13;
    v118 = v5;
    v75 = v73 + v74;
    v76 = *(_QWORD *)(v5 + 520);
    do
    {
      v77 = *(_QWORD *)(v76 + 8);
      v78 = *(_DWORD *)v76;
      v76 += 16;
      sub_B99FD0(v26, v78, v77);
    }
    while ( v75 != v76 );
    goto LABEL_52;
  }
  if ( v129 == v23 )
  {
LABEL_12:
    v26 = v127;
    goto LABEL_13;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v5 + 600) + 120LL))(
          *(_QWORD *)(v5 + 600),
          39,
          v127,
          v129);
  if ( !v26 )
  {
    v154 = 257;
    v51 = sub_BD2C40(72, unk_3F10A14);
    v26 = (__int64)v51;
    if ( v51 )
      sub_B515B0((__int64)v51, v127, v129, (__int64)&v152, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
      *(_QWORD *)(v5 + 608),
      v26,
      v150,
      *(_QWORD *)(v18 + 56),
      *(_QWORD *)(v18 + 64));
    v52 = *(_QWORD *)(v5 + 520);
    v53 = 16LL * *(unsigned int *)(v5 + 528);
    if ( v52 != v52 + v53 )
    {
      v118 = v5;
      v54 = v52 + v53;
      v55 = *(_QWORD *)(v5 + 520);
      do
      {
        v56 = *(_QWORD *)(v55 + 8);
        v57 = *(_DWORD *)v55;
        v55 += 16;
        sub_B99FD0(v26, v57, v56);
      }
      while ( v54 != v55 );
LABEL_52:
      v5 = v118;
    }
  }
LABEL_13:
  if ( sub_D96900(v12) )
  {
    v35 = (__int64 *)sub_BD5C60(v26);
    v28 = sub_ACD720(v35);
  }
  else
  {
    v150[1] = v26;
    v152 = "mul";
    BYTE4(v148) = 0;
    v150[0] = v120;
    v154 = 259;
    v146 = v129;
    v27 = sub_B33D10(v18, 0x171u, (__int64)&v146, 1, (int)v150, 2, v148, (__int64)&v152);
    v154 = 259;
    v152 = "mul.result";
    LODWORD(v150[0]) = 0;
    v26 = sub_94D3D0((unsigned int **)v18, v27, (__int64)v150, 1, (__int64)&v152);
    v154 = 259;
    v152 = "mul.overflow";
    LODWORD(v150[0]) = 1;
    v28 = sub_94D3D0((unsigned int **)v18, v27, (__int64)v150, 1, (__int64)&v152);
  }
  v117 = sub_DBEC00(*(_QWORD *)v5, v12);
  v121 = v117 ^ 1;
  v115 = sub_DBEDC0(*(_QWORD *)v5, v12);
  v130 = v115 ^ 1;
  if ( *(_BYTE *)(v132 + 8) != 14 )
  {
    if ( v121 )
    {
      v151 = 257;
      v29 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v5 + 600) + 32LL))(
              *(_QWORD *)(v5 + 600),
              13,
              v125,
              v26,
              0,
              0);
      if ( !v29 )
      {
        v154 = 257;
        v29 = sub_B504D0(13, (__int64)v125, v26, (__int64)&v152, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
          *(_QWORD *)(v5 + 608),
          v29,
          v150,
          *(_QWORD *)(v18 + 56),
          *(_QWORD *)(v18 + 64));
        v87 = *(_QWORD *)(v5 + 520);
        v88 = 16LL * *(unsigned int *)(v5 + 528);
        if ( v87 != v87 + v88 )
        {
          v137 = v18;
          v89 = v87 + v88;
          v113 = v5;
          v90 = *(_QWORD *)(v5 + 520);
          do
          {
            v91 = *(_QWORD *)(v90 + 8);
            v92 = *(_DWORD *)v90;
            v90 += 16;
            sub_B99FD0(v29, v92, v91);
          }
          while ( v89 != v90 );
          v18 = v137;
          v5 = v113;
        }
      }
      if ( !v130 )
      {
        v31 = 0;
        v154 = 257;
        if ( !a4 )
          goto LABEL_67;
        v38 = sub_92B530((unsigned int **)v18, 0x28u, v29, v125, (__int64)&v152);
        goto LABEL_29;
      }
      goto LABEL_18;
    }
    v29 = 0;
    if ( v130 )
    {
LABEL_18:
      v30 = *(_QWORD *)(v5 + 600);
      v151 = 257;
      v31 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64, _QWORD, _QWORD))(*(_QWORD *)v30 + 32LL))(
                        v30,
                        15,
                        v125,
                        v26,
                        0,
                        0);
      if ( !v31 )
      {
        v154 = 257;
        v135 = sub_B504D0(15, (__int64)v125, v26, (__int64)&v152, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
          *(_QWORD *)(v5 + 608),
          v135,
          v150,
          *(_QWORD *)(v18 + 56),
          *(_QWORD *)(v18 + 64));
        v80 = *(_QWORD *)(v5 + 520);
        v31 = (_QWORD *)v135;
        v81 = v80 + 16LL * *(unsigned int *)(v5 + 528);
        if ( v80 != v81 )
        {
          v136 = v18;
          v82 = (__int64)v31;
          v112 = v5;
          v83 = *(_QWORD *)(v5 + 520);
          v84 = v81;
          do
          {
            v85 = *(_QWORD *)(v83 + 8);
            v86 = *(_DWORD *)v83;
            v83 += 16;
            sub_B99FD0(v82, v86, v85);
          }
          while ( v84 != v83 );
          v31 = (_QWORD *)v82;
          v5 = v112;
          v18 = v136;
        }
      }
      goto LABEL_19;
    }
LABEL_28:
    v38 = 0;
    goto LABEL_29;
  }
  v151 = 257;
  v36 = sub_AD6530(*(_QWORD *)(v26 + 8), v12);
  v37 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v5 + 600) + 32LL))(
          *(_QWORD *)(v5 + 600),
          15,
          v36,
          v26,
          0,
          0);
  if ( !v37 )
  {
    v154 = 257;
    v139 = sub_B504D0(15, v36, v26, (__int64)&v152, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
      *(_QWORD *)(v5 + 608),
      v139,
      v150,
      *(_QWORD *)(v18 + 56),
      *(_QWORD *)(v18 + 64));
    v105 = *(_QWORD *)(v5 + 520);
    v37 = v139;
    v106 = 16LL * *(unsigned int *)(v5 + 528);
    if ( v105 != v105 + v106 )
    {
      v140 = v18;
      v107 = v105 + v106;
      v108 = v37;
      v114 = v5;
      v109 = *(_QWORD *)(v5 + 520);
      do
      {
        v110 = *(_QWORD *)(v109 + 8);
        v111 = *(_DWORD *)v109;
        v109 += 16;
        sub_B99FD0(v108, v111, v110);
      }
      while ( v107 != v109 );
      v37 = v108;
      v5 = v114;
      v18 = v140;
    }
  }
  if ( v121 )
  {
    v138 = v37;
    v154 = 257;
    v104 = sub_F7CA10((__int64 *)v18, (__int64)v125, v26, (__int64)&v152, 0);
    v37 = v138;
    v29 = (__int64)v104;
    if ( !v130 )
    {
      v31 = 0;
      v154 = 257;
      if ( !a4 )
      {
        v38 = sub_92B530((unsigned int **)v18, 0x24u, (__int64)v104, v125, (__int64)&v152);
        goto LABEL_29;
      }
      goto LABEL_21;
    }
  }
  else
  {
    if ( !v130 )
      goto LABEL_28;
    v29 = 0;
  }
  v154 = 257;
  v31 = sub_F7CA10((__int64 *)v18, (__int64)v125, v37, (__int64)&v152, 0);
LABEL_19:
  if ( v121 )
  {
    v154 = 257;
    if ( a4 )
    {
LABEL_21:
      v133 = v31;
      v32 = sub_92B530((unsigned int **)v18, 0x28u, v29, v125, (__int64)&v152);
      v31 = v133;
      v33 = v32;
      if ( v130 )
      {
        v34 = 38;
        v154 = 257;
LABEL_69:
        v72 = sub_92B530((unsigned int **)v18, v34, (__int64)v31, v125, (__int64)&v152);
        v38 = v72;
        if ( !v117 && !v115 )
        {
          v154 = 257;
          v38 = sub_B36550((unsigned int **)v18, (__int64)v128, v72, v33, (__int64)&v152, 0);
        }
        goto LABEL_29;
      }
LABEL_93:
      v38 = v33;
      goto LABEL_29;
    }
LABEL_67:
    v134 = v31;
    v71 = sub_92B530((unsigned int **)v18, 0x24u, v29, v125, (__int64)&v152);
    v31 = v134;
    v33 = v71;
    if ( v130 )
    {
      v154 = 257;
      v34 = 34;
      goto LABEL_69;
    }
    goto LABEL_93;
  }
  v154 = 257;
  if ( !a4 )
  {
    v33 = 0;
    v34 = 34;
    goto LABEL_69;
  }
  v38 = sub_92B530((unsigned int **)v18, 0x26u, (__int64)v31, v125, (__int64)&v152);
LABEL_29:
  v39 = 29;
  v40 = *(_QWORD *)(v5 + 600);
  v151 = 257;
  v41 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v40 + 16LL))(v40, 29, v38, v28);
  if ( !v41 )
  {
    v154 = 257;
    v41 = sub_B504D0(29, v38, v28, (__int64)&v152, 0, 0);
    v39 = v41;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
      *(_QWORD *)(v5 + 608),
      v41,
      v150,
      *(_QWORD *)(v18 + 56),
      *(_QWORD *)(v18 + 64));
    v68 = *(unsigned int **)(v5 + 520);
    v69 = &v68[4 * *(unsigned int *)(v5 + 528)];
    while ( v69 != v68 )
    {
      v70 = *((_QWORD *)v68 + 1);
      v39 = *v68;
      v68 += 4;
      sub_B99FD0(v41, v39, v70);
    }
  }
LABEL_30:
  if ( v122 > v126 )
  {
    v153 = v126;
    if ( v126 > 0x40 )
    {
      sub_C43690((__int64)&v152, -1, 1);
    }
    else
    {
      v43 = (char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v126);
      if ( !v126 )
        v43 = 0;
      v152 = v43;
    }
    sub_C449B0((__int64)&v146, (const void **)&v152, v122);
    if ( v153 > 0x40 && v152 )
      j_j___libc_free_0_0(v152);
    v154 = 257;
    v44 = (__int64 *)sub_BD5C60(a3);
    v45 = (_BYTE *)sub_ACCFD0(v44, (__int64)&v146);
    v46 = sub_92B530((unsigned int **)v18, 0x22u, v127, v45, (__int64)&v152);
    v151 = 257;
    v149 = 257;
    v47 = sub_92B530((unsigned int **)v18, 0x21u, v131, v123, (__int64)&v148);
    v48 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v5 + 600) + 16LL))(
            *(_QWORD *)(v5 + 600),
            28,
            v46,
            v47);
    if ( !v48 )
    {
      v154 = 257;
      v143 = sub_B504D0(28, v46, v47, (__int64)&v152, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
        *(_QWORD *)(v5 + 608),
        v143,
        v150,
        *(_QWORD *)(v18 + 56),
        *(_QWORD *)(v18 + 64));
      v93 = *(_QWORD *)(v5 + 520);
      v48 = v143;
      v94 = v93 + 16LL * *(unsigned int *)(v5 + 528);
      if ( v93 != v94 )
      {
        v144 = v5;
        v95 = *(_QWORD *)(v5 + 520);
        v96 = v94;
        v97 = v48;
        do
        {
          v98 = *(_QWORD *)(v95 + 8);
          v99 = *(_DWORD *)v95;
          v95 += 16;
          sub_B99FD0(v97, v99, v98);
        }
        while ( v96 != v95 );
        v5 = v144;
        v48 = v97;
      }
    }
    v49 = *(_QWORD *)(v5 + 600);
    v151 = 257;
    v39 = 29;
    v142 = v48;
    v50 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v49 + 16LL))(v49, 29, v41, v48);
    if ( !v50 )
    {
      v154 = 257;
      v50 = sub_B504D0(29, v41, v142, (__int64)&v152, 0, 0);
      v39 = v50;
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v5 + 608) + 16LL))(
        *(_QWORD *)(v5 + 608),
        v50,
        v150,
        *(_QWORD *)(v18 + 56),
        *(_QWORD *)(v18 + 64));
      v100 = *(unsigned int **)(v5 + 520);
      v101 = &v100[4 * *(unsigned int *)(v5 + 528)];
      if ( v100 != v101 )
      {
        v102 = *(unsigned int **)(v5 + 520);
        do
        {
          v103 = *((_QWORD *)v102 + 1);
          v39 = *v102;
          v102 += 4;
          sub_B99FD0(v50, v39, v103);
        }
        while ( v101 != v102 );
      }
    }
    if ( v147 > 0x40 && v146 )
      j_j___libc_free_0_0(v146);
    v41 = v50;
  }
  if ( (_BYTE *)v155[0] != v156 )
    _libc_free(v155[0], v39);
  return v41;
}
