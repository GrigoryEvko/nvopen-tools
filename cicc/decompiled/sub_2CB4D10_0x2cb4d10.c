// Function: sub_2CB4D10
// Address: 0x2cb4d10
//
__int64 __fastcall sub_2CB4D10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r8
  __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rdi
  _BYTE *v12; // rsi
  _QWORD *v13; // r15
  __int64 v14; // rdx
  unsigned __int64 v15; // r14
  void **p_s; // r13
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  bool v20; // zf
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  char v26; // al
  unsigned __int64 v27; // rdx
  _QWORD *v28; // r8
  unsigned __int64 v29; // r10
  char *v30; // r15
  char *v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // rsi
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r12
  unsigned int *v39; // rax
  int v40; // ecx
  unsigned int *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // rax
  _BYTE *v46; // r12
  __int64 v47; // rax
  unsigned __int8 *v48; // rbx
  __int64 (__fastcall *v49)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v50; // r12
  __int64 v51; // rax
  unsigned __int8 *v52; // r10
  __int64 (__fastcall *v53)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v54; // rax
  __int64 v55; // rbx
  size_t v56; // r14
  _QWORD *v57; // rsi
  unsigned __int64 v58; // rdi
  _QWORD *v59; // rcx
  unsigned __int64 v60; // rdx
  char *v61; // rax
  _QWORD *v62; // rdx
  _QWORD *v63; // rbx
  unsigned __int64 v64; // rdi
  char *v65; // rdi
  size_t v66; // rdx
  _QWORD *v67; // rbx
  unsigned __int64 v68; // r12
  unsigned __int64 v69; // rdi
  char *v70; // rdi
  size_t v71; // rdx
  _QWORD *v73; // rax
  __int64 v74; // rdx
  unsigned int *v75; // r15
  unsigned int *v76; // r12
  __int64 v77; // rdx
  unsigned int v78; // esi
  _QWORD *v79; // rax
  unsigned int *v80; // r13
  unsigned int *v81; // rbx
  __int64 v82; // rdx
  unsigned int v83; // esi
  __int64 v84; // rax
  unsigned __int64 v85; // rsi
  unsigned __int64 v86; // rax
  __int64 v87; // [rsp+0h] [rbp-220h]
  unsigned __int8 *v88; // [rsp+8h] [rbp-218h]
  __int64 v89; // [rsp+8h] [rbp-218h]
  void **v90; // [rsp+8h] [rbp-218h]
  unsigned __int8 *v91; // [rsp+8h] [rbp-218h]
  __int64 v93; // [rsp+30h] [rbp-1F0h]
  _QWORD *v94; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v95; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v96; // [rsp+48h] [rbp-1D8h]
  __int64 v97; // [rsp+48h] [rbp-1D8h]
  __int64 v98; // [rsp+50h] [rbp-1D0h]
  _BYTE *v99; // [rsp+50h] [rbp-1D0h]
  _QWORD *v100; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v101; // [rsp+50h] [rbp-1D0h]
  _QWORD *v102; // [rsp+58h] [rbp-1C8h]
  unsigned __int64 v103; // [rsp+58h] [rbp-1C8h]
  _QWORD *v104; // [rsp+58h] [rbp-1C8h]
  __int64 v105; // [rsp+68h] [rbp-1B8h] BYREF
  __int64 v106; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v107; // [rsp+78h] [rbp-1A8h]
  _BYTE v108[32]; // [rsp+80h] [rbp-1A0h] BYREF
  __int16 v109; // [rsp+A0h] [rbp-180h]
  _QWORD v110[4]; // [rsp+B0h] [rbp-170h] BYREF
  __int16 v111; // [rsp+D0h] [rbp-150h]
  void *v112; // [rsp+E0h] [rbp-140h]
  unsigned __int64 v113; // [rsp+E8h] [rbp-138h]
  _QWORD *v114; // [rsp+F0h] [rbp-130h] BYREF
  __int128 v115; // [rsp+F8h] [rbp-128h] BYREF
  __int128 v116; // [rsp+108h] [rbp-118h] BYREF
  void *s; // [rsp+120h] [rbp-100h] BYREF
  __int64 v118; // [rsp+128h] [rbp-F8h]
  _QWORD *v119; // [rsp+130h] [rbp-F0h]
  __int128 v120; // [rsp+138h] [rbp-E8h]
  __int128 v121; // [rsp+148h] [rbp-D8h] BYREF
  unsigned int *v122; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v123; // [rsp+168h] [rbp-B8h]
  _BYTE v124[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+190h] [rbp-90h]
  __int64 v126; // [rsp+198h] [rbp-88h]
  __int64 v127; // [rsp+1A0h] [rbp-80h]
  _QWORD *v128; // [rsp+1A8h] [rbp-78h]
  void **v129; // [rsp+1B0h] [rbp-70h]
  void **v130; // [rsp+1B8h] [rbp-68h]
  __int64 v131; // [rsp+1C0h] [rbp-60h]
  int v132; // [rsp+1C8h] [rbp-58h]
  __int16 v133; // [rsp+1CCh] [rbp-54h]
  char v134; // [rsp+1CEh] [rbp-52h]
  __int64 v135; // [rsp+1D0h] [rbp-50h]
  __int64 v136; // [rsp+1D8h] [rbp-48h]
  void *v137; // [rsp+1E0h] [rbp-40h] BYREF
  void *v138; // [rsp+1E8h] [rbp-38h] BYREF

  v112 = (char *)&v116 + 8;
  v2 = *(_QWORD *)(a1 + 80);
  v115 = 0;
  v113 = 1;
  v114 = 0;
  DWORD2(v115) = 1065353216;
  v98 = v2;
  v93 = a1 + 72;
  v116 = 0;
  if ( v2 == a1 + 72 )
  {
    v70 = (char *)&v116 + 8;
    v71 = 8;
  }
  else
  {
    do
    {
      if ( !v98 )
        BUG();
      v3 = *(_QWORD *)(v98 + 32);
      if ( v3 != v98 + 24 )
      {
        while ( 1 )
        {
          v4 = v3 - 24;
          if ( !v3 )
            v4 = 0;
          v5 = sub_2CB2D10(v4);
          v122 = (unsigned int *)v5;
          if ( !v5 )
            goto LABEL_20;
          v6 = *(_QWORD *)(v5 - 32);
          if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(v5 + 80) )
            BUG();
          v96 = *(unsigned int *)(v6 + 36);
          v7 = (_QWORD *)*((_QWORD *)v112 + v96 % v113);
          v8 = 8 * (v96 % v113);
          if ( v7 )
          {
            v9 = (_QWORD *)*v7;
            if ( (_DWORD)v96 == *(_DWORD *)(*v7 + 8LL) )
            {
LABEL_15:
              v11 = *v7 + 16LL;
              if ( *v7 )
              {
                v12 = *(_BYTE **)(*v7 + 24LL);
                if ( v12 != *(_BYTE **)(*v7 + 32LL) )
                  goto LABEL_17;
                goto LABEL_42;
              }
            }
            else
            {
              while ( 1 )
              {
                v10 = (_QWORD *)*v9;
                if ( !*v9 )
                  break;
                v7 = v9;
                if ( v96 % v113 != *((unsigned int *)v10 + 2) % v113 )
                  break;
                v9 = (_QWORD *)*v9;
                if ( (_DWORD)v96 == *((_DWORD *)v10 + 2) )
                  goto LABEL_15;
              }
            }
          }
          v23 = sub_22077B0(0x28u);
          if ( v23 )
            *(_QWORD *)v23 = 0;
          *(_DWORD *)(v23 + 8) = v96;
          v24 = v115;
          *(_QWORD *)(v23 + 16) = 0;
          v25 = v113;
          *(_QWORD *)(v23 + 24) = 0;
          *(_QWORD *)(v23 + 32) = 0;
          v102 = (_QWORD *)v23;
          v26 = sub_222DA10((__int64)&v115 + 8, v25, v24, 1);
          v28 = v102;
          v29 = v27;
          if ( !v26 )
          {
            v30 = (char *)v112;
            v31 = (char *)v112 + v8;
            v32 = *(_QWORD **)((char *)v112 + v8);
            if ( v32 )
              goto LABEL_40;
            goto LABEL_78;
          }
          if ( v27 == 1 )
          {
            *((_QWORD *)&v116 + 1) = 0;
            v30 = (char *)&v116 + 8;
          }
          else
          {
            if ( v27 > 0xFFFFFFFFFFFFFFFLL )
              sub_4261EA((char *)&v115 + 8, v25, v27);
            v56 = 8 * v27;
            v94 = v102;
            v103 = v27;
            v30 = (char *)sub_22077B0(8 * v27);
            memset(v30, 0, v56);
            v28 = v94;
            v29 = v103;
          }
          v57 = v114;
          v114 = 0;
          if ( v57 )
            break;
LABEL_75:
          if ( v112 != (char *)&v116 + 8 )
          {
            v95 = v29;
            v104 = v28;
            j_j___libc_free_0((unsigned __int64)v112);
            v29 = v95;
            v28 = v104;
          }
          v113 = v29;
          v112 = v30;
          v8 = 8 * (v96 % v29);
          v31 = &v30[v8];
          v32 = *(_QWORD **)&v30[v8];
          if ( v32 )
          {
LABEL_40:
            *v28 = *v32;
            **(_QWORD **)v31 = v28;
            goto LABEL_41;
          }
LABEL_78:
          v62 = v114;
          v114 = v28;
          *v28 = v62;
          if ( v62 )
          {
            *(_QWORD *)&v30[8 * (*((unsigned int *)v62 + 2) % v113)] = v28;
            v31 = (char *)v112 + v8;
          }
          *(_QWORD *)v31 = &v114;
LABEL_41:
          v11 = (__int64)(v28 + 2);
          *(_QWORD *)&v115 = v115 + 1;
          v12 = (_BYTE *)v28[3];
          if ( v12 != (_BYTE *)v28[4] )
          {
LABEL_17:
            if ( v12 )
            {
              *(_QWORD *)v12 = v122;
              v12 = *(_BYTE **)(v11 + 8);
            }
            *(_QWORD *)(v11 + 8) = v12 + 8;
            goto LABEL_20;
          }
LABEL_42:
          sub_2CB4B10(v11, v12, &v122);
LABEL_20:
          v3 = *(_QWORD *)(v3 + 8);
          if ( v98 + 24 == v3 )
            goto LABEL_21;
        }
        v58 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v59 = v57;
            v57 = (_QWORD *)*v57;
            v60 = *((unsigned int *)v59 + 2) % v29;
            v61 = &v30[8 * v60];
            if ( !*(_QWORD *)v61 )
              break;
            *v59 = **(_QWORD **)v61;
            **(_QWORD **)v61 = v59;
LABEL_71:
            if ( !v57 )
              goto LABEL_75;
          }
          *v59 = v114;
          v114 = v59;
          *(_QWORD *)v61 = &v114;
          if ( !*v59 )
          {
            v58 = v60;
            goto LABEL_71;
          }
          *(_QWORD *)&v30[8 * v58] = v59;
          v58 = v60;
          if ( !v57 )
            goto LABEL_75;
        }
      }
LABEL_21:
      v98 = *(_QWORD *)(v98 + 8);
    }
    while ( v93 != v98 );
    v13 = v114;
    if ( !v114 )
      goto LABEL_91;
LABEL_25:
    while ( 2 )
    {
      if ( v13[3] - v13[2] > 8u )
      {
        v119 = 0;
        v120 = 0;
        v121 = 0;
        DWORD2(v120) = 1065353216;
        v14 = v13[2];
        s = (char *)&v121 + 8;
        v118 = 1;
        if ( v13[3] == v14 )
        {
          v65 = (char *)&v121 + 8;
          v66 = 8;
          goto LABEL_85;
        }
        v15 = 0;
        p_s = &s;
        while ( 1 )
        {
          v18 = *(_QWORD *)(v14 + 8 * v15++);
          v105 = v18;
          v19 = sub_2CB4CA0(p_s, &v105);
          v14 = v13[2];
          v20 = v19 == 0;
          v21 = v13[3];
          if ( v20 )
            break;
LABEL_28:
          v17 = (v21 - v14) >> 3;
LABEL_29:
          if ( v15 >= v17 )
          {
            v63 = v119;
            while ( v63 )
            {
              v64 = (unsigned __int64)v63;
              v63 = (_QWORD *)*v63;
              j_j___libc_free_0(v64);
            }
            v65 = (char *)s;
            v66 = 8 * v118;
LABEL_85:
            memset(v65, 0, v66);
            *(_QWORD *)&v120 = 0;
            v119 = 0;
            if ( s == (char *)&v121 + 8 )
              goto LABEL_24;
            j_j___libc_free_0((unsigned __int64)s);
            v13 = (_QWORD *)*v13;
            if ( !v13 )
              goto LABEL_87;
            goto LABEL_25;
          }
        }
        v17 = (v21 - v14) >> 3;
        if ( v17 <= v15 )
          goto LABEL_29;
        v22 = v15;
        while ( 1 )
        {
          v106 = *(_QWORD *)(v14 + 8 * v22);
          if ( !sub_2CB4CA0(p_s, &v106) )
          {
            v33 = sub_2CB3BD0(v105, v106, a2);
            if ( v33 )
              break;
          }
          v14 = v13[2];
          ++v22;
          v17 = (v13[3] - v14) >> 3;
          if ( v22 >= v17 )
            goto LABEL_29;
        }
        v34 = v33;
        v128 = (_QWORD *)sub_BD5C60(v33);
        v129 = &v137;
        v130 = &v138;
        v122 = (unsigned int *)v124;
        v123 = 0x200000000LL;
        v137 = &unk_49DA100;
        v125 = 0;
        v131 = 0;
        v126 = 0;
        v132 = 0;
        v133 = 512;
        v134 = 7;
        v135 = 0;
        v136 = 0;
        LOWORD(v127) = 0;
        v138 = &unk_49DA0B0;
        v125 = *(_QWORD *)(v34 + 40);
        v126 = v34 + 24;
        v35 = *(_QWORD *)sub_B46C60(v34);
        v110[0] = v35;
        if ( v35 && (sub_B96E90((__int64)v110, v35, 1), (v38 = v110[0]) != 0) )
        {
          v39 = v122;
          v40 = v123;
          v41 = &v122[4 * (unsigned int)v123];
          if ( v122 != v41 )
          {
            while ( 1 )
            {
              v36 = *v39;
              if ( !(_DWORD)v36 )
                break;
              v39 += 4;
              if ( v41 == v39 )
                goto LABEL_98;
            }
            *((_QWORD *)v39 + 1) = v110[0];
            goto LABEL_51;
          }
LABEL_98:
          if ( (unsigned int)v123 >= (unsigned __int64)HIDWORD(v123) )
          {
            v85 = (unsigned int)v123 + 1LL;
            v86 = v87 & 0xFFFFFFFF00000000LL;
            v87 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v123) < v85 )
            {
              v101 = v86;
              sub_C8D5F0((__int64)&v122, v124, v85, 0x10u, v36, v37);
              v86 = v101;
              v41 = &v122[4 * (unsigned int)v123];
            }
            *(_QWORD *)v41 = v86;
            *((_QWORD *)v41 + 1) = v38;
            v38 = v110[0];
            LODWORD(v123) = v123 + 1;
          }
          else
          {
            if ( v41 )
            {
              *v41 = 0;
              *((_QWORD *)v41 + 1) = v38;
              v40 = v123;
              v38 = v110[0];
            }
            LODWORD(v123) = v40 + 1;
          }
        }
        else
        {
          sub_93FB40((__int64)&v122, 0);
          v38 = v110[0];
        }
        if ( !v38 )
        {
LABEL_52:
          v42 = sub_2CB2E70(v105, v106, (__int64)&v122);
          BYTE4(v107) = 0;
          LODWORD(v107) = 2;
          v43 = v42;
          v44 = sub_BCE1B0(**(__int64 ***)(*(_QWORD *)(v105 + 80) + 16LL), v107);
          v111 = 257;
          v45 = sub_A83570(&v122, v43, v44, (__int64)v110);
          v109 = 257;
          v46 = (_BYTE *)v45;
          v99 = (_BYTE *)v45;
          v47 = sub_BCB2E0(v128);
          v48 = (unsigned __int8 *)sub_ACD640(v47, 0, 0);
          v49 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v129 + 12);
          if ( v49 == sub_948070 )
          {
            if ( *v46 > 0x15u || *v48 > 0x15u )
            {
LABEL_108:
              v111 = 257;
              v79 = sub_BD2C40(72, 2u);
              v50 = (__int64)v79;
              if ( v79 )
                sub_B4DE80((__int64)v79, (__int64)v99, (__int64)v48, (__int64)v110, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v130 + 2))(
                v130,
                v50,
                v108,
                v126,
                v127);
              if ( v122 != &v122[4 * (unsigned int)v123] )
              {
                v90 = p_s;
                v80 = v122;
                v81 = &v122[4 * (unsigned int)v123];
                do
                {
                  v82 = *((_QWORD *)v80 + 1);
                  v83 = *v80;
                  v80 += 4;
                  sub_B99FD0(v50, v83, v82);
                }
                while ( v81 != v80 );
                p_s = v90;
              }
LABEL_57:
              v109 = 257;
              v51 = sub_BCB2E0(v128);
              v52 = (unsigned __int8 *)sub_ACD640(v51, 1, 0);
              v53 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v129 + 12);
              if ( v53 == sub_948070 )
              {
                if ( *v99 > 0x15u || *v52 > 0x15u )
                  goto LABEL_102;
                v88 = v52;
                v54 = sub_AD5840((__int64)v99, v52, 0);
                v52 = v88;
                v55 = v54;
              }
              else
              {
                v91 = v52;
                v84 = v53((__int64)v129, v99, v52);
                v52 = v91;
                v55 = v84;
              }
              if ( v55 )
              {
LABEL_62:
                sub_BD84D0(v105, v50);
                sub_BD84D0(v106, v55);
                sub_2CAFB90((unsigned __int64 *)p_s, (unsigned __int64 *)&v105, 1);
                sub_2CAFB90((unsigned __int64 *)p_s, (unsigned __int64 *)&v106, 1);
                nullsub_61();
                v137 = &unk_49DA100;
                nullsub_63();
                if ( v122 != (unsigned int *)v124 )
                  _libc_free((unsigned __int64)v122);
                v14 = v13[2];
                v21 = v13[3];
                goto LABEL_28;
              }
LABEL_102:
              v89 = (__int64)v52;
              v111 = 257;
              v73 = sub_BD2C40(72, 2u);
              v55 = (__int64)v73;
              if ( v73 )
                sub_B4DE80((__int64)v73, (__int64)v99, v89, (__int64)v110, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v130 + 2))(
                v130,
                v55,
                v108,
                v126,
                v127);
              v74 = 4LL * (unsigned int)v123;
              if ( v122 != &v122[v74] )
              {
                v100 = v13;
                v75 = &v122[v74];
                v97 = v50;
                v76 = v122;
                do
                {
                  v77 = *((_QWORD *)v76 + 1);
                  v78 = *v76;
                  v76 += 4;
                  sub_B99FD0(v55, v78, v77);
                }
                while ( v75 != v76 );
                v13 = v100;
                v50 = v97;
              }
              goto LABEL_62;
            }
            v50 = sub_AD5840((__int64)v99, v48, 0);
          }
          else
          {
            v50 = v49((__int64)v129, v99, v48);
          }
          if ( v50 )
            goto LABEL_57;
          goto LABEL_108;
        }
LABEL_51:
        sub_B91220((__int64)v110, v38);
        goto LABEL_52;
      }
LABEL_24:
      v13 = (_QWORD *)*v13;
      if ( v13 )
        continue;
      break;
    }
LABEL_87:
    v67 = v114;
    while ( v67 )
    {
      v68 = (unsigned __int64)v67;
      v67 = (_QWORD *)*v67;
      v69 = *(_QWORD *)(v68 + 16);
      if ( v69 )
        j_j___libc_free_0(v69);
      j_j___libc_free_0(v68);
    }
LABEL_91:
    v70 = (char *)v112;
    v71 = 8 * v113;
  }
  memset(v70, 0, v71);
  *(_QWORD *)&v115 = 0;
  v114 = 0;
  if ( v112 != (char *)&v116 + 8 )
    j_j___libc_free_0((unsigned __int64)v112);
  return 0;
}
