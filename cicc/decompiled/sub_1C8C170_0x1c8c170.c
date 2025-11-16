// Function: sub_1C8C170
// Address: 0x1c8c170
//
void __fastcall sub_1C8C170(
        _BYTE *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 **v19; // rax
  __int64 v20; // r10
  __int64 v21; // rbx
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // r14
  unsigned __int64 *v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 *v28; // r15
  __int64 v29; // rsi
  unsigned __int8 *v30; // rsi
  __int64 v31; // rsi
  __int64 *v32; // rdi
  int v33; // eax
  const char *v34; // rdx
  __int64 v35; // rcx
  const char *v36; // rdx
  __int64 v37; // rax
  unsigned __int8 *v38; // rsi
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 *v41; // rax
  __int64 *v42; // rbx
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r11
  __int64 v47; // rdx
  __int64 v48; // rcx
  char v49; // al
  char v50; // al
  _QWORD *v51; // rax
  __int64 v52; // r14
  __int64 *v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 *v56; // r15
  __int64 v57; // rsi
  unsigned __int8 *v58; // rsi
  double v59; // xmm4_8
  double v60; // xmm5_8
  __int64 **v61; // rdx
  __int64 v62; // r8
  __int64 v63; // rsi
  __int64 v64; // rdi
  char v65; // bl
  double v66; // xmm4_8
  double v67; // xmm5_8
  const char *v68; // rdx
  __int64 v69; // rdi
  char v70; // bl
  char v71; // bl
  __int64 v72; // rdi
  char v73; // bl
  __int64 v74; // rdi
  const char *v75; // rdx
  __int64 v76; // rcx
  char v77; // al
  char v78; // al
  bool v79; // al
  bool v80; // al
  bool v81; // al
  bool v82; // al
  bool v83; // al
  bool v84; // al
  bool v85; // al
  bool v86; // al
  bool v87; // al
  bool v88; // al
  bool v89; // al
  bool v90; // al
  bool v91; // al
  bool v92; // al
  bool v93; // al
  bool v94; // al
  bool v95; // al
  bool v96; // al
  __int64 v97; // rsi
  unsigned __int8 *v98; // rsi
  __int64 v99; // rsi
  unsigned __int8 *v100; // rsi
  char v101; // al
  __int64 v102; // rsi
  __int64 v103; // r10
  __int64 *v104; // rbx
  __int64 v105; // rcx
  __int64 v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rbx
  unsigned __int8 *v109; // rsi
  __int64 v110; // r8
  __int64 v111; // rax
  __int64 v112; // rsi
  __int64 v113; // rsi
  __int64 v114; // rdx
  unsigned __int8 *v115; // rsi
  __int64 v116; // rax
  __int64 v117; // r10
  __int64 *v118; // r14
  __int64 v119; // rax
  __int64 v120; // rsi
  __int64 v121; // rsi
  unsigned __int8 *v122; // rsi
  __int64 v123; // rsi
  int v124; // edi
  __int64 v125; // r11
  __int64 *v126; // r14
  __int64 v127; // rcx
  __int64 v128; // rax
  __int64 v129; // rsi
  __int64 v130; // r14
  unsigned __int8 *v131; // rsi
  bool v132; // al
  __int64 *v133; // [rsp+8h] [rbp-108h]
  __int64 v134; // [rsp+10h] [rbp-100h]
  __int64 v135; // [rsp+10h] [rbp-100h]
  __int64 v136; // [rsp+10h] [rbp-100h]
  __int64 v137; // [rsp+10h] [rbp-100h]
  __int64 v138; // [rsp+10h] [rbp-100h]
  __int64 v139; // [rsp+10h] [rbp-100h]
  __int64 v140; // [rsp+10h] [rbp-100h]
  __int64 v141; // [rsp+10h] [rbp-100h]
  __int64 v142; // [rsp+10h] [rbp-100h]
  __int64 v143; // [rsp+10h] [rbp-100h]
  __int64 v144; // [rsp+10h] [rbp-100h]
  __int64 v145; // [rsp+10h] [rbp-100h]
  __int64 v146; // [rsp+18h] [rbp-F8h]
  __int64 v147; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v148; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v149[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v150; // [rsp+40h] [rbp-D0h]
  __int64 v151[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v152; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v153; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v154; // [rsp+78h] [rbp-98h]
  __int64 v155[2]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int8 *v156; // [rsp+90h] [rbp-80h] BYREF
  __int64 v157; // [rsp+98h] [rbp-78h]
  __int64 *v158; // [rsp+A0h] [rbp-70h]
  __int64 v159; // [rsp+A8h] [rbp-68h]
  __int64 v160; // [rsp+B0h] [rbp-60h]
  int v161; // [rsp+B8h] [rbp-58h]
  __int64 v162; // [rsp+C0h] [rbp-50h]
  __int64 v163; // [rsp+C8h] [rbp-48h]

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x25:
    case 0x27:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
      return;
    case 0x24:
      v34 = "__nv_add_fp128";
      v35 = 14;
      goto LABEL_38;
    case 0x26:
      v34 = "__nv_sub_fp128";
      v35 = 14;
      goto LABEL_38;
    case 0x28:
      v34 = "__nv_mul_fp128";
      v35 = 14;
      goto LABEL_38;
    case 0x29:
      v36 = "__nv_udiv128";
      goto LABEL_42;
    case 0x2A:
      v36 = "__nv_idiv128";
      goto LABEL_42;
    case 0x2B:
      v34 = "__nv_div_fp128";
      v35 = 14;
      goto LABEL_38;
    case 0x2C:
      v36 = "__nv_urem128";
      goto LABEL_42;
    case 0x2D:
      v36 = "__nv_irem128";
LABEL_42:
      sub_1C8BD70(a1, (__int64 *)a2, (__int64)v36, 12, a3, a4, a5, a6, a7, a8, a9, a10);
      return;
    case 0x2E:
      v34 = "__nv_rem_fp128";
      v35 = 14;
LABEL_38:
      sub_1C8A5C0(a1, (_QWORD *)a2, (__int64)v34, v35, a3, a4, a5, a6, a7, a8, a9, a10);
      return;
    case 0x36:
      if ( !sub_1642F90(*(_QWORD *)a2, 128) && *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 5 )
        return;
      v37 = sub_16498A0(a2);
      v38 = *(unsigned __int8 **)(a2 + 48);
      v156 = 0;
      v159 = v37;
      v39 = *(_QWORD *)(a2 + 40);
      v160 = 0;
      v157 = v39;
      v161 = 0;
      v162 = 0;
      v163 = 0;
      v158 = (__int64 *)(a2 + 24);
      v153 = v38;
      if ( v38 )
      {
        sub_1623A60((__int64)&v153, (__int64)v38, 2);
        if ( v156 )
          sub_161E7C0((__int64)&v156, (__int64)v156);
        v156 = v153;
        if ( v153 )
          sub_1623210((__int64)&v153, v153, (__int64)&v156);
      }
      v40 = (_QWORD *)sub_16498A0(a2);
      v41 = (__int64 *)sub_1643360(v40);
      v42 = sub_16463B0(v41, 2u);
      v43 = *(__int64 **)(a2 - 24);
      v152 = 257;
      v44 = *v43;
      if ( *(_BYTE *)(v44 + 8) == 16 )
        v44 = **(_QWORD **)(v44 + 16);
      v45 = sub_1647190(v42, *(_DWORD *)(v44 + 8) >> 8);
      v46 = *(_QWORD *)(a2 - 24);
      v47 = v45;
      v48 = *(_QWORD *)v46;
      if ( v45 == *(_QWORD *)v46 )
        goto LABEL_67;
      v49 = *(_BYTE *)(v48 + 8);
      if ( v49 == 16 )
        v49 = *(_BYTE *)(**(_QWORD **)(v48 + 16) + 8LL);
      if ( v49 == 15 )
      {
        v101 = *(_BYTE *)(v47 + 8);
        if ( v101 == 16 )
          v101 = *(_BYTE *)(**(_QWORD **)(v47 + 16) + 8LL);
        if ( v101 != 11 )
          goto LABEL_65;
        if ( *(_BYTE *)(v46 + 16) <= 0x10u )
        {
          v46 = sub_15A46C0(45, *(__int64 ****)(a2 - 24), (__int64 **)v47, 0);
          goto LABEL_67;
        }
        v123 = *(_QWORD *)(a2 - 24);
        v124 = 45;
        LOWORD(v155[0]) = 257;
      }
      else
      {
        if ( v49 != 11 )
          goto LABEL_65;
        v50 = *(_BYTE *)(v47 + 8);
        if ( v50 == 16 )
          v50 = *(_BYTE *)(**(_QWORD **)(v47 + 16) + 8LL);
        if ( v50 != 15 )
        {
LABEL_65:
          if ( *(_BYTE *)(v46 + 16) <= 0x10u )
          {
            v46 = sub_15A46C0(47, *(__int64 ****)(a2 - 24), (__int64 **)v47, 0);
            goto LABEL_67;
          }
          v123 = *(_QWORD *)(a2 - 24);
          LOWORD(v155[0]) = 257;
          v124 = 47;
          goto LABEL_193;
        }
        if ( *(_BYTE *)(v46 + 16) <= 0x10u )
        {
          v46 = sub_15A46C0(46, *(__int64 ****)(a2 - 24), (__int64 **)v47, 0);
          goto LABEL_67;
        }
        v123 = *(_QWORD *)(a2 - 24);
        v124 = 46;
        LOWORD(v155[0]) = 257;
      }
LABEL_193:
      v125 = sub_15FDBD0(v124, v123, v47, (__int64)&v153, 0);
      if ( v157 )
      {
        v126 = v158;
        v147 = v125;
        sub_157E9D0(v157 + 40, v125);
        v125 = v147;
        v127 = *v126;
        v128 = *(_QWORD *)(v147 + 24);
        *(_QWORD *)(v147 + 32) = v126;
        v127 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v147 + 24) = v127 | v128 & 7;
        *(_QWORD *)(v127 + 8) = v147 + 24;
        *v126 = *v126 & 7 | (v147 + 24);
      }
      v144 = v125;
      sub_164B780(v125, v151);
      v46 = v144;
      if ( v156 )
      {
        v149[0] = (__int64)v156;
        sub_1623A60((__int64)v149, (__int64)v156, 2);
        v46 = v144;
        v129 = *(_QWORD *)(v144 + 48);
        v130 = v144 + 48;
        if ( v129 )
        {
          sub_161E7C0(v144 + 48, v129);
          v46 = v144;
        }
        v131 = (unsigned __int8 *)v149[0];
        *(_QWORD *)(v46 + 48) = v149[0];
        if ( v131 )
        {
          v145 = v46;
          sub_1623210((__int64)v149, v131, v130);
          v46 = v145;
        }
      }
LABEL_67:
      v136 = v46;
      LOWORD(v155[0]) = 257;
      v51 = sub_1648A60(64, 1u);
      v52 = (__int64)v51;
      if ( v51 )
        sub_15F9210((__int64)v51, (__int64)v42, v136, 0, 0, 0);
      if ( v157 )
      {
        v53 = v158;
        sub_157E9D0(v157 + 40, v52);
        v54 = *(_QWORD *)(v52 + 24);
        v55 = *v53;
        *(_QWORD *)(v52 + 32) = v53;
        v55 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v52 + 24) = v55 | v54 & 7;
        *(_QWORD *)(v55 + 8) = v52 + 24;
        *v53 = *v53 & 7 | (v52 + 24);
      }
      v56 = (__int64 *)(v52 + 48);
      sub_164B780(v52, (__int64 *)&v153);
      if ( v156 )
      {
        v151[0] = (__int64)v156;
        sub_1623A60((__int64)v151, (__int64)v156, 2);
        v57 = *(_QWORD *)(v52 + 48);
        if ( v57 )
          sub_161E7C0(v52 + 48, v57);
        v58 = (unsigned __int8 *)v151[0];
        *(_QWORD *)(v52 + 48) = v151[0];
        if ( v58 )
          sub_1623210((__int64)v151, v58, v52 + 48);
      }
      v153 = (unsigned __int8 *)v155;
      v155[0] = a2;
      v154 = 0x200000001LL;
      sub_14C4AD0(v52, v155, 1);
      sub_15F8F50(v52, 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1);
      v61 = *(__int64 ***)a2;
      v150 = 257;
      if ( v61 == *(__int64 ***)v52 )
      {
        v62 = v52;
      }
      else if ( *(_BYTE *)(v52 + 16) > 0x10u )
      {
        v152 = 257;
        v110 = sub_15FDBD0(47, v52, (__int64)v61, (__int64)v151, 0);
        if ( v157 )
        {
          v139 = v110;
          v133 = v158;
          sub_157E9D0(v157 + 40, v110);
          v110 = v139;
          v111 = *(_QWORD *)(v139 + 24);
          v112 = *v133;
          *(_QWORD *)(v139 + 32) = v133;
          v112 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v139 + 24) = v112 | v111 & 7;
          *(_QWORD *)(v112 + 8) = v139 + 24;
          *v133 = *v133 & 7 | (v139 + 24);
        }
        v140 = v110;
        sub_164B780(v110, v149);
        v62 = v140;
        if ( v156 )
        {
          v148 = v156;
          sub_1623A60((__int64)&v148, (__int64)v156, 2);
          v62 = v140;
          v113 = *(_QWORD *)(v140 + 48);
          v114 = v140 + 48;
          if ( v113 )
          {
            sub_161E7C0(v140 + 48, v113);
            v62 = v140;
            v114 = v140 + 48;
          }
          v115 = v148;
          *(_QWORD *)(v62 + 48) = v148;
          if ( v115 )
          {
            v141 = v62;
            sub_1623210((__int64)&v148, v115, v114);
            v62 = v141;
          }
        }
      }
      else
      {
        v62 = sub_15A46C0(47, (__int64 ***)v52, v61, 0);
      }
      sub_164D160(a2, v62, a3, a4, a5, a6, v59, v60, a9, a10);
      v63 = *(_QWORD *)(a2 + 48);
      v151[0] = v63;
      if ( !v63 )
      {
        if ( v56 == v151 )
          goto LABEL_31;
        v99 = *(_QWORD *)(v52 + 48);
        if ( !v99 )
          goto LABEL_31;
LABEL_158:
        sub_161E7C0(v52 + 48, v99);
        goto LABEL_159;
      }
      sub_1623A60((__int64)v151, v63, 2);
      if ( v56 == v151 )
      {
        if ( v151[0] )
          sub_161E7C0(v52 + 48, v151[0]);
        goto LABEL_31;
      }
      v99 = *(_QWORD *)(v52 + 48);
      if ( v99 )
        goto LABEL_158;
LABEL_159:
      v100 = (unsigned __int8 *)v151[0];
      *(_QWORD *)(v52 + 48) = v151[0];
      if ( v100 )
        sub_1623210((__int64)v151, v100, v52 + 48);
LABEL_31:
      v32 = (__int64 *)v153;
      *a1 = 1;
      if ( v32 != v155 )
        _libc_free((unsigned __int64)v32);
      if ( v156 )
        sub_161E7C0((__int64)&v156, (__int64)v156);
      return;
    case 0x37:
      if ( !sub_1642F90(**(_QWORD **)(a2 - 48), 128) && *(_BYTE *)(**(_QWORD **)(a2 - 48) + 8LL) != 5 )
        return;
      v12 = sub_16498A0(a2);
      v13 = *(unsigned __int8 **)(a2 + 48);
      v156 = 0;
      v159 = v12;
      v14 = *(_QWORD *)(a2 + 40);
      v160 = 0;
      v157 = v14;
      v161 = 0;
      v162 = 0;
      v163 = 0;
      v158 = (__int64 *)(a2 + 24);
      v153 = v13;
      if ( v13 )
      {
        sub_1623A60((__int64)&v153, (__int64)v13, 2);
        if ( v156 )
          sub_161E7C0((__int64)&v156, (__int64)v156);
        v156 = v153;
        if ( v153 )
          sub_1623210((__int64)&v153, v153, (__int64)&v156);
      }
      v15 = (_QWORD *)sub_16498A0(a2);
      v16 = (__int64 *)sub_1643360(v15);
      v17 = sub_16463B0(v16, 2u);
      v152 = 257;
      v18 = **(_QWORD **)(a2 - 24);
      if ( *(_BYTE *)(v18 + 8) == 16 )
        v18 = **(_QWORD **)(v18 + 16);
      v19 = (__int64 **)sub_1647190(v17, *(_DWORD *)(v18 + 8) >> 8);
      v20 = *(_QWORD *)(a2 - 24);
      if ( v19 != *(__int64 ***)v20 )
      {
        if ( *(_BYTE *)(v20 + 16) > 0x10u )
        {
          v102 = *(_QWORD *)(a2 - 24);
          LOWORD(v155[0]) = 257;
          v103 = sub_15FDBD0(47, v102, (__int64)v19, (__int64)&v153, 0);
          if ( v157 )
          {
            v104 = v158;
            v146 = v103;
            sub_157E9D0(v157 + 40, v103);
            v103 = v146;
            v105 = *v104;
            v106 = *(_QWORD *)(v146 + 24);
            *(_QWORD *)(v146 + 32) = v104;
            v105 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v146 + 24) = v105 | v106 & 7;
            *(_QWORD *)(v105 + 8) = v146 + 24;
            *v104 = *v104 & 7 | (v146 + 24);
          }
          v137 = v103;
          sub_164B780(v103, v151);
          v20 = v137;
          if ( v156 )
          {
            v149[0] = (__int64)v156;
            sub_1623A60((__int64)v149, (__int64)v156, 2);
            v20 = v137;
            v107 = *(_QWORD *)(v137 + 48);
            v108 = v137 + 48;
            if ( v107 )
            {
              sub_161E7C0(v137 + 48, v107);
              v20 = v137;
            }
            v109 = (unsigned __int8 *)v149[0];
            *(_QWORD *)(v20 + 48) = v149[0];
            if ( v109 )
            {
              v138 = v20;
              sub_1623210((__int64)v149, v109, v108);
              v20 = v138;
            }
          }
        }
        else
        {
          v20 = sub_15A46C0(47, *(__int64 ****)(a2 - 24), v19, 0);
        }
      }
      v21 = *(_QWORD *)(a2 - 48);
      v152 = 257;
      if ( v17 != *(__int64 **)v21 )
      {
        v134 = v20;
        if ( *(_BYTE *)(v21 + 16) > 0x10u )
        {
          LOWORD(v155[0]) = 257;
          v116 = sub_15FDBD0(47, v21, (__int64)v17, (__int64)&v153, 0);
          v117 = v134;
          v21 = v116;
          if ( v157 )
          {
            v118 = v158;
            sub_157E9D0(v157 + 40, v116);
            v119 = *(_QWORD *)(v21 + 24);
            v117 = v134;
            v120 = *v118;
            *(_QWORD *)(v21 + 32) = v118;
            v120 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v21 + 24) = v120 | v119 & 7;
            *(_QWORD *)(v120 + 8) = v21 + 24;
            *v118 = *v118 & 7 | (v21 + 24);
          }
          v142 = v117;
          sub_164B780(v21, v151);
          v20 = v142;
          if ( v156 )
          {
            v149[0] = (__int64)v156;
            sub_1623A60((__int64)v149, (__int64)v156, 2);
            v121 = *(_QWORD *)(v21 + 48);
            v20 = v142;
            if ( v121 )
            {
              sub_161E7C0(v21 + 48, v121);
              v20 = v142;
            }
            v122 = (unsigned __int8 *)v149[0];
            *(_QWORD *)(v21 + 48) = v149[0];
            if ( v122 )
            {
              v143 = v20;
              sub_1623210((__int64)v149, v122, v21 + 48);
              v20 = v143;
            }
          }
        }
        else
        {
          v22 = sub_15A46C0(47, (__int64 ***)v21, (__int64 **)v17, 0);
          v20 = v134;
          v21 = v22;
        }
      }
      v135 = v20;
      LOWORD(v155[0]) = 257;
      v23 = sub_1648A60(64, 2u);
      v24 = v23;
      if ( v23 )
        sub_15F9650((__int64)v23, v21, v135, 0, 0);
      if ( v157 )
      {
        v25 = (unsigned __int64 *)v158;
        sub_157E9D0(v157 + 40, (__int64)v24);
        v26 = v24[3];
        v27 = *v25;
        v24[4] = v25;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        v24[3] = v27 | v26 & 7;
        *(_QWORD *)(v27 + 8) = v24 + 3;
        *v25 = *v25 & 7 | (unsigned __int64)(v24 + 3);
      }
      v28 = v24 + 6;
      sub_164B780((__int64)v24, (__int64 *)&v153);
      if ( v156 )
      {
        v151[0] = (__int64)v156;
        sub_1623A60((__int64)v151, (__int64)v156, 2);
        v29 = v24[6];
        if ( v29 )
          sub_161E7C0((__int64)(v24 + 6), v29);
        v30 = (unsigned __int8 *)v151[0];
        v24[6] = v151[0];
        if ( v30 )
          sub_1623210((__int64)v151, v30, (__int64)(v24 + 6));
      }
      v153 = (unsigned __int8 *)v155;
      v155[0] = a2;
      v154 = 0x200000001LL;
      sub_14C4AD0((__int64)v24, v155, 1);
      sub_15F9450((__int64)v24, 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1);
      v31 = *(_QWORD *)(a2 + 48);
      v151[0] = v31;
      if ( v31 )
      {
        sub_1623A60((__int64)v151, v31, 2);
        if ( v28 == v151 )
        {
          if ( v151[0] )
            sub_161E7C0((__int64)(v24 + 6), v151[0]);
          goto LABEL_30;
        }
        v97 = v24[6];
        if ( !v97 )
        {
LABEL_154:
          v98 = (unsigned __int8 *)v151[0];
          v24[6] = v151[0];
          if ( v98 )
            sub_1623210((__int64)v151, v98, (__int64)(v24 + 6));
LABEL_30:
          sub_15F20C0((_QWORD *)a2);
          goto LABEL_31;
        }
      }
      else
      {
        if ( v28 == v151 )
          goto LABEL_30;
        v97 = v24[6];
        if ( !v97 )
          goto LABEL_30;
      }
      sub_161E7C0((__int64)(v24 + 6), v97);
      goto LABEL_154;
    case 0x3F:
      v64 = *(_QWORD *)a2;
      v65 = *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL);
      if ( v65 != 5 )
      {
        if ( (unsigned int)sub_16431D0(v64) != 128 )
          return;
        v68 = "__nv_cvt_f32_u128_rz";
        if ( v65 != 2 )
        {
          v68 = "__nv_cvt_f64_u128_rz";
          if ( v65 != 3 )
            return;
        }
        goto LABEL_92;
      }
      v92 = sub_1642F90(v64, 8);
      v75 = "__nv_fp128_to_uint8";
      v76 = 19;
      if ( v92 )
        goto LABEL_112;
      v93 = sub_1642F90(*(_QWORD *)a2, 16);
      v75 = "__nv_fp128_to_uint16";
      if ( v93 )
        goto LABEL_111;
      v94 = sub_1642F90(*(_QWORD *)a2, 32);
      v75 = "__nv_fp128_to_uint32";
      if ( v94 )
        goto LABEL_111;
      v95 = sub_1642F90(*(_QWORD *)a2, 64);
      v75 = "__nv_fp128_to_uint64";
      if ( v95 )
        goto LABEL_111;
      v96 = sub_1642F90(*(_QWORD *)a2, 128);
      v75 = "__nv_fp128_to_uint128";
      v76 = 21;
      if ( !v96 )
        return;
      goto LABEL_112;
    case 0x40:
      v69 = *(_QWORD *)a2;
      v70 = *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL);
      if ( v70 != 5 )
      {
        if ( (unsigned int)sub_16431D0(v69) != 128 )
          return;
        v68 = "__nv_cvt_f32_i128_rz";
        if ( v70 != 2 )
        {
          v68 = "__nv_cvt_f64_i128_rz";
          if ( v70 != 3 )
            return;
        }
        goto LABEL_92;
      }
      v89 = sub_1642F90(v69, 8);
      v75 = "__nv_fp128_to_int8";
      v76 = 18;
      if ( v89 )
        goto LABEL_112;
      v90 = sub_1642F90(*(_QWORD *)a2, 16);
      v75 = "__nv_fp128_to_int16";
      v76 = 19;
      if ( v90 )
        goto LABEL_112;
      v91 = sub_1642F90(*(_QWORD *)a2, 32);
      v75 = "__nv_fp128_to_int32";
      v76 = 19;
      if ( v91 )
        goto LABEL_112;
      if ( sub_1642F90(*(_QWORD *)a2, 64) )
      {
        v75 = "__nv_fp128_to_int64";
        v76 = 19;
        goto LABEL_112;
      }
      v132 = sub_1642F90(*(_QWORD *)a2, 128);
      v75 = "__nv_fp128_to_int128";
      if ( v132 )
        goto LABEL_111;
      return;
    case 0x41:
      v71 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      v72 = **(_QWORD **)(a2 - 24);
      if ( v71 != 5 )
      {
        if ( (unsigned int)sub_16431D0(v72) != 128 )
          return;
        v68 = "__nv_cvt_u128_f32_rn";
        if ( v71 != 2 )
        {
          v68 = "__nv_cvt_u128_f64_rn";
          if ( v71 != 3 )
            return;
        }
        goto LABEL_92;
      }
      v84 = sub_1642F90(v72, 8);
      v75 = "__nv_uint8_to_fp128";
      v76 = 19;
      if ( v84 )
        goto LABEL_112;
      v85 = sub_1642F90(**(_QWORD **)(a2 - 24), 16);
      v75 = "__nv_uint16_to_fp128";
      if ( v85 )
        goto LABEL_111;
      v86 = sub_1642F90(**(_QWORD **)(a2 - 24), 32);
      v75 = "__nv_uint32_to_fp128";
      if ( v86 )
        goto LABEL_111;
      v87 = sub_1642F90(**(_QWORD **)(a2 - 24), 64);
      v75 = "__nv_uint64_to_fp128";
      if ( v87 )
        goto LABEL_111;
      v88 = sub_1642F90(**(_QWORD **)(a2 - 24), 128);
      v75 = "__nv_uint128_to_fp128";
      v76 = 21;
      if ( !v88 )
        return;
      goto LABEL_112;
    case 0x42:
      v73 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      v74 = **(_QWORD **)(a2 - 24);
      if ( v73 != 5 )
      {
        if ( (unsigned int)sub_16431D0(v74) != 128 )
          return;
        v68 = "__nv_cvt_i128_f32_rn";
        if ( v73 != 2 )
        {
          if ( v73 != 3 )
            return;
          v68 = "__nv_cvt_i128_f64_rn";
        }
LABEL_92:
        sub_1C8BF90(a1, (__int64 **)a2, (__int64)v68, 20, a3, a4, a5, a6, v66, v67, a9, a10);
        return;
      }
      v79 = sub_1642F90(v74, 8);
      v75 = "__nv_int8_to_fp128";
      v76 = 18;
      if ( v79 )
        goto LABEL_112;
      v80 = sub_1642F90(**(_QWORD **)(a2 - 24), 16);
      v75 = "__nv_int16_to_fp128";
      v76 = 19;
      if ( v80 )
        goto LABEL_112;
      v81 = sub_1642F90(**(_QWORD **)(a2 - 24), 32);
      v75 = "__nv_int32_to_fp128";
      v76 = 19;
      if ( v81 )
        goto LABEL_112;
      v82 = sub_1642F90(**(_QWORD **)(a2 - 24), 64);
      v75 = "__nv_int64_to_fp128";
      v76 = 19;
      if ( v82 )
        goto LABEL_112;
      v83 = sub_1642F90(**(_QWORD **)(a2 - 24), 128);
      v75 = "__nv_int128_to_fp128";
      if ( v83 )
      {
LABEL_111:
        v76 = 20;
LABEL_112:
        sub_1C8ADC0(a1, (_QWORD *)a2, (__int64)v75, v76, a3, a4, a5, a6, a7, a8, a9, a10);
        return;
      }
      return;
    case 0x43:
      if ( *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL) != 5 )
        return;
      v75 = "__nv_fp128_to_float";
      v76 = 19;
      v77 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      if ( v77 == 2 )
        goto LABEL_112;
      v75 = "__nv_fp128_to_double";
      if ( v77 == 3 )
        goto LABEL_111;
      return;
    case 0x44:
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 5 )
        return;
      v75 = "__nv_float_to_fp128";
      v76 = 19;
      v78 = *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL);
      if ( v78 == 2 )
        goto LABEL_112;
      if ( v78 != 3 )
        return;
      v75 = "__nv_double_to_fp128";
      goto LABEL_111;
    case 0x4C:
      v33 = *(unsigned __int16 *)(a2 + 18);
      BYTE1(v33) &= ~0x80u;
      switch ( v33 )
      {
        case 1:
          v34 = "__nv_fcmp_oeq";
          goto LABEL_137;
        case 2:
          v34 = "__nv_fcmp_ogt";
          goto LABEL_137;
        case 3:
          v34 = "__nv_fcmp_oge";
          goto LABEL_137;
        case 4:
          v34 = "__nv_fcmp_olt";
          goto LABEL_137;
        case 5:
          v34 = "__nv_fcmp_ole";
          goto LABEL_137;
        case 6:
          v34 = "__nv_fcmp_one";
          goto LABEL_137;
        case 7:
          v34 = "__nv_fcmp_ord";
          goto LABEL_137;
        case 8:
          v34 = "__nv_fcmp_uno";
          goto LABEL_137;
        case 9:
          v34 = "__nv_fcmp_ueq";
          goto LABEL_137;
        case 10:
          v34 = "__nv_fcmp_ugt";
          goto LABEL_137;
        case 11:
          v34 = "__nv_fcmp_uge";
          goto LABEL_137;
        case 12:
          v34 = "__nv_fcmp_ult";
          goto LABEL_137;
        case 13:
          v34 = "__nv_fcmp_ule";
          goto LABEL_137;
        case 14:
          v34 = "__nv_fcmp_une";
LABEL_137:
          v35 = 13;
          goto LABEL_38;
        default:
          return;
      }
  }
}
