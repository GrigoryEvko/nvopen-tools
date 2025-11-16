// Function: sub_18BC140
// Address: 0x18bc140
//
char **__fastcall sub_18BC140(
        __int64 a1,
        char ***a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  char **result; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // r15
  char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rbx
  _QWORD *v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // rcx
  _BYTE *v20; // rsi
  __int64 v21; // r14
  __int64 **v22; // rax
  _BYTE *v23; // rdx
  __int64 **v24; // r13
  char **v25; // r12
  char *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rax
  char *v30; // rsi
  __int64 **v31; // rdx
  char *v32; // rsi
  __int64 v33; // rax
  _BYTE *v34; // rsi
  __int64 v35; // rbx
  _BYTE *v36; // rax
  unsigned int i; // r12d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  int v41; // r13d
  __int64 v42; // rax
  __int64 v43; // rdx
  _BYTE *v44; // rsi
  char *v45; // rax
  char v46; // al
  int v47; // ebx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r13
  int v51; // r13d
  __int64 v52; // rax
  __int64 v53; // rdx
  int v54; // eax
  __int64 *v55; // r14
  __int64 v56; // rax
  __int64 v57; // r10
  __int64 v58; // rbx
  _QWORD *v59; // rax
  __int64 v60; // r13
  __int64 v61; // r14
  __int64 *v62; // rbx
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  unsigned __int64 v67; // r13
  int v68; // eax
  char *v69; // rax
  __int64 ***v70; // rax
  __int64 v71; // rax
  char *v72; // rsi
  int j; // ebx
  char *v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 v77; // rax
  double v78; // xmm4_8
  double v79; // xmm5_8
  char *v80; // rax
  int v81; // r14d
  __int64 ****v82; // rax
  __int64 ***v83; // r13
  __int64 v84; // rbx
  __int64 **v85; // rax
  __int64 ***v86; // rax
  char *v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r9
  __int64 *v90; // r12
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  unsigned __int64 *v95; // r12
  __int64 **v96; // rax
  unsigned __int64 v97; // rcx
  __int64 v98; // rsi
  __int64 v99; // rdx
  unsigned __int8 *v100; // rsi
  __int64 v101; // rsi
  __int64 v102; // r10
  __int64 *v103; // rbx
  __int64 v104; // rcx
  __int64 v105; // rax
  __int64 v106; // rsi
  __int64 v107; // r13
  unsigned __int8 *v108; // rsi
  __int64 v109; // [rsp+8h] [rbp-1B8h]
  __int64 v110; // [rsp+10h] [rbp-1B0h]
  __int64 v111; // [rsp+18h] [rbp-1A8h]
  __int64 v112; // [rsp+20h] [rbp-1A0h]
  __int64 *v113; // [rsp+30h] [rbp-190h]
  char *v114; // [rsp+50h] [rbp-170h]
  __int64 **v115; // [rsp+58h] [rbp-168h]
  __int64 v116; // [rsp+58h] [rbp-168h]
  __int64 v117; // [rsp+58h] [rbp-168h]
  __int64 v118; // [rsp+58h] [rbp-168h]
  __int64 v119; // [rsp+58h] [rbp-168h]
  char **v120; // [rsp+68h] [rbp-158h]
  char **v121; // [rsp+88h] [rbp-138h]
  __int64 v122; // [rsp+90h] [rbp-130h] BYREF
  char *v123; // [rsp+98h] [rbp-128h] BYREF
  _QWORD *v124; // [rsp+A0h] [rbp-120h] BYREF
  _BYTE *v125; // [rsp+A8h] [rbp-118h]
  _QWORD *v126; // [rsp+B0h] [rbp-110h]
  __int64 *v127; // [rsp+C0h] [rbp-100h] BYREF
  _BYTE *v128; // [rsp+C8h] [rbp-F8h]
  _BYTE *v129; // [rsp+D0h] [rbp-F0h]
  __int64 v130[2]; // [rsp+E0h] [rbp-E0h] BYREF
  __int16 v131; // [rsp+F0h] [rbp-D0h]
  __int64 v132[2]; // [rsp+100h] [rbp-C0h] BYREF
  __int16 v133; // [rsp+110h] [rbp-B0h]
  char *v134; // [rsp+120h] [rbp-A0h] BYREF
  char *v135; // [rsp+128h] [rbp-98h]
  char *v136; // [rsp+130h] [rbp-90h]
  char *v137; // [rsp+140h] [rbp-80h] BYREF
  __int64 v138; // [rsp+148h] [rbp-78h]
  __int64 *v139; // [rsp+150h] [rbp-70h]
  __int64 v140; // [rsp+158h] [rbp-68h]
  __int64 v141; // [rsp+160h] [rbp-60h]
  int v142; // [rsp+168h] [rbp-58h]
  __int64 v143; // [rsp+170h] [rbp-50h]
  __int64 v144; // [rsp+178h] [rbp-48h]

  result = a2[1];
  v120 = result;
  v121 = *a2;
  if ( *a2 == result )
    return result;
  do
  {
    v11 = (__int64)v121[1];
    v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    v122 = sub_1560340(
             (_QWORD *)(*(_QWORD *)(*(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL) + 112LL),
             -1,
             "target-features",
             0xFu);
    if ( sub_155D460(&v122, 0) )
      goto LABEL_84;
    v13 = (char *)sub_155D8B0(&v122);
    v138 = v14;
    v137 = v13;
    if ( sub_16D20C0((__int64 *)&v137, "+retpoline", 0xAu, 0) == -1 )
      goto LABEL_84;
    v15 = *(_QWORD *)(a1 + 8);
    if ( *(_BYTE *)(v15 + 80) )
    {
      v87 = (char *)sub_1649960(**(_QWORD **)(a1 + 16));
      sub_18B6C20(
        (__int64)v121,
        "branch-funnel",
        13,
        v87,
        v88,
        v89,
        *(__int64 (__fastcall **)(__int64, __int64))(v15 + 88),
        *(_QWORD *)(v15 + 96));
      v15 = *(_QWORD *)(a1 + 8);
    }
    v16 = *(_QWORD *)(v15 + 48);
    v124 = 0;
    v125 = 0;
    v126 = 0;
    v17 = (_QWORD *)sub_22077B0(8);
    v18 = v17;
    if ( v17 )
      *v17 = v16;
    v19 = *(_QWORD *)(v12 + 64);
    v20 = v17 + 1;
    v124 = v17;
    v126 = v17 + 1;
    v21 = (v11 >> 2) & 1;
    v22 = *(__int64 ***)(v19 + 16);
    v125 = v18 + 1;
    if ( &v22[*(unsigned int *)(v19 + 12)] != v22 + 1 )
    {
      v23 = v18 + 1;
      v24 = &v22[*(unsigned int *)(v19 + 12)];
      v25 = (char **)(v22 + 1);
      while ( 1 )
      {
        v26 = *v25;
        v137 = *v25;
        if ( v20 == v23 )
        {
          ++v25;
          sub_1277EB0((__int64)&v124, v20, &v137);
          if ( v24 == (__int64 **)v25 )
            goto LABEL_16;
        }
        else
        {
          if ( v20 )
          {
            *(_QWORD *)v20 = v26;
            v20 = v125;
          }
          ++v25;
          v125 = v20 + 8;
          if ( v24 == (__int64 **)v25 )
          {
LABEL_16:
            v19 = *(_QWORD *)(v12 + 64);
            v18 = v124;
            v22 = *(__int64 ***)(v19 + 16);
            v27 = (v125 - (_BYTE *)v124) >> 3;
            goto LABEL_17;
          }
        }
        v20 = v125;
        v23 = v126;
      }
    }
    v27 = 1;
LABEL_17:
    v28 = (__int64 *)sub_1644EA0(*v22, v18, v27, *(_DWORD *)(v19 + 8) >> 8 != 0);
    v115 = (__int64 **)sub_1646BA0(v28, 0);
    v29 = sub_16498A0(v12);
    v137 = 0;
    v140 = v29;
    v141 = 0;
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v138 = *(_QWORD *)(v12 + 40);
    v139 = (__int64 *)(v12 + 24);
    v30 = *(char **)(v12 + 48);
    v134 = v30;
    if ( v30 )
    {
      sub_1623A60((__int64)&v134, (__int64)v30, 2);
      if ( v137 )
        sub_161E7C0((__int64)&v137, (__int64)v137);
      v137 = v134;
      if ( v134 )
        sub_1623210((__int64)&v134, (unsigned __int8 *)v134, (__int64)&v137);
    }
    v127 = 0;
    v133 = 257;
    v128 = 0;
    v129 = 0;
    v31 = *(__int64 ***)(*(_QWORD *)(a1 + 8) + 48LL);
    v32 = *v121;
    if ( v31 == *(__int64 ***)*v121 )
    {
      v134 = *v121;
      v34 = 0;
    }
    else
    {
      if ( (unsigned __int8)v32[16] > 0x10u )
      {
        LOWORD(v136) = 257;
        v35 = sub_15FDBD0(47, (__int64)v32, (__int64)v31, (__int64)&v134, 0);
        if ( v138 )
        {
          v90 = v139;
          sub_157E9D0(v138 + 40, v35);
          v91 = *(_QWORD *)(v35 + 24);
          v92 = *v90;
          *(_QWORD *)(v35 + 32) = v90;
          v92 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v35 + 24) = v92 | v91 & 7;
          *(_QWORD *)(v92 + 8) = v35 + 24;
          *v90 = *v90 & 7 | (v35 + 24);
        }
        sub_164B780(v35, v132);
        if ( v137 )
        {
          v130[0] = (__int64)v137;
          sub_1623A60((__int64)v130, (__int64)v137, 2);
          v93 = *(_QWORD *)(v35 + 48);
          if ( v93 )
            sub_161E7C0(v35 + 48, v93);
          v94 = (unsigned __int8 *)v130[0];
          *(_QWORD *)(v35 + 48) = v130[0];
          if ( v94 )
            sub_1623210((__int64)v130, v94, v35 + 48);
        }
        v34 = v128;
        v36 = v129;
      }
      else
      {
        v33 = sub_15A46C0(47, (__int64 ***)v32, v31, 0);
        v34 = v128;
        v35 = v33;
        v36 = v129;
      }
      v134 = (char *)v35;
      if ( v36 != v34 )
      {
        if ( v34 )
        {
          *(_QWORD *)v34 = v35;
          v34 = v128;
        }
        v128 = v34 + 8;
        goto LABEL_29;
      }
    }
    sub_12879C0((__int64)&v127, v34, &v134);
LABEL_29:
    for ( i = 0; ; ++i )
    {
      v46 = *(_BYTE *)(v12 + 23);
      v47 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
      if ( !(_BYTE)v21 )
      {
        if ( v46 < 0 )
        {
          v48 = sub_1648A40(v12);
          v50 = v48 + v49;
          if ( *(char *)(v12 + 23) >= 0 )
          {
            if ( (unsigned int)(v50 >> 4) )
LABEL_127:
              BUG();
          }
          else if ( (unsigned int)((v50 - sub_1648A40(v12)) >> 4) )
          {
            if ( *(char *)(v12 + 23) >= 0 )
              goto LABEL_127;
            v51 = *(_DWORD *)(sub_1648A40(v12) + 8);
            if ( *(char *)(v12 + 23) >= 0 )
              BUG();
            v52 = sub_1648A40(v12);
            v54 = *(_DWORD *)(v52 + v53 - 4) - v51;
LABEL_48:
            if ( i == v47 - 3 - v54 )
            {
              v55 = v127;
              v133 = 257;
              v112 = (v128 - (_BYTE *)v127) >> 3;
              v111 = *(_QWORD *)(v12 - 24);
              v56 = *(_QWORD *)(v12 - 48);
              v131 = 257;
              v110 = v56;
              v57 = **(_QWORD **)(a1 + 16);
              if ( v115 != *(__int64 ***)v57 )
              {
                if ( *(_BYTE *)(v57 + 16) > 0x10u )
                {
                  v101 = **(_QWORD **)(a1 + 16);
                  LOWORD(v136) = 257;
                  v102 = sub_15FDBD0(47, v101, (__int64)v115, (__int64)&v134, 0);
                  if ( v138 )
                  {
                    v103 = v139;
                    v117 = v102;
                    sub_157E9D0(v138 + 40, v102);
                    v102 = v117;
                    v104 = *v103;
                    v105 = *(_QWORD *)(v117 + 24);
                    *(_QWORD *)(v117 + 32) = v103;
                    v104 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v117 + 24) = v104 | v105 & 7;
                    *(_QWORD *)(v104 + 8) = v117 + 24;
                    *v103 = *v103 & 7 | (v117 + 24);
                  }
                  v118 = v102;
                  sub_164B780(v102, v130);
                  v57 = v118;
                  if ( v137 )
                  {
                    v123 = v137;
                    sub_1623A60((__int64)&v123, (__int64)v137, 2);
                    v57 = v118;
                    v106 = *(_QWORD *)(v118 + 48);
                    v107 = v118 + 48;
                    if ( v106 )
                    {
                      sub_161E7C0(v118 + 48, v106);
                      v57 = v118;
                    }
                    v108 = (unsigned __int8 *)v123;
                    *(_QWORD *)(v57 + 48) = v123;
                    if ( v108 )
                    {
                      v119 = v57;
                      sub_1623210((__int64)&v123, v108, v107);
                      v57 = v119;
                    }
                  }
                }
                else
                {
                  v57 = sub_15A46C0(47, **(__int64 *****)(a1 + 16), v115, 0);
                }
              }
              v109 = v57;
              LOWORD(v136) = 257;
              v58 = *(_QWORD *)(*(_QWORD *)v57 + 24LL);
              v59 = sub_1648AB0(72, (int)v112 + 3, 0);
              v60 = (__int64)v59;
              if ( v59 )
              {
                v113 = v55;
                v61 = (__int64)v59;
                sub_15F1EA0(
                  (__int64)v59,
                  **(_QWORD **)(v58 + 16),
                  5,
                  (__int64)&v59[-3 * (unsigned int)(v112 + 3)],
                  v112 + 3,
                  0);
                *(_QWORD *)(v60 + 56) = 0;
                sub_15F6500(v60, v58, v109, v110, v111, (__int64)&v134, v113, v112, 0, 0);
              }
              else
              {
                v61 = 0;
              }
              if ( v138 )
              {
                v62 = v139;
                sub_157E9D0(v138 + 40, v60);
                v63 = *(_QWORD *)(v60 + 24);
                v64 = *v62;
                *(_QWORD *)(v60 + 32) = v62;
                v64 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v60 + 24) = v64 | v63 & 7;
                *(_QWORD *)(v64 + 8) = v60 + 24;
                *v62 = *v62 & 7 | (v60 + 24);
              }
              sub_164B780(v61, v132);
              if ( v137 )
              {
                v123 = v137;
                sub_1623A60((__int64)&v123, (__int64)v137, 2);
                v65 = *(_QWORD *)(v60 + 48);
                if ( v65 )
                  sub_161E7C0(v60 + 48, v65);
                v66 = (unsigned __int8 *)v123;
                *(_QWORD *)(v60 + 48) = v123;
                if ( v66 )
                  sub_1623210((__int64)&v123, v66, v60 + 48);
              }
              v67 = v60 & 0xFFFFFFFFFFFFFFFBLL;
              v68 = (*(unsigned __int16 *)(v12 + 18) >> 2) & 0x3FFFDFFF;
              goto LABEL_62;
            }
            goto LABEL_36;
          }
        }
        v54 = 0;
        goto LABEL_48;
      }
      if ( v46 < 0 )
        break;
LABEL_89:
      if ( i == v47 - 1 )
        goto LABEL_90;
LABEL_36:
      v44 = v128;
      v45 = *(char **)(v12 + 24 * (i - (unsigned __int64)(*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
      v134 = v45;
      if ( v128 == v129 )
      {
        sub_12879C0((__int64)&v127, v128, &v134);
      }
      else
      {
        if ( v128 )
        {
          *(_QWORD *)v128 = v45;
          v44 = v128;
        }
        v128 = v44 + 8;
      }
    }
    v38 = sub_1648A40(v12);
    v40 = v38 + v39;
    if ( *(char *)(v12 + 23) >= 0 )
    {
      if ( (unsigned int)(v40 >> 4) )
LABEL_129:
        BUG();
      goto LABEL_89;
    }
    if ( !(unsigned int)((v40 - sub_1648A40(v12)) >> 4) )
      goto LABEL_89;
    if ( *(char *)(v12 + 23) >= 0 )
      goto LABEL_129;
    v41 = *(_DWORD *)(sub_1648A40(v12) + 8);
    if ( *(char *)(v12 + 23) >= 0 )
      BUG();
    v42 = sub_1648A40(v12);
    if ( i != v47 - 1 - (*(_DWORD *)(v42 + v43 - 4) - v41) )
      goto LABEL_36;
LABEL_90:
    v81 = (int)v127;
    v133 = 257;
    v82 = *(__int64 *****)(a1 + 16);
    v131 = 257;
    v83 = *v82;
    v84 = (v128 - (_BYTE *)v127) >> 3;
    v85 = **v82;
    if ( v115 != v85 )
    {
      if ( *((_BYTE *)v83 + 16) > 0x10u )
      {
        LOWORD(v136) = 257;
        v83 = (__int64 ***)sub_15FDBD0(47, (__int64)v83, (__int64)v115, (__int64)&v134, 0);
        if ( v138 )
        {
          v95 = (unsigned __int64 *)v139;
          sub_157E9D0(v138 + 40, (__int64)v83);
          v96 = v83[3];
          v97 = *v95;
          v83[4] = (__int64 **)v95;
          v97 &= 0xFFFFFFFFFFFFFFF8LL;
          v83[3] = (__int64 **)(v97 | (unsigned __int8)v96 & 7);
          *(_QWORD *)(v97 + 8) = v83 + 3;
          *v95 = *v95 & 7 | (unsigned __int64)(v83 + 3);
        }
        sub_164B780((__int64)v83, v130);
        if ( v137 )
        {
          v123 = v137;
          sub_1623A60((__int64)&v123, (__int64)v137, 2);
          v98 = (__int64)v83[6];
          v99 = (__int64)(v83 + 6);
          if ( v98 )
          {
            sub_161E7C0((__int64)(v83 + 6), v98);
            v99 = (__int64)(v83 + 6);
          }
          v100 = (unsigned __int8 *)v123;
          v83[6] = (__int64 **)v123;
          if ( v100 )
            sub_1623210((__int64)&v123, v100, v99);
        }
        v85 = *v83;
      }
      else
      {
        v86 = (__int64 ***)sub_15A46C0(47, v83, v115, 0);
        LODWORD(v83) = (_DWORD)v86;
        v85 = *v86;
      }
    }
    v67 = sub_1285290((__int64 *)&v137, (__int64)v85[3], (int)v83, v81, v84, (__int64)v132, 0) | 4;
    v68 = (*(unsigned __int16 *)(v12 + 18) >> 2) & 0x3FFFDFFF;
LABEL_62:
    *(_WORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 18) = *(_WORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8003 | (4 * v68);
    v69 = *(char **)(v12 + 56);
    v134 = 0;
    v123 = v69;
    v70 = *(__int64 ****)(a1 + 8);
    v135 = 0;
    v136 = 0;
    v130[0] = sub_155CEC0(**v70, 19, 0);
    v71 = sub_155F1F0(***(__int64 ****)(a1 + 8), v130, 1);
    v72 = v135;
    v132[0] = v71;
    if ( v135 == v136 )
    {
      sub_17401C0(&v134, v135, v132);
    }
    else
    {
      if ( v135 )
      {
        *(_QWORD *)v135 = v71;
        v72 = v135;
      }
      v135 = v72 + 8;
    }
    for ( j = 0; (unsigned int)sub_15601D0((__int64)&v123) > j + 2; ++j )
    {
      v75 = sub_1560230(&v123, j);
      v74 = v135;
      v132[0] = v75;
      if ( v135 == v136 )
      {
        sub_17401C0(&v134, v135, v132);
      }
      else
      {
        if ( v135 )
        {
          *(_QWORD *)v135 = v75;
          v74 = v135;
        }
        v135 = v74 + 8;
      }
    }
    v76 = v135 - v134;
    v114 = v134;
    v116 = sub_1560240(&v123);
    v77 = sub_1560250(&v123);
    *(_QWORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 56) = sub_155FDB0(***(__int64 ****)(a1 + 8), v77, v116, v114, v76 >> 3);
    sub_164D160(v12, v67 & 0xFFFFFFFFFFFFFFF8LL, a3, a4, a5, a6, v78, v79, a9, a10);
    sub_15F20C0((_QWORD *)v12);
    v80 = v121[2];
    if ( v80 )
      --*(_DWORD *)v80;
    if ( v134 )
      j_j___libc_free_0(v134, v136 - v134);
    if ( v127 )
      j_j___libc_free_0(v127, v129 - (_BYTE *)v127);
    if ( v137 )
      sub_161E7C0((__int64)&v137, (__int64)v137);
    if ( v124 )
      j_j___libc_free_0(v124, (char *)v126 - (char *)v124);
LABEL_84:
    v121 += 3;
    result = v121;
  }
  while ( v120 != v121 );
  return result;
}
