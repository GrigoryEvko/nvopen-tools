// Function: sub_1B81290
// Address: 0x1b81290
//
unsigned __int64 *__fastcall sub_1B81290(
        __int64 a1,
        __int64 *a2,
        __int64 **a3,
        __int64 a4,
        unsigned __int64 **a5,
        int a6)
{
  unsigned __int64 *result; // rax
  unsigned __int64 *v7; // rcx
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 *v13; // rbx
  _BYTE *v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // rax
  unsigned int v19; // r8d
  _QWORD *v20; // r12
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // rax
  int v24; // r8d
  __int64 *v25; // r10
  __int64 *v26; // rcx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rdi
  unsigned __int64 *v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  char v37; // al
  char *v38; // rdx
  unsigned int *v39; // r15
  __int64 v40; // rbx
  __int64 *v41; // r14
  _QWORD *v42; // r13
  _QWORD *v43; // rax
  _QWORD *v44; // r14
  __int64 v45; // rdi
  unsigned __int64 *v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  unsigned __int64 *v52; // r13
  unsigned __int64 *v53; // r12
  unsigned __int64 *v54; // rbx
  unsigned __int64 v55; // rdi
  __int64 *v56; // rax
  __int64 **v57; // rdi
  __int64 v58; // r12
  unsigned __int64 *v59; // r14
  __int64 v60; // r13
  __int64 *v61; // rbx
  char *v62; // rdx
  char v63; // al
  __int64 *v64; // r15
  __int64 v65; // r12
  _QWORD *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  unsigned __int64 *v75; // rbx
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rsi
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  __int64 v81; // rax
  __int64 *v82; // rax
  __int64 **v83; // rdx
  unsigned __int64 v84; // rcx
  _QWORD *v85; // rax
  _QWORD *v86; // r15
  __int64 v87; // rax
  unsigned __int64 *v88; // r12
  __int64 v89; // rax
  unsigned __int64 v90; // rcx
  __int64 v91; // rsi
  __int64 v92; // rsi
  unsigned __int8 *v93; // rsi
  _QWORD *v94; // rax
  unsigned int *v95; // rsi
  __int64 v96; // rax
  __int64 v97; // rdx
  unsigned __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 *v101; // r15
  __int64 v102; // rax
  __int64 v103; // rcx
  __int64 v104; // rsi
  __int64 v105; // rsi
  unsigned __int8 *v106; // rsi
  _QWORD *v107; // rdi
  __int64 v108; // rax
  __int64 v109; // r12
  unsigned __int64 *v110; // rbx
  char v111; // al
  __int64 v112; // r13
  __int64 v113; // rax
  unsigned int v114; // r14d
  __int64 *v115; // rax
  __int64 v116; // rax
  __int64 v117; // rax
  unsigned __int64 v118; // rbx
  __int64 v119; // r14
  _QWORD *v120; // rax
  _QWORD *v121; // r13
  __int64 v122; // rax
  unsigned __int64 *v123; // r12
  __int64 v124; // rax
  unsigned __int64 v125; // rcx
  __int64 v126; // rsi
  __int64 v127; // rsi
  unsigned __int8 *v128; // rsi
  __int64 *v129; // rax
  unsigned __int64 *v131; // [rsp+10h] [rbp-100h]
  int v134; // [rsp+2Ch] [rbp-E4h]
  int v135; // [rsp+30h] [rbp-E0h]
  unsigned int *v136; // [rsp+30h] [rbp-E0h]
  __int64 v137; // [rsp+38h] [rbp-D8h]
  unsigned int v138; // [rsp+40h] [rbp-D0h]
  __int64 v139; // [rsp+40h] [rbp-D0h]
  __int64 v140; // [rsp+40h] [rbp-D0h]
  __int64 v141; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v142; // [rsp+48h] [rbp-C8h]
  unsigned __int64 *v143; // [rsp+48h] [rbp-C8h]
  __int64 *v145; // [rsp+60h] [rbp-B0h]
  __int64 v146; // [rsp+60h] [rbp-B0h]
  unsigned int *v147; // [rsp+60h] [rbp-B0h]
  __int64 v148; // [rsp+60h] [rbp-B0h]
  unsigned __int64 *v149; // [rsp+68h] [rbp-A8h]
  unsigned int v150; // [rsp+68h] [rbp-A8h]
  __int64 v151; // [rsp+68h] [rbp-A8h]
  unsigned __int8 *v152; // [rsp+78h] [rbp-98h] BYREF
  __int64 v153[2]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v154; // [rsp+90h] [rbp-80h]
  char *v155; // [rsp+A0h] [rbp-70h] BYREF
  char *v156; // [rsp+A8h] [rbp-68h]
  __int16 v157; // [rsp+B0h] [rbp-60h]
  __int64 v158[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v159; // [rsp+D0h] [rbp-40h]

  result = *a5;
  v7 = a5[1];
  v131 = v7;
  if ( v7 != *a5 )
  {
    v9 = 0x2E8BA2E8BA2E8BA3LL * (v7 - result);
    if ( a6 == 1 )
    {
      if ( v9 != *(_QWORD *)(a1 + 168) )
        goto LABEL_4;
      v56 = (__int64 *)sub_1643340((_QWORD *)a2[3]);
      v57 = (__int64 **)sub_16463B0(v56, v9);
    }
    else
    {
      if ( v9 != *(_QWORD *)(a1 + 160) )
      {
LABEL_4:
        v145 = a2;
        v149 = *a5;
        while ( 1 )
        {
          v142 = v149[10];
          v10 = *(_BYTE *)(a4 + 16);
          if ( v10 )
          {
            if ( v10 == 1 )
            {
              v153[0] = (__int64)".gep";
              v154 = 259;
            }
            else
            {
              if ( *(_BYTE *)(a4 + 17) == 1 )
              {
                v11 = *(_QWORD *)a4;
              }
              else
              {
                v11 = a4;
                v10 = 2;
              }
              v153[0] = v11;
              v153[1] = (__int64)".gep";
              LOBYTE(v154) = v10;
              HIBYTE(v154) = 3;
            }
          }
          else
          {
            v154 = 256;
          }
          v12 = *((unsigned int *)v149 + 10);
          v13 = (__int64 *)v149[4];
          v14 = *(_BYTE **)(a1 + 88);
          v15 = *(_QWORD *)(a1 + 96);
          if ( v14[16] > 0x10u )
            goto LABEL_15;
          if ( *((_DWORD *)v149 + 10) )
            break;
LABEL_98:
          v83 = (__int64 **)v149[4];
          v84 = *((unsigned int *)v149 + 10);
          BYTE4(v158[0]) = 0;
          v20 = (_QWORD *)sub_15A2E80(v15, (__int64)v14, v83, v84, 1u, (__int64)v158, 0);
LABEL_33:
          v37 = *(_BYTE *)(a4 + 16);
          if ( v37 )
          {
            if ( v37 == 1 )
            {
              v155 = ".extract";
              v157 = 259;
            }
            else
            {
              if ( *(_BYTE *)(a4 + 17) == 1 )
              {
                v38 = *(char **)a4;
              }
              else
              {
                v38 = (char *)a4;
                v37 = 2;
              }
              v155 = v38;
              v156 = ".extract";
              LOBYTE(v157) = v37;
              HIBYTE(v157) = 3;
            }
          }
          else
          {
            v157 = 256;
          }
          v39 = (unsigned int *)*v149;
          v40 = *((unsigned int *)v149 + 2);
          v41 = *a3;
          if ( *((_BYTE *)*a3 + 16) > 0x10u )
          {
            v159 = 257;
            v69 = sub_1648A60(88, 1u);
            v42 = v69;
            if ( v69 )
            {
              v140 = (__int64)v69;
              v70 = sub_15FB2A0(*v41, v39, v40);
              sub_15F1EA0((__int64)v42, v70, 62, (__int64)(v42 - 3), 1, 0);
              if ( *(v42 - 3) )
              {
                v71 = *(v42 - 2);
                v72 = *(v42 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v72 = v71;
                if ( v71 )
                  *(_QWORD *)(v71 + 16) = *(_QWORD *)(v71 + 16) & 3LL | v72;
              }
              *(v42 - 3) = v41;
              v73 = v41[1];
              *(v42 - 2) = v73;
              if ( v73 )
                *(_QWORD *)(v73 + 16) = (unsigned __int64)(v42 - 2) | *(_QWORD *)(v73 + 16) & 3LL;
              *(v42 - 1) = (unsigned __int64)(v41 + 1) | *(v42 - 1) & 3LL;
              v41[1] = (__int64)(v42 - 3);
              v42[7] = v42 + 9;
              v42[8] = 0x400000000LL;
              sub_15FB110((__int64)v42, v39, v40, (__int64)v158);
            }
            else
            {
              v140 = 0;
            }
            v74 = v145[1];
            if ( v74 )
            {
              v75 = (unsigned __int64 *)v145[2];
              sub_157E9D0(v74 + 40, (__int64)v42);
              v76 = v42[3];
              v77 = *v75;
              v42[4] = v75;
              v77 &= 0xFFFFFFFFFFFFFFF8LL;
              v42[3] = v77 | v76 & 7;
              *(_QWORD *)(v77 + 8) = v42 + 3;
              *v75 = *v75 & 7 | (unsigned __int64)(v42 + 3);
            }
            sub_164B780(v140, (__int64 *)&v155);
            v78 = *v145;
            if ( *v145 )
            {
              v152 = (unsigned __int8 *)*v145;
              sub_1623A60((__int64)&v152, v78, 2);
              v79 = v42[6];
              if ( v79 )
                sub_161E7C0((__int64)(v42 + 6), v79);
              v80 = v152;
              v42[6] = v152;
              if ( v80 )
                sub_1623210((__int64)&v152, v80, (__int64)(v42 + 6));
            }
          }
          else
          {
            v42 = (_QWORD *)sub_15A3AE0(*a3, (unsigned int *)*v149, *((unsigned int *)v149 + 2), 0);
          }
          v159 = 257;
          v43 = sub_1648A60(64, 2u);
          v44 = v43;
          if ( v43 )
            sub_15F9650((__int64)v43, (__int64)v42, (__int64)v20, 0, 0);
          v45 = v145[1];
          if ( v45 )
          {
            v46 = (unsigned __int64 *)v145[2];
            sub_157E9D0(v45 + 40, (__int64)v44);
            v47 = v44[3];
            v48 = *v46;
            v44[4] = v46;
            v48 &= 0xFFFFFFFFFFFFFFF8LL;
            v44[3] = v48 | v47 & 7;
            *(_QWORD *)(v48 + 8) = v44 + 3;
            *v46 = *v46 & 7 | (unsigned __int64)(v44 + 3);
          }
          sub_164B780((__int64)v44, v158);
          v49 = *v145;
          if ( *v145 )
          {
            v152 = (unsigned __int8 *)*v145;
            sub_1623A60((__int64)&v152, v49, 2);
            v50 = v44[6];
            if ( v50 )
              sub_161E7C0((__int64)(v44 + 6), v50);
            v51 = v152;
            v44[6] = v152;
            if ( v51 )
              sub_1623210((__int64)&v152, v51, (__int64)(v44 + 6));
          }
          sub_15F9450((__int64)v44, v142);
          v149 += 11;
          if ( v131 == v149 )
            goto LABEL_50;
        }
        v16 = 0;
        while ( *(_BYTE *)(v13[v16] + 16) <= 0x10u )
        {
          if ( v12 == ++v16 )
            goto LABEL_98;
        }
LABEL_15:
        v17 = *((_DWORD *)v149 + 10) + 1;
        v159 = 257;
        if ( !v15 )
        {
          v81 = *(_QWORD *)v14;
          if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
            v81 = **(_QWORD **)(v81 + 16);
          v15 = *(_QWORD *)(v81 + 24);
        }
        v138 = v17;
        v18 = sub_1648A60(72, v17);
        v19 = v138;
        v20 = v18;
        if ( v18 )
        {
          v139 = (__int64)v18;
          v21 = *(_QWORD *)v14;
          v137 = (__int64)&v20[-3 * v19];
          if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
            v21 = **(_QWORD **)(v21 + 16);
          v134 = v19;
          v135 = *(_DWORD *)(v21 + 8) >> 8;
          v22 = (__int64 *)sub_15F9F50(v15, (__int64)v13, v12);
          v23 = (__int64 *)sub_1646BA0(v22, v135);
          v24 = v134;
          v25 = v23;
          if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
          {
            v82 = sub_16463B0(v23, *(_QWORD *)(*(_QWORD *)v14 + 32LL));
            v24 = v134;
            v25 = v82;
          }
          else
          {
            v26 = &v13[v12];
            if ( v13 != v26 )
            {
              v27 = v13;
              while ( 1 )
              {
                v28 = *(_QWORD *)*v27;
                if ( *(_BYTE *)(v28 + 8) == 16 )
                  break;
                if ( v26 == ++v27 )
                  goto LABEL_25;
              }
              v29 = sub_16463B0(v25, *(_QWORD *)(v28 + 32));
              v24 = v134;
              v25 = v29;
            }
          }
LABEL_25:
          sub_15F1EA0((__int64)v20, (__int64)v25, 32, v137, v24, 0);
          v20[7] = v15;
          v20[8] = sub_15F9F50(v15, (__int64)v13, v12);
          sub_15F9CE0((__int64)v20, (__int64)v14, v13, v12, (__int64)v158);
        }
        else
        {
          v139 = 0;
        }
        sub_15FA2E0((__int64)v20, 1);
        v30 = v145[1];
        if ( v30 )
        {
          v31 = (unsigned __int64 *)v145[2];
          sub_157E9D0(v30 + 40, (__int64)v20);
          v32 = v20[3];
          v33 = *v31;
          v20[4] = v31;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          v20[3] = v33 | v32 & 7;
          *(_QWORD *)(v33 + 8) = v20 + 3;
          *v31 = *v31 & 7 | (unsigned __int64)(v20 + 3);
        }
        sub_164B780(v139, v153);
        v34 = *v145;
        if ( *v145 )
        {
          v155 = (char *)*v145;
          sub_1623A60((__int64)&v155, v34, 2);
          v35 = v20[6];
          if ( v35 )
            sub_161E7C0((__int64)(v20 + 6), v35);
          v36 = (unsigned __int8 *)v155;
          v20[6] = v155;
          if ( v36 )
            sub_1623210((__int64)&v155, v36, (__int64)(v20 + 6));
        }
        goto LABEL_33;
      }
      v151 = 0x2E8BA2E8BA2E8BA3LL * (v7 - result);
      v129 = (__int64 *)sub_1643330((_QWORD *)a2[3]);
      v57 = (__int64 **)sub_16463B0(v129, v151);
    }
    v58 = sub_1599EF0(v57);
    v143 = a5[1];
    if ( *a5 != v143 )
    {
      v59 = *a5;
      v60 = v58;
      v61 = a2;
      v150 = 0;
      do
      {
        v63 = *(_BYTE *)(a4 + 16);
        if ( v63 )
        {
          if ( v63 == 1 )
          {
            v155 = ".extract";
            v157 = 259;
          }
          else
          {
            if ( *(_BYTE *)(a4 + 17) == 1 )
            {
              v62 = *(char **)a4;
            }
            else
            {
              v62 = (char *)a4;
              v63 = 2;
            }
            v155 = v62;
            v156 = ".extract";
            LOBYTE(v157) = v63;
            HIBYTE(v157) = 3;
          }
        }
        else
        {
          v157 = 256;
        }
        v64 = *a3;
        if ( *((_BYTE *)*a3 + 16) > 0x10u )
        {
          v141 = *((unsigned int *)v59 + 2);
          v147 = (unsigned int *)*v59;
          v159 = 257;
          v94 = sub_1648A60(88, 1u);
          v65 = (__int64)v94;
          if ( v94 )
          {
            v95 = v147;
            v136 = v147;
            v148 = (__int64)v94;
            v96 = sub_15FB2A0(*v64, v95, v141);
            sub_15F1EA0(v65, v96, 62, v65 - 24, 1, 0);
            if ( *(_QWORD *)(v65 - 24) )
            {
              v97 = *(_QWORD *)(v65 - 16);
              v98 = *(_QWORD *)(v65 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v98 = v97;
              if ( v97 )
                *(_QWORD *)(v97 + 16) = *(_QWORD *)(v97 + 16) & 3LL | v98;
            }
            *(_QWORD *)(v65 - 24) = v64;
            v99 = v64[1];
            *(_QWORD *)(v65 - 16) = v99;
            if ( v99 )
              *(_QWORD *)(v99 + 16) = (v65 - 16) | *(_QWORD *)(v99 + 16) & 3LL;
            *(_QWORD *)(v65 - 8) = (unsigned __int64)(v64 + 1) | *(_QWORD *)(v65 - 8) & 3LL;
            v64[1] = v65 - 24;
            *(_QWORD *)(v65 + 56) = v65 + 72;
            *(_QWORD *)(v65 + 64) = 0x400000000LL;
            sub_15FB110(v65, v136, v141, (__int64)v158);
          }
          else
          {
            v148 = 0;
          }
          v100 = v61[1];
          if ( v100 )
          {
            v101 = (__int64 *)v61[2];
            sub_157E9D0(v100 + 40, v65);
            v102 = *(_QWORD *)(v65 + 24);
            v103 = *v101;
            *(_QWORD *)(v65 + 32) = v101;
            v103 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v65 + 24) = v103 | v102 & 7;
            *(_QWORD *)(v103 + 8) = v65 + 24;
            *v101 = *v101 & 7 | (v65 + 24);
          }
          sub_164B780(v148, (__int64 *)&v155);
          v104 = *v61;
          if ( *v61 )
          {
            v153[0] = *v61;
            sub_1623A60((__int64)v153, v104, 2);
            v105 = *(_QWORD *)(v65 + 48);
            if ( v105 )
              sub_161E7C0(v65 + 48, v105);
            v106 = (unsigned __int8 *)v153[0];
            *(_QWORD *)(v65 + 48) = v153[0];
            if ( v106 )
              sub_1623210((__int64)v153, v106, v65 + 48);
          }
        }
        else
        {
          v65 = sub_15A3AE0(*a3, (unsigned int *)*v59, *((unsigned int *)v59 + 2), 0);
        }
        v66 = (_QWORD *)v61[3];
        v157 = 257;
        v67 = sub_1643350(v66);
        v68 = sub_159C470(v67, v150, 0);
        if ( *(_BYTE *)(v60 + 16) > 0x10u || *(_BYTE *)(v65 + 16) > 0x10u || *(_BYTE *)(v68 + 16) > 0x10u )
        {
          v146 = v68;
          v159 = 257;
          v85 = sub_1648A60(56, 3u);
          v86 = v85;
          if ( v85 )
            sub_15FA480((__int64)v85, (__int64 *)v60, v65, v146, (__int64)v158, 0);
          v87 = v61[1];
          if ( v87 )
          {
            v88 = (unsigned __int64 *)v61[2];
            sub_157E9D0(v87 + 40, (__int64)v86);
            v89 = v86[3];
            v90 = *v88;
            v86[4] = v88;
            v90 &= 0xFFFFFFFFFFFFFFF8LL;
            v86[3] = v90 | v89 & 7;
            *(_QWORD *)(v90 + 8) = v86 + 3;
            *v88 = *v88 & 7 | (unsigned __int64)(v86 + 3);
          }
          sub_164B780((__int64)v86, (__int64 *)&v155);
          v91 = *v61;
          if ( *v61 )
          {
            v153[0] = *v61;
            sub_1623A60((__int64)v153, v91, 2);
            v92 = v86[6];
            if ( v92 )
              sub_161E7C0((__int64)(v86 + 6), v92);
            v93 = (unsigned __int8 *)v153[0];
            v86[6] = v153[0];
            if ( v93 )
              sub_1623210((__int64)v153, v93, (__int64)(v86 + 6));
          }
          v60 = (__int64)v86;
        }
        else
        {
          v60 = sub_15A3890((__int64 *)v60, v65, v68, 0);
        }
        ++v150;
        v59 += 11;
      }
      while ( v143 != v59 );
      v58 = v60;
      a2 = v61;
    }
    v107 = (_QWORD *)a2[3];
    v159 = 257;
    v108 = sub_1643350(v107);
    v109 = sub_17FE280(a2, v58, v108, v158);
    v110 = *a5;
    v111 = *(_BYTE *)(a4 + 16);
    if ( v111 )
    {
      if ( v111 == 1 )
      {
        v158[0] = (__int64)".gep";
        v159 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a4 + 17) == 1 )
          a4 = *(_QWORD *)a4;
        else
          v111 = 2;
        LOBYTE(v159) = v111;
        HIBYTE(v159) = 3;
        v158[0] = a4;
        v158[1] = (__int64)".gep";
      }
    }
    else
    {
      v159 = 256;
    }
    v112 = sub_128B460(
             a2,
             *(_QWORD *)(a1 + 96),
             *(_BYTE **)(a1 + 88),
             (__int64 **)v110[4],
             *((unsigned int *)v110 + 10),
             (__int64)v158);
    v113 = **(_QWORD **)(a1 + 88);
    if ( *(_BYTE *)(v113 + 8) == 16 )
      v113 = **(_QWORD **)(v113 + 16);
    v114 = *(_DWORD *)(v113 + 8);
    v115 = (__int64 *)sub_1643350((_QWORD *)a2[3]);
    v116 = sub_1646BA0(v115, v114 >> 8);
    v159 = 257;
    v117 = sub_17FE280(a2, v112, v116, v158);
    v118 = v110[10];
    v119 = v117;
    v159 = 257;
    v120 = sub_1648A60(64, 2u);
    v121 = v120;
    if ( v120 )
      sub_15F9650((__int64)v120, v109, v119, 0, 0);
    v122 = a2[1];
    if ( v122 )
    {
      v123 = (unsigned __int64 *)a2[2];
      sub_157E9D0(v122 + 40, (__int64)v121);
      v124 = v121[3];
      v125 = *v123;
      v121[4] = v123;
      v125 &= 0xFFFFFFFFFFFFFFF8LL;
      v121[3] = v125 | v124 & 7;
      *(_QWORD *)(v125 + 8) = v121 + 3;
      *v123 = *v123 & 7 | (unsigned __int64)(v121 + 3);
    }
    sub_164B780((__int64)v121, v158);
    v126 = *a2;
    if ( *a2 )
    {
      v155 = (char *)*a2;
      sub_1623A60((__int64)&v155, v126, 2);
      v127 = v121[6];
      if ( v127 )
        sub_161E7C0((__int64)(v121 + 6), v127);
      v128 = (unsigned __int8 *)v155;
      v121[6] = v155;
      if ( v128 )
        sub_1623210((__int64)&v155, v128, (__int64)(v121 + 6));
    }
    sub_15F9450((__int64)v121, v118);
LABEL_50:
    if ( *(_BYTE *)(a1 + 184) )
      *(_BYTE *)(a1 + 184) = 0;
    result = (unsigned __int64 *)a5;
    v52 = *a5;
    v53 = a5[1];
    if ( *a5 != v53 )
    {
      v54 = *a5;
      do
      {
        v55 = v54[4];
        if ( (unsigned __int64 *)v55 != v54 + 6 )
          _libc_free(v55);
        if ( (unsigned __int64 *)*v54 != v54 + 2 )
          _libc_free(*v54);
        v54 += 11;
      }
      while ( v53 != v54 );
      result = (unsigned __int64 *)a5;
      a5[1] = v52;
    }
  }
  return result;
}
