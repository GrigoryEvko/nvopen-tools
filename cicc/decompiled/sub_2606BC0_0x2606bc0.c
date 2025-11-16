// Function: sub_2606BC0
// Address: 0x2606bc0
//
void __fastcall sub_2606BC0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, unsigned int *a5)
{
  __int64 v5; // r12
  __int64 *v6; // rax
  const char *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  _QWORD *v11; // r12
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rax
  const char *v14; // rsi
  _QWORD *v15; // r15
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 *v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  const char *v24; // r12
  const char *v25; // rbx
  _QWORD *v26; // rdi
  __int64 *v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  _BYTE *v36; // rsi
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // rdx
  __int64 v40; // rcx
  _BYTE *v41; // rsi
  __int64 v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r8
  __int64 v45; // r9
  int v46; // edx
  _QWORD *v47; // rax
  _QWORD *v48; // r13
  _QWORD *v49; // rbx
  _QWORD *v50; // rdi
  char *v51; // rsi
  __int64 v52; // rbx
  unsigned __int64 v53; // r12
  __int64 v54; // rsi
  __int64 v55; // rdi
  __int64 *v56; // rax
  __int64 v57; // r10
  __int64 *v58; // r13
  __int64 *v59; // rbx
  __int64 v60; // r14
  __int64 *v61; // r15
  __int64 v62; // rcx
  __int64 v63; // r13
  __int64 v64; // rsi
  unsigned int v65; // edx
  _QWORD *v66; // rax
  __int64 v67; // rdi
  __int64 v68; // r11
  _QWORD *v69; // rdi
  int v70; // eax
  __int64 *v71; // rdx
  __int64 *v72; // r13
  __int64 v73; // rax
  __int64 *v74; // rbx
  unsigned int v75; // esi
  __int64 v76; // rdi
  unsigned int v77; // ecx
  __int64 *v78; // rdx
  __int64 v79; // r9
  __int64 v80; // r14
  unsigned __int16 v81; // r15
  _QWORD *v82; // rdi
  __int64 v83; // r15
  int v84; // eax
  int v85; // edi
  __int64 v86; // rsi
  __int64 v87; // rdx
  const char *v88; // rax
  __int64 v89; // r10
  int v90; // r9d
  const char *v91; // r8
  unsigned int v92; // esi
  __int64 v93; // rdi
  int v94; // eax
  int v95; // eax
  char *v96; // rax
  int v97; // edx
  int v98; // r11d
  int v99; // r10d
  __int128 v100; // [rsp-30h] [rbp-1A0h]
  __int128 v101; // [rsp-30h] [rbp-1A0h]
  __int64 v103; // [rsp+20h] [rbp-150h]
  __int64 v105; // [rsp+30h] [rbp-140h]
  __int64 *v106; // [rsp+38h] [rbp-138h]
  __int64 v108; // [rsp+50h] [rbp-120h]
  __int64 v109; // [rsp+58h] [rbp-118h]
  unsigned int v110; // [rsp+58h] [rbp-118h]
  __int64 v111; // [rsp+60h] [rbp-110h]
  __int64 v112; // [rsp+68h] [rbp-108h]
  __int64 v113; // [rsp+68h] [rbp-108h]
  _QWORD *v114; // [rsp+70h] [rbp-100h]
  unsigned __int16 v115; // [rsp+70h] [rbp-100h]
  __int64 v116; // [rsp+70h] [rbp-100h]
  __int64 v117; // [rsp+78h] [rbp-F8h]
  __int64 v118; // [rsp+88h] [rbp-E8h]
  __int64 v119; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v120; // [rsp+98h] [rbp-D8h] BYREF
  unsigned __int64 v121; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v122; // [rsp+A8h] [rbp-C8h]
  __int64 v123; // [rsp+B0h] [rbp-C0h]
  __int64 v124; // [rsp+C0h] [rbp-B0h] BYREF
  _QWORD *v125; // [rsp+C8h] [rbp-A8h]
  __int64 v126; // [rsp+D0h] [rbp-A0h]
  unsigned int v127; // [rsp+D8h] [rbp-98h]
  const char *v128; // [rsp+E0h] [rbp-90h] BYREF
  __int64 *v129; // [rsp+E8h] [rbp-88h] BYREF
  __int128 v130; // [rsp+F0h] [rbp-80h]
  __int64 v131; // [rsp+100h] [rbp-70h]
  const char *v132; // [rsp+110h] [rbp-60h] BYREF
  char *v133; // [rsp+118h] [rbp-58h]
  __int128 v134; // [rsp+120h] [rbp-50h]
  __int64 v135; // [rsp+130h] [rbp-40h]

  v5 = (__int64)a3;
  sub_25FBE50(a1, a2, a3, *a5);
  v121 = 0;
  v103 = a1 + 56;
  v6 = *(__int64 **)v5;
  v122 = 0;
  v7 = *(const char **)(v5 + 56);
  v123 = 0;
  v106 = (__int64 *)(v5 + 72);
  v109 = *v6;
  v8 = *(_QWORD *)(*v6 + 248);
  v112 = v8 + 72;
  v117 = *(_QWORD *)(v8 + 80);
  if ( v8 + 72 != v117 )
  {
    v108 = v5;
    do
    {
      v9 = v117;
      v10 = v117 - 24;
      v11 = (_QWORD *)(v117 + 24);
      v12 = (_QWORD *)(v117 - 24);
      v117 = *(_QWORD *)(v117 + 8);
      sub_AA4A70(v12);
      sub_AA4C60(v10, (__int64)v7, 0);
      v13 = *(_QWORD *)(v9 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v11 == (_QWORD *)v13 )
        goto LABEL_149;
      if ( !v13 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_149:
        BUG();
      if ( *(_BYTE *)(v13 - 24) == 30 )
      {
        v14 = 0;
        if ( (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) != 0 )
          v14 = *(const char **)(v13 - 32LL * (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) - 24);
        v128 = v14;
        v129 = (__int64 *)v10;
        sub_2603560((__int64)&v132, (__int64)v106, (__int64 *)&v128, (__int64 *)&v129);
      }
      v132 = 0;
      v133 = 0;
      *(_QWORD *)&v134 = 0;
      v15 = *(_QWORD **)(v9 + 32);
      if ( v15 != v11 )
      {
        while ( 1 )
        {
          if ( !v15 )
          {
            sub_B44570(0);
            BUG();
          }
          sub_B44570((__int64)(v15 - 3));
          if ( *((_BYTE *)v15 - 24) != 85 )
          {
            v128 = 0;
            if ( v15 + 3 != &v128 )
            {
              v16 = v15[3];
              if ( v16 )
              {
                sub_B91220((__int64)(v15 + 3), v16);
                v17 = (unsigned __int8 *)v128;
                v15[3] = v128;
                if ( v17 )
                  sub_B976B0((__int64)&v128, v17, (__int64)(v15 + 3));
              }
            }
            v128 = v7;
            sub_AE8EA0((__int64)(v15 - 3), (__int64 (__fastcall *)(__int64))sub_25F5A20, (__int64)&v128);
            goto LABEL_17;
          }
          v18 = *(v15 - 7);
          if ( v18 )
          {
            if ( !*(_BYTE *)v18
              && *(_QWORD *)(v18 + 24) == v15[7]
              && (*(_BYTE *)(v18 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v18 + 36) - 68) <= 3 )
            {
              break;
            }
          }
          v19 = sub_B92180((__int64)v7);
          if ( !v19 )
            goto LABEL_17;
          v20 = (__int64 *)sub_B2BE50((__int64)v7);
          v21 = sub_B01860(v20, 0, 0, v19, 0, 0, 0, 1);
          sub_B10CB0(&v128, (__int64)v21);
          if ( v15 + 3 == &v128 )
          {
            if ( v128 )
              sub_B91220((__int64)&v128, (__int64)v128);
            goto LABEL_17;
          }
          v22 = v15[3];
          if ( v22 )
            sub_B91220((__int64)(v15 + 3), v22);
          v23 = (unsigned __int8 *)v128;
          v15[3] = v128;
          if ( v23 )
          {
            sub_B976B0((__int64)&v128, v23, (__int64)(v15 + 3));
            v15 = (_QWORD *)v15[1];
            if ( v15 == v11 )
            {
LABEL_29:
              v24 = v132;
              v25 = v133;
              if ( v132 != v133 )
              {
                do
                {
                  v26 = *(_QWORD **)v24;
                  v24 += 8;
                  sub_B43D60(v26);
                }
                while ( v25 != v24 );
                v24 = v132;
              }
              if ( v24 )
                j_j___libc_free_0((unsigned __int64)v24);
              goto LABEL_34;
            }
          }
          else
          {
LABEL_17:
            v15 = (_QWORD *)v15[1];
            if ( v15 == v11 )
              goto LABEL_29;
          }
        }
        v128 = (const char *)(v15 - 3);
        v51 = v133;
        if ( v133 == (char *)v134 )
        {
          sub_249A840((__int64)&v132, v133, &v128);
        }
        else
        {
          if ( v133 )
          {
            *(_QWORD *)v133 = v15 - 3;
            v51 = v133;
          }
          v133 = v51 + 8;
        }
        goto LABEL_17;
      }
LABEL_34:
      ;
    }
    while ( v112 != v117 );
    v5 = v108;
    v8 = *(_QWORD *)(v109 + 248);
  }
  v128 = *(const char **)(v8 + 120);
  v132 = (const char *)sub_A74680(&v128);
  v27 = (__int64 *)sub_A73280((__int64 *)&v132);
  v31 = sub_A73290((__int64 *)&v132);
  while ( (__int64 *)v31 != v27 )
  {
    v32 = *v27++;
    sub_B2CDC0(*(_QWORD *)(v5 + 56), v32);
  }
  v33 = *(_QWORD *)(v5 + 56);
  LOWORD(v135) = 259;
  *((_QWORD *)&v100 + 1) = v133;
  *(_QWORD *)&v100 = "output_block_0";
  v132 = "output_block_0";
  v128 = 0;
  v129 = 0;
  *(_QWORD *)&v130 = 0;
  DWORD2(v130) = 0;
  sub_26028B0((__int64)v106, (__int64)&v128, v33, v28, v29, v30, v100, v134, v135);
  *(_DWORD *)(v109 + 28) = 0;
  sub_2605530((__int64 *)v109, (__int64)&v128, v103, 1);
  sub_25FC850(v109, (__int64 *)&v128, v34, v35);
  if ( !(unsigned __int8)sub_25F8840((__int64)&v128, v109) )
  {
    v132 = 0;
    v133 = 0;
    *(_QWORD *)&v134 = 0;
    DWORD2(v134) = 0;
    sub_25FE4C0(&v121, (__int64)&v132);
    sub_C7D6A0((__int64)v133, 16LL * DWORD2(v134), 8);
    if ( (_DWORD)v130 )
    {
      v71 = v129;
      v72 = &v129[2 * DWORD2(v130)];
      if ( v129 != v72 )
      {
        while ( 1 )
        {
          v73 = *v71;
          v74 = v71;
          if ( *v71 != -4096 && v73 != -8192 )
            break;
          v71 += 2;
          if ( v72 == v71 )
            goto LABEL_39;
        }
        while ( v72 != v74 )
        {
          v75 = *(_DWORD *)(v5 + 96);
          v76 = *(_QWORD *)(v5 + 80);
          if ( v75 )
          {
            v77 = (v75 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
            v78 = (__int64 *)(v76 + 16LL * v77);
            v79 = *v78;
            if ( v73 == *v78 )
              goto LABEL_113;
            v97 = 1;
            while ( v79 != -4096 )
            {
              v99 = v97 + 1;
              v77 = (v75 - 1) & (v97 + v77);
              v78 = (__int64 *)(v76 + 16LL * v77);
              v79 = *v78;
              if ( v73 == *v78 )
                goto LABEL_113;
              v97 = v99;
            }
          }
          v78 = (__int64 *)(v76 + 16LL * v75);
LABEL_113:
          v80 = v78[1];
          sub_B43C20((__int64)&v132, v74[1]);
          v81 = (unsigned __int16)v133;
          v116 = (__int64)v132;
          v82 = sub_BD2C40(72, 1u);
          if ( v82 )
            sub_B4C8F0((__int64)v82, v80, 1u, v116, v81);
          v83 = v122;
          v84 = *(_DWORD *)(v122 - 8);
          if ( v84 )
          {
            v85 = v84 - 1;
            v86 = *(_QWORD *)(v122 - 24);
            LODWORD(v87) = (v84 - 1) & (((unsigned int)*v74 >> 9) ^ ((unsigned int)*v74 >> 4));
            v88 = (const char *)(v86 + 16LL * (unsigned int)v87);
            v89 = *(_QWORD *)v88;
            if ( *v74 == *(_QWORD *)v88 )
              goto LABEL_117;
            v98 = 1;
            v91 = 0;
            while ( v89 != -4096 )
            {
              if ( !v91 && v89 == -8192 )
                v91 = v88;
              v87 = v85 & (unsigned int)(v87 + v98);
              v88 = (const char *)(v86 + 16 * v87);
              v89 = *(_QWORD *)v88;
              if ( *v74 == *(_QWORD *)v88 )
                goto LABEL_117;
              ++v98;
            }
            if ( !v91 )
              v91 = v88;
          }
          else
          {
            v91 = 0;
          }
          v132 = v91;
          v92 = *(_DWORD *)(v122 - 8);
          v93 = v122 - 32;
          v94 = *(_DWORD *)(v122 - 16);
          ++*(_QWORD *)(v122 - 32);
          v95 = v94 + 1;
          if ( 4 * v95 >= 3 * v92 )
          {
            v92 *= 2;
LABEL_140:
            sub_26026D0(v93, v92);
            sub_25FD8E0(v83 - 32, v74, &v132);
            v95 = *(_DWORD *)(v83 - 16) + 1;
            goto LABEL_127;
          }
          if ( v92 - *(_DWORD *)(v83 - 12) - v95 <= v92 >> 3 )
            goto LABEL_140;
LABEL_127:
          *(_DWORD *)(v83 - 16) = v95;
          v96 = (char *)v132;
          if ( *(_QWORD *)v132 != -4096 )
            --*(_DWORD *)(v83 - 12);
          *(_QWORD *)v96 = *v74;
          *((_QWORD *)v96 + 1) = v74[1];
LABEL_117:
          v74 += 2;
          if ( v74 == v72 )
            break;
          while ( 1 )
          {
            v73 = *v74;
            if ( *v74 != -4096 && v73 != -8192 )
              break;
            v74 += 2;
            if ( v72 == v74 )
              goto LABEL_39;
          }
        }
      }
    }
  }
LABEL_39:
  *(_QWORD *)(v109 + 240) = sub_25FDAE0((_QWORD **)a2, v109);
  v36 = *(_BYTE **)(a4 + 8);
  if ( v36 == *(_BYTE **)(a4 + 16) )
  {
    sub_9CC5C0(a4, v36, (_QWORD *)(v109 + 248));
  }
  else
  {
    if ( v36 )
    {
      *(_QWORD *)v36 = *(_QWORD *)(v109 + 248);
      v36 = *(_BYTE **)(a4 + 8);
    }
    *(_QWORD *)(a4 + 8) = v36 + 8;
  }
  sub_C7D6A0((__int64)v129, 16LL * DWORD2(v130), 8);
  v39 = *(__int64 **)v5;
  if ( *(_QWORD *)(v5 + 8) - *(_QWORD *)v5 > 8u )
  {
    v40 = 1;
    v110 = 1;
    do
    {
      v42 = v39[v40];
      sub_A755B0(*(_QWORD *)(v5 + 56), *(_QWORD *)(v42 + 248));
      v43 = *(_QWORD *)(v5 + 56);
      v128 = "output_block_";
      LODWORD(v130) = v110;
      LOWORD(v131) = 2307;
      *((_QWORD *)&v101 + 1) = v129;
      *(_QWORD *)&v101 = "output_block_";
      v124 = 0;
      v125 = 0;
      v126 = 0;
      v127 = 0;
      sub_26028B0((__int64)v106, (__int64)&v124, v43, 2307, v44, v45, v101, v130, v131);
      sub_2605530((__int64 *)v42, (__int64)&v124, v103, 0);
      if ( (unsigned __int8)sub_25F8840((__int64)&v124, v42) )
        goto LABEL_52;
      v118 = sub_25FC950((__int64)&v124, (__int64 *)&v121);
      if ( BYTE4(v118) )
      {
        v46 = v126;
        *(_DWORD *)(v42 + 28) = v118;
        if ( v46 )
        {
          v47 = v125;
          v48 = &v125[2 * v127];
          if ( v125 != v48 )
          {
            while ( 1 )
            {
              v49 = v47;
              if ( *v47 != -4096 && *v47 != -8192 )
                break;
              v47 += 2;
              if ( v48 == v47 )
                goto LABEL_52;
            }
            if ( v48 != v47 )
            {
              do
              {
                v50 = (_QWORD *)v49[1];
                v49 += 2;
                sub_AA5450(v50);
                if ( v49 == v48 )
                  break;
                while ( *v49 == -8192 || *v49 == -4096 )
                {
                  v49 += 2;
                  if ( v48 == v49 )
                    goto LABEL_52;
                }
              }
              while ( v49 != v48 );
            }
          }
        }
      }
      else
      {
        *(_DWORD *)(v42 + 28) = (__int64)(v122 - v121) >> 5;
        v132 = 0;
        v133 = 0;
        *(_QWORD *)&v134 = 0;
        DWORD2(v134) = 0;
        sub_25FE4C0(&v121, (__int64)&v132);
        sub_C7D6A0((__int64)v133, 16LL * DWORD2(v134), 8);
        if ( (_DWORD)v126 )
        {
          v56 = v125;
          v57 = 2LL * v127;
          v58 = &v125[v57];
          if ( v125 != &v125[v57] )
          {
            while ( 1 )
            {
              v59 = v56;
              if ( *v56 != -8192 && *v56 != -4096 )
                break;
              v56 += 2;
              if ( v58 == v56 )
                goto LABEL_52;
            }
            if ( v58 != v56 )
            {
              v105 = v42;
              v60 = *v56;
              v61 = &v125[v57];
              while ( 1 )
              {
                v62 = *(unsigned int *)(v5 + 96);
                v63 = v59[1];
                v64 = *(_QWORD *)(v5 + 80);
                if ( !(_DWORD)v62 )
                  goto LABEL_103;
                v65 = (v62 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
                v66 = (_QWORD *)(v64 + 16LL * v65);
                v67 = *v66;
                if ( *v66 != v60 )
                  break;
LABEL_92:
                v114 = v66;
                sub_B43C20((__int64)&v132, v59[1]);
                v68 = v114[1];
                v113 = (__int64)v132;
                v115 = (unsigned __int16)v133;
                v111 = v68;
                v69 = sub_BD2C40(72, 1u);
                if ( v69 )
                  sub_B4C8F0((__int64)v69, v111, 1u, v113, v115);
                v59 += 2;
                v119 = v60;
                v120 = v63;
                sub_2603560((__int64)&v132, v122 - 32, &v119, &v120);
                if ( v59 == v61 )
                  goto LABEL_98;
                while ( *v59 == -8192 || *v59 == -4096 )
                {
                  v59 += 2;
                  if ( v61 == v59 )
                    goto LABEL_98;
                }
                if ( v61 == v59 )
                {
LABEL_98:
                  v42 = v105;
                  goto LABEL_52;
                }
                v60 = *v59;
              }
              v70 = 1;
              while ( v67 != -4096 )
              {
                v90 = v70 + 1;
                v65 = (v62 - 1) & (v70 + v65);
                v66 = (_QWORD *)(v64 + 16LL * v65);
                v67 = *v66;
                if ( v60 == *v66 )
                  goto LABEL_92;
                v70 = v90;
              }
LABEL_103:
              v66 = (_QWORD *)(v64 + 16 * v62);
              goto LABEL_92;
            }
          }
        }
      }
LABEL_52:
      *(_QWORD *)(v42 + 240) = sub_25FDAE0((_QWORD **)a2, v42);
      v41 = *(_BYTE **)(a4 + 8);
      if ( v41 == *(_BYTE **)(a4 + 16) )
      {
        sub_9CC5C0(a4, v41, (_QWORD *)(v42 + 248));
      }
      else
      {
        if ( v41 )
        {
          *(_QWORD *)v41 = *(_QWORD *)(v42 + 248);
          v41 = *(_BYTE **)(a4 + 8);
        }
        *(_QWORD *)(a4 + 8) = v41 + 8;
      }
      sub_C7D6A0((__int64)v125, 16LL * v127, 8);
      v39 = *(__int64 **)v5;
      v40 = ++v110;
    }
    while ( v110 < (unsigned __int64)((__int64)(*(_QWORD *)(v5 + 8) - *(_QWORD *)v5) >> 3) );
  }
  sub_2602DE0((_QWORD **)a2, v5, v106, (__int64 *)&v121, v37, v38);
  v52 = v122;
  v53 = v121;
  ++*a5;
  if ( v52 != v53 )
  {
    do
    {
      v54 = *(unsigned int *)(v53 + 24);
      v55 = *(_QWORD *)(v53 + 8);
      v53 += 32LL;
      sub_C7D6A0(v55, 16 * v54, 8);
    }
    while ( v52 != v53 );
    v53 = v121;
  }
  if ( v53 )
    j_j___libc_free_0(v53);
}
