// Function: sub_1F66030
// Address: 0x1f66030
//
void __fastcall sub_1F66030(
        __int64 a1,
        __int64 a2,
        char a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  __int64 ****v16; // rbx
  __int64 ****v17; // r13
  __int64 ***v18; // r12
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r12
  _QWORD *v23; // r13
  _QWORD *v24; // rbx
  _QWORD *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // r12
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  int v34; // r8d
  int v35; // r9d
  unsigned int v36; // r14d
  __int64 v37; // rax
  __int64 **v38; // rax
  __int64 v39; // r12
  __int64 *v40; // rdx
  __int64 v41; // r15
  _QWORD *v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rsi
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  _QWORD *v48; // r12
  unsigned __int64 v49; // r15
  _QWORD *v50; // r14
  __int64 v51; // rcx
  __int64 v52; // r12
  __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  _QWORD *v59; // r14
  __int64 v60; // rax
  __int64 v61; // rcx
  unsigned __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned __int64 v65; // r15
  _QWORD *v66; // r14
  int v67; // r9d
  _QWORD *v68; // r12
  unsigned __int64 *v69; // r8
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rdx
  unsigned __int64 v73; // rsi
  int v74; // edi
  __int64 v75; // rcx
  __int64 v76; // rdx
  const char *v77; // rax
  __int64 v78; // rcx
  _QWORD *v79; // rdx
  __int64 v80; // rcx
  unsigned __int64 v81; // rax
  __int64 v82; // rax
  _QWORD *v83; // r12
  unsigned int v84; // r14d
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // r15
  __int64 v88; // r14
  __int64 v89; // r15
  __int64 v90; // rax
  __int64 v91; // rsi
  unsigned __int64 v92; // rcx
  __int64 v93; // rax
  unsigned __int64 v94; // rdi
  unsigned __int64 v95; // r14
  unsigned __int64 **v96; // rax
  __int64 v97; // rax
  __int64 *v98; // rbx
  __int64 *v99; // r12
  _BYTE *v100; // rsi
  __int64 v101; // rax
  _QWORD *v102; // r15
  unsigned int v103; // r14d
  const char *v104; // rax
  const char *v105; // rdx
  __int64 v106; // rax
  __int64 v107; // r9
  _QWORD *v108; // rax
  const char *v109; // rax
  const char *v110; // rdx
  __int64 v111; // r12
  _QWORD *v112; // rax
  double v113; // xmm4_8
  double v114; // xmm5_8
  __int64 v115; // r13
  int v116; // r10d
  char *v117; // r14
  int v118; // eax
  __int64 v119; // rdx
  _QWORD *v120; // rax
  _QWORD *v121; // r12
  int v122; // edx
  _QWORD *v123; // rdi
  int v124; // esi
  char *v125; // rcx
  char *v126; // rdx
  int v127; // r15d
  int v128; // ecx
  _QWORD *v129; // rsi
  _QWORD *v130; // rax
  int v131; // r9d
  unsigned __int64 v132; // r15
  __int64 v133; // rcx
  __int64 v134; // rdx
  unsigned __int64 v135; // rcx
  __int64 v136; // [rsp+10h] [rbp-1B0h]
  _QWORD *v138; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v139; // [rsp+20h] [rbp-1A0h]
  __int64 v140; // [rsp+28h] [rbp-198h]
  _QWORD *v141; // [rsp+30h] [rbp-190h]
  unsigned __int64 v142; // [rsp+30h] [rbp-190h]
  __int64 v145; // [rsp+48h] [rbp-178h]
  __int64 v146; // [rsp+50h] [rbp-170h]
  __int64 v147; // [rsp+58h] [rbp-168h]
  _QWORD *v148; // [rsp+60h] [rbp-160h]
  __int64 **v149; // [rsp+60h] [rbp-160h]
  __int64 v150; // [rsp+60h] [rbp-160h]
  __int64 v151; // [rsp+68h] [rbp-158h]
  _QWORD *v152; // [rsp+78h] [rbp-148h] BYREF
  _QWORD *v153; // [rsp+80h] [rbp-140h] BYREF
  __int64 v154; // [rsp+88h] [rbp-138h]
  __int64 *v155; // [rsp+90h] [rbp-130h] BYREF
  const char *v156; // [rsp+98h] [rbp-128h]
  __int16 v157; // [rsp+A0h] [rbp-120h]
  __int64 **v158; // [rsp+B0h] [rbp-110h] BYREF
  const char *v159; // [rsp+B8h] [rbp-108h]
  __int64 v160; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v161; // [rsp+C8h] [rbp-F8h]
  __int64 ****v162; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v163; // [rsp+108h] [rbp-B8h]
  _BYTE v164[176]; // [rsp+110h] [rbp-B0h] BYREF

  v162 = (__int64 ****)v164;
  v163 = 0x1000000000LL;
  v151 = *(_QWORD *)(a2 + 80);
  v147 = a2 + 72;
  if ( v151 == a2 + 72 )
    return;
  do
  {
    v11 = v151;
    v12 = v151 - 24;
    v13 = v151 - 24;
    v151 = *(_QWORD *)(v151 + 8);
    v14 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v13) + 16) - 34;
    if ( (unsigned int)v14 > 0x36 )
      continue;
    v15 = 0x40018000000001LL;
    if ( !_bittest64(&v15, v14) || a3 && *(_BYTE *)(sub_157ED20(v12) + 16) != 34 )
      continue;
    v140 = v11 + 16;
    v145 = *(_QWORD *)(v11 + 24);
    if ( v11 + 16 == v145 )
      continue;
LABEL_14:
    v136 = *(_QWORD *)(v145 + 8);
    v146 = v145 - 24;
    if ( *(_BYTE *)(v145 - 8) != 77 )
      continue;
    v22 = *(_QWORD *)(v145 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(sub_157ED20(v22) + 16) - 25 <= 9 )
    {
      v158 = 0;
      v159 = 0;
      v160 = 0;
      LODWORD(v161) = 0;
      if ( !*(_QWORD *)(v145 - 16) )
      {
        j___libc_free_0(0);
        goto LABEL_38;
      }
      v148 = 0;
      v23 = *(_QWORD **)(v145 - 16);
      while ( 1 )
      {
        v24 = v23;
        v23 = (_QWORD *)v23[1];
        v25 = sub_1648700((__int64)v24);
        v26 = (__int64)v25;
        if ( *((_BYTE *)v25 + 16) != 77 )
          break;
        v44 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v25[5]) + 16) - 34;
        if ( (unsigned int)v44 <= 0x36 )
        {
          v45 = 0x40018000000001LL;
          if ( _bittest64(&v45, v44) )
            goto LABEL_27;
        }
        if ( !v148 )
          goto LABEL_90;
LABEL_45:
        v46 = sub_1648700((__int64)v24);
        v26 = (__int64)v46;
        if ( *((_BYTE *)v46 + 16) != 77 )
        {
LABEL_20:
          v153 = sub_1649960(v146);
          v157 = 773;
          v155 = (__int64 *)&v153;
          v154 = v27;
          v156 = ".wineh.reload";
          v28 = sub_1648A60(64, 1u);
          v29 = v28;
          if ( v28 )
          {
            sub_15F90C0((__int64)v28, *(_QWORD *)(*v148 + 24LL), (__int64)v148, (__int64)&v155, 0, v26);
            if ( *v24 )
            {
              v30 = v24[1];
              v31 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v31 = v30;
              if ( v30 )
                *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
            }
            *v24 = v29;
            v32 = v29[1];
            v24[1] = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = (unsigned __int64)(v24 + 1) | *(_QWORD *)(v32 + 16) & 3LL;
            v24[2] = (unsigned __int64)(v29 + 1) | v24[2] & 3LL;
            v29[1] = v24;
          }
          else if ( *v24 )
          {
            v134 = v24[1];
            v135 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v135 = v134;
            if ( v134 )
              *(_QWORD *)(v134 + 16) = v135 | *(_QWORD *)(v134 + 16) & 3LL;
            *v24 = 0;
          }
          goto LABEL_27;
        }
        if ( (*((_BYTE *)v46 + 23) & 0x40) != 0 )
          v47 = (_QWORD *)*(v46 - 1);
        else
          v47 = &v46[-3 * (*((_DWORD *)v46 + 5) & 0xFFFFFFF)];
        v48 = (_QWORD *)v47[3 * *((unsigned int *)v46 + 14) + 1 + -1431655765 * (unsigned int)(v24 - v47)];
        v49 = sub_157EBA0((__int64)v48);
        if ( *(_BYTE *)(v49 + 16) == 33 )
        {
          v152 = *(_QWORD **)(v26 + 40);
          v153 = (_QWORD *)sub_1AA91E0(v48, v152, 0, 0);
          v50 = (_QWORD *)sub_157EBA0((__int64)v48);
          sub_15F2070(v50);
          sub_15F2070((_QWORD *)v49);
          sub_157E9D0((__int64)(v48 + 5), v49);
          v51 = v48[5];
          *(_QWORD *)(v49 + 32) = v48 + 5;
          v51 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v49 + 24) = v51 | *(_QWORD *)(v49 + 24) & 7LL;
          *(_QWORD *)(v51 + 8) = v49 + 24;
          v48[5] = v48[5] & 7LL | (v49 + 24);
          v52 = (__int64)v153;
          v141 = v153 + 5;
          sub_157E9D0((__int64)(v153 + 5), (__int64)v50);
          v53 = *(_QWORD *)(v52 + 40);
          v50[4] = v141;
          v53 &= 0xFFFFFFFFFFFFFFF8LL;
          v50[3] = v53 | v50[3] & 7LL;
          *(_QWORD *)(v53 + 8) = v50 + 3;
          v54 = v152;
          *(_QWORD *)(v52 + 40) = *(_QWORD *)(v52 + 40) & 7LL | (unsigned __int64)(v50 + 3);
          if ( *(v50 - 3) )
          {
            v55 = *(v50 - 2);
            v56 = *(v50 - 1) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v56 = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
          }
          *(v50 - 3) = v54;
          if ( v54 )
          {
            v57 = v54[1];
            *(v50 - 2) = v57;
            if ( v57 )
              *(_QWORD *)(v57 + 16) = (unsigned __int64)(v50 - 2) | *(_QWORD *)(v57 + 16) & 3LL;
            v58 = *(v50 - 1);
            v59 = v50 - 3;
            v59[2] = (unsigned __int64)(v54 + 1) | v58 & 3;
            v54[1] = v59;
          }
          v60 = (__int64)v153;
          if ( *(_QWORD *)(v49 - 24) )
          {
            v61 = *(_QWORD *)(v49 - 16);
            v62 = *(_QWORD *)(v49 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v62 = v61;
            if ( v61 )
              *(_QWORD *)(v61 + 16) = *(_QWORD *)(v61 + 16) & 3LL | v62;
          }
          *(_QWORD *)(v49 - 24) = v60;
          if ( v60 )
          {
            v63 = *(_QWORD *)(v60 + 8);
            *(_QWORD *)(v49 - 16) = v63;
            if ( v63 )
              *(_QWORD *)(v63 + 16) = (v49 - 16) | *(_QWORD *)(v63 + 16) & 3LL;
            v64 = *(_QWORD *)(v49 - 8);
            v65 = v49 - 24;
            *(_QWORD *)(v65 + 16) = (v60 + 8) | v64 & 3;
            *(_QWORD *)(v60 + 8) = v65;
          }
          v66 = sub_1F61110(a1 + 168, (__int64 *)&v153);
          v68 = sub_1F61110(a1 + 168, (__int64 *)&v152);
          v69 = v68 + 1;
          v70 = v68[1];
          if ( v68 + 1 != v66 + 1 )
          {
            v71 = v66[1];
            v72 = (v71 >> 2) & 1;
            v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v70 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_106;
            if ( (v70 & 4) != 0 )
            {
              v74 = *(_DWORD *)(v73 + 8);
              if ( v74 )
              {
                if ( (_BYTE)v72 )
                {
                  sub_1F5FFE0(v71 & 0xFFFFFFFFFFFFFFF8LL, v73, v72, v71 & 0xFFFFFFFFFFFFFFF8LL, (int)v69, v67);
                  v69 = v68 + 1;
                  v73 = v68[1] & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (v68[1] & 4) == 0 )
                    goto LABEL_69;
LABEL_119:
                  v69 = *(unsigned __int64 **)v73;
                  v75 = *(_QWORD *)v73 + 8LL * *(unsigned int *)(v73 + 8);
LABEL_120:
                  if ( (unsigned __int64 *)v75 != v69 )
                  {
                    v138 = v24;
                    v98 = (__int64 *)v69;
                    v99 = (__int64 *)v75;
                    do
                    {
                      v155 = (__int64 *)*v98;
                      v101 = sub_1F65D50(a1 + 200, (__int64 *)&v155);
                      v100 = *(_BYTE **)(v101 + 8);
                      if ( v100 == *(_BYTE **)(v101 + 16) )
                      {
                        sub_1292090(v101, v100, &v153);
                      }
                      else
                      {
                        if ( v100 )
                        {
                          *(_QWORD *)v100 = v153;
                          v100 = *(_BYTE **)(v101 + 8);
                        }
                        *(_QWORD *)(v101 + 8) = v100 + 8;
                      }
                      ++v98;
                    }
                    while ( v99 != v98 );
                    v24 = v138;
                  }
                  goto LABEL_78;
                }
                if ( v74 == 1 )
                {
                  v73 = **(_QWORD **)v73;
LABEL_154:
                  v66[1] = v73;
                  v70 = v68[1];
                  v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
                }
                else
                {
                  v139 = v70 & 0xFFFFFFFFFFFFFFF8LL;
                  v130 = (_QWORD *)sub_22077B0(48);
                  v69 = v68 + 1;
                  v132 = (unsigned __int64)v130;
                  if ( v130 )
                  {
                    *v130 = v130 + 2;
                    v130[1] = 0x400000000LL;
                    v133 = *(unsigned int *)(v139 + 8);
                    if ( (_DWORD)v133 )
                    {
                      sub_1F5FFE0((__int64)v130, v139, (__int64)(v130 + 2), v133, (_DWORD)v68 + 8, v131);
                      v69 = v68 + 1;
                    }
                  }
                  v66[1] = v132 | 4;
                  v70 = v68[1];
                  v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
                }
LABEL_118:
                if ( (v70 & 4) != 0 )
                  goto LABEL_119;
LABEL_69:
                if ( v73 )
                {
                  v75 = (__int64)(v68 + 2);
                  goto LABEL_120;
                }
LABEL_78:
                v48 = v153;
                goto LABEL_79;
              }
LABEL_106:
              if ( (_BYTE)v72 )
              {
                v92 = v71 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v92 )
                {
                  *(_DWORD *)(v92 + 8) = 0;
                  v70 = v68[1];
                  v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
                }
              }
              else
              {
                v66[1] = 0;
                v70 = v68[1];
                v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
              }
              goto LABEL_118;
            }
            if ( !(_BYTE)v72 )
              goto LABEL_154;
            *(_DWORD *)((v71 & 0xFFFFFFFFFFFFFFF8LL) + 8) = 0;
            v93 = v68[1];
            v94 = v66[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v93 & 4) != 0 )
            {
              v96 = (unsigned __int64 **)(v93 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_113:
              v95 = **v96;
            }
            else
            {
              v95 = v93 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v93 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              {
                v96 = 0;
                goto LABEL_113;
              }
            }
            v97 = *(unsigned int *)(v94 + 8);
            if ( (unsigned int)v97 >= *(_DWORD *)(v94 + 12) )
            {
              sub_16CD150(v94, (const void *)(v94 + 16), 0, 8, (int)v69, v67);
              v69 = v68 + 1;
              v97 = *(unsigned int *)(v94 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v94 + 8 * v97) = v95;
            ++*(_DWORD *)(v94 + 8);
            v70 = v68[1];
          }
          v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_118;
        }
LABEL_79:
        if ( !(_DWORD)v161 )
        {
          v158 = (__int64 **)((char *)v158 + 1);
          goto LABEL_156;
        }
        LODWORD(v76) = (v161 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v77 = &v159[16 * (unsigned int)v76];
        v78 = *(_QWORD *)v77;
        if ( *(_QWORD **)v77 != v48 )
        {
          v116 = 1;
          v117 = 0;
          while ( v78 != -8 )
          {
            if ( !v117 && v78 == -16 )
              v117 = (char *)v77;
            v76 = ((_DWORD)v161 - 1) & (unsigned int)(v76 + v116);
            v77 = &v159[16 * v76];
            v78 = *(_QWORD *)v77;
            if ( *(_QWORD **)v77 == v48 )
              goto LABEL_81;
            ++v116;
          }
          if ( !v117 )
            v117 = (char *)v77;
          v158 = (__int64 **)((char *)v158 + 1);
          v118 = v160 + 1;
          if ( 4 * ((int)v160 + 1) < (unsigned int)(3 * v161) )
          {
            if ( (int)v161 - HIDWORD(v160) - v118 <= (unsigned int)v161 >> 3 )
            {
              sub_141A900((__int64)&v158, v161);
              if ( !(_DWORD)v161 )
              {
LABEL_196:
                LODWORD(v160) = v160 + 1;
                BUG();
              }
              v126 = 0;
              v127 = (v161 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
              v128 = 1;
              v118 = v160 + 1;
              v117 = (char *)&v159[16 * v127];
              v129 = *(_QWORD **)v117;
              if ( *(_QWORD **)v117 != v48 )
              {
                while ( v129 != (_QWORD *)-8LL )
                {
                  if ( v129 == (_QWORD *)-16LL && !v126 )
                    v126 = v117;
                  v127 = (v161 - 1) & (v128 + v127);
                  v117 = (char *)&v159[16 * v127];
                  v129 = *(_QWORD **)v117;
                  if ( *(_QWORD **)v117 == v48 )
                    goto LABEL_145;
                  ++v128;
                }
                if ( v126 )
                  v117 = v126;
              }
            }
            goto LABEL_145;
          }
LABEL_156:
          sub_141A900((__int64)&v158, 2 * v161);
          if ( !(_DWORD)v161 )
            goto LABEL_196;
          v122 = (v161 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v118 = v160 + 1;
          v117 = (char *)&v159[16 * v122];
          v123 = *(_QWORD **)v117;
          if ( *(_QWORD **)v117 != v48 )
          {
            v124 = 1;
            v125 = 0;
            while ( v123 != (_QWORD *)-8LL )
            {
              if ( v123 == (_QWORD *)-16LL && !v125 )
                v125 = v117;
              v122 = (v161 - 1) & (v124 + v122);
              v117 = (char *)&v159[16 * v122];
              v123 = *(_QWORD **)v117;
              if ( *(_QWORD **)v117 == v48 )
                goto LABEL_145;
              ++v124;
            }
            if ( v125 )
              v117 = v125;
          }
LABEL_145:
          LODWORD(v160) = v118;
          if ( *(_QWORD *)v117 != -8 )
            --HIDWORD(v160);
          *(_QWORD *)v117 = v48;
          *((_QWORD *)v117 + 1) = 0;
LABEL_148:
          v153 = sub_1649960(v146);
          v154 = v119;
          v155 = (__int64 *)&v153;
          v157 = 773;
          v156 = ".wineh.reload";
          v142 = sub_157EBA0((__int64)v48);
          v120 = sub_1648A60(64, 1u);
          v121 = v120;
          if ( v120 )
            sub_15F90C0((__int64)v120, *(_QWORD *)(*v148 + 24LL), (__int64)v148, (__int64)&v155, 0, v142);
          *((_QWORD *)v117 + 1) = v121;
          v79 = v121;
          if ( !*v24 )
            goto LABEL_85;
          goto LABEL_83;
        }
LABEL_81:
        v79 = (_QWORD *)*((_QWORD *)v77 + 1);
        if ( !v79 )
        {
          v117 = (char *)v77;
          goto LABEL_148;
        }
        if ( !*v24 )
        {
          *v24 = v79;
          goto LABEL_86;
        }
LABEL_83:
        v80 = v24[1];
        v81 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v81 = v80;
        if ( v80 )
          *(_QWORD *)(v80 + 16) = *(_QWORD *)(v80 + 16) & 3LL | v81;
LABEL_85:
        *v24 = v79;
        if ( v79 )
        {
LABEL_86:
          v82 = v79[1];
          v24[1] = v82;
          if ( v82 )
            *(_QWORD *)(v82 + 16) = (unsigned __int64)(v24 + 1) | *(_QWORD *)(v82 + 16) & 3LL;
          v24[2] = (unsigned __int64)(v79 + 1) | v24[2] & 3LL;
          v79[1] = v24;
          if ( !v23 )
          {
LABEL_28:
            v33 = (__int64)v148;
            j___libc_free_0(v159);
            goto LABEL_29;
          }
        }
        else
        {
LABEL_27:
          if ( !v23 )
            goto LABEL_28;
        }
      }
      if ( v148 )
        goto LABEL_20;
LABEL_90:
      v83 = *(_QWORD **)(v145 - 24);
      v84 = *(_DWORD *)(*(_QWORD *)(a1 + 160) + 4LL);
      v153 = sub_1649960(v146);
      v155 = (__int64 *)&v153;
      v156 = ".wineh.spillslot";
      v154 = v85;
      v86 = *(_QWORD *)(a2 + 80);
      v157 = 773;
      if ( !v86 )
        BUG();
      v87 = *(_QWORD *)(v86 + 24);
      if ( v87 )
        v87 -= 24;
      v148 = sub_1648A60(64, 1u);
      if ( v148 )
        sub_15F8BC0((__int64)v148, v83, v84, 0, (__int64)&v155, v87);
      goto LABEL_45;
    }
    v102 = *(_QWORD **)(v145 - 24);
    v103 = *(_DWORD *)(*(_QWORD *)(a1 + 160) + 4LL);
    v104 = sub_1649960(v146);
    v158 = &v155;
    v155 = (__int64 *)v104;
    v159 = ".wineh.spillslot";
    v156 = v105;
    v106 = *(_QWORD *)(a2 + 80);
    LOWORD(v160) = 773;
    if ( !v106 )
      BUG();
    v107 = *(_QWORD *)(v106 + 24);
    if ( v107 )
      v107 -= 24;
    v150 = v107;
    v108 = sub_1648A60(64, 1u);
    v33 = (__int64)v108;
    if ( v108 )
      sub_15F8BC0((__int64)v108, v102, v103, 0, (__int64)&v158, v150);
    v109 = sub_1649960(v146);
    v158 = &v155;
    v155 = (__int64 *)v109;
    LOWORD(v160) = 773;
    v156 = v110;
    v159 = ".wineh.reload";
    v111 = sub_157EE30(v22);
    if ( v111 )
      v111 -= 24;
    v112 = sub_1648A60(64, 1u);
    v115 = (__int64)v112;
    if ( v112 )
      sub_15F90E0((__int64)v112, v33, (__int64)&v158, v111);
    sub_164D160(v146, v115, a4, a5, a6, a7, v113, v114, a10, a11);
LABEL_29:
    if ( !v33 )
      goto LABEL_38;
    v36 = 1;
    v158 = (__int64 **)&v160;
    v149 = (__int64 **)&v160;
    v160 = *(_QWORD *)(v145 + 16);
    v161 = v145 - 24;
    v159 = (const char *)0x400000001LL;
    while ( 1 )
    {
LABEL_31:
      v37 = v36--;
      v38 = &v149[2 * v37 - 2];
      v39 = (__int64)v38[1];
      v40 = *v38;
      LODWORD(v159) = v36;
      if ( *(_BYTE *)(v39 + 16) == 77 && *(__int64 **)(v39 + 40) == v40 )
      {
        if ( (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) != 0 )
        {
          v88 = 0;
          v89 = 8LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
          do
          {
            if ( (*(_BYTE *)(v39 + 23) & 0x40) != 0 )
              v90 = *(_QWORD *)(v39 - 8);
            else
              v90 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
            v91 = *(_QWORD *)(v90 + 3 * v88);
            if ( *(_BYTE *)(v91 + 16) != 9 )
              sub_1F601F0(*(_QWORD *)(v88 + v90 + 24LL * *(unsigned int *)(v39 + 56) + 8), v91, v33, (__int64)&v158);
            v88 += 8;
          }
          while ( v89 != v88 );
LABEL_74:
          v36 = (unsigned int)v159;
          v149 = v158;
        }
        goto LABEL_75;
      }
      v41 = v40[1];
      if ( v41 )
        break;
LABEL_75:
      if ( !v36 )
        goto LABEL_36;
    }
    do
    {
      v42 = sub_1648700(v41);
      if ( (unsigned __int8)(*((_BYTE *)v42 + 16) - 25) <= 9u )
      {
LABEL_72:
        sub_1F601F0(v42[5], v39, v33, (__int64)&v158);
        while ( 1 )
        {
          v41 = *(_QWORD *)(v41 + 8);
          if ( !v41 )
            goto LABEL_74;
          v42 = sub_1648700(v41);
          if ( (unsigned __int8)(*((_BYTE *)v42 + 16) - 25) <= 9u )
            goto LABEL_72;
        }
      }
      v41 = *(_QWORD *)(v41 + 8);
    }
    while ( v41 );
    if ( v36 )
      goto LABEL_31;
LABEL_36:
    if ( v149 != (__int64 **)&v160 )
      _libc_free((unsigned __int64)v149);
LABEL_38:
    v43 = (unsigned int)v163;
    if ( (unsigned int)v163 >= HIDWORD(v163) )
    {
      sub_16CD150((__int64)&v162, v164, 0, 8, v34, v35);
      v43 = (unsigned int)v163;
    }
    v162[v43] = (__int64 ***)v146;
    LODWORD(v163) = v163 + 1;
    if ( v140 != v136 )
    {
      v145 = v136;
      goto LABEL_14;
    }
  }
  while ( v147 != v151 );
  v16 = v162;
  v17 = &v162[(unsigned int)v163];
  if ( v162 != v17 )
  {
    do
    {
      v18 = *v16++;
      v19 = sub_1599EF0(*v18);
      sub_164D160((__int64)v18, v19, a4, a5, a6, a7, v20, v21, a10, a11);
      sub_15F20C0(v18);
    }
    while ( v17 != v16 );
    v17 = v162;
  }
  if ( v17 != (__int64 ****)v164 )
    _libc_free((unsigned __int64)v17);
}
