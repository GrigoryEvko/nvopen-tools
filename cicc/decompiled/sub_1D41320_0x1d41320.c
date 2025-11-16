// Function: sub_1D41320
// Address: 0x1d41320
//
_QWORD *__fastcall sub_1D41320(
        __int64 a1,
        __int64 a2,
        const void **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int64 a10,
        __int64 a11,
        const void *a12,
        __int64 a13)
{
  bool v14; // zf
  const void *v15; // r10
  __int64 v16; // r8
  __int64 (*v17)(); // rax
  __int16 v18; // di
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // si
  __int64 v22; // r14
  int v23; // edx
  __int16 v24; // r11
  __int16 v25; // r10
  _BYTE *v26; // r8
  __int64 v27; // rax
  char v28; // bl
  char v29; // di
  __int64 v30; // r9
  __int64 j; // rdx
  __int64 v32; // r15
  _DWORD *v33; // rax
  int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // r9
  unsigned __int16 *v37; // r10
  unsigned __int8 *v38; // r9
  __int64 v39; // rcx
  unsigned __int8 v40; // dl
  unsigned __int64 v41; // rax
  int v42; // edx
  char v43; // r11
  int v44; // eax
  unsigned __int64 v45; // r8
  _QWORD *v46; // rcx
  unsigned __int8 v47; // bl
  const void **v48; // r12
  __int64 *v49; // rax
  __int128 v50; // rax
  __int64 v51; // r14
  _QWORD *v52; // r12
  __int64 v54; // r15
  __int64 v55; // rsi
  __int64 v56; // r15
  int v57; // esi
  _QWORD *v58; // rax
  char *v59; // rdi
  _QWORD *v60; // rax
  unsigned int v61; // edx
  unsigned int v62; // r15d
  __int64 v63; // rax
  int v64; // edx
  unsigned int v65; // edx
  _QWORD *v66; // rbx
  unsigned int v67; // r8d
  int v68; // esi
  int *v69; // rax
  int v70; // edx
  int v71; // edi
  unsigned int v72; // eax
  __int64 v73; // rax
  _QWORD *v74; // rax
  unsigned int v75; // edx
  unsigned int v76; // ebx
  int v77; // esi
  int *v78; // rax
  int *v79; // r8
  int v80; // edx
  int v81; // edi
  unsigned int v82; // esi
  __int64 v83; // rdx
  int v84; // ecx
  int v85; // r8d
  __int64 v86; // rax
  unsigned __int64 v87; // rdi
  __int64 v88; // r8
  unsigned __int64 i; // rax
  __int64 v90; // rdx
  unsigned __int64 v91; // rdx
  _DWORD *v92; // rcx
  unsigned __int64 v93; // rdx
  __int64 v94; // rsi
  __int64 v95; // rax
  unsigned __int64 v96; // rdi
  __int64 v97; // rsi
  unsigned __int64 k; // rax
  __int64 v99; // rdx
  unsigned __int64 v100; // rdx
  int *v101; // rcx
  int v102; // edx
  unsigned __int64 v103; // rdx
  __int64 v104; // r10
  void *v105; // rax
  unsigned __int64 v106; // rcx
  __int64 v107; // r8
  void *v108; // r12
  __int64 v109; // rdx
  __int64 v110; // r9
  const void **v111; // r13
  unsigned __int64 v112; // r15
  int v113; // r14d
  __int128 v114; // rdi
  __int64 v115; // r13
  __int64 v116; // rsi
  unsigned __int8 *v117; // rsi
  int v118; // eax
  __int64 v119; // r15
  __int64 v120; // rdi
  unsigned __int64 v121; // r15
  __int64 v122; // rax
  int v123; // eax
  __int64 v124; // rax
  unsigned int v125; // ebx
  int v126; // eax
  bool v127; // al
  unsigned __int16 *v128; // [rsp+0h] [rbp-1A0h]
  unsigned __int16 *src; // [rsp+10h] [rbp-190h]
  int srca; // [rsp+10h] [rbp-190h]
  unsigned int v131; // [rsp+18h] [rbp-188h]
  int v132; // [rsp+18h] [rbp-188h]
  unsigned __int8 *v133; // [rsp+18h] [rbp-188h]
  unsigned __int8 *v134; // [rsp+18h] [rbp-188h]
  __int64 v135; // [rsp+18h] [rbp-188h]
  __int64 v137; // [rsp+30h] [rbp-170h]
  unsigned int v138; // [rsp+40h] [rbp-160h]
  unsigned int v139; // [rsp+44h] [rbp-15Ch]
  __int64 v140; // [rsp+48h] [rbp-158h]
  unsigned int v141; // [rsp+48h] [rbp-158h]
  unsigned __int8 v142; // [rsp+48h] [rbp-158h]
  __int64 v143; // [rsp+48h] [rbp-158h]
  __int64 v145; // [rsp+58h] [rbp-148h]
  char v146; // [rsp+58h] [rbp-148h]
  _QWORD *v147; // [rsp+58h] [rbp-148h]
  _QWORD *v148; // [rsp+58h] [rbp-148h]
  unsigned int v149; // [rsp+58h] [rbp-148h]
  unsigned __int8 v150; // [rsp+58h] [rbp-148h]
  __int64 v151; // [rsp+70h] [rbp-130h] BYREF
  const void **v152; // [rsp+78h] [rbp-128h]
  __int64 *v153; // [rsp+80h] [rbp-120h] BYREF
  unsigned __int8 *v154; // [rsp+88h] [rbp-118h] BYREF
  __int64 v155; // [rsp+90h] [rbp-110h] BYREF
  unsigned __int64 v156; // [rsp+98h] [rbp-108h]
  __int64 v157; // [rsp+A0h] [rbp-100h]
  unsigned __int64 v158; // [rsp+A8h] [rbp-F8h]
  void *v159; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v160; // [rsp+B8h] [rbp-E8h]
  _BYTE v161[32]; // [rsp+C0h] [rbp-E0h] BYREF
  _DWORD *v162; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v163; // [rsp+E8h] [rbp-B8h]
  _DWORD v164[44]; // [rsp+F0h] [rbp-B0h] BYREF

  v14 = *(_WORD *)(a5 + 24) == 48;
  v151 = a2;
  v152 = a3;
  v15 = a12;
  v137 = a6;
  v140 = a5;
  v138 = a6;
  v145 = a10;
  v139 = a11;
  if ( !v14 || *(_WORD *)(a10 + 24) != 48 )
  {
    v159 = v161;
    v16 = (4 * a13) >> 2;
    v160 = 0x800000000LL;
    if ( (unsigned __int64)(4 * a13) > 0x20 )
    {
      sub_16CD150((__int64)&v159, v161, (4 * a13) >> 2, 4, v16, a6);
      v16 = (4 * a13) >> 2;
      v15 = a12;
      v59 = (char *)v159 + 4 * (unsigned int)v160;
    }
    else
    {
      if ( !(4 * a13) )
      {
        LODWORD(v160) = (4 * a13) >> 2;
        if ( a10 != a5 )
          goto LABEL_7;
LABEL_6:
        if ( v138 == (_DWORD)a11 )
        {
          v162 = 0;
          LODWORD(v163) = 0;
          v60 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v162, v151, (__int64)v152, a6);
          v62 = v61;
          if ( v162 )
          {
            v147 = v60;
            sub_161E7C0((__int64)&v162, (__int64)v162);
            v60 = v147;
          }
          v145 = (__int64)v60;
          v139 = v62;
          if ( (_DWORD)a13 )
          {
            v63 = 0;
            do
            {
              v64 = *(_DWORD *)((char *)v159 + v63);
              if ( (int)a13 <= v64 )
                *(_DWORD *)((char *)v159 + v63) = v64 - a13;
              v63 += 4;
            }
            while ( 4LL * (unsigned int)(a13 - 1) + 4 != v63 );
          }
        }
LABEL_7:
        if ( *(_WORD *)(a5 + 24) == 48 )
        {
          v77 = v160;
          v78 = (int *)v159;
          if ( (_DWORD)v160 )
          {
            v79 = (int *)((char *)v159 + 4 * (unsigned int)(v160 - 1) + 4);
            do
            {
              v80 = *v78;
              if ( *v78 >= 0 )
              {
                v81 = v80 - v77;
                if ( v80 < v77 )
                  v81 = v77 + v80;
                *v78 = v81;
              }
              ++v78;
            }
            while ( v79 != v78 );
          }
          v82 = v139;
          v139 = v138;
          v138 = v82;
          v140 = v145;
          v145 = a5;
        }
        v17 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 840LL);
        if ( v17 != sub_1D12E10 && (unsigned __int8)v17() )
        {
          if ( *(_WORD *)(v140 + 24) == 104 )
          {
            v162 = 0;
            v163 = 0;
            v164[0] = 0;
            v86 = sub_1D1AA50(v140, (__int64)&v162, v83, v84, v85, a6);
            v87 = (unsigned __int64)v162;
            if ( v86 && (int)a13 > 0 )
            {
              v88 = (unsigned int)(a13 - 1);
              for ( i = 0; ; i = v91 )
              {
                v92 = (char *)v159 + 4 * i;
                v93 = (unsigned int)*v92;
                if ( (int)a13 <= (int)v93 || (v93 & 0x80000000) != 0LL )
                  goto LABEL_113;
                v94 = *(_QWORD *)(v87 + 8LL * ((unsigned int)v93 >> 6));
                if ( !_bittest64(&v94, v93) )
                  break;
                *v92 = -1;
                v91 = i + 1;
                v87 = (unsigned __int64)v162;
                if ( v88 == i )
                  goto LABEL_119;
LABEL_114:
                ;
              }
              v90 = *(_QWORD *)(v87 + 8LL * ((unsigned int)i >> 6));
              if ( !_bittest64(&v90, i) )
              {
                *v92 = i;
                v87 = (unsigned __int64)v162;
              }
LABEL_113:
              v91 = i + 1;
              if ( v88 == i )
                goto LABEL_119;
              goto LABEL_114;
            }
LABEL_119:
            _libc_free(v87);
          }
          v18 = *(_WORD *)(v145 + 24);
          if ( v18 != 104 )
          {
LABEL_10:
            if ( !(_DWORD)a13 )
              goto LABEL_57;
            v19 = (unsigned int)(a13 - 1);
            v20 = 0;
            v21 = 1;
            v22 = 4 * v19 + 4;
            a6 = 1;
            do
            {
              while ( 1 )
              {
                v23 = *(_DWORD *)((char *)v159 + v20);
                if ( (int)a13 > v23 )
                  break;
                if ( v18 == 48 )
                  *(_DWORD *)((char *)v159 + v20) = -1;
                else
                  a6 = 0;
                v20 += 4;
                if ( v22 == v20 )
                  goto LABEL_19;
              }
              if ( v23 >= 0 )
                v21 = 0;
              v20 += 4;
            }
            while ( v22 != v20 );
LABEL_19:
            if ( (_BYTE)a6 && v21 )
              goto LABEL_57;
            if ( v18 != 48 && (_BYTE)a6 )
            {
              v162 = 0;
              LODWORD(v163) = 0;
              v74 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v162, v151, (__int64)v152, a6);
              v76 = v75;
              if ( v162 )
              {
                v148 = v74;
                sub_161E7C0((__int64)&v162, (__int64)v162);
                v74 = v148;
              }
              v145 = (__int64)v74;
              v139 = v76;
            }
            else if ( v21 )
            {
              v162 = 0;
              LODWORD(v163) = 0;
              v66 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v162, v151, (__int64)v152, a6);
              v67 = v65;
              if ( v162 )
              {
                v141 = v65;
                sub_161E7C0((__int64)&v162, (__int64)v162);
                v67 = v141;
              }
              v68 = v160;
              v69 = (int *)v159;
              if ( (_DWORD)v160 )
              {
                a6 = (__int64)v159 + 4 * (unsigned int)(v160 - 1) + 4;
                do
                {
                  v70 = *v69;
                  if ( *v69 >= 0 )
                  {
                    v71 = v70 - v68;
                    if ( v70 < v68 )
                      v71 = v68 + v70;
                    *v69 = v71;
                  }
                  ++v69;
                }
                while ( (int *)a6 != v69 );
              }
              v72 = v139;
              v139 = v67;
              v138 = v72;
              v73 = v145;
              v145 = (__int64)v66;
              v140 = v73;
            }
            v24 = *(_WORD *)(v145 + 24);
            v25 = *(_WORD *)(v140 + 24);
            if ( v25 == 48 && v24 == 48 )
            {
LABEL_57:
              v162 = 0;
              LODWORD(v163) = 0;
              v52 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v162, v151, (__int64)v152, a6);
              if ( v162 )
                sub_161E7C0((__int64)&v162, (__int64)v162);
              goto LABEL_59;
            }
            v26 = v159;
            v27 = 0;
            v28 = 1;
            v29 = 1;
            v30 = *(unsigned int *)v159;
            for ( j = v30; ; j = *((unsigned int *)v159 + ++v27) )
            {
              if ( (_DWORD)j != (_DWORD)v27 && (int)j >= 0 )
                v29 = 0;
              if ( (_DWORD)v30 != (_DWORD)j )
                v28 = 0;
              if ( v19 == v27 )
                break;
            }
            if ( v29 )
            {
              v52 = (_QWORD *)v140;
LABEL_60:
              if ( v26 != v161 )
                _libc_free((unsigned __int64)v26);
              return v52;
            }
            if ( v24 != 48 )
              goto LABEL_63;
            v32 = v140;
            if ( v25 == 158 )
            {
              do
              {
                v33 = *(_DWORD **)(v32 + 32);
                v32 = *(_QWORD *)v33;
                v34 = v33[2];
                v25 = *(_WORD *)(*(_QWORD *)v33 + 24LL);
              }
              while ( v25 == 158 );
            }
            else
            {
              v34 = v138;
            }
            if ( v25 != 104 )
            {
LABEL_63:
              v163 = 0x2000000000LL;
              v162 = v164;
              v155 = v140;
              v156 = v138 | v137 & 0xFFFFFFFF00000000LL;
              v157 = v145;
              v158 = v139 | a11 & 0xFFFFFFFF00000000LL;
              v54 = sub_1D29190(a1, v151, (__int64)v152, v138, (__int64)v26, v30);
              sub_16BD430((__int64)&v162, 110);
              v55 = v54;
              v56 = 0;
              sub_16BD4C0((__int64)&v162, v55);
              sub_16BD4C0((__int64)&v162, v155);
              sub_16BD430((__int64)&v162, v156);
              sub_16BD4C0((__int64)&v162, v157);
              sub_16BD430((__int64)&v162, v158);
              do
              {
                v57 = *(_DWORD *)((char *)v159 + v56);
                v56 += 4;
                sub_16BD3E0((__int64)&v162, v57);
              }
              while ( v56 != v22 );
              v153 = 0;
              v58 = sub_1D17920(a1, (__int64)&v162, a4, (__int64 *)&v153);
              if ( v58 )
              {
                v52 = v58;
                goto LABEL_67;
              }
              v105 = (void *)sub_145CBF0((__int64 *)(a1 + 360), 4LL * (int)a13, 4);
              v108 = v105;
              v109 = 4LL * (unsigned int)v160;
              if ( v109 )
                memmove(v105, v159, v109);
              v110 = (unsigned __int8)v151;
              v111 = v152;
              v112 = *(_QWORD *)(a1 + 208);
              v113 = *(_DWORD *)(a4 + 8);
              if ( v112 )
              {
                *(_QWORD *)(a1 + 208) = *(_QWORD *)v112;
              }
              else
              {
                v119 = *(_QWORD *)(a1 + 216);
                v120 = *(_QWORD *)(a1 + 224);
                *(_QWORD *)(a1 + 296) += 112LL;
                v109 = v120 - v119;
                v106 = ((v119 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v119 + 112;
                if ( v106 <= v120 - v119 )
                {
                  v112 = (v119 + 7) & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(a1 + 216) = v112 + 112;
                }
                else
                {
                  v142 = v110;
                  v121 = 0x40000000000LL;
                  v149 = *(_DWORD *)(a1 + 240);
                  if ( v149 >> 7 < 0x1E )
                    v121 = 4096LL << (v149 >> 7);
                  v122 = malloc(v121);
                  v109 = v149;
                  v110 = v142;
                  if ( !v122 )
                  {
                    sub_16BD1C0("Allocation failed", 1u);
                    v110 = v142;
                    v109 = *(unsigned int *)(a1 + 240);
                    v122 = 0;
                  }
                  if ( *(_DWORD *)(a1 + 244) <= (unsigned int)v109 )
                  {
                    v143 = v122;
                    v150 = v110;
                    sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 8, v107, v110);
                    v110 = v150;
                    v109 = *(unsigned int *)(a1 + 240);
                    v122 = v143;
                  }
                  v106 = *(_QWORD *)(a1 + 232);
                  *(_QWORD *)(v106 + 8 * v109) = v122;
                  *(_QWORD *)(a1 + 224) = v122 + v121;
                  ++*(_DWORD *)(a1 + 240);
                  v112 = (v122 + 7) & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(a1 + 216) = v112 + 112;
                }
                if ( !v112 )
                  goto LABEL_143;
              }
              *((_QWORD *)&v114 + 1) = v111;
              *(_QWORD *)&v114 = (unsigned __int8)v110;
              v115 = sub_1D274F0(v114, v109, v106, v107, v110);
              v116 = *(_QWORD *)a4;
              v154 = (unsigned __int8 *)v116;
              if ( v116 )
                sub_1623A60((__int64)&v154, v116, 2);
              *(_QWORD *)v112 = 0;
              v117 = v154;
              *(_QWORD *)(v112 + 8) = 0;
              *(_QWORD *)(v112 + 16) = 0;
              *(_WORD *)(v112 + 24) = 110;
              *(_DWORD *)(v112 + 28) = -1;
              *(_QWORD *)(v112 + 32) = 0;
              *(_QWORD *)(v112 + 40) = v115;
              *(_QWORD *)(v112 + 48) = 0;
              *(_QWORD *)(v112 + 56) = 0x100000000LL;
              *(_DWORD *)(v112 + 64) = v113;
              *(_QWORD *)(v112 + 72) = v117;
              if ( v117 )
                sub_1623210((__int64)&v154, v117, v112 + 72);
              *(_WORD *)(v112 + 80) &= 0xF000u;
              *(_WORD *)(v112 + 26) = 0;
              *(_QWORD *)(v112 + 88) = v108;
LABEL_143:
              v52 = (_QWORD *)v112;
              sub_1D23B60(a1, v112, (__int64)&v155, 2);
              sub_16BDA20((__int64 *)(a1 + 320), (__int64 *)v112, v153);
              sub_1D172A0(a1, v112);
LABEL_67:
              if ( v162 != v164 )
                _libc_free((unsigned __int64)v162);
LABEL_59:
              v26 = v159;
              goto LABEL_60;
            }
            v131 = v34;
            v162 = 0;
            v163 = 0;
            v164[0] = 0;
            v35 = sub_1D1AA50(v32, (__int64)&v162, j, v34, (int)v159, v30);
            v37 = (unsigned __int16 *)v35;
            if ( v35 && *(_WORD *)(v35 + 24) == 48 )
            {
              v155 = 0;
              LODWORD(v156) = 0;
              v52 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v155, v151, (__int64)v152, v36);
              if ( v155 )
                sub_161E7C0((__int64)&v155, v155);
              goto LABEL_54;
            }
            v38 = *(unsigned __int8 **)(v32 + 40);
            v39 = 16LL * v131;
            v40 = v38[v39];
            v41 = *(_QWORD *)&v38[v39 + 8];
            LOBYTE(v155) = v40;
            v156 = v41;
            if ( v40 )
            {
              v42 = word_42E7700[(unsigned __int8)(v40 - 14)];
            }
            else
            {
              src = v37;
              v133 = v38;
              v118 = sub_1F58D30(&v155);
              v38 = v133;
              v37 = src;
              v42 = v118;
            }
            v43 = v151;
            if ( (_BYTE)v151 )
            {
              v44 = word_42E7700[(unsigned __int8)(v151 - 14)];
            }
            else
            {
              v128 = v37;
              srca = v42;
              v134 = v38;
              v44 = sub_1F58D30(&v151);
              v37 = v128;
              v43 = 0;
              v42 = srca;
              v38 = v134;
            }
            if ( v37 )
            {
              v45 = (unsigned __int64)v162;
              if ( !((unsigned int)(v164[0] + 63) >> 6) )
              {
LABEL_160:
                if ( v44 == v42
                  || ((v123 = v37[12], v123 == 10) || v123 == 32)
                  && ((v124 = *((_QWORD *)v37 + 11), v125 = *(_DWORD *)(v124 + 32), v125 <= 0x40)
                    ? (v127 = *(_QWORD *)(v124 + 24) == 0)
                    : (v135 = (__int64)v162, v126 = sub_16A57B0(v124 + 24), v45 = v135, v127 = v125 == v126),
                      v127) )
                {
                  v52 = (_QWORD *)v140;
                  goto LABEL_55;
                }
LABEL_84:
                _libc_free(v45);
                goto LABEL_63;
              }
              v46 = v162;
              while ( !*v46 )
              {
                if ( &v162[2 * ((unsigned int)(v164[0] + 63) >> 6)] == (_DWORD *)++v46 )
                  goto LABEL_160;
              }
            }
            if ( v44 == v42 && v28 )
            {
              v146 = v43;
              v47 = *v38;
              v48 = (const void **)*((_QWORD *)v38 + 1);
              v49 = (__int64 *)(*(_QWORD *)(v32 + 32) + 40LL * *(unsigned int *)v159);
              *(_QWORD *)&v50 = sub_1D35F20((__int64 *)a1, *v38, v48, a4, *v49, v49[1], a7, a8, a9);
              v51 = v50;
              if ( v47 != v146 || v152 != v48 && !v146 )
                v51 = sub_1D309E0(
                        (__int64 *)a1,
                        158,
                        a4,
                        (unsigned int)v151,
                        v152,
                        0,
                        a7,
                        a8,
                        *(double *)a9.m128i_i64,
                        v50);
              v52 = (_QWORD *)v51;
LABEL_54:
              v45 = (unsigned __int64)v162;
LABEL_55:
              _libc_free(v45);
              v26 = v159;
              goto LABEL_60;
            }
            v45 = (unsigned __int64)v162;
            goto LABEL_84;
          }
          v162 = 0;
          v163 = 0;
          v164[0] = 0;
          v95 = sub_1D1AA50(v145, (__int64)&v162, v83, v84, v85, a6);
          v96 = (unsigned __int64)v162;
          if ( v95 && (int)a13 > 0 )
          {
            v97 = (unsigned int)(a13 - 1);
            for ( k = 0; ; k = v100 )
            {
              v101 = (int *)((char *)v159 + 4 * k);
              v102 = *v101;
              if ( (int)a13 > *v101 || v102 >= 2 * (int)a13 )
                goto LABEL_126;
              v103 = (unsigned int)(v102 - a13);
              v104 = *(_QWORD *)(v96 + 8LL * ((unsigned int)v103 >> 6));
              if ( !_bittest64(&v104, v103) )
                break;
              *v101 = -1;
              v100 = k + 1;
              v96 = (unsigned __int64)v162;
              if ( v97 == k )
                goto LABEL_132;
LABEL_127:
              ;
            }
            v99 = *(_QWORD *)(v96 + 8LL * ((unsigned int)k >> 6));
            if ( !_bittest64(&v99, k) )
            {
              *v101 = a13 + k;
              v96 = (unsigned __int64)v162;
            }
LABEL_126:
            v100 = k + 1;
            if ( v97 == k )
              goto LABEL_132;
            goto LABEL_127;
          }
LABEL_132:
          _libc_free(v96);
        }
        v18 = *(_WORD *)(v145 + 24);
        goto LABEL_10;
      }
      v59 = v161;
    }
    v132 = v16;
    memcpy(v59, v15, 4 * a13);
    LODWORD(v160) = v160 + v132;
    if ( a10 != a5 )
      goto LABEL_7;
    goto LABEL_6;
  }
  v162 = 0;
  LODWORD(v163) = 0;
  v52 = sub_1D2B300((_QWORD *)a1, 0x30u, (__int64)&v162, a2, (__int64)a3, a6);
  if ( v162 )
    sub_161E7C0((__int64)&v162, (__int64)v162);
  return v52;
}
