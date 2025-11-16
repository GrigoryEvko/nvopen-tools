// Function: sub_215FB20
// Address: 0x215fb20
//
__int64 __fastcall sub_215FB20(
        __int64 a1,
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
  __int64 v10; // r15
  __int64 v11; // r15
  __int64 v13; // r12
  __int64 v14; // rbx
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  bool v17; // al
  size_t v18; // r9
  unsigned __int8 v19; // r12
  _QWORD *v20; // r14
  __int16 v21; // r12
  unsigned int v22; // esi
  int v23; // r8d
  __int64 v24; // rax
  int v25; // r8d
  int v26; // esi
  __int64 v27; // r9
  unsigned int v28; // edx
  _QWORD *v29; // rcx
  _QWORD *v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int8 *v36; // rsi
  _QWORD *v37; // r12
  _QWORD *v38; // rbx
  unsigned __int64 v39; // r13
  __int64 v40; // r14
  __int64 v41; // r15
  __int64 v42; // rcx
  __int64 v43; // rdx
  _QWORD *v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rsi
  unsigned __int64 v47; // rcx
  __int64 v48; // rcx
  int v49; // r14d
  __int64 v50; // rdx
  _QWORD *v51; // rax
  __int64 v52; // r8
  unsigned int v53; // ecx
  unsigned int v54; // edx
  __int64 v55; // r8
  _QWORD *v56; // rbx
  __int64 v57; // rcx
  _QWORD *v58; // r12
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // eax
  _QWORD *v62; // rbx
  _QWORD *v63; // r12
  __int64 v64; // rsi
  _QWORD *v65; // rax
  _QWORD *v66; // rdx
  char v67; // cl
  int v68; // ecx
  unsigned int v69; // eax
  _QWORD *v71; // rbx
  _QWORD *v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  _QWORD *v76; // rbx
  _QWORD *v77; // r12
  __int64 v78; // rsi
  __int64 v79; // rax
  _QWORD *v80; // rdx
  _QWORD *v81; // rbx
  _QWORD *v82; // r13
  __int64 v83; // rcx
  _QWORD *v84; // rcx
  __int64 v85; // rax
  _QWORD *v86; // r14
  _QWORD *v87; // rbx
  __int64 *v88; // r12
  __int64 ***k; // r13
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  bool v93; // zf
  unsigned __int8 **v94; // rsi
  bool v95; // al
  __int64 v96; // rax
  double v97; // xmm4_8
  double v98; // xmm5_8
  unsigned __int64 v99; // rdx
  const char *v100; // r10
  size_t v101; // r9
  __int64 v102; // rax
  __int64 *v103; // rdx
  unsigned __int8 *v104; // rax
  __int64 *v105; // rdi
  __int64 v106; // rax
  __int64 v107; // r14
  __int64 v108; // rax
  _QWORD *v109; // r12
  int v110; // edx
  __int64 v111; // rdx
  unsigned __int64 *v112; // rdi
  unsigned __int8 **v113; // rdx
  __int64 v114; // rax
  _QWORD *v115; // rdi
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // r8
  int v119; // edi
  _QWORD *v120; // rcx
  int v121; // esi
  _QWORD *v122; // rcx
  __int64 v123; // rdx
  __int64 v124; // r8
  int v125; // esi
  __int64 v126; // rdx
  __int64 v127; // rdi
  _QWORD *v128; // r12
  _QWORD *v129; // rbx
  __int64 v130; // rax
  __int64 v131; // rdx
  int v132; // eax
  unsigned int v133; // eax
  _QWORD *v134; // r8
  unsigned __int64 v135; // rdx
  unsigned __int64 v136; // rax
  _QWORD *v137; // rax
  __int64 v138; // rdx
  _QWORD *v139; // rdx
  char v140; // cl
  __int64 v141; // r9
  unsigned int v142; // ecx
  _QWORD *v143; // rdx
  __int64 v144; // r8
  int v145; // edi
  int v146; // edi
  int v147; // edx
  __int64 v148; // rdx
  unsigned __int64 *v149; // r12
  __int64 v150; // rax
  int v151; // esi
  int v152; // esi
  __int64 v153; // r8
  _QWORD *v154; // r9
  unsigned int v155; // edx
  int v156; // ecx
  __int64 v157; // rdi
  _QWORD *v158; // r14
  _QWORD *v159; // rbx
  char v160; // al
  __int64 v161; // rax
  __int64 i; // [rsp+8h] [rbp-118h]
  size_t n; // [rsp+10h] [rbp-110h]
  size_t na; // [rsp+10h] [rbp-110h]
  size_t nb; // [rsp+10h] [rbp-110h]
  size_t nc; // [rsp+10h] [rbp-110h]
  void *src; // [rsp+18h] [rbp-108h]
  _QWORD *srca; // [rsp+18h] [rbp-108h]
  unsigned __int64 *srcb; // [rsp+18h] [rbp-108h]
  const char *srcc; // [rsp+18h] [rbp-108h]
  char v171; // [rsp+20h] [rbp-100h]
  char v173; // [rsp+30h] [rbp-F0h]
  __int64 v174; // [rsp+38h] [rbp-E8h]
  __int64 v175; // [rsp+38h] [rbp-E8h]
  int v176; // [rsp+38h] [rbp-E8h]
  unsigned __int8 **v177; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v178; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v179; // [rsp+50h] [rbp-D0h]
  __int64 v180; // [rsp+58h] [rbp-C8h]
  __int64 v181; // [rsp+60h] [rbp-C0h]
  unsigned __int8 *v182; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v183; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v184; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v185; // [rsp+88h] [rbp-98h]
  unsigned __int8 **j; // [rsp+90h] [rbp-90h]
  unsigned __int8 *v187; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v188; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v189; // [rsp+B0h] [rbp-70h]
  __int64 v190; // [rsp+B8h] [rbp-68h]
  __int64 v191; // [rsp+C0h] [rbp-60h]
  _QWORD *v192; // [rsp+C8h] [rbp-58h]
  __int64 v193; // [rsp+D0h] [rbp-50h]
  __int64 v194; // [rsp+D8h] [rbp-48h]
  char v195; // [rsp+E0h] [rbp-40h]
  char v196; // [rsp+E9h] [rbp-37h]

  v10 = a1;
  v174 = a2 + 8;
  if ( *(_QWORD *)(a2 + 16) != a2 + 8 )
  {
    v11 = *(_QWORD *)(a2 + 16);
    while ( 1 )
    {
      v13 = v11;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !(*(_DWORD *)(*(_QWORD *)(v13 - 56) + 8LL) >> 8) )
      {
        v14 = v13 - 56;
        if ( !(unsigned __int8)sub_1C2E830(v13 - 56)
          && !(unsigned __int8)sub_1C2E860(v13 - 56)
          && !(unsigned __int8)sub_1C2E890(v13 - 56) )
        {
          v15 = sub_1649960(v13 - 56);
          if ( v16 <= 4 || *(_DWORD *)v15 != 1836477548 || v15[4] != 46 )
            break;
        }
      }
LABEL_3:
      if ( v11 == v174 )
      {
        v10 = a1;
        goto LABEL_27;
      }
    }
    src = *(void **)(v13 - 32);
    v171 = *(_BYTE *)(v13 + 24) & 1;
    v173 = *(_BYTE *)(v13 - 24) & 0xF;
    v17 = sub_15E4F60(v13 - 56);
    v18 = 0;
    if ( !v17 )
      v18 = *(_QWORD *)(v13 - 80);
    n = v18;
    LOWORD(v189) = 257;
    v19 = *(_BYTE *)(v13 - 23);
    v20 = sub_1648A60(88, 1u);
    v21 = (v19 >> 2) & 7;
    if ( v20 )
      sub_15E51E0((__int64)v20, a2, (__int64)src, v171, v173, n, (__int64)&v187, v14, v21, 1u, 0);
    sub_15E6480((__int64)v20, v14);
    sub_1628980((__int64)v20, v14, 0);
    v188 = 2;
    v189 = 0;
    v190 = v14;
    if ( v14 != -16 && v14 != -8 )
      sub_164C220((__int64)&v188);
    v22 = *(_DWORD *)(a1 + 184);
    v191 = a1 + 160;
    v187 = (unsigned __int8 *)&unk_49F8530;
    if ( v22 )
    {
      v24 = v190;
      v141 = *(_QWORD *)(a1 + 168);
      v142 = (v22 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
      v143 = (_QWORD *)(v141 + 48LL * v142);
      v144 = v143[3];
      if ( v190 == v144 )
      {
LABEL_261:
        v30 = v143;
        goto LABEL_262;
      }
      v145 = 1;
      v30 = 0;
      while ( v144 != -8 )
      {
        if ( !v30 && v144 == -16 )
          v30 = v143;
        v142 = (v22 - 1) & (v145 + v142);
        v143 = (_QWORD *)(v141 + 48LL * v142);
        v144 = v143[3];
        if ( v190 == v144 )
          goto LABEL_261;
        ++v145;
      }
      v146 = *(_DWORD *)(a1 + 176);
      if ( !v30 )
        v30 = v143;
      ++*(_QWORD *)(a1 + 160);
      v147 = v146 + 1;
      if ( 4 * (v146 + 1) < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a1 + 180) - v147 > v22 >> 3 )
          goto LABEL_272;
        sub_1CB10E0(a1 + 160, v22);
        v151 = *(_DWORD *)(a1 + 184);
        if ( v151 )
        {
          v24 = v190;
          v152 = v151 - 1;
          v153 = *(_QWORD *)(a1 + 168);
          v154 = 0;
          v155 = v152 & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
          v156 = 1;
          v30 = (_QWORD *)(v153 + 48LL * v155);
          v157 = v30[3];
          if ( v190 != v157 )
          {
            while ( v157 != -8 )
            {
              if ( !v154 && v157 == -16 )
                v154 = v30;
              v155 = v152 & (v156 + v155);
              v30 = (_QWORD *)(v153 + 48LL * v155);
              v157 = v30[3];
              if ( v190 == v157 )
                goto LABEL_286;
              ++v156;
            }
            if ( v154 )
              v30 = v154;
          }
          goto LABEL_286;
        }
        goto LABEL_317;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 160);
    }
    sub_1CB10E0(a1 + 160, 2 * v22);
    v23 = *(_DWORD *)(a1 + 184);
    if ( v23 )
    {
      v24 = v190;
      v25 = v23 - 1;
      v26 = 1;
      v27 = *(_QWORD *)(a1 + 168);
      v28 = v25 & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
      v29 = 0;
      v30 = (_QWORD *)(v27 + 48LL * v28);
      v31 = v30[3];
      if ( v190 != v31 )
      {
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v29 )
            v29 = v30;
          v28 = v25 & (v26 + v28);
          v30 = (_QWORD *)(v27 + 48LL * v28);
          v31 = v30[3];
          if ( v190 == v31 )
            goto LABEL_286;
          ++v26;
        }
        if ( v29 )
          v30 = v29;
      }
      goto LABEL_286;
    }
LABEL_317:
    v24 = v190;
    v30 = 0;
LABEL_286:
    v147 = *(_DWORD *)(a1 + 176) + 1;
LABEL_272:
    *(_DWORD *)(a1 + 176) = v147;
    if ( v30[3] == -8 )
    {
      v148 = -8;
      v149 = v30 + 1;
      if ( v24 != -8 )
      {
LABEL_277:
        v30[3] = v24;
        if ( v24 != -8 && v24 != 0 && v24 != -16 )
          sub_1649AC0(v149, v188 & 0xFFFFFFFFFFFFFFF8LL);
        v148 = v190;
      }
    }
    else
    {
      --*(_DWORD *)(a1 + 180);
      v148 = v30[3];
      if ( v24 != v148 )
      {
        v149 = v30 + 1;
        if ( v148 != -8 && v148 != 0 && v148 != -16 )
        {
          sub_1649B30(v30 + 1);
          v24 = v190;
        }
        goto LABEL_277;
      }
    }
    v150 = v191;
    v30[5] = 0;
    v30[4] = v150;
    v24 = v148;
LABEL_262:
    v187 = (unsigned __int8 *)&unk_49EE2B0;
    if ( v24 != -8 && v24 != 0 && v24 != -16 )
      sub_1649B30(&v188);
    v30[5] = v20;
    goto LABEL_3;
  }
LABEL_27:
  if ( !*(_DWORD *)(v10 + 176) )
    return 0;
  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    v32 = i - 56;
    if ( !i )
      v32 = 0;
    v175 = v32;
    if ( !sub_15E4F60(v32) )
    {
      v33 = *(_QWORD *)(v32 + 80);
      if ( v33 )
        v33 -= 24;
      v34 = sub_157ED60(v33);
      v35 = sub_16498A0(v34);
      v187 = 0;
      v190 = v35;
      v191 = 0;
      LODWORD(v192) = 0;
      v193 = 0;
      v194 = 0;
      v188 = *(_QWORD *)(v34 + 40);
      v189 = v34 + 24;
      v36 = *(unsigned __int8 **)(v34 + 48);
      v182 = v36;
      if ( v36 )
      {
        sub_1623A60((__int64)&v182, (__int64)v36, 2);
        if ( v187 )
          sub_161E7C0((__int64)&v187, (__int64)v187);
        v187 = v182;
        if ( v182 )
          sub_1623210((__int64)&v182, v182, (__int64)&v187);
      }
      na = v32 + 72;
      srca = *(_QWORD **)(v32 + 80);
      if ( (_QWORD *)(v32 + 72) != srca )
      {
        while ( 1 )
        {
          if ( !srca )
            BUG();
          v37 = (_QWORD *)srca[3];
          if ( srca + 2 != v37 )
            break;
LABEL_62:
          srca = (_QWORD *)srca[1];
          if ( (_QWORD *)na == srca )
            goto LABEL_63;
        }
        while ( 1 )
        {
          if ( !v37 )
            BUG();
          if ( (*((_DWORD *)v37 - 1) & 0xFFFFFFF) != 0 )
            break;
LABEL_61:
          v37 = (_QWORD *)v37[1];
          if ( srca + 2 == v37 )
            goto LABEL_62;
        }
        v38 = v37 - 3;
        v39 = 0;
        v40 = v10;
        v41 = 24LL * (*((_DWORD *)v37 - 1) & 0xFFFFFFF);
        while ( 1 )
        {
          if ( (*((_BYTE *)v37 - 1) & 0x40) != 0 )
          {
            v42 = *(_QWORD *)(*(v37 - 4) + v39);
            if ( *(_BYTE *)(v42 + 16) > 0x10u )
              goto LABEL_48;
          }
          else
          {
            v42 = v38[v39 / 8 + -3 * (*((_DWORD *)v37 - 1) & 0xFFFFFFF)];
            if ( *(_BYTE *)(v42 + 16) > 0x10u )
              goto LABEL_48;
          }
          v43 = sub_215E100(v40, (_QWORD **)a2, v175, v42, (__int64)&v187, *(double *)a3.m128_u64, a4, a5);
          if ( (*((_BYTE *)v37 - 1) & 0x40) != 0 )
            v44 = (_QWORD *)*(v37 - 4);
          else
            v44 = &v38[-3 * (*((_DWORD *)v37 - 1) & 0xFFFFFFF)];
          v45 = &v44[v39 / 8];
          if ( *v45 )
          {
            v46 = v45[1];
            v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v47 = v46;
            if ( v46 )
              *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
          }
          *v45 = v43;
          if ( !v43 )
          {
LABEL_48:
            v39 += 24LL;
            if ( v41 == v39 )
              goto LABEL_60;
            continue;
          }
          v48 = *(_QWORD *)(v43 + 8);
          v45[1] = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
          v39 += 24LL;
          v45[2] = (v43 + 8) | v45[2] & 3;
          *(_QWORD *)(v43 + 8) = v45;
          if ( v41 == v39 )
          {
LABEL_60:
            v10 = v40;
            goto LABEL_61;
          }
        }
      }
LABEL_63:
      v49 = *(_DWORD *)(v10 + 256);
      ++*(_QWORD *)(v10 + 240);
      if ( v49 || *(_DWORD *)(v10 + 260) )
      {
        v50 = *(unsigned int *)(v10 + 264);
        v51 = *(_QWORD **)(v10 + 248);
        v178 = 2;
        v179 = 0;
        v52 = 3 * v50;
        v53 = v50;
        v54 = 4 * v49;
        v180 = -8;
        v55 = 2 * v52;
        if ( (unsigned int)(4 * v49) < 0x40 )
          v54 = 64;
        v56 = &v51[v55];
        if ( v53 > v54 )
        {
          v181 = 0;
          v183 = 2;
          v184 = 0;
          v177 = (unsigned __int8 **)&unk_4A01B30;
          v185 = -16;
          v182 = (unsigned __int8 *)&unk_4A01B30;
          j = 0;
          v128 = &v51[v55];
          v129 = v51;
          do
          {
            v130 = v129[3];
            *v129 = &unk_49EE2B0;
            if ( v130 != 0 && v130 != -8 && v130 != -16 )
              sub_1649B30(v129 + 1);
            v129 += 6;
          }
          while ( v128 != v129 );
          v182 = (unsigned __int8 *)&unk_49EE2B0;
          if ( v185 != -8 && v185 != 0 && v185 != -16 )
            sub_1649B30(&v183);
          v177 = (unsigned __int8 **)&unk_49EE2B0;
          if ( v180 != 0 && v180 != -8 && v180 != -16 )
            sub_1649B30(&v178);
          v131 = *(unsigned int *)(v10 + 264);
          if ( v49 )
          {
            v132 = 64;
            if ( v49 != 1 )
            {
              _BitScanReverse(&v133, v49 - 1);
              v132 = 1 << (33 - (v133 ^ 0x1F));
              if ( v132 < 64 )
                v132 = 64;
            }
            v134 = *(_QWORD **)(v10 + 248);
            if ( (_DWORD)v131 == v132 )
            {
              *(_QWORD *)(v10 + 256) = 0;
              v158 = v134;
              v183 = 2;
              v184 = 0;
              v159 = &v134[6 * v131];
              v185 = -8;
              v182 = (unsigned __int8 *)&unk_4A01B30;
              j = 0;
              do
              {
                if ( v158 )
                {
                  v160 = v183;
                  v158[2] = 0;
                  v158[1] = v160 & 6;
                  v161 = v185;
                  v93 = v185 == -8;
                  v158[3] = v185;
                  if ( v161 != 0 && !v93 && v161 != -16 )
                    sub_1649AC0(v158 + 1, v183 & 0xFFFFFFFFFFFFFFF8LL);
                  *v158 = &unk_4A01B30;
                  v158[4] = j;
                }
                v158 += 6;
              }
              while ( v159 != v158 );
              v182 = (unsigned __int8 *)&unk_49EE2B0;
              if ( v185 != -8 && v185 != 0 && v185 != -16 )
                sub_1649B30(&v183);
            }
            else
            {
              v176 = v132;
              j___libc_free_0(*(_QWORD *)(v10 + 248));
              v135 = ((((((((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                        | (4 * v176 / 3u + 1)
                        | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                      | (4 * v176 / 3u + 1)
                      | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 8)
                    | (((((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                      | (4 * v176 / 3u + 1)
                      | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                    | (4 * v176 / 3u + 1)
                    | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 16;
              v136 = (v135
                    | (((((((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                        | (4 * v176 / 3u + 1)
                        | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                      | (4 * v176 / 3u + 1)
                      | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 8)
                    | (((((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                      | (4 * v176 / 3u + 1)
                      | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v176 / 3u + 1) | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1)) >> 2)
                    | (4 * v176 / 3u + 1)
                    | ((unsigned __int64)(4 * v176 / 3u + 1) >> 1))
                   + 1;
              *(_DWORD *)(v10 + 264) = v136;
              v137 = (_QWORD *)sub_22077B0(48 * v136);
              v138 = *(unsigned int *)(v10 + 264);
              *(_QWORD *)(v10 + 256) = 0;
              *(_QWORD *)(v10 + 248) = v137;
              v183 = 2;
              v182 = (unsigned __int8 *)&unk_4A01B30;
              v184 = 0;
              v139 = &v137[6 * v138];
              v185 = -8;
              for ( j = 0; v139 != v137; v137 += 6 )
              {
                if ( v137 )
                {
                  v140 = v183;
                  v137[2] = 0;
                  v137[3] = -8;
                  *v137 = &unk_4A01B30;
                  v137[1] = v140 & 6;
                  v137[4] = j;
                }
              }
            }
          }
          else if ( (_DWORD)v131 )
          {
            j___libc_free_0(*(_QWORD *)(v10 + 248));
            *(_QWORD *)(v10 + 248) = 0;
            *(_QWORD *)(v10 + 256) = 0;
            *(_DWORD *)(v10 + 264) = 0;
          }
          else
          {
            *(_QWORD *)(v10 + 256) = 0;
          }
        }
        else
        {
          v181 = 0;
          v183 = 2;
          v184 = 0;
          v177 = (unsigned __int8 **)&unk_4A01B30;
          v185 = -16;
          v182 = (unsigned __int8 *)&unk_4A01B30;
          j = 0;
          if ( v51 == v56 )
          {
            *(_QWORD *)(v10 + 256) = 0;
          }
          else
          {
            v57 = -8;
            v58 = v51;
            while ( 1 )
            {
              v59 = v58[3];
              if ( v59 != v57 )
              {
                if ( v59 != -8 && v59 != 0 && v59 != -16 )
                {
                  sub_1649B30(v58 + 1);
                  v57 = v180;
                }
                v58[3] = v57;
                if ( v57 != -8 && v57 != 0 && v57 != -16 )
                  sub_1649AC0(v58 + 1, v178 & 0xFFFFFFFFFFFFFFF8LL);
                v58[4] = v181;
              }
              v58 += 6;
              if ( v56 == v58 )
                break;
              v57 = v180;
            }
            v60 = v185;
            *(_QWORD *)(v10 + 256) = 0;
            v182 = (unsigned __int8 *)&unk_49EE2B0;
            if ( v60 != 0 && v60 != -8 && v60 != -16 )
              sub_1649B30(&v183);
          }
          v177 = (unsigned __int8 **)&unk_49EE2B0;
          if ( v180 != 0 && v180 != -8 && v180 != -16 )
            sub_1649B30(&v178);
        }
      }
      if ( *(_BYTE *)(v10 + 304) )
      {
        v61 = *(_DWORD *)(v10 + 296);
        if ( v61 )
        {
          v62 = *(_QWORD **)(v10 + 280);
          v63 = &v62[2 * v61];
          do
          {
            if ( *v62 != -8 && *v62 != -4 )
            {
              v64 = v62[1];
              if ( v64 )
                sub_161E7C0((__int64)(v62 + 1), v64);
            }
            v62 += 2;
          }
          while ( v63 != v62 );
        }
        j___libc_free_0(*(_QWORD *)(v10 + 280));
        *(_BYTE *)(v10 + 304) = 0;
      }
      if ( v187 )
        sub_161E7C0((__int64)&v187, (__int64)v187);
    }
  }
  v187 = 0;
  LODWORD(v190) = 128;
  v65 = (_QWORD *)sub_22077B0(0x2000);
  v189 = 0;
  v188 = (__int64)v65;
  v183 = 2;
  v66 = &v65[8 * (unsigned __int64)(unsigned int)v190];
  v182 = (unsigned __int8 *)&unk_49E6B50;
  v184 = 0;
  v185 = -8;
  for ( j = 0; v66 != v65; v65 += 8 )
  {
    if ( v65 )
    {
      v67 = v183;
      v65[2] = 0;
      v65[3] = -8;
      *v65 = &unk_49E6B50;
      v65[1] = v67 & 6;
      v65[4] = j;
    }
  }
  v68 = *(_DWORD *)(v10 + 176);
  v195 = 0;
  v196 = 1;
  if ( !v68 )
    goto LABEL_103;
  v79 = *(unsigned int *)(v10 + 184);
  v80 = *(_QWORD **)(v10 + 168);
  v81 = v80;
  v82 = &v80[6 * v79];
  if ( v80 == v82 )
    goto LABEL_136;
  while ( 1 )
  {
    v83 = v81[3];
    if ( v83 != -16 && v83 != -8 )
      break;
    v81 += 6;
    if ( v82 == v81 )
      goto LABEL_136;
  }
  if ( v82 != v81 )
  {
    while ( 1 )
    {
      v106 = v81[3];
      v107 = v81[5];
      v183 = 2;
      v184 = 0;
      if ( v106 )
      {
        v185 = v106;
        if ( v106 != -8 && v106 != -16 )
          sub_164C220((__int64)&v183);
      }
      else
      {
        v185 = 0;
      }
      j = &v187;
      v182 = (unsigned __int8 *)&unk_49E6B50;
      if ( !(_DWORD)v190 )
        break;
      v108 = v185;
      v117 = ((_DWORD)v190 - 1) & (((unsigned int)v185 >> 9) ^ ((unsigned int)v185 >> 4));
      v109 = (_QWORD *)(v188 + (v117 << 6));
      v118 = v109[3];
      if ( v185 == v118 )
        goto LABEL_193;
      v119 = 1;
      v120 = 0;
      while ( v118 != -8 )
      {
        if ( !v120 && v118 == -16 )
          v120 = v109;
        LODWORD(v117) = (v190 - 1) & (v119 + v117);
        v109 = (_QWORD *)(v188 + ((unsigned __int64)(unsigned int)v117 << 6));
        v118 = v109[3];
        if ( v185 == v118 )
          goto LABEL_193;
        ++v119;
      }
      if ( v120 )
        v109 = v120;
      ++v187;
      v110 = v189 + 1;
      if ( 4 * ((int)v189 + 1) >= (unsigned int)(3 * v190) )
        goto LABEL_180;
      if ( (int)v190 - HIDWORD(v189) - v110 <= (unsigned int)v190 >> 3 )
      {
        sub_12E48B0((__int64)&v187, v190);
        if ( (_DWORD)v190 )
        {
          v108 = v185;
          v121 = 1;
          v122 = 0;
          LODWORD(v123) = (v190 - 1) & (((unsigned int)v185 >> 9) ^ ((unsigned int)v185 >> 4));
          v109 = (_QWORD *)(v188 + ((unsigned __int64)(unsigned int)v123 << 6));
          v124 = v109[3];
          if ( v185 != v124 )
          {
            while ( v124 != -8 )
            {
              if ( !v122 && v124 == -16 )
                v122 = v109;
              v123 = ((_DWORD)v190 - 1) & (unsigned int)(v123 + v121);
              v109 = (_QWORD *)(v188 + (v123 << 6));
              v124 = v109[3];
              if ( v185 == v124 )
                goto LABEL_182;
              ++v121;
            }
LABEL_221:
            if ( v122 )
              v109 = v122;
          }
LABEL_182:
          v110 = v189 + 1;
          goto LABEL_183;
        }
LABEL_181:
        v108 = v185;
        v109 = 0;
        goto LABEL_182;
      }
LABEL_183:
      LODWORD(v189) = v110;
      v111 = v109[3];
      v112 = v109 + 1;
      if ( v111 == -8 )
      {
        if ( v108 != -8 )
          goto LABEL_188;
      }
      else
      {
        --HIDWORD(v189);
        if ( v108 != v111 )
        {
          if ( v111 && v111 != -16 )
          {
            sub_1649B30(v112);
            v108 = v185;
            v112 = v109 + 1;
          }
LABEL_188:
          v109[3] = v108;
          if ( v108 != -8 && v108 != 0 && v108 != -16 )
            sub_1649AC0(v112, v183 & 0xFFFFFFFFFFFFFFF8LL);
          v108 = v185;
        }
      }
      v113 = j;
      v109[5] = 6;
      v109[6] = 0;
      v109[4] = v113;
      v109[7] = 0;
LABEL_193:
      v182 = (unsigned __int8 *)&unk_49EE2B0;
      if ( v108 != -8 && v108 != 0 && v108 != -16 )
        sub_1649B30(&v183);
      v114 = v109[7];
      v115 = v109 + 5;
      if ( v114 != v107 )
      {
        if ( v114 != -8 && v114 != 0 && v114 != -16 )
        {
          sub_1649B30(v115);
          v115 = v109 + 5;
        }
        v109[7] = v107;
        if ( v107 != -8 && v107 != 0 && v107 != -16 )
          sub_164C220((__int64)v115);
      }
      v81 += 6;
      if ( v81 != v82 )
      {
        while ( 1 )
        {
          v116 = v81[3];
          if ( v116 != -8 && v116 != -16 )
            break;
          v81 += 6;
          if ( v82 == v81 )
            goto LABEL_207;
        }
        if ( v82 != v81 )
          continue;
      }
LABEL_207:
      if ( *(_DWORD *)(v10 + 176) )
      {
        v80 = *(_QWORD **)(v10 + 168);
        v79 = *(unsigned int *)(v10 + 184);
        goto LABEL_136;
      }
LABEL_103:
      if ( v195 )
        goto LABEL_123;
LABEL_104:
      v69 = v190;
      if ( (_DWORD)v190 )
        goto LABEL_106;
      goto LABEL_105;
    }
    ++v187;
LABEL_180:
    sub_12E48B0((__int64)&v187, 2 * v190);
    if ( (_DWORD)v190 )
    {
      v108 = v185;
      v125 = 1;
      v122 = 0;
      LODWORD(v126) = (v190 - 1) & (((unsigned int)v185 >> 9) ^ ((unsigned int)v185 >> 4));
      v109 = (_QWORD *)(v188 + ((unsigned __int64)(unsigned int)v126 << 6));
      v127 = v109[3];
      if ( v127 != v185 )
      {
        while ( v127 != -8 )
        {
          if ( v127 == -16 && !v122 )
            v122 = v109;
          v126 = ((_DWORD)v190 - 1) & (unsigned int)(v126 + v125);
          v109 = (_QWORD *)(v188 + (v126 << 6));
          v127 = v109[3];
          if ( v185 == v127 )
            goto LABEL_182;
          ++v125;
        }
        goto LABEL_221;
      }
      goto LABEL_182;
    }
    goto LABEL_181;
  }
LABEL_136:
  v84 = &v80[6 * v79];
  if ( v80 == v84 )
    goto LABEL_103;
  do
  {
    v85 = v80[3];
    if ( v85 != -8 && v85 != -16 )
    {
      if ( v80 == v84 )
        goto LABEL_103;
      v86 = v84;
      while ( 1 )
      {
        v87 = v80 + 6;
        v88 = (__int64 *)v80[3];
        for ( k = (__int64 ***)v80[5]; v87 != v86; v87 += 6 )
        {
          v90 = v87[3];
          if ( v90 != -16 && v90 != -8 )
            break;
        }
        v183 = 2;
        v184 = 0;
        v185 = -16;
        v182 = (unsigned __int8 *)&unk_49F8530;
        j = 0;
        v91 = v80[3];
        if ( v91 == -16 )
        {
          v80[4] = 0;
          goto LABEL_156;
        }
        if ( !v91 || v91 == -8 )
        {
          v80[3] = -16;
          v94 = j;
          v95 = v185 != -16 && v185 != -8 && v185 != 0;
        }
        else
        {
          nb = (size_t)v80;
          srcb = v80 + 1;
          sub_1649B30(v80 + 1);
          v92 = v185;
          v93 = v185 == 0;
          *(_QWORD *)(nb + 24) = v185;
          if ( v92 == -8 || v93 || v92 == -16 )
          {
            *(_QWORD *)(nb + 32) = j;
            goto LABEL_156;
          }
          sub_1649AC0(srcb, v183 & 0xFFFFFFFFFFFFFFF8LL);
          v94 = j;
          v80 = (_QWORD *)nb;
          v95 = v185 != -16 && v185 != -8 && v185 != 0;
        }
        v80[4] = v94;
        v182 = (unsigned __int8 *)&unk_49EE2B0;
        if ( v95 )
          sub_1649B30(&v183);
LABEL_156:
        --*(_DWORD *)(v10 + 176);
        ++*(_DWORD *)(v10 + 180);
        v96 = sub_15A4A70(k, *v88);
        sub_164D160((__int64)v88, v96, a3, a4, a5, a6, v97, v98, a9, a10);
        v100 = sub_1649960((__int64)v88);
        v101 = v99;
        if ( v100 )
        {
          v177 = (unsigned __int8 **)v99;
          v102 = v99;
          v182 = (unsigned __int8 *)&v184;
          if ( v99 > 0xF )
          {
            nc = v99;
            srcc = v100;
            v104 = (unsigned __int8 *)sub_22409D0(&v182, &v177, 0);
            v100 = srcc;
            v101 = nc;
            v182 = v104;
            v105 = (__int64 *)v104;
            v184 = (__int64)v177;
          }
          else
          {
            if ( v99 == 1 )
            {
              LOBYTE(v184) = *v100;
              v103 = &v184;
LABEL_160:
              v183 = v102;
              *((_BYTE *)v103 + v102) = 0;
              goto LABEL_161;
            }
            if ( !v99 )
            {
              v103 = &v184;
              goto LABEL_160;
            }
            v105 = &v184;
          }
          memcpy(v105, v100, v101);
          v102 = (__int64)v177;
          v103 = (__int64 *)v182;
          goto LABEL_160;
        }
        LOBYTE(v184) = 0;
        v183 = 0;
        v182 = (unsigned __int8 *)&v184;
LABEL_161:
        sub_15E55B0((__int64)v88);
        LOWORD(v179) = 260;
        v177 = &v182;
        sub_164B780((__int64)k, (__int64 *)&v177);
        if ( v182 != (unsigned __int8 *)&v184 )
          j_j___libc_free_0(v182, v184 + 1);
        if ( v87 == v86 )
          goto LABEL_103;
        v80 = v87;
      }
    }
    v80 += 6;
  }
  while ( v84 != v80 );
  if ( !v195 )
    goto LABEL_104;
LABEL_123:
  if ( (_DWORD)v194 )
  {
    v76 = v192;
    v77 = &v192[2 * (unsigned int)v194];
    do
    {
      if ( *v76 != -8 && *v76 != -4 )
      {
        v78 = v76[1];
        if ( v78 )
          sub_161E7C0((__int64)(v76 + 1), v78);
      }
      v76 += 2;
    }
    while ( v77 != v76 );
  }
  j___libc_free_0(v192);
  v69 = v190;
  if ( (_DWORD)v190 )
  {
LABEL_106:
    v71 = (_QWORD *)v188;
    v178 = 2;
    v179 = 0;
    v72 = (_QWORD *)(v188 + ((unsigned __int64)v69 << 6));
    v180 = -8;
    v177 = (unsigned __int8 **)&unk_49E6B50;
    v182 = (unsigned __int8 *)&unk_49E6B50;
    v73 = -8;
    v181 = 0;
    v183 = 2;
    v184 = 0;
    v185 = -16;
    j = 0;
    while ( 1 )
    {
      v74 = v71[3];
      if ( v73 != v74 && v74 != v185 )
      {
        v75 = v71[7];
        if ( v75 != 0 && v75 != -8 && v75 != -16 )
        {
          sub_1649B30(v71 + 5);
          v74 = v71[3];
        }
      }
      *v71 = &unk_49EE2B0;
      if ( v74 != 0 && v74 != -8 && v74 != -16 )
        sub_1649B30(v71 + 1);
      v71 += 8;
      if ( v72 == v71 )
        break;
      v73 = v180;
    }
    v182 = (unsigned __int8 *)&unk_49EE2B0;
    if ( v185 != 0 && v185 != -8 && v185 != -16 )
      sub_1649B30(&v183);
    v177 = (unsigned __int8 **)&unk_49EE2B0;
    if ( v180 != -8 && v180 != 0 && v180 != -16 )
      sub_1649B30(&v178);
  }
LABEL_105:
  j___libc_free_0(v188);
  return 1;
}
