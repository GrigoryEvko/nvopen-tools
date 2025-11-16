// Function: sub_259C9D0
// Address: 0x259c9d0
//
__int64 __fastcall sub_259C9D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        char a6,
        __int64 (__fastcall *a7)(__int64, __int64 *, _QWORD),
        __int64 a8,
        bool *a9,
        __int64 *a10,
        unsigned __int8 (__fastcall *a11)(__int64, __int64 *),
        __int64 a12)
{
  __int64 v14; // rbx
  char v15; // r14
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r14
  unsigned __int8 *v19; // rax
  __int64 v20; // rax
  char v21; // cl
  __int64 v22; // rax
  _BYTE *v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // r12
  __int64 v26; // rax
  unsigned __int8 v27; // al
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // edx
  __int64 *v31; // rcx
  __int64 v32; // r8
  unsigned int *v33; // rsi
  __int64 v34; // r14
  unsigned int *v35; // rdi
  __int64 v36; // rdx
  const __m128i *v37; // rax
  const __m128i *v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 *v43; // rdx
  __int64 v44; // rdx
  char *v45; // rax
  char *v46; // r15
  char *v47; // r12
  __int64 v48; // r14
  __int64 **v49; // r12
  unsigned int v50; // eax
  __int64 *v51; // rbx
  char v52; // r13
  __int64 v53; // rdx
  __int64 v54; // rcx
  char v55; // r8
  unsigned int v56; // r13d
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // r9
  char v61; // r15
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 v64; // r13
  char *v65; // r14
  __int64 v66; // r15
  char *v67; // rax
  __m128i v68; // xmm2
  __m128i v69; // xmm0
  __int64 v70; // rcx
  __int64 *v71; // rsi
  __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rbx
  bool v77; // r13
  __int64 v78; // r12
  unsigned __int8 v79; // bl
  unsigned int *v80; // rax
  __int64 v81; // r15
  __int64 v82; // rax
  __int64 v83; // rcx
  int v84; // eax
  __int64 v85; // rdx
  int v86; // edx
  __int64 v87; // rax
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  unsigned __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // rax
  __int64 v94; // r12
  __m128i v95; // xmm3
  __m128i v96; // xmm0
  int v97; // ecx
  int v98; // r9d
  __int64 *v99; // rax
  __int64 v100; // [rsp-8h] [rbp-2F8h]
  __int64 v101; // [rsp+18h] [rbp-2D8h]
  __int64 v102; // [rsp+20h] [rbp-2D0h]
  __int64 v103; // [rsp+28h] [rbp-2C8h]
  __int64 v104; // [rsp+30h] [rbp-2C0h]
  __int64 v105; // [rsp+30h] [rbp-2C0h]
  __int64 v106; // [rsp+38h] [rbp-2B8h]
  __int64 v107; // [rsp+38h] [rbp-2B8h]
  _QWORD *v108; // [rsp+38h] [rbp-2B8h]
  __int64 v109; // [rsp+40h] [rbp-2B0h]
  __int64 v110; // [rsp+40h] [rbp-2B0h]
  char v111; // [rsp+48h] [rbp-2A8h]
  __int64 v112; // [rsp+48h] [rbp-2A8h]
  unsigned __int8 (__fastcall *v113)(__int64, __int64, __int64, __int64, __m128i *); // [rsp+48h] [rbp-2A8h]
  char v114; // [rsp+48h] [rbp-2A8h]
  char v115; // [rsp+48h] [rbp-2A8h]
  __int64 v116; // [rsp+58h] [rbp-298h]
  __int64 v117; // [rsp+58h] [rbp-298h]
  char v118; // [rsp+60h] [rbp-290h]
  __int64 v120; // [rsp+90h] [rbp-260h]
  char v121; // [rsp+A0h] [rbp-250h]
  __int64 **v122; // [rsp+A0h] [rbp-250h]
  __int64 v124; // [rsp+B0h] [rbp-240h]
  unsigned __int8 v125; // [rsp+B9h] [rbp-237h]
  char v126; // [rsp+BAh] [rbp-236h]
  char v128; // [rsp+BCh] [rbp-234h] BYREF
  char v129; // [rsp+C2h] [rbp-22Eh] BYREF
  char v130; // [rsp+C3h] [rbp-22Dh] BYREF
  char v131; // [rsp+C4h] [rbp-22Ch] BYREF
  char v132; // [rsp+C5h] [rbp-22Bh] BYREF
  char v133; // [rsp+C6h] [rbp-22Ah] BYREF
  char v134; // [rsp+C7h] [rbp-229h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-228h] BYREF
  __m128i v136; // [rsp+D0h] [rbp-220h] BYREF
  __int64 (__fastcall *v137)(__int64 *, __m128i *, int); // [rsp+E0h] [rbp-210h]
  void *v138; // [rsp+E8h] [rbp-208h]
  _QWORD v139[2]; // [rsp+F0h] [rbp-200h] BYREF
  __int64 *v140; // [rsp+100h] [rbp-1F0h]
  __int64 *v141; // [rsp+108h] [rbp-1E8h]
  __int64 v142; // [rsp+110h] [rbp-1E0h] BYREF
  __int64 v143; // [rsp+118h] [rbp-1D8h]
  __int64 (__fastcall *v144)(__int64 *, __m128i *, int); // [rsp+120h] [rbp-1D0h]
  void *v145; // [rsp+128h] [rbp-1C8h]
  char v146; // [rsp+130h] [rbp-1C0h]
  __m128i v147; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v148; // [rsp+150h] [rbp-1A0h]
  void *v149; // [rsp+158h] [rbp-198h]
  char v150; // [rsp+160h] [rbp-190h] BYREF
  __m128i v151; // [rsp+180h] [rbp-170h] BYREF
  __int64 v152; // [rsp+190h] [rbp-160h]
  __int64 *v153; // [rsp+198h] [rbp-158h]
  __int64 v154; // [rsp+1A0h] [rbp-150h]
  __int64 v155; // [rsp+1A8h] [rbp-148h]
  char *v156; // [rsp+1B0h] [rbp-140h]
  char *v157; // [rsp+1B8h] [rbp-138h]
  char *v158; // [rsp+1C0h] [rbp-130h]
  __int64 v159; // [rsp+1D0h] [rbp-120h] BYREF
  char *v160; // [rsp+1D8h] [rbp-118h]
  __int64 v161; // [rsp+1E0h] [rbp-110h]
  int v162; // [rsp+1E8h] [rbp-108h]
  char v163; // [rsp+1ECh] [rbp-104h]
  char v164; // [rsp+1F0h] [rbp-100h] BYREF
  __int64 **v165; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v166; // [rsp+238h] [rbp-B8h]
  _BYTE v167[176]; // [rsp+240h] [rbp-B0h] BYREF

  v14 = a2;
  v128 = a5;
  v159 = 0;
  v161 = 8;
  *a9 = 0;
  v160 = &v164;
  v165 = (__int64 **)v167;
  v162 = 0;
  v163 = 1;
  v166 = 0x800000000LL;
  v124 = sub_B43CB0(a4);
  sub_250D230((unsigned __int64 *)&v151, v124, 4, 0);
  v15 = sub_259B8C0(a2, a3, &v151, 1, &v129, 0, 0);
  sub_250D230((unsigned __int64 *)&v151, v124, 4, 0);
  v16 = sub_2567630(a2, &v151, a3, 2, 0);
  v130 = v15;
  v135 = v16;
  if ( v16 )
  {
    v17 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v16 + 112LL))(v16, *(_QWORD *)(a4 + 40));
    v131 = v17;
    if ( v135 && a6 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v135 + 120LL))(v135, a2, a4) )
      {
        v132 = 1;
LABEL_7:
        sub_250ED80(a2, v135, a3, 1);
        goto LABEL_8;
      }
      v132 = 0;
      v17 = v131;
    }
    else
    {
      v132 = 0;
    }
    if ( !v17 )
      goto LABEL_8;
    goto LABEL_7;
  }
  v131 = 0;
  v132 = 0;
LABEL_8:
  v18 = *(_QWORD *)(a2 + 208);
  v19 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  v133 = sub_25282F0(a2, v19, a1);
  v151.m128i_i64[0] = (__int64)&v133;
  v151.m128i_i64[1] = (__int64)&v130;
  v153 = &v135;
  v152 = v124;
  v155 = a3;
  v156 = &v132;
  v157 = &v128;
  v158 = &v131;
  v154 = a2;
  sub_250D230((unsigned __int64 *)&v147, v124, 4, 0);
  sub_259B4A0(a2, a1, &v147, 1, &v134, 0, 0);
  v20 = sub_251B1C0(*(_QWORD *)(a2 + 208), v124);
  v21 = v128;
  v121 = *(_BYTE *)(v20 + 114);
  if ( v128 )
    v21 = v134;
  v126 = v21;
  v22 = sub_2554D30(*(_QWORD *)(v18 + 240), v124, 0);
  v137 = 0;
  v120 = v22;
  v23 = (_BYTE *)sub_250D070((_QWORD *)(a1 + 72));
  if ( *v23 == 60 )
  {
    v94 = sub_B43CB0((__int64)v23);
    v118 = *(_BYTE *)(sub_251B1C0(*(_QWORD *)(a2 + 208), v94) + 114);
    sub_250D230((unsigned __int64 *)&v147, v94, 4, 0);
    if ( (unsigned __int8)sub_259B4A0(a2, a1, &v147, 1, &v142, 0, 0) )
    {
      v147.m128i_i64[0] = v94;
      v95 = _mm_load_si128(&v136);
      v96 = _mm_load_si128(&v147);
      v148 = (__int64)v137;
      v137 = (__int64 (__fastcall *)(__int64 *, __m128i *, int))sub_2535B10;
      v147 = v95;
      v149 = v138;
      v138 = sub_2535290;
      v136 = v96;
      sub_A17130((__int64)&v147);
    }
  }
  else
  {
    v24 = sub_250D070((_QWORD *)(a1 + 72));
    v25 = v24;
    if ( *(_BYTE *)v24 > 3u )
      goto LABEL_16;
    v118 = sub_250C0F0(*(_QWORD **)(v24 + 40));
    if ( !v118 )
      goto LABEL_16;
    v26 = *(_QWORD *)(v25 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
      v26 = **(_QWORD **)(v26 + 16);
    if ( (unsigned int)((*(_DWORD *)(v26 + 8) >> 8) - 3) <= 2 )
    {
      v147.m128i_i64[0] = a2;
      v68 = _mm_load_si128(&v136);
      v69 = _mm_load_si128(&v147);
      v148 = (__int64)v137;
      v137 = (__int64 (__fastcall *)(__int64 *, __m128i *, int))sub_2535B40;
      v147 = v68;
      v149 = v138;
      v138 = sub_2568BA0;
      v136 = v69;
      sub_A17130((__int64)&v147);
    }
    else
    {
LABEL_16:
      v118 = 0;
    }
  }
  v147.m128i_i64[0] = 0;
  v147.m128i_i64[1] = (__int64)&v150;
  v27 = *(_BYTE *)(a1 + 393);
  v148 = 4;
  LODWORD(v149) = 0;
  BYTE4(v149) = 1;
  v125 = v27;
  if ( !v27 || *(_QWORD *)(a1 + 376) || *(_DWORD *)(a1 + 296) )
    goto LABEL_83;
  v28 = *(unsigned int *)(a1 + 280);
  v29 = *(_QWORD *)(a1 + 264);
  if ( !(_DWORD)v28 )
    goto LABEL_46;
  v30 = (v28 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v31 = (__int64 *)(v29 + 72LL * v30);
  v32 = *v31;
  if ( a4 != *v31 )
  {
    v97 = 1;
    while ( v32 != -4096 )
    {
      v98 = v97 + 1;
      v30 = (v28 - 1) & (v97 + v30);
      v31 = (__int64 *)(v29 + 72LL * v30);
      v32 = *v31;
      if ( a4 == *v31 )
        goto LABEL_22;
      v97 = v98;
    }
    goto LABEL_46;
  }
LABEL_22:
  if ( v31 == (__int64 *)(v29 + 72 * v28) )
    goto LABEL_46;
  v33 = (unsigned int *)v31[1];
  v34 = *a10;
  v35 = &v33[*((unsigned int *)v31 + 4)];
  if ( v33 == v35 )
    goto LABEL_43;
  do
  {
    v36 = *(_QWORD *)(a1 + 96) + 112LL * *v33;
    v37 = *(const __m128i **)(v36 + 32);
    v38 = &v37[*(unsigned int *)(v36 + 40)];
    if ( v37 != v38 )
    {
      while ( 1 )
      {
        v42 = v37->m128i_i64[0];
        if ( v37->m128i_i64[0] == 0xFFFFFFFF80000000LL )
          goto LABEL_39;
        if ( v34 == 0xFFFFFFFF80000000LL )
          break;
        if ( v42 == 0x7FFFFFFF || v34 == 0x7FFFFFFF )
        {
          v58 = a10[1];
          *a10 = 0x7FFFFFFF;
          if ( v58 == 0x7FFFFFFF || (v59 = v37->m128i_i64[1], v59 == 0x7FFFFFFF) )
          {
            a10[1] = 0x7FFFFFFF;
            v34 = 0x7FFFFFFF;
            goto LABEL_41;
          }
          v34 = 0x7FFFFFFF;
          if ( v58 < v59 )
            v58 = v37->m128i_i64[1];
          a10[1] = v58;
        }
        else
        {
          v39 = a10[1];
          if ( v39 == 0x7FFFFFFF || v37->m128i_i64[1] == 0x7FFFFFFF )
          {
            a10[1] = 0x7FFFFFFF;
            if ( v34 > v37->m128i_i64[0] )
              v34 = v37->m128i_i64[0];
            *a10 = v34;
            if ( v34 == 0x7FFFFFFF )
              goto LABEL_41;
          }
          else
          {
            if ( v34 > v42 )
              v34 = v37->m128i_i64[0];
            *a10 = v34;
            v40 = v34 + v39;
            v41 = v37->m128i_i64[0] + v37->m128i_i64[1];
            if ( v41 < v40 )
              v41 = v40;
            a10[1] = v41 - v34;
          }
        }
LABEL_35:
        if ( v38 == ++v37 )
          goto LABEL_41;
      }
      v34 = v37->m128i_i64[0];
      *(__m128i *)a10 = _mm_loadu_si128(v37);
LABEL_39:
      if ( v34 == 0x7FFFFFFF && a10[1] == 0x7FFFFFFF )
        goto LABEL_41;
      goto LABEL_35;
    }
LABEL_41:
    ++v33;
  }
  while ( v35 != v33 );
  if ( !*(_BYTE *)(a1 + 393) )
  {
LABEL_83:
    v56 = 0;
    goto LABEL_84;
  }
LABEL_43:
  if ( *(_QWORD *)(a1 + 376) || *(_DWORD *)(a1 + 296) )
    goto LABEL_83;
  v43 = *(__int64 **)(a1 + 232);
  if ( *(_DWORD *)(a1 + 240) )
  {
    v70 = a10[1];
    v141 = &v43[12 * *(unsigned int *)(a1 + 248)];
    v140 = v43;
    v101 = v70;
    v139[0] = a1 + 224;
    v139[1] = *(_QWORD *)(a1 + 224);
    sub_255DC40((__int64)v139);
    v71 = v140;
    v102 = *(_QWORD *)(a1 + 232) + 96LL * *(unsigned int *)(a1 + 248);
    if ( (__int64 *)v102 != v140 )
    {
      v117 = a1;
      v103 = v14;
      v105 = v34;
      v72 = v109;
      while ( 1 )
      {
        v73 = *v71;
        v74 = v71[1];
        if ( v105 == 0x7FFFFFFF || v101 == 0x7FFFFFFF )
        {
LABEL_143:
          v77 = 0;
          if ( v73 == v105 )
            v77 = v101 == v74 && v101 != 0x7FFFFFFF && v105 != 0x7FFFFFFF;
          goto LABEL_145;
        }
        if ( v73 == 0x7FFFFFFF )
          break;
        if ( v74 == 0x7FFFFFFF || v73 + v74 > v105 && v73 < v105 + v101 )
          goto LABEL_143;
LABEL_137:
        v71 = v141;
        v75 = v140 + 12;
        v140 = v75;
        if ( v75 != v141 )
        {
          while ( 1 )
          {
            v76 = 0x7FFFFFFFFFFFFFFFLL;
            if ( *v75 != 0x7FFFFFFFFFFFFFFFLL )
            {
              v76 = 0x7FFFFFFFFFFFFFFELL;
              if ( *v75 != 0x7FFFFFFFFFFFFFFELL )
                break;
            }
            if ( v75[1] != v76 )
              break;
            v75 += 12;
            v140 = v75;
            if ( v75 == v141 )
              goto LABEL_141;
          }
          v71 = v140;
        }
LABEL_141:
        if ( (__int64 *)v102 == v71 )
        {
          v14 = v103;
          goto LABEL_46;
        }
      }
      v77 = 0;
LABEL_145:
      if ( v71[11] )
      {
        v78 = v71[9];
        v79 = 0;
        v110 = (__int64)(v71 + 7);
      }
      else
      {
        v78 = v71[2];
        v79 = v125;
        v110 = v78 + 4LL * *((unsigned int *)v71 + 6);
      }
      if ( !v79 )
      {
LABEL_148:
        if ( v78 != v110 )
        {
          v80 = (unsigned int *)(v78 + 32);
          goto LABEL_150;
        }
        goto LABEL_137;
      }
      while ( 1 )
      {
        if ( v78 == v110 )
          goto LABEL_137;
        v80 = (unsigned int *)v78;
LABEL_150:
        v81 = *(_QWORD *)(v117 + 96) + 112LL * *v80;
        v82 = sub_B43CB0(*(_QWORD *)(v81 + 8));
        v83 = v82;
        if ( !v121
          || v124 == v82
          || !v118
          || (v106 = v82, v93 = sub_251B1C0(*(_QWORD *)(v103 + 208), v82), v83 = v106, !*(_BYTE *)(v93 + 114)) )
        {
          if ( v77 )
          {
            v84 = *(_DWORD *)(v81 + 96);
            if ( (v84 & 1) != 0 )
            {
              v85 = *(_QWORD *)(v81 + 8);
              if ( a4 != v85 && ((v84 & 8) != 0 || *(_BYTE *)a4 == 61 && v84 == 17) )
              {
                v107 = v83;
                sub_BED950((__int64)&v142, (__int64)&v147, v85);
                v83 = v107;
              }
            }
          }
          if ( v128 )
          {
            v86 = *(_DWORD *)(v81 + 96);
            if ( (v86 & 8) != 0 || v86 == 17 )
              goto LABEL_201;
          }
          if ( !a6 )
            goto LABEL_169;
          v86 = *(_DWORD *)(v81 + 96);
          if ( (v86 & 4) == 0 )
            goto LABEL_169;
          if ( v128 )
          {
LABEL_201:
            if ( v120
              && v77
              && v124 == v83
              && (v86 & 1) != 0
              && (unsigned __int8)sub_B19DB0(v120, *(_QWORD *)(v81 + 8), a4) )
            {
              v108 = sub_AE6EC0((__int64)&v159, v81);
              v143 = sub_254BB00((__int64)&v159);
              v142 = (__int64)v108;
              sub_254BBF0((__int64)&v142);
            }
          }
          LOBYTE(v72) = v77;
          v87 = sub_B43CB0(*(_QWORD *)(v81 + 8));
          v130 &= v124 == v87;
          v90 = (unsigned int)v166;
          v91 = (unsigned int)v166 + 1LL;
          if ( v91 > HIDWORD(v166) )
          {
            sub_C8D5F0((__int64)&v165, v167, v91, 0x10u, v88, v89);
            v90 = (unsigned int)v166;
          }
          v92 = (__int64 *)&v165[2 * v90];
          *v92 = v81;
          v92[1] = v72;
          LODWORD(v166) = v166 + 1;
        }
LABEL_169:
        if ( !v79 )
        {
          v78 = sub_220EF30(v78);
          goto LABEL_148;
        }
        v78 += 4;
      }
    }
  }
LABEL_46:
  v44 = HIDWORD(v161);
  *a9 = v162 != HIDWORD(v161);
  v45 = v160;
  if ( !v163 )
    v44 = (unsigned int)v161;
  v46 = &v160[8 * v44];
  if ( v160 == v46 )
    goto LABEL_51;
  while ( 1 )
  {
    v47 = v45;
    if ( *(_QWORD *)v45 < 0xFFFFFFFFFFFFFFFELL )
      break;
    v45 += 8;
    if ( v46 == v45 )
      goto LABEL_51;
  }
  if ( v45 == v46 )
  {
LABEL_51:
    v48 = 0;
  }
  else
  {
    v64 = 0;
    v65 = &v160[8 * v44];
    v66 = *(_QWORD *)v45;
    do
    {
      if ( !v64 || (unsigned __int8)sub_B19DB0(v120, v64, *(_QWORD *)(v66 + 8)) )
        v64 = *(_QWORD *)(v66 + 8);
      v67 = v47 + 8;
      if ( v47 + 8 == v65 )
        break;
      while ( 1 )
      {
        v66 = *(_QWORD *)v67;
        v47 = v67;
        if ( *(_QWORD *)v67 < 0xFFFFFFFFFFFFFFFELL )
          break;
        v67 += 8;
        if ( v65 == v67 )
          goto LABEL_123;
      }
    }
    while ( v67 != v65 );
LABEL_123:
    v48 = v64;
  }
  v49 = v165;
  v122 = &v165[2 * (unsigned int)v166];
  if ( v122 != v165 )
  {
    v116 = v14;
    while ( 1 )
    {
      if ( !v130 && !v133 && !v135 )
        goto LABEL_60;
      v51 = *v49;
      if ( a11 && a11(a12, *v49) )
        goto LABEL_61;
      if ( !*(_BYTE *)v151.m128i_i64[0]
        && !*(_BYTE *)v151.m128i_i64[1]
        && !(unsigned __int8)sub_2567770((__int64)&v151, v51[1])
        && (v51[1] == *v51
         || !*(_BYTE *)v151.m128i_i64[0]
         && !*(_BYTE *)v151.m128i_i64[1]
         && !(unsigned __int8)sub_2567770((__int64)&v151, *v51)) )
      {
        goto LABEL_60;
      }
      v52 = v128;
      if ( a6 )
      {
        v144 = 0;
        if ( v137 )
        {
          v137(&v142, &v136, 2);
          v145 = v138;
          v144 = v137;
        }
        v114 = sub_2529340(v116, a4, v51[1], a3, (__int64)&v147, (__int64)&v142);
        sub_A17130((__int64)&v142);
        v55 = v114 ^ 1;
        if ( !v52 )
          goto LABEL_73;
        v52 = v114 ^ 1;
      }
      else if ( !v128 )
      {
        goto LABEL_61;
      }
      v144 = 0;
      if ( v137 )
      {
        v137(&v142, &v136, 2);
        v145 = v138;
        v144 = v137;
      }
      v111 = sub_2529340(v116, v51[1], a4, a3, (__int64)&v147, (__int64)&v142);
      sub_A17130((__int64)&v142);
      v55 = v52;
      if ( !v111 )
        goto LABEL_73;
      if ( *a9 && v124 != sub_B43CB0(v51[1]) )
      {
        sub_250D230((unsigned __int64 *)&v142, v124, 4, 0);
        v60 = sub_25289A0(v116, v142, v143, a3, 1, 0, 1);
        v53 = v100;
        v112 = v60;
        if ( v60 )
        {
          sub_BED950((__int64)&v142, (__int64)&v147, a4);
          v61 = v146;
          v104 = v112;
          v113 = *(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __m128i *))(*(_QWORD *)v112
                                                                                                 + 112LL);
          v62 = sub_B43CB0(v51[1]);
          if ( !v113(v104, v116, v48, v62, &v147) )
          {
            v55 = v52;
            if ( v61 )
              goto LABEL_105;
            goto LABEL_73;
          }
          if ( v61 )
          {
            v55 = 0;
LABEL_105:
            if ( BYTE4(v149) )
            {
              v54 = v147.m128i_i64[1] + 8LL * HIDWORD(v148);
              v53 = HIDWORD(v148);
              if ( v147.m128i_i64[1] != v54 )
              {
                v63 = (__int64 *)v147.m128i_i64[1];
                while ( a4 != *v63 )
                {
                  if ( (__int64 *)v54 == ++v63 )
                    goto LABEL_73;
                }
                --HIDWORD(v148);
                v53 = *(_QWORD *)(v147.m128i_i64[1] + 8LL * HIDWORD(v148));
                *v63 = v53;
                ++v147.m128i_i64[0];
              }
            }
            else
            {
              v115 = v55;
              v99 = sub_C8CA60((__int64)&v147, a4);
              v55 = v115;
              if ( v99 )
              {
                *v99 = -2;
                LODWORD(v149) = (_DWORD)v149 + 1;
                ++v147.m128i_i64[0];
              }
            }
LABEL_73:
            if ( v55 )
              goto LABEL_61;
          }
        }
      }
      if ( v120 && v126 && (unsigned __int8)sub_B19060((__int64)&v159, (__int64)v51, v53, v54) && v51[1] != v48 )
      {
        v49 += 2;
        if ( v122 == v49 )
          break;
      }
      else
      {
LABEL_60:
        v50 = a7(a8, *v49, *((unsigned __int8 *)v49 + 8));
        if ( !(_BYTE)v50 )
        {
          v56 = v50;
          goto LABEL_80;
        }
LABEL_61:
        v49 += 2;
        if ( v122 == v49 )
          break;
      }
    }
  }
  v56 = v125;
LABEL_80:
  if ( !BYTE4(v149) )
    _libc_free(v147.m128i_u64[1]);
LABEL_84:
  sub_A17130((__int64)&v136);
  if ( v165 != (__int64 **)v167 )
    _libc_free((unsigned __int64)v165);
  if ( !v163 )
    _libc_free((unsigned __int64)v160);
  return v56;
}
