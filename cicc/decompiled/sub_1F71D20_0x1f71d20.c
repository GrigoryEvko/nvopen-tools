// Function: sub_1F71D20
// Address: 0x1f71d20
//
__int64 *__fastcall sub_1F71D20(
        _QWORD *a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        _DWORD *a5,
        __m128 a6,
        double a7,
        __m128i a8)
{
  unsigned __int64 v8; // r11
  __int64 *result; // rax
  int v10; // r9d
  bool v11; // zf
  _QWORD *v12; // rcx
  unsigned int v13; // r15d
  unsigned __int64 v15; // rax
  char v16; // dl
  __int16 v17; // ax
  bool v18; // cc
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rbx
  unsigned __int64 v24; // r12
  __int64 *v25; // rax
  __int64 *v26; // rsi
  __int64 *v27; // rcx
  unsigned __int64 v28; // rdi
  __int64 **v29; // r14
  __int64 **v30; // rdi
  __int64 v31; // rdx
  const __m128i *v32; // r14
  __int64 v33; // rax
  __int8 *v34; // rdx
  __int64 v35; // rax
  char v36; // al
  char v37; // di
  __int64 v38; // rax
  int v39; // eax
  unsigned __int64 v40; // r11
  char v41; // di
  __int64 v42; // rax
  int v43; // eax
  unsigned __int64 v44; // r11
  char v45; // al
  int v46; // r10d
  bool v47; // al
  bool v48; // dl
  int v49; // r14d
  int v50; // edx
  int v51; // edi
  int v52; // edi
  int v53; // esi
  bool v54; // si
  unsigned __int8 v55; // di
  bool v56; // cl
  char v57; // cl
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // r14
  unsigned int v62; // edx
  __int64 v63; // rcx
  __int64 v64; // rdi
  unsigned __int64 v65; // r10
  _QWORD *v66; // rax
  _DWORD *v67; // rdi
  const __m128i *v68; // rbx
  __int64 v69; // r15
  __int64 *v70; // r13
  __int64 **v71; // r14
  __int64 v72; // rsi
  __int64 v73; // r12
  const __m128i *v74; // r14
  __int64 v75; // rax
  __int64 **v76; // rdx
  __int64 v77; // rdi
  char v78; // al
  __int64 v79; // rdi
  __int64 *v80; // rdx
  _QWORD *v81; // rcx
  unsigned __int64 v82; // rax
  __int64 v83; // r10
  __int64 v84; // r14
  __int64 v85; // rsi
  __int64 v86; // r8
  __int64 v87; // r10
  __int64 v88; // rcx
  __int64 v89; // rsi
  __int64 v90; // r10
  __int64 v91; // rcx
  __int64 v92; // rdx
  unsigned __int64 v93; // rax
  char v94; // al
  __int64 v95; // rax
  _DWORD *v96; // r10
  __int128 v97; // [rsp-10h] [rbp-330h]
  unsigned __int64 v98; // [rsp+0h] [rbp-320h]
  unsigned __int64 v99; // [rsp+0h] [rbp-320h]
  unsigned int v100; // [rsp+10h] [rbp-310h]
  unsigned __int64 v101; // [rsp+10h] [rbp-310h]
  char v102; // [rsp+1Fh] [rbp-301h]
  __int64 *v106; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v107; // [rsp+48h] [rbp-2D8h]
  unsigned int v108; // [rsp+48h] [rbp-2D8h]
  __int64 *v109; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v110; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v111; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v112; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v113; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v114; // [rsp+48h] [rbp-2D8h]
  __int64 v115; // [rsp+58h] [rbp-2C8h] BYREF
  _QWORD v116[6]; // [rsp+60h] [rbp-2C0h] BYREF
  unsigned __int64 v117; // [rsp+90h] [rbp-290h] BYREF
  __int64 v118; // [rsp+98h] [rbp-288h]
  __int64 v119; // [rsp+A0h] [rbp-280h]
  __int64 v120; // [rsp+A8h] [rbp-278h]
  __int64 v121; // [rsp+B0h] [rbp-270h]
  _QWORD v122[2]; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v123; // [rsp+D0h] [rbp-250h]
  int v124; // [rsp+D8h] [rbp-248h]
  __int64 v125; // [rsp+F0h] [rbp-230h] BYREF
  __int64 v126; // [rsp+F8h] [rbp-228h]
  __int64 v127; // [rsp+100h] [rbp-220h]
  int v128; // [rsp+108h] [rbp-218h]
  __int64 **v129; // [rsp+120h] [rbp-200h] BYREF
  __int64 v130; // [rsp+128h] [rbp-1F8h]
  _BYTE v131[128]; // [rsp+130h] [rbp-1F0h] BYREF
  _QWORD *v132; // [rsp+1B0h] [rbp-170h] BYREF
  unsigned int v133; // [rsp+1B8h] [rbp-168h]
  unsigned int v134; // [rsp+1BCh] [rbp-164h]
  _QWORD v135[16]; // [rsp+1C0h] [rbp-160h] BYREF
  __int64 v136; // [rsp+240h] [rbp-E0h] BYREF
  __int64 *v137; // [rsp+248h] [rbp-D8h]
  __int64 *v138; // [rsp+250h] [rbp-D0h]
  __int64 v139; // [rsp+258h] [rbp-C8h]
  int v140; // [rsp+260h] [rbp-C0h]
  _BYTE v141[184]; // [rsp+268h] [rbp-B8h] BYREF

  result = a3;
  v10 = *((_DWORD *)a1 + 5);
  if ( !v10 )
    return result;
  v134 = 8;
  v129 = (__int64 **)v131;
  v130 = 0x800000000LL;
  v132 = v135;
  v137 = (__int64 *)v141;
  v138 = (__int64 *)v141;
  v136 = 0;
  v11 = *(_WORD *)(a2 + 24) == 185;
  v102 = 0;
  v139 = 16;
  v140 = 0;
  if ( v11 )
    v102 = ((*(_BYTE *)(a2 + 26) >> 3) ^ 1) & 1;
  v12 = v135;
  v13 = 0;
  v133 = 1;
  v135[0] = a3;
  v135[1] = a4;
  LODWORD(v15) = 1;
  while ( 1 )
  {
    v21 = (__int64)&v12[2 * (unsigned int)v15 - 2];
    v22 = *(unsigned int *)(v21 + 8);
    v23 = *(_QWORD *)v21;
    v133 = v15 - 1;
    v24 = v8 & 0xFFFFFFFF00000000LL | v22;
    v8 = v24;
    if ( *(_DWORD *)(a1[1] + 81496LL) < v13 )
      break;
    v25 = v137;
    if ( v138 != v137 )
      goto LABEL_6;
    v26 = &v137[HIDWORD(v139)];
    if ( v137 != v26 )
    {
      v27 = 0;
      while ( v23 != *v25 )
      {
        if ( *v25 == -2 )
          v27 = v25;
        if ( v26 == ++v25 )
        {
          if ( !v27 )
            goto LABEL_95;
          *v27 = v23;
          --v140;
          ++v136;
          v17 = *(_WORD *)(v23 + 24);
          v18 = v17 <= 47;
          if ( v17 != 47 )
            goto LABEL_8;
          goto LABEL_83;
        }
      }
      goto LABEL_25;
    }
LABEL_95:
    if ( HIDWORD(v139) < (unsigned int)v139 )
    {
      ++HIDWORD(v139);
      *v26 = v23;
      ++v136;
    }
    else
    {
LABEL_6:
      sub_16CCBA0((__int64)&v136, v23);
      v8 = v24;
      if ( !v16 )
        goto LABEL_25;
    }
    v17 = *(_WORD *)(v23 + 24);
    v18 = v17 <= 47;
    if ( v17 == 47 )
    {
LABEL_83:
      v68 = *(const __m128i **)(v23 + 32);
      v15 = v133;
      if ( v133 >= v134 )
      {
        v113 = v8;
        sub_16CD150((__int64)&v132, v135, 0, 16, (int)a5, v10);
        v15 = v133;
        v8 = v113;
      }
      a6 = (__m128)_mm_loadu_si128(v68);
      goto LABEL_86;
    }
LABEL_8:
    if ( !v18 )
    {
      if ( (unsigned __int16)(v17 - 185) > 1u )
        goto LABEL_12;
      v31 = 80;
      if ( v17 == 185 )
      {
        if ( (*(_BYTE *)(v23 + 26) & 8) != 0 || !v102 )
        {
          v31 = 40;
          goto LABEL_36;
        }
        goto LABEL_91;
      }
LABEL_36:
      v32 = *(const __m128i **)(v23 + 32);
      v33 = 80;
      v34 = &v32->m128i_i8[v31];
      if ( *(_WORD *)(a2 + 24) != 186 )
        v33 = 40;
      v35 = *(_QWORD *)(a2 + 32) + v33;
      if ( *(_QWORD *)v35 != *(_QWORD *)v34 || *(_DWORD *)(v35 + 8) != *((_DWORD *)v34 + 2) )
      {
        v36 = *(_BYTE *)(a2 + 26);
        if ( (v36 & 8) == 0 || (*(_BYTE *)(v23 + 26) & 8) == 0 )
        {
          if ( (v36 & 0x40) != 0 && (*(_BYTE *)(*(_QWORD *)(v23 + 104) + 32LL) & 2) != 0
            || (*(_BYTE *)(v23 + 26) & 0x40) != 0 && (*(_BYTE *)(*(_QWORD *)(a2 + 104) + 32LL) & 2) != 0 )
          {
            goto LABEL_92;
          }
          v37 = *(_BYTE *)(a2 + 88);
          v38 = *(_QWORD *)(a2 + 96);
          LOBYTE(v125) = v37;
          v126 = v38;
          if ( v37 )
          {
            v39 = sub_1F6C8D0(v37);
          }
          else
          {
            v107 = v8;
            v39 = sub_1F58D40((__int64)&v125);
            v40 = v107;
          }
          v41 = *(_BYTE *)(v23 + 88);
          v100 = (unsigned int)(v39 + 7) >> 3;
          v42 = *(_QWORD *)(v23 + 96);
          LOBYTE(v125) = v41;
          v126 = v42;
          if ( v41 )
          {
            v43 = sub_1F6C8D0(v41);
          }
          else
          {
            v112 = v40;
            v43 = sub_1F58D40((__int64)&v125);
            v44 = v112;
          }
          v98 = v44;
          v108 = (unsigned int)(v43 + 7) >> 3;
          sub_2043720(v122, a2, *a1);
          sub_2043720(&v125, v23, *a1);
          v8 = v98;
          if ( !v122[0] || !v125 )
            goto LABEL_69;
          v45 = sub_2043540(v122, &v125, *a1, &v115);
          v8 = v98;
          if ( v45 )
          {
            if ( v100 <= v115 || v115 + v108 <= 0 )
              goto LABEL_91;
          }
          else
          {
            v46 = *(unsigned __int16 *)(v122[0] + 24LL);
            v10 = *(unsigned __int16 *)(v125 + 24);
            LOBYTE(a5) = (_WORD)v46 == 14;
            v47 = (_WORD)v10 == 36 || (_WORD)v10 == 14;
            v48 = (_WORD)v46 == 14 || (_WORD)v46 == 36;
            if ( v48 && v47 )
            {
              if ( v122[0] == v125
                || (v49 = *(_DWORD *)(v122[0] + 84LL), v49 < 0)
                && (v50 = -*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 56LL) + 32LL), v49 >= v50)
                && (v51 = *(_DWORD *)(v125 + 84), v51 < 0)
                && v50 <= v51 )
              {
                v48 = (_WORD)v10 == 36 || (_WORD)v10 == 14;
                goto LABEL_60;
              }
LABEL_91:
              v32 = *(const __m128i **)(v23 + 32);
LABEL_92:
              v15 = v133;
              if ( v133 >= v134 )
              {
                v114 = v8;
                sub_16CD150((__int64)&v132, v135, 0, 16, (int)a5, v10);
                v15 = v133;
                v8 = v114;
              }
              a6 = (__m128)_mm_loadu_si128(v32);
LABEL_86:
              *(__m128 *)&v132[2 * v15] = a6;
              LODWORD(v15) = ++v133;
              goto LABEL_87;
            }
LABEL_60:
            v52 = v46 - 34;
            LOBYTE(v52) = (unsigned __int16)(v46 - 34) <= 1u;
            v53 = v46 - 12;
            LOBYTE(v53) = (unsigned __int16)(v46 - 12) <= 1u;
            LODWORD(a5) = v53 | v52;
            v54 = (unsigned __int16)(*(_WORD *)(v125 + 24) - 12) <= 1u
               || (unsigned __int16)(*(_WORD *)(v125 + 24) - 34) <= 1u;
            v55 = v46 == 16 || v46 == 38;
            v56 = v10 == 16;
            v11 = v10 == 38;
            v10 = v123;
            v57 = v11 || v56;
            if ( v127 == v123 && (v10 = v124, v128 == v124)
              || (unsigned __int8)(v54 ^ (unsigned __int8)a5) | v47 ^ v48
              || v57 != v55 )
            {
              if ( ((unsigned __int8)a5 | v55 || v48) && (v47 || v54 || v57) )
                goto LABEL_91;
            }
LABEL_69:
            v58 = *(_QWORD *)(v23 + 104);
            v59 = *(_QWORD *)(a2 + 104);
            v60 = *(_QWORD *)(v58 + 8);
            v61 = *(_QWORD *)(v59 + 8);
            v62 = (unsigned int)(1 << *(_WORD *)(v59 + 34)) >> 1;
            v63 = (unsigned int)(1 << *(_WORD *)(v58 + 34)) >> 1;
            if ( v62 == (_DWORD)v63 && v61 != v60 )
            {
              v10 = v100;
              if ( v100 == v108 && v100 < v62 )
              {
                v64 = v61 % v62;
                if ( v60 % v63 >= v64 + v100 || v64 >= v60 % v63 + v108 )
                  goto LABEL_91;
              }
            }
            v99 = v8;
            v65 = sub_16D5D50();
            v8 = v99;
            v66 = *(_QWORD **)&dword_4FA0208[2];
            if ( !*(_QWORD *)&dword_4FA0208[2] )
              goto LABEL_118;
            v67 = dword_4FA0208;
            do
            {
              if ( v65 > v66[4] )
              {
                v66 = (_QWORD *)v66[3];
              }
              else
              {
                v67 = v66;
                v66 = (_QWORD *)v66[2];
              }
            }
            while ( v66 );
            if ( v67 == dword_4FA0208 )
              goto LABEL_118;
            if ( v65 < *((_QWORD *)v67 + 4) )
              goto LABEL_118;
            v95 = *((_QWORD *)v67 + 7);
            a5 = v67 + 12;
            if ( !v95 )
              goto LABEL_118;
            v96 = v67 + 12;
            do
            {
              if ( *(_DWORD *)(v95 + 32) < dword_4FCE988 )
              {
                v95 = *(_QWORD *)(v95 + 24);
              }
              else
              {
                v96 = (_DWORD *)v95;
                v95 = *(_QWORD *)(v95 + 16);
              }
            }
            while ( v95 );
            if ( v96 == a5 || dword_4FCE988 < v96[8] || (v78 = byte_4FCEA20, (int)v96[9] <= 0) )
            {
LABEL_118:
              v77 = *(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 16LL);
              v78 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v77 + 272LL))(v77);
              v8 = v99;
            }
            if ( v78 )
            {
              v79 = a1[111];
              if ( v79 )
              {
                v80 = *(__int64 **)(a2 + 104);
                if ( (*v80 & 4) == 0 && (*v80 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                {
                  v81 = *(_QWORD **)(v23 + 104);
                  if ( (*v81 & 4) == 0 )
                  {
                    v82 = *v81 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v82 )
                    {
                      v83 = v60;
                      if ( v61 <= v60 )
                        v83 = v61;
                      v84 = v100 + v61 - v83;
                      v85 = v108 + v60 - v83;
                      if ( byte_4FCE940 )
                      {
                        v86 = v81[5];
                        v87 = v81[6];
                        v117 = *v81 & 0xFFFFFFFFFFFFFFF8LL;
                        v88 = v81[7];
                        v118 = v85;
                        v119 = v86;
                        v89 = v80[6];
                        v120 = v87;
                        v90 = v80[7];
                        v121 = v88;
                        v91 = v80[5];
                      }
                      else
                      {
                        v118 = v85;
                        v90 = 0;
                        v89 = 0;
                        v91 = 0;
                        v117 = v82;
                        v119 = 0;
                        v120 = 0;
                        v121 = 0;
                      }
                      v92 = *v80;
                      v116[3] = v89;
                      v111 = v8;
                      v116[1] = v84;
                      v93 = v92 & 0xFFFFFFFFFFFFFFF8LL;
                      v116[2] = v91;
                      if ( (v92 & 4) != 0 )
                        v93 = 0;
                      v116[4] = v90;
                      v116[0] = v93;
                      v94 = sub_134CB50(v79, (__int64)v116, (__int64)&v117);
                      v8 = v111;
                      if ( !v94 )
                        goto LABEL_91;
                    }
                  }
                }
              }
            }
          }
        }
      }
LABEL_12:
      v20 = (unsigned int)v130;
      if ( (unsigned int)v130 >= HIDWORD(v130) )
      {
        v110 = v8;
        sub_16CD150((__int64)&v129, v131, 0, 16, (int)a5, v10);
        v20 = (unsigned int)v130;
        v8 = v110;
      }
      v15 = (unsigned __int64)&v129[2 * v20];
      *(_QWORD *)v15 = v23;
      *(_QWORD *)(v15 + 8) = v24;
      LODWORD(v15) = v133;
      LODWORD(v130) = v130 + 1;
      goto LABEL_15;
    }
    if ( v17 != 1 )
    {
      if ( v17 != 2 )
        goto LABEL_12;
      v19 = *(_DWORD *)(v23 + 56);
      if ( v19 > 0x10 )
        goto LABEL_12;
      v15 = v133;
      if ( v19 )
      {
        v73 = 40LL * (v19 - 1);
        do
        {
          v74 = (const __m128i *)(v73 + *(_QWORD *)(v23 + 32));
          if ( v134 <= (unsigned int)v15 )
          {
            v101 = v8;
            sub_16CD150((__int64)&v132, v135, 0, 16, (int)a5, v10);
            v15 = v133;
            v8 = v101;
          }
          a6 = (__m128)_mm_loadu_si128(v74);
          v73 -= 40;
          *(__m128 *)&v132[2 * v15] = a6;
          v15 = ++v133;
        }
        while ( v73 != -40 );
      }
LABEL_87:
      ++v13;
LABEL_15:
      if ( !(_DWORD)v15 )
        goto LABEL_26;
      goto LABEL_16;
    }
LABEL_25:
    LODWORD(v15) = v133;
    if ( !v133 )
    {
LABEL_26:
      v28 = (unsigned __int64)v138;
      v29 = (__int64 **)a1;
      if ( v138 == v137 )
        goto LABEL_28;
      goto LABEL_27;
    }
LABEL_16:
    v12 = v132;
  }
  v29 = (__int64 **)a1;
  v75 = 0;
  LODWORD(v130) = 0;
  if ( !HIDWORD(v130) )
  {
    sub_16CD150((__int64)&v129, v131, 0, 16, 0, v10);
    v75 = 2LL * (unsigned int)v130;
  }
  v76 = v129;
  v129[v75] = a3;
  v76[v75 + 1] = a4;
  v28 = (unsigned __int64)v138;
  LODWORD(v130) = v130 + 1;
  if ( v138 != v137 )
LABEL_27:
    _libc_free(v28);
LABEL_28:
  if ( v132 != v135 )
    _libc_free((unsigned __int64)v132);
  v30 = v129;
  if ( (_DWORD)v130 )
  {
    if ( (unsigned int)v130 == 1 )
    {
      result = *v129;
    }
    else
    {
      v69 = (unsigned int)v130;
      v70 = *v29;
      v71 = v129;
      v72 = *(_QWORD *)(a2 + 72);
      v136 = v72;
      if ( v72 )
        sub_1623A60((__int64)&v136, v72, 2);
      *((_QWORD *)&v97 + 1) = v69;
      *(_QWORD *)&v97 = v71;
      LODWORD(v137) = *(_DWORD *)(a2 + 64);
      result = sub_1D359D0(v70, 2, (__int64)&v136, 1, 0, 0, *(double *)a6.m128_u64, a7, a8, v97);
      if ( v136 )
      {
        v109 = result;
        sub_161E7C0((__int64)&v136, v136);
        result = v109;
      }
      v30 = v129;
    }
  }
  else
  {
    result = *v29 + 11;
  }
  if ( v30 != (__int64 **)v131 )
  {
    v106 = result;
    _libc_free((unsigned __int64)v30);
    return v106;
  }
  return result;
}
