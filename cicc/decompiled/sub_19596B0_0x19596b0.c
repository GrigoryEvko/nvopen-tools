// Function: sub_19596B0
// Address: 0x19596b0
//
__int64 __fastcall sub_19596B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r15
  int v18; // esi
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 *v22; // r9
  __int64 v23; // r8
  __int64 v24; // rax
  char v25; // di
  unsigned int v26; // esi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // r12
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r10
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // r8d
  __int64 *v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // rcx
  __int64 v43; // rdi
  unsigned __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rbx
  __int64 v50; // rsi
  unsigned int v51; // ecx
  _QWORD *v52; // rax
  __int64 v53; // rdx
  const char *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rbx
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rax
  const __m128i *v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rcx
  double v64; // xmm4_8
  double v65; // xmm5_8
  unsigned __int64 v66; // rbx
  __int64 v67; // rbx
  __int64 v68; // r13
  __int64 v69; // r15
  _QWORD *v70; // rcx
  _QWORD *v71; // rax
  int v72; // r8d
  int v73; // r9d
  int v75; // esi
  int v76; // r14d
  __int64 v77; // r15
  const char *v78; // rax
  __int64 v79; // rdx
  unsigned int v80; // edi
  _QWORD *v81; // rax
  __int64 v82; // r9
  __int64 v83; // rdx
  unsigned __int32 i; // eax
  __int64 v85; // rsi
  unsigned int v86; // edx
  __int64 *v87; // rax
  __int64 v88; // rcx
  int v89; // r11d
  _QWORD *v90; // r10
  int v91; // edx
  __int64 v92; // rcx
  __int64 v93; // rdi
  int v94; // r11d
  _QWORD *v95; // r10
  __int64 v96; // r14
  int v97; // r10d
  __int64 v98; // rcx
  int v99; // r8d
  _QWORD *v100; // rcx
  int v101; // eax
  __int64 v102; // rdx
  __int64 v103; // r9
  int v104; // edi
  _QWORD *v105; // rsi
  int v106; // edi
  __int64 v107; // rdx
  __int64 v108; // r9
  int v109; // r11d
  __int64 *v110; // r10
  int v111; // ecx
  int v112; // r10d
  int v113; // ecx
  __int64 v114; // rdi
  __int64 *v115; // r13
  _QWORD *v116; // [rsp+18h] [rbp-178h]
  __int64 v118; // [rsp+30h] [rbp-160h]
  __int64 v120; // [rsp+40h] [rbp-150h]
  const __m128i *v121; // [rsp+50h] [rbp-140h] BYREF
  __m128i *v122; // [rsp+58h] [rbp-138h]
  const __m128i *v123; // [rsp+60h] [rbp-130h]
  __int64 v124; // [rsp+70h] [rbp-120h] BYREF
  __int64 v125; // [rsp+78h] [rbp-118h]
  __int64 v126; // [rsp+80h] [rbp-110h]
  unsigned int v127; // [rsp+88h] [rbp-108h]
  __int64 v128[8]; // [rsp+90h] [rbp-100h] BYREF
  __m128i v129; // [rsp+D0h] [rbp-C0h] BYREF
  _QWORD v130[22]; // [rsp+E0h] [rbp-B0h] BYREF

  v11 = *(unsigned int *)(a3 + 8);
  v12 = *(__int64 **)a3;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  if ( v11 == 1 )
    v13 = *v12;
  else
    v13 = sub_1956A20((__int64)a1, a2, v12, v11, ".thr_comm");
  v129.m128i_i64[0] = v13;
  v129.m128i_i64[1] = a2 | 4;
  sub_19541D0((__int64)&v121, &v129);
  v14 = sub_157EBA0(v13);
  v116 = (_QWORD *)v14;
  if ( *(_BYTE *)(v14 + 16) != 26 || (v118 = v13, (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != 1) )
  {
    v15 = sub_1AA91E0(v13, a2, 0, 0);
    v129.m128i_i64[0] = v13;
    v16 = v15;
    v118 = v15;
    v129.m128i_i64[1] = v15 & 0xFFFFFFFFFFFFFFFBLL;
    sub_19541D0((__int64)&v121, &v129);
    v129.m128i_i64[1] = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v129.m128i_i64[0] = v16;
    sub_19541D0((__int64)&v121, &v129);
    v129.m128i_i64[0] = v13;
    v129.m128i_i64[1] = a2 | 4;
    sub_19541D0((__int64)&v121, &v129);
    v116 = (_QWORD *)sub_157EBA0(v16);
  }
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v17 = *(_QWORD *)(a2 + 48);
  v127 = 0;
  while ( 1 )
  {
    if ( !v17 )
      BUG();
    if ( *(_BYTE *)(v17 - 8) != 77 )
      break;
    v18 = v127;
    v19 = v17 - 24;
    v128[0] = v17 - 24;
    v20 = v17 - 24;
    if ( !v127 )
    {
      ++v124;
LABEL_155:
      v18 = 2 * v127;
LABEL_156:
      sub_19566A0((__int64)&v124, v18);
      sub_1954890((__int64)&v124, v128, &v129);
      v22 = (__int64 *)v129.m128i_i64[0];
      v20 = v128[0];
      v111 = v126 + 1;
      goto LABEL_151;
    }
    v21 = (v127 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v22 = (__int64 *)(v125 + 16LL * v21);
    v23 = *v22;
    if ( v19 == *v22 )
      goto LABEL_11;
    v109 = 1;
    v110 = 0;
    while ( v23 != -8 )
    {
      if ( v23 == -16 && !v110 )
        v110 = v22;
      v21 = (v127 - 1) & (v109 + v21);
      v22 = (__int64 *)(v125 + 16LL * v21);
      v23 = *v22;
      if ( v19 == *v22 )
        goto LABEL_11;
      ++v109;
    }
    if ( v110 )
      v22 = v110;
    ++v124;
    v111 = v126 + 1;
    if ( 4 * ((int)v126 + 1) >= 3 * v127 )
      goto LABEL_155;
    if ( v127 - HIDWORD(v126) - v111 <= v127 >> 3 )
      goto LABEL_156;
LABEL_151:
    LODWORD(v126) = v111;
    if ( *v22 != -8 )
      --HIDWORD(v126);
    *v22 = v20;
    v22[1] = 0;
LABEL_11:
    v24 = 0x17FFFFFFE8LL;
    v25 = *(_BYTE *)(v17 - 1) & 0x40;
    v26 = *(_DWORD *)(v17 - 4) & 0xFFFFFFF;
    if ( v26 )
    {
      v27 = 24LL * *(unsigned int *)(v17 + 32) + 8;
      v28 = 0;
      do
      {
        v29 = v19 - 24LL * v26;
        if ( v25 )
          v29 = *(_QWORD *)(v17 - 32);
        if ( v118 == *(_QWORD *)(v29 + v27) )
        {
          v24 = 24 * v28;
          goto LABEL_18;
        }
        ++v28;
        v27 += 8;
      }
      while ( v26 != (_DWORD)v28 );
      v24 = 0x17FFFFFFE8LL;
    }
LABEL_18:
    if ( v25 )
      v30 = *(_QWORD *)(v17 - 32);
    else
      v30 = v19 - 24LL * v26;
    v22[1] = *(_QWORD *)(v30 + v24);
    v17 = *(_QWORD *)(v17 + 8);
  }
  v120 = a2 + 40;
  while ( v17 != v120 )
  {
    v31 = v17 - 24;
    if ( !v17 )
      v31 = 0;
    v32 = sub_15F4880(v31);
    v33 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
    if ( v33 )
    {
      v34 = 0;
      v35 = 24LL * v33;
      do
      {
        if ( (*(_BYTE *)(v32 + 23) & 0x40) != 0 )
          v36 = *(_QWORD *)(v32 - 8);
        else
          v36 = v32 - 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF);
        v37 = (_QWORD *)(v34 + v36);
        v38 = *v37;
        if ( *(_BYTE *)(*v37 + 16LL) > 0x17u && v127 )
        {
          v39 = (v127 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v40 = (__int64 *)(v125 + 16LL * v39);
          v41 = *v40;
          if ( v38 == *v40 )
          {
LABEL_30:
            if ( v40 != (__int64 *)(v125 + 16LL * v127) )
            {
              v42 = v40[1];
              v43 = v37[1];
              v44 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v44 = v43;
              if ( v43 )
                *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
              *v37 = v42;
              if ( v42 )
              {
                v45 = *(_QWORD *)(v42 + 8);
                v37[1] = v45;
                if ( v45 )
                  *(_QWORD *)(v45 + 16) = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v45 + 16) & 3LL;
                v37[2] = (v42 + 8) | v37[2] & 3LL;
                *(_QWORD *)(v42 + 8) = v37;
              }
            }
          }
          else
          {
            v75 = 1;
            while ( v41 != -8 )
            {
              v76 = v75 + 1;
              v39 = (v127 - 1) & (v75 + v39);
              v40 = (__int64 *)(v125 + 16LL * v39);
              v41 = *v40;
              if ( v38 == *v40 )
                goto LABEL_30;
              v75 = v76;
            }
          }
        }
        v34 += 24;
      }
      while ( v35 != v34 );
    }
    v46 = sub_157EB90(a2);
    v129.m128i_i64[0] = sub_1632FA0(v46);
    v47 = *a1;
    v130[2] = v32;
    v130[0] = 0;
    v129.m128i_i64[1] = v47;
    v130[1] = 0;
    v49 = sub_13E3350(v32, &v129, 0, 1, v48);
    if ( !v49 )
    {
      v128[0] = v31;
      if ( v127 )
      {
        v86 = (v127 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v87 = (__int64 *)(v125 + 16LL * v86);
        v88 = *v87;
        if ( v31 == *v87 )
        {
LABEL_93:
          v87[1] = v32;
LABEL_44:
          v54 = sub_1649960(v31);
          v129.m128i_i64[0] = (__int64)v128;
          v128[0] = (__int64)v54;
          v128[1] = v55;
          LOWORD(v130[0]) = 261;
          sub_164B780(v32, v129.m128i_i64);
          sub_157E9D0(v118 + 40, v32);
          v56 = v116[3];
          *(_QWORD *)(v32 + 32) = v116 + 3;
          v56 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v32 + 24) = v56 | *(_QWORD *)(v32 + 24) & 7LL;
          *(_QWORD *)(v56 + 8) = v32 + 24;
          v116[3] = v116[3] & 7LL | (v32 + 24);
          if ( (*(_DWORD *)(v32 + 20) & 0xFFFFFFF) != 0 )
          {
            v57 = 0;
            v58 = 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF);
            do
            {
              if ( (*(_BYTE *)(v32 + 23) & 0x40) != 0 )
                v59 = *(_QWORD *)(v32 - 8);
              else
                v59 = v32 - 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF);
              v60 = *(_QWORD *)(v59 + v57);
              if ( *(_BYTE *)(v60 + 16) == 18 )
              {
                v129.m128i_i64[0] = v118;
                v61 = v122;
                v129.m128i_i64[1] = v60 & 0xFFFFFFFFFFFFFFFBLL;
                if ( v122 == v123 )
                {
                  sub_17F2860(&v121, v122, &v129);
                }
                else
                {
                  if ( v122 )
                  {
                    a4 = (__m128)_mm_loadu_si128(&v129);
                    *v122 = (__m128i)a4;
                    v61 = v122;
                  }
                  v122 = (__m128i *)&v61[1];
                }
              }
              v57 += 24;
            }
            while ( v57 != v58 );
          }
          goto LABEL_57;
        }
        v112 = 1;
        while ( v88 != -8 )
        {
          if ( v88 == -16 && !v49 )
            v49 = (__int64)v87;
          v86 = (v127 - 1) & (v112 + v86);
          v87 = (__int64 *)(v125 + 16LL * v86);
          v88 = *v87;
          if ( v31 == *v87 )
            goto LABEL_93;
          ++v112;
        }
        if ( v49 )
          v87 = (__int64 *)v49;
        ++v124;
        v113 = v126 + 1;
        if ( 4 * ((int)v126 + 1) < 3 * v127 )
        {
          v114 = v31;
          if ( v127 - HIDWORD(v126) - v113 > v127 >> 3 )
          {
LABEL_163:
            LODWORD(v126) = v113;
            if ( *v87 != -8 )
              --HIDWORD(v126);
            *v87 = v114;
            v87[1] = 0;
            goto LABEL_93;
          }
          sub_19566A0((__int64)&v124, v127);
LABEL_168:
          sub_1954890((__int64)&v124, v128, &v129);
          v87 = (__int64 *)v129.m128i_i64[0];
          v114 = v128[0];
          v113 = v126 + 1;
          goto LABEL_163;
        }
      }
      else
      {
        ++v124;
      }
      sub_19566A0((__int64)&v124, 2 * v127);
      goto LABEL_168;
    }
    v50 = v127;
    if ( v127 )
    {
      v51 = (v127 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v52 = (_QWORD *)(v125 + 16LL * v51);
      v53 = *v52;
      if ( v31 == *v52 )
        goto LABEL_43;
      v89 = 1;
      v90 = 0;
      while ( v53 != -8 )
      {
        if ( !v90 && v53 == -16 )
          v90 = v52;
        v51 = (v127 - 1) & (v89 + v51);
        v52 = (_QWORD *)(v125 + 16LL * v51);
        v53 = *v52;
        if ( v31 == *v52 )
          goto LABEL_43;
        ++v89;
      }
      if ( v90 )
        v52 = v90;
      ++v124;
      v91 = v126 + 1;
      if ( 4 * ((int)v126 + 1) < 3 * v127 )
      {
        if ( v127 - HIDWORD(v126) - v91 <= v127 >> 3 )
        {
          sub_19566A0((__int64)&v124, v127);
          if ( !v127 )
          {
LABEL_210:
            LODWORD(v126) = v126 + 1;
            BUG();
          }
          v50 = 0;
          LODWORD(v96) = (v127 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v97 = 1;
          v91 = v126 + 1;
          v52 = (_QWORD *)(v125 + 16LL * (unsigned int)v96);
          v98 = *v52;
          if ( v31 != *v52 )
          {
            while ( v98 != -8 )
            {
              if ( v98 == -16 && !v50 )
                v50 = (__int64)v52;
              v96 = (v127 - 1) & ((_DWORD)v96 + v97);
              v52 = (_QWORD *)(v125 + 16 * v96);
              v98 = *v52;
              if ( v31 == *v52 )
                goto LABEL_103;
              ++v97;
            }
            if ( v50 )
              v52 = (_QWORD *)v50;
          }
        }
        goto LABEL_103;
      }
    }
    else
    {
      ++v124;
    }
    sub_19566A0((__int64)&v124, 2 * v127);
    if ( !v127 )
      goto LABEL_210;
    v50 = v127 - 1;
    LODWORD(v92) = v50 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v91 = v126 + 1;
    v52 = (_QWORD *)(v125 + 16LL * (unsigned int)v92);
    v93 = *v52;
    if ( v31 != *v52 )
    {
      v94 = 1;
      v95 = 0;
      while ( v93 != -8 )
      {
        if ( !v95 && v93 == -16 )
          v95 = v52;
        v92 = (unsigned int)v50 & ((_DWORD)v92 + v94);
        v52 = (_QWORD *)(v125 + 16 * v92);
        v93 = *v52;
        if ( v31 == *v52 )
          goto LABEL_103;
        ++v94;
      }
      if ( v95 )
        v52 = v95;
    }
LABEL_103:
    LODWORD(v126) = v91;
    if ( *v52 != -8 )
      --HIDWORD(v126);
    *v52 = v31;
    v52[1] = 0;
LABEL_43:
    v52[1] = v49;
    if ( (unsigned __int8)sub_15F3040(v32) || sub_15F3330(v32) )
      goto LABEL_44;
    sub_164BEC0(v32, v50, v62, v63, a4, a5, a6, a7, v64, v65, a10, a11);
LABEL_57:
    v17 = *(_QWORD *)(v17 + 8);
  }
  v66 = sub_157EBA0(a2);
  sub_19523F0(*(_QWORD *)(v66 - 24), a2, v118, (__int64)&v124);
  sub_19523F0(*(_QWORD *)(v66 - 48), a2, v118, (__int64)&v124);
  sub_1B3B830(v128, 0);
  v67 = *(_QWORD *)(a2 + 48);
  v129.m128i_i64[0] = (__int64)v130;
  v129.m128i_i64[1] = 0x1000000000LL;
  if ( v67 != v120 )
  {
    while ( 1 )
    {
      if ( !v67 )
        BUG();
      v68 = *(_QWORD *)(v67 - 16);
      v69 = v129.m128i_u32[2];
      if ( v68 )
        break;
LABEL_73:
      if ( (_DWORD)v69 )
      {
        v77 = v67 - 24;
        v78 = sub_1649960(v67 - 24);
        sub_1B3B8C0(v128, *(_QWORD *)(v67 - 24), v78, v79);
        sub_1B3BE00(v128, a2, v67 - 24);
        if ( v127 )
        {
          v80 = (v127 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v81 = (_QWORD *)(v125 + 16LL * v80);
          v82 = *v81;
          if ( v77 == *v81 )
          {
            v83 = v81[1];
LABEL_88:
            sub_1B3BE00(v128, v118, v83);
            for ( i = v129.m128i_u32[2]; v129.m128i_i32[2]; i = v129.m128i_u32[2] )
            {
              v85 = *(_QWORD *)(v129.m128i_i64[0] + 8LL * i - 8);
              v129.m128i_i32[2] = i - 1;
              sub_1B420C0(v128, v85);
            }
            goto LABEL_74;
          }
          v99 = 1;
          v100 = 0;
          while ( v82 != -8 )
          {
            if ( v100 || v82 != -16 )
              v81 = v100;
            v80 = (v127 - 1) & (v99 + v80);
            v115 = (__int64 *)(v125 + 16LL * v80);
            v82 = *v115;
            if ( v77 == *v115 )
            {
              v83 = v115[1];
              goto LABEL_88;
            }
            ++v99;
            v100 = v81;
            v81 = (_QWORD *)(v125 + 16LL * v80);
          }
          if ( !v100 )
            v100 = v81;
          ++v124;
          v101 = v126 + 1;
          if ( 4 * ((int)v126 + 1) < 3 * v127 )
          {
            if ( v127 - HIDWORD(v126) - v101 > v127 >> 3 )
              goto LABEL_126;
            sub_19566A0((__int64)&v124, v127);
            if ( !v127 )
            {
LABEL_207:
              LODWORD(v126) = v126 + 1;
              BUG();
            }
            v106 = 1;
            v105 = 0;
            LODWORD(v107) = (v127 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
            v101 = v126 + 1;
            v100 = (_QWORD *)(v125 + 16LL * (unsigned int)v107);
            v108 = *v100;
            if ( v77 == *v100 )
              goto LABEL_126;
            while ( v108 != -8 )
            {
              if ( !v105 && v108 == -16 )
                v105 = v100;
              v107 = (v127 - 1) & ((_DWORD)v107 + v106);
              v100 = (_QWORD *)(v125 + 16 * v107);
              v108 = *v100;
              if ( v77 == *v100 )
                goto LABEL_126;
              ++v106;
            }
            goto LABEL_142;
          }
        }
        else
        {
          ++v124;
        }
        sub_19566A0((__int64)&v124, 2 * v127);
        if ( !v127 )
          goto LABEL_207;
        LODWORD(v102) = (v127 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
        v101 = v126 + 1;
        v100 = (_QWORD *)(v125 + 16LL * (unsigned int)v102);
        v103 = *v100;
        if ( v77 == *v100 )
          goto LABEL_126;
        v104 = 1;
        v105 = 0;
        while ( v103 != -8 )
        {
          if ( !v105 && v103 == -16 )
            v105 = v100;
          v102 = (v127 - 1) & ((_DWORD)v102 + v104);
          v100 = (_QWORD *)(v125 + 16 * v102);
          v103 = *v100;
          if ( v77 == *v100 )
            goto LABEL_126;
          ++v104;
        }
LABEL_142:
        if ( v105 )
          v100 = v105;
LABEL_126:
        LODWORD(v126) = v101;
        if ( *v100 != -8 )
          --HIDWORD(v126);
        *v100 = v77;
        v83 = 0;
        v100[1] = 0;
        goto LABEL_88;
      }
LABEL_74:
      v67 = *(_QWORD *)(v67 + 8);
      if ( v67 == v120 )
        goto LABEL_75;
    }
    while ( 1 )
    {
      v71 = sub_1648700(v68);
      if ( *((_BYTE *)v71 + 16) == 77 )
        break;
      if ( a2 == v71[5] )
        goto LABEL_67;
      if ( v129.m128i_i32[3] <= (unsigned int)v69 )
        goto LABEL_71;
LABEL_66:
      *(_QWORD *)(v129.m128i_i64[0] + 8 * v69) = v68;
      v69 = (unsigned int)++v129.m128i_i32[2];
LABEL_67:
      v68 = *(_QWORD *)(v68 + 8);
      if ( !v68 )
        goto LABEL_73;
    }
    if ( (*((_BYTE *)v71 + 23) & 0x40) != 0 )
      v70 = (_QWORD *)*(v71 - 1);
    else
      v70 = &v71[-3 * (*((_DWORD *)v71 + 5) & 0xFFFFFFF)];
    if ( a2 == v70[3 * *((unsigned int *)v71 + 14) + 1 + -1431655765 * (unsigned int)((v68 - (__int64)v70) >> 3)] )
      goto LABEL_67;
    if ( v129.m128i_i32[3] > (unsigned int)v69 )
      goto LABEL_66;
LABEL_71:
    sub_16CD150((__int64)&v129, v130, 0, 8, v72, v73);
    v69 = v129.m128i_u32[2];
    goto LABEL_66;
  }
LABEL_75:
  sub_157F2D0(a2, v118, 1);
  sub_15F20C0(v116);
  sub_15CD9D0(a1[3], v121->m128i_i64, v122 - v121);
  if ( (_QWORD *)v129.m128i_i64[0] != v130 )
    _libc_free(v129.m128i_u64[0]);
  sub_1B3B860(v128);
  j___libc_free_0(v125);
  if ( v121 )
    j_j___libc_free_0(v121, (char *)v123 - (char *)v121);
  return 1;
}
