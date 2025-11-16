// Function: sub_3751010
// Address: 0x3751010
//
void __fastcall sub_3751010(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r13
  __int64 (__fastcall *v9)(__int64, __int64, _QWORD, __int64, unsigned __int64); // rbx
  __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  char v17; // al
  unsigned int v18; // eax
  __int64 v19; // rsi
  unsigned int v20; // r14d
  __int64 v21; // rax
  unsigned int v22; // ecx
  __int64 *v23; // rdx
  __int64 v24; // r8
  int v25; // eax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned int v28; // ecx
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // r8
  unsigned __int8 *v31; // rsi
  int v32; // edx
  __int64 v33; // rdi
  const void **v34; // r9
  __int64 (*v35)(); // rax
  unsigned __int64 v36; // r8
  int v37; // eax
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned int v40; // r9d
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  int v45; // edx
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // rdi
  bool v48; // cc
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // r14
  unsigned __int8 *v51; // rsi
  int v52; // edx
  __int64 v53; // rdi
  const void **v54; // r9
  __int64 (*v55)(); // rax
  unsigned int v56; // eax
  __int64 v57; // rdx
  unsigned __int64 v58; // rax
  unsigned int v59; // eax
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rcx
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  unsigned int v66; // eax
  __int64 v67; // rax
  unsigned __int64 v68; // r8
  unsigned __int64 v69; // r10
  unsigned __int64 v70; // r9
  __int64 v71; // rdi
  unsigned __int64 v72; // rcx
  unsigned __int64 v73; // rbx
  unsigned __int64 v74; // r12
  unsigned __int64 v75; // r13
  unsigned int v76; // eax
  unsigned int v77; // eax
  int v78; // edx
  int v79; // r9d
  int *v80; // rax
  unsigned __int64 v81; // rbx
  unsigned __int64 v82; // r12
  unsigned __int64 v83; // rdi
  unsigned __int64 v84; // rdi
  int v85; // esi
  __int64 v86; // rax
  unsigned int v87; // ecx
  unsigned int v88; // edx
  unsigned int v89; // ecx
  unsigned __int64 v90; // rdx
  unsigned int v91; // esi
  unsigned __int64 v92; // rdx
  unsigned __int64 v93; // rdi
  unsigned __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rcx
  __int64 v97; // rdi
  int v98; // edx
  unsigned __int64 v99; // rsi
  char v100; // al
  int v101; // edx
  unsigned __int64 v102; // rsi
  unsigned int v103; // edi
  char v104; // al
  __int128 v105; // [rsp-10h] [rbp-120h]
  unsigned int v106; // [rsp+4h] [rbp-10Ch]
  unsigned int v107; // [rsp+4h] [rbp-10Ch]
  unsigned int v108; // [rsp+4h] [rbp-10Ch]
  __int64 v109; // [rsp+8h] [rbp-108h]
  unsigned int v110; // [rsp+10h] [rbp-100h]
  unsigned __int64 v111; // [rsp+10h] [rbp-100h]
  unsigned int v112; // [rsp+10h] [rbp-100h]
  unsigned __int64 v113; // [rsp+10h] [rbp-100h]
  __int64 v114; // [rsp+10h] [rbp-100h]
  unsigned __int64 v115; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v116; // [rsp+18h] [rbp-F8h]
  __int64 v117; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v118; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v119; // [rsp+20h] [rbp-F0h]
  __int64 v120; // [rsp+20h] [rbp-F0h]
  int v121; // [rsp+28h] [rbp-E8h]
  int v122; // [rsp+28h] [rbp-E8h]
  int v123; // [rsp+28h] [rbp-E8h]
  unsigned int v124; // [rsp+28h] [rbp-E8h]
  __int64 v125; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v126; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v127; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v128; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v129; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v130; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v131; // [rsp+30h] [rbp-E0h]
  unsigned int v132; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v133; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v134; // [rsp+30h] [rbp-E0h]
  unsigned int v135; // [rsp+30h] [rbp-E0h]
  __int64 v136; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v137; // [rsp+30h] [rbp-E0h]
  int v138; // [rsp+30h] [rbp-E0h]
  __int64 v139; // [rsp+38h] [rbp-D8h]
  const void **v140; // [rsp+38h] [rbp-D8h]
  unsigned int v141; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v142; // [rsp+38h] [rbp-D8h]
  int v143; // [rsp+38h] [rbp-D8h]
  unsigned int v144; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v145; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v146; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v147; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v148; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v149; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int8 *v150; // [rsp+58h] [rbp-B8h] BYREF
  __m128i v151; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int64 v152; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v153; // [rsp+78h] [rbp-98h]
  __int64 v154; // [rsp+80h] [rbp-90h]
  __int64 v155; // [rsp+88h] [rbp-88h]
  unsigned __int64 v156; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v157; // [rsp+98h] [rbp-78h]
  const __m128i *v158[2]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE v159[16]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v160; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v161; // [rsp+C8h] [rbp-48h]
  __int64 v162; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v163; // [rsp+D8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v2 + 8) == 12 )
  {
    v4 = *(__int64 **)(a1 + 8);
    v5 = a2;
    v158[0] = (const __m128i *)v159;
    v158[1] = (const __m128i *)0x100000000LL;
    v6 = sub_2E79000(v4);
    v7 = *(_QWORD *)(a1 + 16);
    LOBYTE(v161) = 0;
    *((_QWORD *)&v105 + 1) = v161;
    v160 = 0;
    *(_QWORD *)&v105 = 0;
    sub_34B8C80(v7, v6, v2, (__int64)v158, 0, 0, v105);
    v8 = *(_QWORD *)(a1 + 16);
    v151 = _mm_loadu_si128(v158[0]);
    v9 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, unsigned __int64))(*(_QWORD *)v8 + 736LL);
    BYTE2(v160) = 0;
    v10 = sub_BD5C60(a2);
    v11 = v9(v8, v10, v151.m128i_u32[0], v151.m128i_i64[1], v160);
    if ( v11 != 1 )
      goto LABEL_4;
    v12 = *(_QWORD *)(a1 + 16);
    v13 = sub_BD5C60(a2);
    v14 = sub_2FE98B0(v12, v13, v151.m128i_u32[0], v151.m128i_u64[1]);
    v151.m128i_i64[1] = 0;
    v151.m128i_i16[0] = v14;
    if ( v14 )
    {
      if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
        BUG();
      v67 = 16LL * (v14 - 1);
      v16 = *(_QWORD *)&byte_444C4A0[v67];
      v17 = byte_444C4A0[v67 + 8];
    }
    else
    {
      v154 = sub_3007260((__int64)&v151);
      v155 = v15;
      v16 = v154;
      v17 = v155;
    }
    v160 = v16;
    LOBYTE(v161) = v17;
    v18 = sub_CA1930(&v160);
    v19 = *(_QWORD *)(a1 + 128);
    v20 = v18;
    v21 = *(unsigned int *)(a1 + 144);
    if ( !(_DWORD)v21 )
      goto LABEL_4;
    v22 = (v21 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v23 = (__int64 *)(v19 + 16LL * v22);
    v24 = *v23;
    if ( v5 != *v23 )
    {
      v78 = 1;
      while ( v24 != -4096 )
      {
        v79 = v78 + 1;
        v22 = (v21 - 1) & (v78 + v22);
        v23 = (__int64 *)(v19 + 16LL * v22);
        v24 = *v23;
        if ( v5 == *v23 )
          goto LABEL_10;
        v78 = v79;
      }
      goto LABEL_4;
    }
LABEL_10:
    if ( v23 == (__int64 *)(v19 + 16 * v21) || (v25 = *((_DWORD *)v23 + 2)) == 0 )
    {
LABEL_4:
      if ( v158[0] != (const __m128i *)v159 )
        _libc_free((unsigned __int64)v158[0]);
      return;
    }
    v26 = v25 & 0x7FFFFFFF;
    v27 = *(unsigned int *)(a1 + 1096);
    v28 = v26 + 1;
    if ( (int)v26 + 1 > (unsigned int)v27 )
    {
      v68 = v28;
      v29 = *(_QWORD *)(a1 + 1088);
      if ( v28 == v27 )
        goto LABEL_14;
      v69 = v29 + 40 * v27;
      if ( v28 < v27 )
      {
        if ( v29 + 40LL * v28 != v69 )
        {
          v141 = v26;
          v135 = v26 + 1;
          v81 = v29 + 40LL * v28;
          v125 = v5;
          v82 = v29 + 40 * v27;
          do
          {
            v82 -= 40LL;
            if ( *(_DWORD *)(v82 + 32) > 0x40u )
            {
              v83 = *(_QWORD *)(v82 + 24);
              if ( v83 )
                j_j___libc_free_0_0(v83);
            }
            if ( *(_DWORD *)(v82 + 16) > 0x40u )
            {
              v84 = *(_QWORD *)(v82 + 8);
              if ( v84 )
                j_j___libc_free_0_0(v84);
            }
          }
          while ( v81 != v82 );
          v11 = 1;
          v26 = v141;
          v28 = v135;
          v5 = v125;
          v29 = *(_QWORD *)(a1 + 1088);
        }
        *(_DWORD *)(a1 + 1096) = v28;
LABEL_14:
        v144 = v20;
        v30 = v29 + 40 * v26;
        v31 = **(unsigned __int8 ***)(v5 - 8);
        v149 = v31;
        v32 = *v31;
        if ( (unsigned int)(v32 - 12) <= 1 || (_BYTE)v32 == 5 )
        {
LABEL_77:
          *(_DWORD *)v30 = *(_DWORD *)v30 & 0x80000000 | 1;
          LODWORD(v161) = v20;
          if ( v20 > 0x40 )
          {
            v148 = v30;
            sub_C43690((__int64)&v160, 0, 0);
            v163 = v20;
            sub_C43690((__int64)&v162, 0, 0);
            v30 = v148;
          }
          else
          {
            v160 = 0;
            v163 = v20;
            v162 = 0;
          }
          if ( *(_DWORD *)(v30 + 16) > 0x40u )
          {
            v64 = *(_QWORD *)(v30 + 8);
            if ( v64 )
            {
              v145 = v30;
              j_j___libc_free_0_0(v64);
              v30 = v145;
            }
          }
          *(_QWORD *)(v30 + 8) = v160;
          *(_DWORD *)(v30 + 16) = v161;
          LODWORD(v161) = 0;
          if ( *(_DWORD *)(v30 + 32) > 0x40u && (v65 = *(_QWORD *)(v30 + 24)) != 0 )
          {
            v146 = v30;
            j_j___libc_free_0_0(v65);
            v66 = v161;
            *(_QWORD *)(v146 + 24) = v162;
            *(_DWORD *)(v146 + 32) = v163;
            if ( v66 > 0x40 && v160 )
              j_j___libc_free_0_0(v160);
          }
          else
          {
            *(_QWORD *)(v30 + 24) = v162;
            *(_DWORD *)(v30 + 32) = v163;
          }
          goto LABEL_4;
        }
        v139 = a1 + 120;
        if ( (_BYTE)v32 != 17 )
        {
          v133 = v30;
          v80 = (int *)sub_374DAD0(v139, (__int64 *)&v149);
          v36 = v133;
          if ( *v80 >= 0 || (v95 = sub_374D270(a1, *v80, v20), v36 = v133, (v96 = v95) == 0) )
          {
LABEL_110:
            *(_BYTE *)(v36 + 3) &= ~0x80u;
            goto LABEL_4;
          }
          v97 = v133 + 8;
          *(_DWORD *)v133 = *(_DWORD *)v95 & 0x7FFFFFFF | *(_DWORD *)v133 & 0x80000000;
          v48 = *(_DWORD *)(v133 + 16) <= 0x40u;
          *(_BYTE *)(v133 + 3) = *(_BYTE *)(v95 + 3) & 0x80 | *(_BYTE *)(v133 + 3) & 0x7F;
          if ( v48 && *(_DWORD *)(v95 + 16) <= 0x40u )
          {
            *(_QWORD *)(v133 + 8) = *(_QWORD *)(v95 + 8);
            *(_DWORD *)(v133 + 16) = *(_DWORD *)(v95 + 16);
          }
          else
          {
            v127 = v133;
            v136 = v95;
            sub_C43990(v97, v95 + 8);
            v36 = v127;
            v96 = v136;
          }
          if ( *(_DWORD *)(v36 + 32) <= 0x40u && *(_DWORD *)(v96 + 32) <= 0x40u )
          {
            *(_QWORD *)(v36 + 24) = *(_QWORD *)(v96 + 24);
            *(_DWORD *)(v36 + 32) = *(_DWORD *)(v96 + 32);
          }
          else
          {
            v137 = v36;
            sub_C43990(v36 + 24, v96 + 24);
            v36 = v137;
          }
          goto LABEL_40;
        }
        v33 = *(_QWORD *)(a1 + 16);
        v34 = (const void **)(v31 + 24);
        v153 = 1;
        v152 = 0;
        v35 = *(__int64 (**)())(*(_QWORD *)v33 + 1464LL);
        if ( v35 == sub_2FE34B0 || (v128 = v30, v100 = v35(), v30 = v128, v34 = (const void **)(v31 + 24), !v100) )
        {
          v128 = v30;
          sub_C449B0((__int64)&v160, v34, v20);
          v36 = v128;
          if ( v153 <= 0x40 )
            goto LABEL_21;
        }
        else
        {
          sub_C44830((__int64)&v160, (_DWORD *)v31 + 6, v20);
          v36 = v128;
          if ( v153 <= 0x40 )
            goto LABEL_21;
        }
        if ( v152 )
        {
          j_j___libc_free_0_0(v152);
          v36 = v128;
        }
LABEL_21:
        v152 = v160;
        v37 = v161;
        v153 = v161;
        v38 = 1LL << ((unsigned __int8)v161 - 1);
        if ( (unsigned int)v161 > 0x40 )
        {
          v134 = v36;
          if ( (*(_QWORD *)(v160 + 8LL * ((unsigned int)(v161 - 1) >> 6)) & v38) != 0 )
            v37 = sub_C44500((__int64)&v152);
          else
            v37 = sub_C444A0((__int64)&v152);
          v36 = v134;
        }
        else if ( (v38 & v160) != 0 )
        {
          if ( (_DWORD)v161 )
          {
            v37 = 64;
            if ( v160 << (64 - (unsigned __int8)v161) != -1 )
            {
              _BitScanReverse64(&v39, ~(v160 << (64 - (unsigned __int8)v161)));
              v37 = v39 ^ 0x3F;
            }
          }
        }
        else
        {
          v98 = 64;
          if ( v160 )
          {
            _BitScanReverse64(&v99, v160);
            v98 = v99 ^ 0x3F;
          }
          v37 = v161 + v98 - 64;
        }
        *(_DWORD *)v36 = *(_DWORD *)v36 & 0x80000000 | v37 & 0x7FFFFFFF;
        v40 = v153;
        LODWORD(v161) = v153;
        if ( v153 > 0x40 )
        {
          v126 = v36;
          sub_C43780((__int64)&v160, (const void **)&v152);
          v40 = v153;
          v36 = v126;
          v157 = v153;
          if ( v153 <= 0x40 )
          {
            v41 = v152;
          }
          else
          {
            sub_C43780((__int64)&v156, (const void **)&v152);
            v40 = v157;
            v36 = v126;
            if ( v157 > 0x40 )
            {
              sub_C43D10((__int64)&v156);
              v40 = v157;
              v44 = v156;
              v36 = v126;
LABEL_31:
              v45 = v161;
              v46 = v160;
              if ( *(_DWORD *)(v36 + 16) > 0x40u )
              {
                v47 = *(_QWORD *)(v36 + 8);
                if ( v47 )
                {
                  v110 = v40;
                  v115 = v44;
                  v118 = v160;
                  v121 = v161;
                  v129 = v36;
                  j_j___libc_free_0_0(v47);
                  v40 = v110;
                  v44 = v115;
                  v46 = v118;
                  v45 = v121;
                  v36 = v129;
                }
              }
              v48 = *(_DWORD *)(v36 + 32) <= 0x40u;
              *(_QWORD *)(v36 + 8) = v44;
              *(_DWORD *)(v36 + 16) = v40;
              if ( !v48 )
              {
                v49 = *(_QWORD *)(v36 + 24);
                if ( v49 )
                {
                  v119 = v46;
                  v122 = v45;
                  v130 = v36;
                  j_j___libc_free_0_0(v49);
                  v46 = v119;
                  v45 = v122;
                  v36 = v130;
                }
              }
              *(_QWORD *)(v36 + 24) = v46;
              *(_DWORD *)(v36 + 32) = v45;
              if ( v153 > 0x40 && v152 )
              {
                v131 = v36;
                j_j___libc_free_0_0(v152);
                v36 = v131;
              }
LABEL_40:
              v123 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
              if ( v123 != 1 )
              {
                v132 = v20;
                v50 = v36;
                while ( 1 )
                {
                  v51 = *(unsigned __int8 **)(*(_QWORD *)(v5 - 8) + 32LL * v11);
                  v150 = v51;
                  v52 = *v51;
                  if ( (unsigned int)(v52 - 12) <= 1 || (_BYTE)v52 == 5 )
                  {
                    v30 = v50;
                    v20 = v132;
                    goto LABEL_77;
                  }
                  if ( (_BYTE)v52 != 17 )
                  {
                    v85 = *(_DWORD *)sub_374DAD0(v139, (__int64 *)&v150);
                    if ( v85 >= 0 || (v86 = sub_374D270(a1, v85, v144)) == 0 )
                    {
                      v36 = v50;
                      goto LABEL_110;
                    }
                    v87 = *(_DWORD *)v86 & 0x7FFFFFFF;
                    if ( v87 > (*(_DWORD *)v50 & 0x7FFFFFFFu) )
                      v87 = *(_DWORD *)v50 & 0x7FFFFFFF;
                    v88 = v87 | *(_DWORD *)v50 & 0x80000000;
                    v89 = *(_DWORD *)(v50 + 32);
                    *(_DWORD *)v50 = v88;
                    LODWORD(v161) = v89;
                    if ( v89 <= 0x40 )
                    {
                      v90 = *(_QWORD *)(v50 + 24);
                      goto LABEL_131;
                    }
                    v117 = v86;
                    sub_C43780((__int64)&v160, (const void **)(v50 + 24));
                    v89 = v161;
                    v86 = v117;
                    if ( (unsigned int)v161 <= 0x40 )
                    {
                      v90 = v160;
LABEL_131:
                      v116 = *(_QWORD *)(v86 + 24) & v90;
                      v160 = v116;
                    }
                    else
                    {
                      v114 = v117;
                      sub_C43B90(&v160, (__int64 *)(v117 + 24));
                      v89 = v161;
                      v116 = v160;
                      v86 = v114;
                    }
                    LODWORD(v161) = 0;
                    v91 = *(_DWORD *)(v50 + 16);
                    v157 = v91;
                    if ( v91 > 0x40 )
                    {
                      v107 = v89;
                      v109 = v86;
                      sub_C43780((__int64)&v156, (const void **)(v50 + 8));
                      v91 = v157;
                      v89 = v107;
                      if ( v157 <= 0x40 )
                      {
                        v103 = v161;
                        v92 = *(_QWORD *)(v109 + 8) & v156;
                      }
                      else
                      {
                        sub_C43B90(&v156, (__int64 *)(v109 + 8));
                        v91 = v157;
                        v92 = v156;
                        v103 = v161;
                        v89 = v107;
                      }
                      if ( v103 > 0x40 && v160 )
                      {
                        v108 = v89;
                        v113 = v92;
                        j_j___libc_free_0_0(v160);
                        v89 = v108;
                        v92 = v113;
                      }
                    }
                    else
                    {
                      v92 = *(_QWORD *)(v86 + 8) & *(_QWORD *)(v50 + 8);
                    }
                    if ( *(_DWORD *)(v50 + 16) > 0x40u )
                    {
                      v93 = *(_QWORD *)(v50 + 8);
                      if ( v93 )
                      {
                        v106 = v89;
                        v111 = v92;
                        j_j___libc_free_0_0(v93);
                        v89 = v106;
                        v92 = v111;
                      }
                    }
                    v48 = *(_DWORD *)(v50 + 32) <= 0x40u;
                    *(_QWORD *)(v50 + 8) = v92;
                    *(_DWORD *)(v50 + 16) = v91;
                    if ( !v48 )
                    {
                      v94 = *(_QWORD *)(v50 + 24);
                      if ( v94 )
                      {
                        v112 = v89;
                        j_j___libc_free_0_0(v94);
                        v89 = v112;
                      }
                    }
                    *(_DWORD *)(v50 + 32) = v89;
                    *(_QWORD *)(v50 + 24) = v116;
                    goto LABEL_74;
                  }
                  v53 = *(_QWORD *)(a1 + 16);
                  v54 = (const void **)(v51 + 24);
                  v153 = 1;
                  v152 = 0;
                  v55 = *(__int64 (**)())(*(_QWORD *)v53 + 1464LL);
                  if ( v55 == sub_2FE34B0 || (v104 = v55(), v54 = (const void **)(v51 + 24), !v104) )
                  {
                    sub_C449B0((__int64)&v160, v54, v144);
                    if ( v153 <= 0x40 )
                      goto LABEL_49;
                  }
                  else
                  {
                    sub_C44830((__int64)&v160, (_DWORD *)v51 + 6, v144);
                    if ( v153 <= 0x40 )
                      goto LABEL_49;
                  }
                  if ( v152 )
                    j_j___libc_free_0_0(v152);
LABEL_49:
                  v152 = v160;
                  v56 = v161;
                  v153 = v161;
                  v57 = 1LL << ((unsigned __int8)v161 - 1);
                  if ( (unsigned int)v161 > 0x40 )
                  {
                    if ( (*(_QWORD *)(v160 + 8LL * ((unsigned int)(v161 - 1) >> 6)) & v57) != 0 )
                      v56 = sub_C44500((__int64)&v152);
                    else
                      v56 = sub_C444A0((__int64)&v152);
                  }
                  else if ( (v57 & v160) != 0 )
                  {
                    if ( (_DWORD)v161 )
                    {
                      _BitScanReverse64(&v58, ~(v160 << (64 - (unsigned __int8)v161)));
                      v56 = v58 ^ 0x3F;
                      if ( v160 << (64 - (unsigned __int8)v161) == -1 )
                        v56 = 64;
                    }
                  }
                  else
                  {
                    v101 = 64;
                    if ( v160 )
                    {
                      _BitScanReverse64(&v102, v160);
                      v101 = v102 ^ 0x3F;
                    }
                    v56 = v161 + v101 - 64;
                  }
                  if ( (*(_DWORD *)v50 & 0x7FFFFFFFu) <= v56 )
                    v56 = *(_DWORD *)v50 & 0x7FFFFFFF;
                  *(_DWORD *)v50 = v56 & 0x7FFFFFFF | *(_DWORD *)v50 & 0x80000000;
                  v59 = v153;
                  v157 = v153;
                  if ( v153 <= 0x40 )
                  {
                    v60 = v152;
LABEL_58:
                    v61 = ~v60;
                    v62 = 0;
                    if ( v59 )
                      v62 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v59;
                    v63 = v62 & v61;
                    v156 = v63;
                    goto LABEL_61;
                  }
                  sub_C43780((__int64)&v156, (const void **)&v152);
                  v59 = v157;
                  if ( v157 <= 0x40 )
                  {
                    v60 = v156;
                    goto LABEL_58;
                  }
                  sub_C43D10((__int64)&v156);
                  v59 = v157;
                  v63 = v156;
LABEL_61:
                  LODWORD(v161) = v59;
                  v160 = v63;
                  v157 = 0;
                  if ( *(_DWORD *)(v50 + 16) > 0x40u )
                    sub_C43B90((_QWORD *)(v50 + 8), (__int64 *)&v160);
                  else
                    *(_QWORD *)(v50 + 8) &= v63;
                  if ( (unsigned int)v161 > 0x40 && v160 )
                    j_j___libc_free_0_0(v160);
                  if ( v157 > 0x40 && v156 )
                    j_j___libc_free_0_0(v156);
                  if ( *(_DWORD *)(v50 + 32) > 0x40u )
                    sub_C43B90((_QWORD *)(v50 + 24), (__int64 *)&v152);
                  else
                    *(_QWORD *)(v50 + 24) &= v152;
                  if ( v153 > 0x40 && v152 )
                    j_j___libc_free_0_0(v152);
LABEL_74:
                  if ( ++v11 == v123 )
                    goto LABEL_4;
                }
              }
              goto LABEL_4;
            }
            v41 = v156;
          }
        }
        else
        {
          v41 = v152;
          v160 = v152;
        }
        v42 = ~v41;
        v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v40;
        if ( !v40 )
          v43 = 0;
        v44 = v43 & v42;
        goto LABEL_31;
      }
      v70 = *(unsigned int *)(a1 + 1100);
      v71 = a1 + 1088;
      v72 = a1 + 1104;
      v147 = v68 - v27;
      if ( v68 > v70 )
      {
        if ( v29 > v72 || v69 <= v72 )
        {
          v143 = v26;
          sub_342D2F0(v71, v68, v27, v72, v68, v70);
          v27 = *(unsigned int *)(a1 + 1096);
          v29 = *(_QWORD *)(a1 + 1088);
          LODWORD(v26) = v143;
          v72 = a1 + 1104;
        }
        else
        {
          v138 = v26;
          v142 = v72 - v29;
          sub_342D2F0(v71, v68, v27, v72 - v29, v68, v70);
          v29 = *(_QWORD *)(a1 + 1088);
          v27 = *(unsigned int *)(a1 + 1096);
          LODWORD(v26) = v138;
          v72 = v29 + v142;
        }
      }
      v73 = v72;
      v120 = v5;
      v74 = v147;
      v75 = v29 + 40 * v27;
      v140 = (const void **)(v72 + 24);
      v124 = v26;
      do
      {
        if ( v75 )
        {
          *(_DWORD *)v75 = *(_DWORD *)v73;
          v77 = *(_DWORD *)(v73 + 16);
          *(_DWORD *)(v75 + 16) = v77;
          if ( v77 <= 0x40 )
            *(_QWORD *)(v75 + 8) = *(_QWORD *)(v73 + 8);
          else
            sub_C43780(v75 + 8, (const void **)(v73 + 8));
          v76 = *(_DWORD *)(v73 + 32);
          *(_DWORD *)(v75 + 32) = v76;
          if ( v76 > 0x40 )
            sub_C43780(v75 + 24, v140);
          else
            *(_QWORD *)(v75 + 24) = *(_QWORD *)(v73 + 24);
        }
        v75 += 40LL;
        --v74;
      }
      while ( v74 );
      v11 = 1;
      *(_DWORD *)(a1 + 1096) += v147;
      v26 = v124;
      v5 = v120;
    }
    v29 = *(_QWORD *)(a1 + 1088);
    goto LABEL_14;
  }
}
