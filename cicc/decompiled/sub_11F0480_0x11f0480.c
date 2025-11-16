// Function: sub_11F0480
// Address: 0x11f0480
//
__int64 __fastcall sub_11F0480(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v7; // r13
  __int64 v9; // rbx
  unsigned int v10; // ebx
  __int64 v11; // rdi
  int v12; // eax
  bool v13; // al
  bool v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  char v17; // bl
  _QWORD *v18; // r12
  __int64 v19; // rbx
  unsigned int *v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // esi
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 **v25; // r10
  _BYTE *v26; // rbx
  __int64 v27; // rdi
  __int64 (__fastcall *v28)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v29; // rax
  __int64 v30; // r15
  _QWORD *v31; // rax
  unsigned int *v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 **v37; // r12
  __int64 v38; // rax
  char v39; // bl
  _QWORD *v40; // rax
  unsigned __int64 v41; // r14
  unsigned int *v42; // rbx
  __int64 v43; // r15
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // rdi
  __int64 (__fastcall *v47)(__int64, unsigned int, _BYTE *, __int64); // rax
  _QWORD *v48; // rax
  unsigned int *v49; // rbx
  __int64 v50; // r12
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r14
  __int64 v58; // r12
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __m128i v64; // xmm0
  __m128i v65; // xmm1
  __m128i v66; // xmm2
  unsigned int v67; // r9d
  char *v68; // r12
  char **v69; // rsi
  __m128i *v70; // r12
  __m128i *v71; // rbx
  __m128i *v72; // rdi
  __int64 v73; // rdi
  __int64 v74; // r12
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v78; // rdi
  __int64 v79; // rbx
  unsigned __int64 v80; // r12
  unsigned __int64 v81; // r10
  __int64 v82; // r12
  __int64 v83; // rax
  __int32 v84; // esi
  __int64 v85; // rdx
  unsigned __int64 v86; // r10
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  int v91; // eax
  char *v92; // rax
  __int64 **v93; // r14
  __int64 v94; // rbx
  unsigned int v95; // eax
  __int64 v96; // r10
  _BYTE *v97; // rax
  __int64 v98; // rax
  unsigned __int64 v99; // rcx
  unsigned __int64 v100; // rax
  __int64 v101; // rdi
  __int64 (__fastcall *v102)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rax
  const __m128i *v106; // rbx
  __m128i *v107; // r12
  unsigned int v108; // r15d
  const __m128i *v109; // r13
  __int64 v110; // rax
  char *v111; // rbx
  char *v112; // rdi
  unsigned int *v113; // r14
  __int64 v114; // rbx
  __int64 v115; // rdx
  unsigned int v116; // esi
  __int64 v117; // [rsp+8h] [rbp-418h]
  unsigned int v118; // [rsp+10h] [rbp-410h]
  __int64 v119; // [rsp+10h] [rbp-410h]
  __int32 v120; // [rsp+18h] [rbp-408h]
  char **v121; // [rsp+20h] [rbp-400h]
  __int64 *v122; // [rsp+28h] [rbp-3F8h]
  unsigned __int64 v123; // [rsp+28h] [rbp-3F8h]
  __int64 v124; // [rsp+28h] [rbp-3F8h]
  __int64 v125; // [rsp+30h] [rbp-3F0h]
  __int64 v126; // [rsp+30h] [rbp-3F0h]
  __int64 v127; // [rsp+30h] [rbp-3F0h]
  unsigned __int64 v128; // [rsp+30h] [rbp-3F0h]
  __int64 v129; // [rsp+30h] [rbp-3F0h]
  unsigned __int8 *v130; // [rsp+38h] [rbp-3E8h]
  __int64 v131; // [rsp+38h] [rbp-3E8h]
  __int64 **v132; // [rsp+38h] [rbp-3E8h]
  __int64 v133; // [rsp+38h] [rbp-3E8h]
  __int64 **v134; // [rsp+38h] [rbp-3E8h]
  unsigned int v135; // [rsp+38h] [rbp-3E8h]
  __int64 v136; // [rsp+38h] [rbp-3E8h]
  unsigned __int64 v137; // [rsp+38h] [rbp-3E8h]
  __int64 v138; // [rsp+40h] [rbp-3E0h] BYREF
  int v139; // [rsp+48h] [rbp-3D8h]
  __int64 v140; // [rsp+50h] [rbp-3D0h]
  unsigned __int64 v141; // [rsp+60h] [rbp-3C0h] BYREF
  unsigned __int32 v142; // [rsp+68h] [rbp-3B8h]
  __int64 v143; // [rsp+70h] [rbp-3B0h]
  unsigned int v144; // [rsp+78h] [rbp-3A8h]
  __int16 v145; // [rsp+80h] [rbp-3A0h]
  char *v146; // [rsp+90h] [rbp-390h] BYREF
  unsigned __int32 v147; // [rsp+98h] [rbp-388h]
  __int8 v148; // [rsp+9Ch] [rbp-384h]
  __int64 v149; // [rsp+A0h] [rbp-380h]
  __m128i v150; // [rsp+A8h] [rbp-378h]
  __int64 v151; // [rsp+B8h] [rbp-368h]
  __m128i v152; // [rsp+C0h] [rbp-360h]
  __m128i v153; // [rsp+D0h] [rbp-350h]
  __m128i *v154; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v155; // [rsp+E8h] [rbp-338h]
  _BYTE v156[320]; // [rsp+F0h] [rbp-330h] BYREF
  char v157; // [rsp+230h] [rbp-1F0h]
  int v158; // [rsp+234h] [rbp-1ECh]
  __int64 v159; // [rsp+238h] [rbp-1E8h]
  __m128i v160; // [rsp+240h] [rbp-1E0h] BYREF
  __int64 v161; // [rsp+250h] [rbp-1D0h]
  __m128i v162; // [rsp+258h] [rbp-1C8h] BYREF
  __int64 v163; // [rsp+268h] [rbp-1B8h]
  __m128i v164; // [rsp+270h] [rbp-1B0h] BYREF
  __m128i v165; // [rsp+280h] [rbp-1A0h] BYREF
  const __m128i *v166; // [rsp+290h] [rbp-190h]
  unsigned int v167; // [rsp+298h] [rbp-188h]
  char v168; // [rsp+2A0h] [rbp-180h] BYREF
  char v169; // [rsp+3E0h] [rbp-40h]
  int v170; // [rsp+3E4h] [rbp-3Ch]
  __int64 v171; // [rsp+3E8h] [rbp-38h]

  v5 = a2;
  v7 = a3;
  v9 = a1;
  v130 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v125 = sub_BCD140(*(_QWORD **)(a3 + 72), a4);
  if ( (unsigned __int8)sub_988330(a2) )
  {
    if ( a5 )
    {
      v160 = (__m128i)*(unsigned __int64 *)(a1 + 16);
      v161 = 0;
      v162 = 0u;
      v163 = 0;
      v164 = 0u;
      v165.m128i_i16[0] = 257;
      if ( !(unsigned __int8)sub_9B6260(a5, &v160, 0) )
        goto LABEL_4;
    }
    v36 = *(_QWORD *)(v7 + 48);
    v37 = *(__int64 ***)(a2 + 8);
    v145 = 257;
    v146 = "char0";
    v150.m128i_i16[4] = 259;
    v38 = sub_AA4E30(v36);
    v39 = sub_AE5020(v38, v125);
    v162.m128i_i16[4] = 257;
    v40 = sub_BD2C40(80, unk_3F10A14);
    v41 = (unsigned __int64)v40;
    if ( v40 )
      sub_B4D190((__int64)v40, v125, (__int64)v130, (__int64)&v160, 0, v39, 0, 0);
    (*(void (__fastcall **)(_QWORD, unsigned __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
      *(_QWORD *)(v7 + 88),
      v41,
      &v146,
      *(_QWORD *)(v7 + 56),
      *(_QWORD *)(v7 + 64));
    v42 = *(unsigned int **)v7;
    v43 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
    if ( *(_QWORD *)v7 != v43 )
    {
      do
      {
        v44 = *((_QWORD *)v42 + 1);
        v45 = *v42;
        v42 += 4;
        sub_B99FD0(v41, v45, v44);
      }
      while ( (unsigned int *)v43 != v42 );
    }
    if ( v37 == *(__int64 ***)(v41 + 8) )
      return v41;
    v46 = *(_QWORD *)(v7 + 80);
    v47 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v46 + 120LL);
    if ( v47 == sub_920130 )
    {
      if ( *(_BYTE *)v41 > 0x15u )
        goto LABEL_39;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v30 = sub_ADAB70(39, v41, v37, 0);
      else
        v30 = sub_AA93C0(0x27u, v41, (__int64)v37);
    }
    else
    {
      v30 = v47(v46, 39u, (_BYTE *)v41, (__int64)v37);
    }
    if ( v30 )
      return v30;
LABEL_39:
    v162.m128i_i16[4] = 257;
    v48 = sub_BD2C40(72, unk_3F10A14);
    v30 = (__int64)v48;
    if ( v48 )
      sub_B515B0((__int64)v48, v41, (__int64)v37, (__int64)&v160, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
      *(_QWORD *)(v7 + 88),
      v30,
      &v141,
      *(_QWORD *)(v7 + 56),
      *(_QWORD *)(v7 + 64));
    v49 = *(unsigned int **)v7;
    v50 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
    if ( *(_QWORD *)v7 != v50 )
    {
      do
      {
        v51 = *((_QWORD *)v49 + 1);
        v52 = *v49;
        v49 += 4;
        sub_B99FD0(v30, v52, v51);
      }
      while ( (unsigned int *)v50 != v49 );
    }
    return v30;
  }
  if ( !a5 )
  {
    v53 = sub_98B430((__int64)v130, a4);
    if ( v53 )
      return sub_AD64C0(*(_QWORD *)(a2 + 8), v53 - 1, 0);
    v55 = *v130;
    if ( *v130 <= 0x1Cu )
    {
      if ( (_BYTE)v55 != 5 || *((_WORD *)v130 + 1) != 34 )
        return 0;
    }
    else if ( (_BYTE)v55 != 63 )
    {
LABEL_48:
      if ( (_BYTE)v55 == 86 )
      {
        v126 = sub_98B430(*((_QWORD *)v130 - 8), a4);
        v56 = sub_98B430(*((_QWORD *)v130 - 4), a4);
        v57 = v56;
        if ( v126 )
        {
          if ( v56 )
          {
            v58 = **(_QWORD **)(v9 + 56);
            v122 = *(__int64 **)(v9 + 56);
            v59 = sub_B2BE50(v58);
            if ( !sub_B6EA50(v59) )
            {
              v104 = sub_B2BE50(v58);
              v105 = sub_B6F970(v104);
              if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v105 + 48LL))(v105) )
              {
LABEL_66:
                v73 = *(_QWORD *)(v5 + 8);
                v162.m128i_i16[4] = 257;
                v74 = sub_AD64C0(v73, v57 - 1, 0);
                v75 = sub_AD64C0(*(_QWORD *)(v5 + 8), v126 - 1, 0);
                return sub_B36550((unsigned int **)v7, *((_QWORD *)v130 - 12), v75, v74, (__int64)&v160, 0);
              }
            }
            sub_B174A0((__int64)&v160, (__int64)"instcombine", (__int64)"simplify-libcalls", 17, v5);
            v60 = (__int64)"folded strlen(select) to select of constants";
            sub_B18290((__int64)&v160, "folded strlen(select) to select of constants", 0x2Cu);
            v64 = _mm_loadu_si128(&v162);
            v65 = _mm_loadu_si128(&v164);
            v66 = _mm_loadu_si128(&v165);
            v147 = v160.m128i_u32[2];
            v67 = v167;
            v150 = v64;
            v148 = v160.m128i_i8[12];
            v152 = v65;
            v149 = v161;
            v153 = v66;
            v146 = (char *)&unk_49D9D40;
            v151 = v163;
            v154 = (__m128i *)v156;
            v155 = 0x400000000LL;
            if ( v167 )
            {
              v60 = v167;
              if ( v167 > 4 )
              {
                v118 = v167;
                sub_11F02D0((__int64)&v154, v167, v61, v62, v63, v167);
                v67 = v118;
              }
              v68 = (char *)v166;
              v106 = v166;
              if ( v166 != &v166[5 * v167] )
              {
                v119 = v5;
                v107 = v154;
                v108 = v67;
                v117 = v7;
                v109 = &v166[5 * v167];
                do
                {
                  if ( v107 )
                  {
                    v107->m128i_i64[0] = (__int64)v107[1].m128i_i64;
                    sub_11DA140(v107->m128i_i64, v106->m128i_i64[0], v106->m128i_i64[0] + v106->m128i_i64[1]);
                    v107[2].m128i_i64[0] = (__int64)v107[3].m128i_i64;
                    v60 = v106[2].m128i_i64[0];
                    sub_11DA140(v107[2].m128i_i64, (_BYTE *)v60, v60 + v106[2].m128i_i64[1]);
                    v107[4] = _mm_loadu_si128(v106 + 4);
                  }
                  v106 += 5;
                  v107 += 5;
                }
                while ( v109 != v106 );
                v67 = v108;
                v7 = v117;
                v5 = v119;
                v68 = (char *)v166;
              }
              LODWORD(v155) = v67;
              v157 = v169;
              v158 = v170;
              v159 = v171;
              v146 = (char *)&unk_49D9D78;
              v160.m128i_i64[0] = (__int64)&unk_49D9D40;
              v110 = 80LL * v167;
              v111 = &v68[v110];
              if ( v68 == &v68[v110] )
                goto LABEL_55;
              do
              {
                v111 -= 80;
                v112 = (char *)*((_QWORD *)v111 + 4);
                if ( v112 != v111 + 48 )
                {
                  v60 = *((_QWORD *)v111 + 6) + 1LL;
                  j_j___libc_free_0(v112, v60);
                }
                if ( *(char **)v111 != v111 + 16 )
                {
                  v60 = *((_QWORD *)v111 + 2) + 1LL;
                  j_j___libc_free_0(*(_QWORD *)v111, v60);
                }
              }
              while ( v68 != v111 );
            }
            else
            {
              v157 = v169;
              v158 = v170;
              v159 = v171;
              v146 = (char *)&unk_49D9D78;
            }
            v68 = (char *)v166;
LABEL_55:
            if ( v68 != &v168 )
              _libc_free(v68, v60);
            v69 = &v146;
            sub_1049740(v122, (__int64)&v146);
            v70 = v154;
            v146 = (char *)&unk_49D9D40;
            v71 = &v154[5 * (unsigned int)v155];
            if ( v154 != v71 )
            {
              do
              {
                v71 -= 5;
                v72 = (__m128i *)v71[2].m128i_i64[0];
                if ( v72 != &v71[3] )
                {
                  v69 = (char **)(v71[3].m128i_i64[0] + 1);
                  j_j___libc_free_0(v72, v69);
                }
                if ( (__m128i *)v71->m128i_i64[0] != &v71[1] )
                {
                  v69 = (char **)(v71[1].m128i_i64[0] + 1);
                  j_j___libc_free_0(v71->m128i_i64[0], v69);
                }
              }
              while ( v70 != v71 );
              v70 = v154;
            }
            if ( v70 != (__m128i *)v156 )
              _libc_free(v70, v69);
            goto LABEL_66;
          }
        }
      }
      return 0;
    }
    if ( !(unsigned __int8)sub_98AB00((__int64)v130, a4, v54) )
      return a5;
    if ( !(unsigned __int8)sub_98AE20(*(_QWORD *)&v130[-32 * (*((_DWORD *)v130 + 1) & 0x7FFFFFF)], &v138, a4, 0) )
    {
LABEL_91:
      LOBYTE(v55) = *v130;
      goto LABEL_48;
    }
    v78 = v138;
    if ( v138 )
    {
      if ( v140 )
      {
        v127 = v9;
        v79 = v140;
        v80 = 0;
        while ( 1 )
        {
          if ( !sub_AC5320(v78, (int)v80 + v139) )
          {
            v9 = v127;
            v81 = v80;
            goto LABEL_82;
          }
          if ( v79 == ++v80 )
            return 0;
          v78 = v138;
        }
      }
      return a5;
    }
    v81 = 0;
LABEL_82:
    v123 = v81;
    v82 = *(_QWORD *)&v130[32 * (2LL - (*((_DWORD *)v130 + 1) & 0x7FFFFFF))];
    sub_9AC3E0((__int64)&v141, v82, *(_QWORD *)(v9 + 16), 0, 0, a2, 0, 1);
    v83 = sub_BB5290((__int64)v130);
    v84 = v142;
    v85 = *(_QWORD *)(v83 + 32);
    v86 = v123;
    v87 = 1LL << ((unsigned __int8)v142 - 1);
    if ( v142 > 0x40 )
    {
      if ( (*(_QWORD *)(v141 + 8LL * ((v142 - 1) >> 6)) & v87) == 0 )
        goto LABEL_84;
      v160.m128i_i32[2] = v142;
      v124 = v85;
      v128 = v86;
      sub_C43780((__int64)&v160, (const void **)&v141);
      v84 = v160.m128i_i32[2];
      v86 = v128;
      v85 = v124;
      if ( v160.m128i_i32[2] > 0x40u )
      {
        sub_C43D10((__int64)&v160);
        v86 = v128;
        v85 = v124;
        v147 = v160.m128i_u32[2];
        v146 = (char *)v160.m128i_i64[0];
        v120 = v160.m128i_i32[2];
        if ( v160.m128i_i32[2] <= 0x40u )
        {
          if ( v128 >= v160.m128i_i64[0] )
            goto LABEL_109;
        }
        else
        {
          v121 = (char **)v160.m128i_i64[0];
          v91 = sub_C444A0((__int64)&v146);
          v86 = v128;
          v85 = v124;
          if ( (unsigned int)(v120 - v91) <= 0x40 )
          {
            v92 = *v121;
            goto LABEL_107;
          }
        }
LABEL_124:
        if ( **(_BYTE **)&v130[-32 * (*((_DWORD *)v130 + 1) & 0x7FFFFFF)] != 3 || v86 != v85 - 1 )
        {
          if ( v147 > 0x40 && v146 )
            j_j___libc_free_0_0(v146);
LABEL_85:
          if ( v144 > 0x40 && v143 )
            j_j___libc_free_0_0(v143);
          if ( v142 > 0x40 && v141 )
            j_j___libc_free_0_0(v141);
          goto LABEL_91;
        }
LABEL_108:
        if ( v147 > 0x40 && v146 )
        {
          v137 = v86;
          j_j___libc_free_0_0(v146);
          v86 = v137;
        }
LABEL_109:
        v93 = *(__int64 ***)(v5 + 8);
        v129 = v86;
        v150.m128i_i16[4] = 257;
        v94 = *(_QWORD *)(v82 + 8);
        v135 = sub_BCB060(v94);
        v95 = sub_BCB060((__int64)v93);
        v96 = v129;
        if ( v135 < v95 )
        {
          v100 = sub_11DB4B0((__int64 *)v7, 0x28u, v82, v93, (__int64)&v146, 0, v160.m128i_i32[0], 0);
          v93 = *(__int64 ***)(v5 + 8);
          v96 = v129;
          v82 = v100;
          goto LABEL_112;
        }
        if ( v135 == v95 || v93 == (__int64 **)v94 )
          goto LABEL_112;
        v101 = *(_QWORD *)(v7 + 80);
        v102 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v101 + 120LL);
        if ( v102 == sub_920130 )
        {
          if ( *(_BYTE *)v82 > 0x15u )
            goto LABEL_160;
          if ( (unsigned __int8)sub_AC4810(0x26u) )
            v103 = sub_ADAB70(38, v82, v93, 0);
          else
            v103 = sub_AA93C0(0x26u, v82, (__int64)v93);
          v96 = v129;
        }
        else
        {
          v103 = v102(v101, 38u, (_BYTE *)v82, (__int64)v93);
          v96 = v129;
        }
        if ( v103 )
        {
          v93 = *(__int64 ***)(v5 + 8);
          v82 = v103;
LABEL_112:
          v162.m128i_i16[4] = 257;
          v97 = (_BYTE *)sub_AD64C0((__int64)v93, v96, 0);
          a5 = sub_929DE0((unsigned int **)v7, v97, (_BYTE *)v82, (__int64)&v160, 0, 0);
          if ( v144 > 0x40 && v143 )
            j_j___libc_free_0_0(v143);
          if ( v142 > 0x40 && v141 )
            j_j___libc_free_0_0(v141);
          return a5;
        }
LABEL_160:
        v136 = v96;
        v162.m128i_i16[4] = 257;
        v82 = sub_B51D30(38, v82, (__int64)v93, (__int64)&v160, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
          *(_QWORD *)(v7 + 88),
          v82,
          &v146,
          *(_QWORD *)(v7 + 56),
          *(_QWORD *)(v7 + 64));
        v113 = *(unsigned int **)v7;
        v96 = v136;
        v114 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
        if ( *(_QWORD *)v7 != v114 )
        {
          do
          {
            v115 = *((_QWORD *)v113 + 1);
            v116 = *v113;
            v113 += 4;
            sub_B99FD0(v82, v116, v115);
          }
          while ( (unsigned int *)v114 != v113 );
          v96 = v136;
        }
        v93 = *(__int64 ***)(v5 + 8);
        goto LABEL_112;
      }
      v88 = v160.m128i_i64[0];
    }
    else
    {
      v88 = v141;
      if ( (v141 & v87) == 0 )
      {
LABEL_84:
        if ( **(_BYTE **)&v130[-32 * (*((_DWORD *)v130 + 1) & 0x7FFFFFF)] == 3 && v123 == v85 - 1 )
          goto LABEL_109;
        goto LABEL_85;
      }
    }
    v98 = ~v88;
    v147 = v84;
    v99 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v84;
    if ( !v84 )
      v99 = 0;
    v92 = (char *)(v99 & v98);
    v146 = v92;
LABEL_107:
    if ( v86 >= (unsigned __int64)v92 )
      goto LABEL_108;
    goto LABEL_124;
  }
LABEL_4:
  if ( *(_BYTE *)a5 != 17 )
  {
LABEL_67:
    v76 = sub_98B430((__int64)v130, a4);
    if ( v76 )
    {
      v89 = sub_AD64C0(*(_QWORD *)(a2 + 8), v76 - 1, 0);
      HIDWORD(v146) = 0;
      v162.m128i_i16[4] = 257;
      return sub_B33C40(v7, 0x16Eu, v89, a5, (unsigned int)v146, (__int64)&v160);
    }
    return 0;
  }
  v10 = *(_DWORD *)(a5 + 32);
  v11 = a5 + 24;
  if ( v10 <= 0x40 )
  {
    v13 = *(_QWORD *)(a5 + 24) == 0;
  }
  else
  {
    v12 = sub_C444A0(v11);
    v11 = a5 + 24;
    v13 = v10 == v12;
  }
  if ( !v13 )
  {
    if ( v10 <= 0x40 )
      v14 = *(_QWORD *)(a5 + 24) == 1;
    else
      v14 = v10 - 1 == (unsigned int)sub_C444A0(v11);
    if ( v14 )
    {
      v15 = *(_QWORD *)(v7 + 48);
      v146 = "strnlen.char0";
      v150.m128i_i16[4] = 259;
      v16 = sub_AA4E30(v15);
      v17 = sub_AE5020(v16, v125);
      v162.m128i_i16[4] = 257;
      v18 = sub_BD2C40(80, unk_3F10A14);
      if ( v18 )
        sub_B4D190((__int64)v18, v125, (__int64)v130, (__int64)&v160, 0, v17, 0, 0);
      (*(void (__fastcall **)(_QWORD, _QWORD *, char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
        *(_QWORD *)(v7 + 88),
        v18,
        &v146,
        *(_QWORD *)(v7 + 56),
        *(_QWORD *)(v7 + 64));
      if ( *(_QWORD *)v7 != *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8) )
      {
        v131 = v7;
        v19 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
        v20 = *(unsigned int **)v7;
        do
        {
          v21 = *((_QWORD *)v20 + 1);
          v22 = *v20;
          v20 += 4;
          sub_B99FD0((__int64)v18, v22, v21);
        }
        while ( (unsigned int *)v19 != v20 );
        v7 = v131;
      }
      v23 = (_BYTE *)sub_AD64C0(v125, 0, 0);
      v160.m128i_i64[0] = (__int64)"strnlen.char0cmp";
      v162.m128i_i16[4] = 259;
      v24 = sub_92B530((unsigned int **)v7, 0x21u, (__int64)v18, v23, (__int64)&v160);
      v150.m128i_i16[4] = 257;
      v25 = *(__int64 ***)(v5 + 8);
      v26 = (_BYTE *)v24;
      if ( v25 == *(__int64 ***)(v24 + 8) )
        return v24;
      v27 = *(_QWORD *)(v7 + 80);
      v28 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v27 + 120LL);
      if ( v28 == sub_920130 )
      {
        if ( *v26 > 0x15u )
        {
LABEL_24:
          v133 = (__int64)v25;
          v162.m128i_i16[4] = 257;
          v31 = sub_BD2C40(72, unk_3F10A14);
          v30 = (__int64)v31;
          if ( v31 )
            sub_B515B0((__int64)v31, (__int64)v26, v133, (__int64)&v160, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
            *(_QWORD *)(v7 + 88),
            v30,
            &v146,
            *(_QWORD *)(v7 + 56),
            *(_QWORD *)(v7 + 64));
          v32 = *(unsigned int **)v7;
          v33 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
          if ( *(_QWORD *)v7 != v33 )
          {
            do
            {
              v34 = *((_QWORD *)v32 + 1);
              v35 = *v32;
              v32 += 4;
              sub_B99FD0(v30, v35, v34);
            }
            while ( (unsigned int *)v33 != v32 );
          }
          return v30;
        }
        v132 = *(__int64 ***)(v5 + 8);
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v29 = sub_ADAB70(39, (unsigned __int64)v26, v132, 0);
        else
          v29 = sub_AA93C0(0x27u, (unsigned __int64)v26, (__int64)v132);
        v25 = v132;
        v30 = v29;
      }
      else
      {
        v134 = *(__int64 ***)(v5 + 8);
        v90 = v28(v27, 39u, v26, (__int64)v25);
        v25 = v134;
        v30 = v90;
      }
      if ( !v30 )
        goto LABEL_24;
      return v30;
    }
    goto LABEL_67;
  }
  return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
}
