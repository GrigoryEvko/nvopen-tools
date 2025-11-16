// Function: sub_27F17A0
// Address: 0x27f17a0
//
__int64 __fastcall sub_27F17A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        _QWORD *a12,
        __int64 a13,
        __int64 *a14,
        char a15,
        char a16)
{
  __int64 v16; // rbx
  __int64 v17; // r12
  unsigned int v18; // r13d
  __int64 v19; // r14
  __int64 v20; // r12
  unsigned __int8 v21; // cl
  __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  unsigned __int16 v25; // r15
  __int64 v27; // r8
  __int64 v28; // r9
  __m128i v29; // xmm1
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __m128i v39; // xmm4
  __m128i v40; // xmm5
  __m128i v41; // xmm6
  unsigned __int64 *v42; // rbx
  __int64 v43; // rax
  unsigned __int64 *v44; // r14
  unsigned __int64 v45; // rdi
  unsigned __int64 *v46; // rbx
  unsigned __int64 *v47; // r14
  unsigned __int64 v48; // rdi
  __int64 *v49; // r14
  __int64 *v50; // rbx
  _BYTE *v51; // rsi
  __int64 *v52; // rax
  unsigned __int16 v53; // r15
  bool v54; // al
  unsigned int v55; // eax
  unsigned __int8 v56; // di
  __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned __int64 v59; // rax
  int v60; // edx
  __int64 v61; // rax
  bool v62; // cf
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  unsigned __int8 v65; // r15
  __int64 **v66; // rax
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __m128i v70; // xmm3
  unsigned __int64 v71; // rax
  int v72; // edx
  __int64 v73; // rax
  unsigned int v74; // eax
  unsigned __int8 v75; // di
  unsigned __int8 *v76; // r14
  unsigned __int64 v77; // rax
  int v78; // edx
  __int64 v79; // r8
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  unsigned int v82; // eax
  __int64 *v83; // r15
  __int64 v84; // rax
  __int64 *v85; // rdi
  __int64 v86; // rax
  __int64 *v87; // rbx
  int v88; // eax
  _QWORD *v89; // rax
  __m128i v90; // kr00_16
  __m128i v91; // kr10_16
  __int64 *v92; // rbx
  __int64 v93; // r14
  __int64 v94; // r14
  __int64 v95; // rax
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  signed __int64 v99; // rax
  unsigned __int8 *v100; // r14
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 v103; // rdx
  unsigned __int8 *v104; // rdx
  unsigned __int64 v105; // rax
  int v106; // edx
  unsigned __int64 v107; // rax
  __int64 v108; // rbx
  _QWORD *v109; // rax
  __int16 v110; // ax
  __int64 v111; // rsi
  unsigned __int8 *v112; // rsi
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v121; // [rsp+40h] [rbp-6E0h]
  __int64 v122; // [rsp+48h] [rbp-6D8h]
  unsigned __int8 *v123; // [rsp+58h] [rbp-6C8h]
  __int64 v124; // [rsp+60h] [rbp-6C0h]
  int v125; // [rsp+68h] [rbp-6B8h]
  char v126; // [rsp+70h] [rbp-6B0h]
  unsigned __int8 v127; // [rsp+78h] [rbp-6A8h]
  char v128; // [rsp+84h] [rbp-69Ch]
  unsigned __int8 v129; // [rsp+85h] [rbp-69Bh]
  char v130; // [rsp+86h] [rbp-69Ah]
  unsigned __int8 v131; // [rsp+87h] [rbp-699h]
  __int64 **v132; // [rsp+88h] [rbp-698h]
  __int64 v133; // [rsp+98h] [rbp-688h]
  __m128i v134; // [rsp+A0h] [rbp-680h] BYREF
  __m128i v135; // [rsp+B0h] [rbp-670h] BYREF
  unsigned __int8 *v136; // [rsp+C8h] [rbp-658h] BYREF
  _QWORD **v137; // [rsp+D0h] [rbp-650h] BYREF
  _BYTE *v138; // [rsp+D8h] [rbp-648h]
  _BYTE *i; // [rsp+E0h] [rbp-640h]
  __m128i v140; // [rsp+F0h] [rbp-630h] BYREF
  __m128i v141; // [rsp+100h] [rbp-620h]
  unsigned __int8 *v142[4]; // [rsp+110h] [rbp-610h] BYREF
  __int16 v143; // [rsp+130h] [rbp-5F0h]
  __int64 v144[8]; // [rsp+140h] [rbp-5E0h] BYREF
  unsigned __int8 **v145; // [rsp+180h] [rbp-5A0h] BYREF
  __int64 v146; // [rsp+188h] [rbp-598h]
  unsigned __int8 *v147; // [rsp+190h] [rbp-590h] BYREF
  __m128i v148; // [rsp+198h] [rbp-588h]
  __int64 v149; // [rsp+1A8h] [rbp-578h]
  __m128i v150; // [rsp+1B0h] [rbp-570h]
  __m128i v151; // [rsp+1C0h] [rbp-560h]
  unsigned __int64 *v152; // [rsp+1D0h] [rbp-550h] BYREF
  __int64 v153; // [rsp+1D8h] [rbp-548h]
  _BYTE v154[324]; // [rsp+1E0h] [rbp-540h] BYREF
  int v155; // [rsp+324h] [rbp-3FCh]
  __int64 v156; // [rsp+328h] [rbp-3F8h]
  __int64 *v157; // [rsp+330h] [rbp-3F0h] BYREF
  __int64 v158; // [rsp+338h] [rbp-3E8h]
  unsigned __int8 *v159; // [rsp+340h] [rbp-3E0h]
  __m128i v160; // [rsp+348h] [rbp-3D8h] BYREF
  __int64 v161; // [rsp+358h] [rbp-3C8h]
  __m128i v162; // [rsp+360h] [rbp-3C0h] BYREF
  __m128i v163; // [rsp+370h] [rbp-3B0h] BYREF
  unsigned __int64 *v164; // [rsp+380h] [rbp-3A0h] BYREF
  _OWORD v165[2]; // [rsp+388h] [rbp-398h] BYREF
  __int64 v166; // [rsp+3A8h] [rbp-378h]
  bool v167; // [rsp+3B0h] [rbp-370h]
  __int64 *v168; // [rsp+3B8h] [rbp-368h]
  __int64 v169; // [rsp+3C0h] [rbp-360h]
  char v170; // [rsp+4D0h] [rbp-250h]
  int v171; // [rsp+4D4h] [rbp-24Ch]
  __int64 v172; // [rsp+4D8h] [rbp-248h]
  __int64 *v173; // [rsp+4E0h] [rbp-240h] BYREF
  __int64 v174; // [rsp+4E8h] [rbp-238h]
  _BYTE v175[560]; // [rsp+4F0h] [rbp-230h] BYREF
  __int64 v176; // [rsp+750h] [rbp+30h]

  v16 = a11;
  v123 = **(unsigned __int8 ***)(a1 + 32);
  v140 = 0u;
  v174 = 0x4000000000LL;
  v124 = sub_D4B130(a11);
  v173 = (__int64 *)v175;
  v141 = 0u;
  v122 = sub_AA4E30(v124);
  if ( a16 )
  {
LABEL_77:
    v125 = 1;
    goto LABEL_3;
  }
  v125 = 2;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a13 + 8LL))(a13) )
  {
    v76 = sub_98ACB0(v123, 6u);
    if ( (unsigned __int8)sub_CF7590(v76, &v157) )
    {
      if ( !(_BYTE)v157 )
        goto LABEL_100;
      v58 = **(_QWORD **)(a11 + 32);
      v59 = *(_QWORD *)(v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v59 == v58 + 48 )
      {
        v63 = 0;
      }
      else
      {
        if ( !v59 )
          BUG();
        v60 = *(unsigned __int8 *)(v59 - 24);
        v61 = v59 - 24;
        v62 = (unsigned int)(v60 - 30) < 0xB;
        v63 = 0;
        if ( v62 )
          v63 = v61;
      }
      if ( !(unsigned __int8)sub_D13FF0((__int64)v76, 1, v63, (__int64)a7, 0, 0, 0) )
      {
LABEL_100:
        v125 = 2;
        goto LABEL_3;
      }
    }
    goto LABEL_77;
  }
LABEL_3:
  v17 = *(_QWORD *)(a1 + 32);
  v121 = v17 + 8LL * *(unsigned int *)(a1 + 40);
  if ( v17 == v121 )
    goto LABEL_15;
  v18 = 0;
  v133 = *(_QWORD *)(a1 + 32);
  v131 = 0;
  v132 = 0;
  v127 = 0;
  v129 = 0;
  v126 = 0;
  v130 = 0;
  v128 = 0;
  do
  {
    v19 = *(_QWORD *)(*(_QWORD *)v133 + 16LL);
    if ( v19 )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v19 + 24);
        v21 = *(_BYTE *)v20;
        if ( *(_BYTE *)v20 <= 0x1Cu )
          goto LABEL_23;
        v22 = *(_QWORD *)(v20 + 40);
        if ( *(_BYTE *)(v16 + 84) )
        {
          v23 = *(_QWORD **)(v16 + 64);
          v24 = &v23[*(unsigned int *)(v16 + 76)];
          if ( v23 == v24 )
            goto LABEL_23;
          while ( v22 != *v23 )
          {
            if ( v24 == ++v23 )
              goto LABEL_23;
          }
          if ( v21 == 61 )
            goto LABEL_13;
        }
        else
        {
          if ( !sub_C8CA60(v16 + 56, v22) )
            goto LABEL_23;
          v21 = *(_BYTE *)v20;
          if ( *(_BYTE *)v20 == 61 )
          {
LABEL_13:
            v25 = *(_WORD *)(v20 + 2);
            if ( ((v25 >> 7) & 6) != 0 || (v25 & 1) != 0 )
              goto LABEL_15;
            LOBYTE(v64) = sub_B46500((unsigned __int8 *)v20);
            v129 |= v64;
            v127 |= (unsigned __int8)v64 ^ 1;
            _BitScanReverse64(&v64, 1LL << (v25 >> 1));
            v65 = 63 - (v64 ^ 0x3F);
            if ( !v130 )
              v130 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(*(_QWORD *)a13 + 24LL))(
                       a13,
                       v20,
                       a7,
                       v16);
            if ( !(_BYTE)v18 || v65 > v131 )
            {
              v71 = *(_QWORD *)(v124 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v71 == v124 + 48 )
              {
                v73 = 0;
              }
              else
              {
                if ( !v71 )
                  BUG();
                v72 = *(unsigned __int8 *)(v71 - 24);
                v73 = v71 - 24;
                if ( (unsigned int)(v72 - 30) >= 0xB )
                  v73 = 0;
              }
              v74 = sub_27F0080((unsigned __int8 *)v20, (__int64)a7, a9, v16, a13, a14, v73, a8, a15, 0);
              if ( (_BYTE)v74 )
              {
                v75 = v131;
                v18 = v74;
                if ( v131 < v65 )
                  v75 = v65;
                v131 = v75;
              }
            }
            v126 = 1;
            goto LABEL_83;
          }
        }
        if ( v21 != 62 || (unsigned int)sub_BD2910(v19) != 1 )
          goto LABEL_23;
        v53 = *(_WORD *)(v20 + 2);
        if ( ((v53 >> 7) & 6) != 0 || (v53 & 1) != 0 )
          goto LABEL_15;
        v54 = sub_B46500((unsigned __int8 *)v20);
        v129 |= v54;
        v127 |= !v54;
        v55 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(*(_QWORD *)a13 + 24LL))(
                a13,
                v20,
                a7,
                v16);
        v128 |= v55;
        if ( !(_BYTE)v55 )
          break;
        v56 = v131;
        v18 = v55;
        _BitScanReverse64((unsigned __int64 *)&v57, 1LL << (v53 >> 1));
        if ( v131 < (unsigned __int8)(63 - (v57 ^ 0x3F)) )
          v56 = 63 - (v57 ^ 0x3F);
        v131 = v56;
        if ( v125 == 2 )
          v125 = 0;
LABEL_83:
        if ( v132 )
        {
          if ( *(_BYTE *)v20 == 61 )
            v66 = *(__int64 ***)(v20 + 8);
          else
            v66 = *(__int64 ***)(*(_QWORD *)(v20 - 64) + 8LL);
          if ( v66 != v132 )
            goto LABEL_15;
        }
        else if ( *(_BYTE *)v20 == 61 )
        {
          v132 = *(__int64 ***)(v20 + 8);
        }
        else
        {
          v132 = *(__int64 ***)(*(_QWORD *)(v20 - 64) + 8LL);
        }
        v30 = (unsigned int)v174;
        if ( (_DWORD)v174 )
        {
          if ( v140.m128i_i64[0] || __PAIR128__(v140.m128i_u64[1], 0) != v141.m128i_u64[0] || v141.m128i_i64[1] )
          {
            sub_B91FC0((__int64 *)&v157, v20);
            sub_E01E30(v134.m128i_i64, v140.m128i_i64, (__int64 *)&v157, v67, v68, v69);
            v70 = _mm_loadu_si128(&v135);
            v30 = (unsigned int)v174;
            v140 = _mm_loadu_si128(&v134);
            v141 = v70;
          }
        }
        else
        {
          sub_B91FC0(v134.m128i_i64, v20);
          v29 = _mm_loadu_si128(&v135);
          v30 = (unsigned int)v174;
          v140 = _mm_loadu_si128(&v134);
          v141 = v29;
        }
        if ( v30 + 1 > (unsigned __int64)HIDWORD(v174) )
        {
          sub_C8D5F0((__int64)&v173, v175, v30 + 1, 8u, v27, v28);
          v30 = (unsigned int)v174;
        }
        v173[v30] = v20;
        LODWORD(v174) = v174 + 1;
LABEL_23:
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_24;
      }
      if ( v125 != 2 )
      {
LABEL_108:
        if ( !(_BYTE)v18 )
        {
          v77 = *(_QWORD *)(v124 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v77 == v124 + 48 )
          {
            v79 = 0;
          }
          else
          {
            if ( !v77 )
              BUG();
            v78 = *(unsigned __int8 *)(v77 - 24);
            v79 = 0;
            v80 = v77 - 24;
            if ( (unsigned int)(v78 - 30) < 0xB )
              v79 = v80;
          }
          _BitScanReverse64(&v81, 1LL << (*(_WORD *)(v20 + 2) >> 1));
          LOBYTE(v82) = sub_D305E0(
                          *(_QWORD *)(v20 - 32),
                          *(_QWORD *)(*(_QWORD *)(v20 - 64) + 8LL),
                          63 - (v81 ^ 0x3F),
                          v122,
                          v79,
                          a8,
                          a7,
                          a9);
          v18 = v82;
        }
        goto LABEL_83;
      }
      v83 = *(__int64 **)a2;
      v84 = 8LL * *(unsigned int *)(a2 + 8);
      v85 = (__int64 *)(*(_QWORD *)a2 + v84);
      v86 = v84 >> 5;
      if ( v86 )
      {
        v176 = v16;
        v87 = &v83[4 * v86];
        while ( 1 )
        {
          if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), *v83) )
          {
            v16 = v176;
            goto LABEL_122;
          }
          if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), v83[1]) )
            break;
          if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), v83[2]) )
          {
            v16 = v176;
            v83 += 2;
            goto LABEL_122;
          }
          if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), v83[3]) )
          {
            v16 = v176;
            v83 += 3;
            goto LABEL_122;
          }
          v83 += 4;
          if ( v87 == v83 )
          {
            v16 = v176;
            goto LABEL_153;
          }
        }
        v16 = v176;
        ++v83;
        goto LABEL_122;
      }
LABEL_153:
      v99 = (char *)v85 - (char *)v83;
      if ( (char *)v85 - (char *)v83 != 16 )
      {
        if ( v99 != 24 )
        {
          if ( v99 != 8 )
          {
LABEL_156:
            v125 = 0;
            goto LABEL_108;
          }
LABEL_175:
          if ( (unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), *v83) )
            goto LABEL_156;
LABEL_122:
          v88 = 0;
          if ( v85 != v83 )
            v88 = 2;
          v125 = v88;
          goto LABEL_108;
        }
        if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), *v83) )
          goto LABEL_122;
        ++v83;
      }
      if ( !(unsigned __int8)sub_B19720((__int64)a7, *(_QWORD *)(v20 + 40), *v83) )
        goto LABEL_122;
      ++v83;
      goto LABEL_175;
    }
LABEL_24:
    v133 += 8;
  }
  while ( v121 != v133 );
  if ( (v127 & v129) != 0 )
    goto LABEL_15;
  if ( v129 )
  {
    v31 = sub_9208B0(v122, (__int64)v132);
    v158 = v32;
    v157 = (__int64 *)((unsigned __int64)(v31 + 7) >> 3);
    if ( sub_CA1930(&v157) > (unsigned __int64)(1LL << v131) )
      goto LABEL_15;
  }
  if ( !(_BYTE)v18 )
    goto LABEL_15;
  if ( v125 == 2 )
  {
    v100 = sub_98ACB0(v123, 6u);
    if ( (unsigned __int8)sub_CF7600(v100, &v157) )
    {
      if ( !(_BYTE)v157 || sub_D30730((__int64)v123, (__int64)v132, v122, 0, 0, 0, 0) )
      {
        if ( (unsigned __int8)sub_CF70D0(v100) )
        {
          v101 = **(_QWORD **)(v16 + 32);
          v102 = *(_QWORD *)(v101 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v102 == v101 + 48 )
          {
            v103 = 0;
          }
          else
          {
            if ( !v102 )
              BUG();
            v103 = v102 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v102 - 24) - 30 >= 0xB )
              v103 = 0;
          }
          if ( !(unsigned __int8)sub_D13FF0((__int64)v100, 1, v103, (__int64)a7, 0, 0, 0) )
            goto LABEL_170;
        }
        if ( (unsigned __int8)sub_DF9BA0(a10) || (_BYTE)qword_4FFE4E8 )
LABEL_170:
          v125 = 0;
      }
    }
  }
  if ( v125 && !v126 )
  {
LABEL_15:
    v18 = 0;
    goto LABEL_16;
  }
  v33 = *a14;
  v34 = sub_B2BE50(*a14);
  if ( sub_B6EA50(v34)
    || (v114 = sub_B2BE50(v33),
        v115 = sub_B6F970(v114),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v115 + 48LL))(v115)) )
  {
    sub_B174A0((__int64)&v157, (__int64)"licm", (__int64)"PromoteLoopAccessesToScalar", 27, *v173);
    sub_B18290((__int64)&v157, "Moving accesses to memory location out of the loop", 0x32u);
    v39 = _mm_loadu_si128(&v160);
    v40 = _mm_loadu_si128(&v162);
    LODWORD(v146) = v158;
    v41 = _mm_loadu_si128(&v163);
    v148 = v39;
    BYTE4(v146) = BYTE4(v158);
    v150 = v40;
    v147 = v159;
    v145 = (unsigned __int8 **)&unk_49D9D40;
    v151 = v41;
    v149 = v161;
    v152 = (unsigned __int64 *)v154;
    v153 = 0x400000000LL;
    if ( LODWORD(v165[0]) )
      sub_27EFAF0((__int64)&v152, (__int64)&v164, v35, v36, v37, v38);
    v42 = v164;
    v154[320] = v170;
    v155 = v171;
    v156 = v172;
    v145 = (unsigned __int8 **)&unk_49D9D78;
    v157 = (__int64 *)&unk_49D9D40;
    v43 = 10LL * LODWORD(v165[0]);
    v44 = &v164[v43];
    if ( v164 != &v164[v43] )
    {
      do
      {
        v44 -= 10;
        v45 = v44[4];
        if ( (unsigned __int64 *)v45 != v44 + 6 )
          j_j___libc_free_0(v45);
        if ( (unsigned __int64 *)*v44 != v44 + 2 )
          j_j___libc_free_0(*v44);
      }
      while ( v42 != v44 );
      v42 = v164;
    }
    if ( v42 != (unsigned __int64 *)((char *)v165 + 8) )
      _libc_free((unsigned __int64)v42);
    sub_1049740(a14, (__int64)&v145);
    v46 = v152;
    v145 = (unsigned __int8 **)&unk_49D9D40;
    v47 = &v152[10 * (unsigned int)v153];
    if ( v152 != v47 )
    {
      do
      {
        v47 -= 10;
        v48 = v47[4];
        if ( (unsigned __int64 *)v48 != v47 + 6 )
          j_j___libc_free_0(v48);
        if ( (unsigned __int64 *)*v47 != v47 + 2 )
          j_j___libc_free_0(*v47);
      }
      while ( v46 != v47 );
      v46 = v152;
    }
    if ( v46 != (unsigned __int64 *)v154 )
      _libc_free((unsigned __int64)v46);
  }
  v137 = 0;
  v138 = 0;
  v49 = &v173[(unsigned int)v174];
  v50 = v173;
  for ( i = 0; v49 != v50; ++v50 )
  {
    v52 = (__int64 *)sub_B10CD0(*v50 + 48);
    v51 = v138;
    v157 = v52;
    if ( v138 == i )
    {
      sub_27F1610((__int64)&v137, v138, &v157);
    }
    else
    {
      if ( v138 )
      {
        *(_QWORD *)v138 = v52;
        v51 = v138;
      }
      v138 = v51 + 8;
    }
  }
  v89 = sub_B026E0(v137, (v138 - (_BYTE *)v137) >> 3);
  sub_B10CB0(&v136, (__int64)v89);
  v145 = &v147;
  v146 = 0x1000000000LL;
  sub_11D2BF0((__int64)v144, (__int64)&v145);
  if ( v128 )
  {
    v90 = v140;
    v91 = v141;
  }
  else
  {
    v90 = 0u;
    v91 = 0u;
  }
  v142[0] = v136;
  if ( v136 )
    sub_B96E90((__int64)v142, (__int64)v136, 1);
  v92 = v173;
  v93 = (unsigned int)v174;
  sub_11D3120(&v157, (_BYTE **)v173, (unsigned int)v174, v144, 0, 0);
  v157 = (__int64 *)&off_4A21100;
  v163.m128i_i64[1] = (__int64)v142[0];
  v159 = v123;
  v160.m128i_i64[0] = a2;
  v160.m128i_i64[1] = a3;
  v161 = a4;
  v162.m128i_i64[0] = a5;
  v162.m128i_i64[1] = (__int64)a12;
  v163.m128i_i64[0] = a6;
  if ( v142[0] )
    sub_B976B0((__int64)v142, v142[0], (__int64)&v163.m128i_i64[1]);
  v168 = v92;
  v169 = v93;
  LOBYTE(v164) = v131;
  BYTE1(v164) = v129;
  v165[0] = v90;
  v165[1] = v91;
  v166 = a13;
  v167 = v125 == 0;
  if ( v128 != 1 || v126 )
  {
    v142[0] = (unsigned __int8 *)sub_BD5D20((__int64)v123);
    v142[1] = v104;
    v143 = 773;
    v142[2] = (unsigned __int8 *)".promoted";
    v105 = *(_QWORD *)(v124 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v105 == v124 + 48 )
    {
      v107 = 0;
    }
    else
    {
      if ( !v105 )
        BUG();
      v106 = *(unsigned __int8 *)(v105 - 24);
      v107 = v105 - 24;
      if ( (unsigned int)(v106 - 30) >= 0xB )
        v107 = 0;
    }
    v108 = v107 + 24;
    v109 = sub_BD2C40(80, unk_3F10A14);
    v94 = (__int64)v109;
    if ( v109 )
      sub_B4D230((__int64)v109, (__int64)v132, (__int64)v123, (__int64)v142, v108, 0);
    v110 = *(_WORD *)(v94 + 2);
    if ( v129 )
    {
      v110 &= 0xFC7Fu;
      LOBYTE(v110) = v110 | 0x80;
    }
    *(_WORD *)(v94 + 2) = v110 & 0xFF81 | (2 * v131);
    v142[0] = 0;
    if ( (unsigned __int8 **)(v94 + 48) != v142 )
    {
      v111 = *(_QWORD *)(v94 + 48);
      if ( v111 )
      {
        sub_B91220(v94 + 48, v111);
        v112 = v142[0];
        *(unsigned __int8 **)(v94 + 48) = v142[0];
        if ( v112 )
          sub_B976B0((__int64)v142, v112, v94 + 48);
      }
    }
    if ( (v140.m128i_i64[0] || __PAIR128__(v140.m128i_u64[1], 0) != v141.m128i_u64[0] || v141.m128i_i64[1]) && v130 )
      sub_B9A100(v94, v140.m128i_i64);
    v113 = sub_D694D0(a12, v94, 0, *(_QWORD *)(v94 + 40), 1u, 1u);
    sub_D73680(a12, v113, 1);
    sub_11D33F0(v144, v124, v94);
  }
  else
  {
    v94 = 0;
    v95 = sub_ACADE0(v132);
    sub_11D33F0(v144, v124, v95);
  }
  if ( byte_4F8F8E8[0] )
    nullsub_390();
  sub_11D7E80(&v157, (__int64)&v173);
  if ( byte_4F8F8E8[0] )
    nullsub_390();
  if ( v94 && !*(_QWORD *)(v94 + 16) )
    sub_27EC480((_QWORD *)v94, a13, a12, v96, v97, v98);
  v157 = (__int64 *)&off_4A21100;
  if ( v163.m128i_i64[1] )
    sub_B91220((__int64)&v163.m128i_i64[1], v163.m128i_i64[1]);
  sub_11D2C20(v144);
  if ( v145 != &v147 )
    _libc_free((unsigned __int64)v145);
  if ( v136 )
    sub_B91220((__int64)&v136, (__int64)v136);
  if ( v137 )
    j_j___libc_free_0((unsigned __int64)v137);
LABEL_16:
  if ( v173 != (__int64 *)v175 )
    _libc_free((unsigned __int64)v173);
  return v18;
}
