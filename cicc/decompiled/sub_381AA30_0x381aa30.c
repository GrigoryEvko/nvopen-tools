// Function: sub_381AA30
// Address: 0x381aa30
//
void __fastcall sub_381AA30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  const __m128i *v7; // roff
  __m128i v8; // xmm0
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int16 v11; // dx
  unsigned int v12; // r15d
  int v13; // r12d
  int v14; // eax
  unsigned int v15; // r12d
  __int16 *v16; // rax
  __int64 v17; // r15
  unsigned __int16 v18; // r9
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // r15
  __int128 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rax
  _QWORD *v28; // r15
  __int64 v29; // r10
  unsigned __int64 v30; // r8
  unsigned int v31; // ecx
  __int64 v32; // r9
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int16 v36; // ax
  __int64 v37; // rdx
  __int64 v38; // r11
  bool v39; // al
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rdi
  unsigned int v45; // eax
  unsigned int v46; // eax
  unsigned __int16 *v49; // rax
  unsigned __int16 v50; // r14
  int v51; // edx
  _QWORD *v52; // r13
  __int128 v53; // rax
  int v54; // edx
  unsigned int v55; // esi
  unsigned __int64 v56; // rsi
  unsigned __int64 v57; // rcx
  unsigned int v58; // esi
  unsigned __int64 v59; // rax
  unsigned int v60; // eax
  unsigned int v61; // eax
  unsigned __int16 *v64; // rax
  unsigned int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // r8
  unsigned __int8 *v68; // r14
  __int64 v69; // rdx
  __int64 v70; // r15
  __int128 v71; // rax
  __int64 v72; // r14
  __int64 v73; // rdx
  __int64 v74; // r15
  __int128 v75; // rax
  __int64 v76; // rax
  int v77; // edx
  __int64 v78; // r9
  unsigned int v79; // r11d
  __int64 v80; // r10
  __int64 v81; // rsi
  _QWORD *v82; // rdi
  __m128i v83; // xmm2
  int v84; // edx
  unsigned __int16 *v85; // rax
  unsigned int v86; // eax
  __int64 v87; // rdx
  __int64 v88; // r9
  __int64 v89; // rsi
  int v90; // r11d
  unsigned int v91; // r14d
  _QWORD *v92; // rdi
  __m128i v93; // xmm4
  unsigned __int8 *v94; // rax
  __int128 v95; // kr00_16
  __int64 v96; // rbx
  int v97; // edx
  _QWORD *v98; // r15
  __int128 v99; // rax
  __int64 v100; // rax
  _QWORD *v101; // rbx
  __int64 v102; // r14
  __int64 v103; // rdx
  __int64 v104; // r15
  __int128 v105; // rax
  __int128 v106; // rax
  __int64 v107; // rax
  _QWORD *v108; // rdi
  __int64 v109; // r14
  __int64 v110; // rdx
  __int64 v111; // r15
  __m128i v112; // xmm7
  __int64 v113; // r9
  __int128 v114; // rax
  int v115; // edx
  unsigned int v116; // eax
  __int128 v117; // rax
  unsigned int v119; // edx
  unsigned int v122; // edx
  __int128 v124; // [rsp-20h] [rbp-1B0h]
  __int128 v125; // [rsp-10h] [rbp-1A0h]
  __int128 v126; // [rsp+0h] [rbp-190h]
  __int128 v127; // [rsp+0h] [rbp-190h]
  __int128 v128; // [rsp+0h] [rbp-190h]
  __int128 v129; // [rsp+0h] [rbp-190h]
  unsigned __int16 v130; // [rsp+10h] [rbp-180h]
  unsigned int v131; // [rsp+10h] [rbp-180h]
  unsigned int v132; // [rsp+10h] [rbp-180h]
  __int64 v133; // [rsp+18h] [rbp-178h]
  __int64 v134; // [rsp+18h] [rbp-178h]
  __int64 v135; // [rsp+18h] [rbp-178h]
  __int64 (__fastcall *v136)(__int64, __int64, __int64, __int64, __int64); // [rsp+20h] [rbp-170h]
  __int128 v137; // [rsp+20h] [rbp-170h]
  __int64 v138; // [rsp+20h] [rbp-170h]
  unsigned int v139; // [rsp+20h] [rbp-170h]
  __int128 v140; // [rsp+20h] [rbp-170h]
  __int128 v141; // [rsp+20h] [rbp-170h]
  __int64 v142; // [rsp+30h] [rbp-160h]
  _QWORD *v143; // [rsp+30h] [rbp-160h]
  __int64 v144; // [rsp+30h] [rbp-160h]
  __int64 v145; // [rsp+30h] [rbp-160h]
  __int64 v146; // [rsp+30h] [rbp-160h]
  _QWORD *v147; // [rsp+30h] [rbp-160h]
  __int64 v148; // [rsp+30h] [rbp-160h]
  unsigned int v149; // [rsp+30h] [rbp-160h]
  __int128 v152; // [rsp+50h] [rbp-140h]
  __int128 v153; // [rsp+50h] [rbp-140h]
  __int128 v154; // [rsp+50h] [rbp-140h]
  __int64 v155; // [rsp+60h] [rbp-130h]
  unsigned int v156; // [rsp+60h] [rbp-130h]
  __int64 v157; // [rsp+60h] [rbp-130h]
  __int64 v158; // [rsp+70h] [rbp-120h]
  __int64 v159; // [rsp+70h] [rbp-120h]
  unsigned int v160; // [rsp+70h] [rbp-120h]
  __int64 v161; // [rsp+F0h] [rbp-A0h] BYREF
  int v162; // [rsp+F8h] [rbp-98h]
  __m128i v163; // [rsp+100h] [rbp-90h] BYREF
  __int128 v164; // [rsp+110h] [rbp-80h] BYREF
  __m128i v165; // [rsp+120h] [rbp-70h] BYREF
  __int128 v166; // [rsp+130h] [rbp-60h] BYREF
  __m128i v167; // [rsp+140h] [rbp-50h] BYREF
  __m128i v168; // [rsp+150h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v161 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v161, v6, 1);
  v162 = *(_DWORD *)(a2 + 72);
  v7 = *(const __m128i **)(a2 + 40);
  v8 = _mm_loadu_si128(v7);
  v9 = v7[2].m128i_u64[1];
  v155 = v7[3].m128i_i64[0];
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *(_WORD *)v10;
  v167.m128i_i64[1] = *(_QWORD *)(v10 + 8);
  v167.m128i_i16[0] = v11;
  v12 = (unsigned int)sub_32844A0((unsigned __int16 *)&v167, v6) >> 1;
  if ( (unsigned int)sub_33D4D80(a1[1], v8.m128i_i64[0], v8.m128i_i64[1], 0) > v12
    && (unsigned int)sub_33D4D80(a1[1], v9, v155, 0) > v12 )
  {
    *(_QWORD *)&v164 = 0;
    DWORD2(v164) = 0;
    v165.m128i_i64[0] = 0;
    v165.m128i_i32[2] = 0;
    *(_QWORD *)&v166 = 0;
    DWORD2(v166) = 0;
    v167.m128i_i64[0] = 0;
    v167.m128i_i32[2] = 0;
    sub_375E510((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1], (__int64)&v164, (__int64)&v165);
    sub_375E510((__int64)a1, v9, v155, (__int64)&v166, (__int64)&v167);
    v49 = (unsigned __int16 *)(*(_QWORD *)(v164 + 48) + 16LL * DWORD2(v164));
    v50 = *v49;
    v158 = *((_QWORD *)v49 + 1);
    *(_QWORD *)a3 = sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v161, *v49, v158, v158, v164, v166);
    *(_DWORD *)(a3 + 8) = v51;
    v52 = (_QWORD *)a1[1];
    *(_QWORD *)&v53 = sub_3400E40((__int64)v52, v12 - 1, v50, v158, (__int64)&v161, v8);
    *(_QWORD *)a4 = sub_3406EB0(v52, 0xBFu, (__int64)&v161, v50, v158, v158, *(_OWORD *)a3, v53);
    *(_DWORD *)(a4 + 8) = v54;
    goto LABEL_33;
  }
  v13 = *(_DWORD *)(a2 + 24);
  if ( v13 == 181 )
  {
    if ( sub_33CF170(v9) )
      goto LABEL_56;
    v13 = *(_DWORD *)(a2 + 24);
    if ( v13 != 180 )
      goto LABEL_6;
LABEL_54:
    if ( !sub_33CF460(v9) )
    {
      v13 = *(_DWORD *)(a2 + 24);
      goto LABEL_6;
    }
LABEL_56:
    *(_QWORD *)&v164 = 0;
    v163.m128i_i64[0] = 0;
    v163.m128i_i32[2] = 0;
    DWORD2(v164) = 0;
    v165.m128i_i64[0] = 0;
    v165.m128i_i32[2] = 0;
    *(_QWORD *)&v166 = 0;
    DWORD2(v166) = 0;
    sub_375E510((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1], (__int64)&v163, (__int64)&v164);
    sub_375E510((__int64)a1, v9, v155, (__int64)&v165, (__int64)&v166);
    v64 = (unsigned __int16 *)(*(_QWORD *)(v163.m128i_i64[0] + 48) + 16LL * v163.m128i_u32[2]);
    v139 = *v64;
    v146 = *((_QWORD *)v64 + 1);
    v65 = sub_38137B0(*a1, a1[1], v139, v146);
    v159 = v66;
    v67 = v146;
    v134 = v146;
    v131 = v139;
    v147 = (_QWORD *)a1[1];
    v156 = v65;
    v68 = sub_3400BD0((__int64)v147, 0, (__int64)&v161, v139, v67, 0, v8, 0);
    v70 = v69;
    v140 = v164;
    *(_QWORD *)&v71 = sub_33ED040(v147, 0x14u);
    *((_QWORD *)&v124 + 1) = v70;
    *(_QWORD *)&v124 = v68;
    v72 = sub_340F900(v147, 0xD0u, (__int64)&v161, v156, v159, *((__int64 *)&v140 + 1), v140, v124, v71);
    v74 = v73;
    v148 = a1[1];
    if ( *(_DWORD *)(a2 + 24) == 180 )
    {
      *(_QWORD *)&v117 = sub_34015B0(v148, (__int64)&v161, v131, v134, 0, 0, v8);
      v76 = sub_3288B20(v148, (int)&v161, v131, v134, v72, v74, *(_OWORD *)&v163, v117, 0);
      v80 = v134;
      v79 = v131;
    }
    else
    {
      *(_QWORD *)&v75 = sub_3400BD0(v148, 0, (__int64)&v161, v131, v134, 0, v8, 0);
      v76 = sub_3288B20(v148, (int)&v161, v131, v134, v72, v74, v75, *(_OWORD *)&v163, 0);
      v79 = v131;
      v80 = v134;
    }
    *(_QWORD *)a3 = v76;
    *(_DWORD *)(a3 + 8) = v77;
    v81 = *(unsigned int *)(a2 + 24);
    v82 = (_QWORD *)a1[1];
    *((_QWORD *)&v127 + 1) = 2;
    v83 = _mm_loadu_si128((const __m128i *)&v166);
    *(_QWORD *)&v127 = &v167;
    v167 = _mm_loadu_si128((const __m128i *)&v164);
    v168 = v83;
    *(_QWORD *)a4 = sub_33FC220(v82, v81, (__int64)&v161, v79, v80, v78, v127);
    *(_DWORD *)(a4 + 8) = v84;
    goto LABEL_33;
  }
  if ( v13 == 180 )
    goto LABEL_54;
LABEL_6:
  v14 = *(_DWORD *)(v9 + 24);
  if ( v14 == 35 || v14 == 11 )
  {
    v43 = *(_QWORD *)(v9 + 96);
    v44 = v43 + 24;
    if ( (unsigned int)(v13 - 182) > 1 )
    {
      if ( v13 == 181 )
      {
        v116 = *(_DWORD *)(v43 + 32);
        if ( v116 <= 0x40 )
        {
          _RCX = *(_QWORD *)(v43 + 24);
          v122 = 64;
          __asm { tzcnt   rsi, rcx }
          if ( _RCX )
            v122 = _RSI;
          if ( v116 > v122 )
            v116 = v122;
        }
        else
        {
          v116 = sub_C44590(v44);
        }
        v15 = (v12 <= v116) + 18;
        goto LABEL_13;
      }
      if ( v13 <= 181 )
      {
        if ( v13 != 180 )
          goto LABEL_88;
        if ( *(_DWORD *)(v43 + 32) > 0x40u )
        {
          v61 = sub_C445E0(v44);
        }
        else
        {
          v61 = 64;
          _RDX = ~*(_QWORD *)(v43 + 24);
          __asm { tzcnt   rcx, rdx }
          if ( _RDX )
            v61 = _RCX;
        }
        v15 = (v12 <= v61) + 20;
        goto LABEL_13;
      }
LABEL_41:
      if ( v13 != 183 )
        goto LABEL_88;
      v60 = *(_DWORD *)(v43 + 32);
      if ( v60 <= 0x40 )
      {
        _RCX = *(_QWORD *)(v43 + 24);
        v119 = 64;
        __asm { tzcnt   rsi, rcx }
        if ( _RCX )
          v119 = _RSI;
        if ( v60 > v119 )
          v60 = v119;
      }
      else
      {
        v60 = sub_C44590(v44);
      }
      v15 = (v12 <= v60) + 10;
      goto LABEL_13;
    }
    v45 = *(_DWORD *)(v43 + 32);
    if ( v45 <= 0x40 )
    {
      if ( v45 )
      {
        v55 = 64;
        if ( *(_QWORD *)(v43 + 24) << (64 - (unsigned __int8)v45) != -1 )
        {
          _BitScanReverse64(&v56, ~(*(_QWORD *)(v43 + 24) << (64 - (unsigned __int8)v45)));
          v55 = v56 ^ 0x3F;
        }
      }
      else
      {
        v55 = 0;
      }
      if ( v12 > v55 )
      {
        v57 = *(_QWORD *)(v43 + 24);
        v58 = v45 - 64;
        if ( v57 )
        {
          _BitScanReverse64(&v59, v57);
          v45 = v58 + (v59 ^ 0x3F);
        }
LABEL_22:
        if ( v12 > v45 )
        {
          if ( v13 == 182 )
          {
            if ( *(_DWORD *)(v43 + 32) > 0x40u )
            {
              v46 = sub_C445E0(v44);
            }
            else
            {
              v46 = 64;
              _RDX = ~*(_QWORD *)(v43 + 24);
              __asm { tzcnt   rcx, rdx }
              if ( _RDX )
                v46 = _RCX;
            }
            v15 = (v12 <= v46) + 12;
            goto LABEL_13;
          }
          goto LABEL_41;
        }
      }
    }
    else
    {
      v138 = *(_QWORD *)(v9 + 96);
      v144 = v43 + 24;
      if ( v12 > (unsigned int)sub_C44500(v44) )
      {
        v45 = sub_C444A0(v144);
        v43 = v138;
        v44 = v144;
        goto LABEL_22;
      }
    }
    v163.m128i_i64[0] = 0;
    v163.m128i_i32[2] = 0;
    *(_QWORD *)&v164 = 0;
    DWORD2(v164) = 0;
    v165.m128i_i64[0] = 0;
    v165.m128i_i32[2] = 0;
    *(_QWORD *)&v166 = 0;
    DWORD2(v166) = 0;
    sub_375E510((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1], (__int64)&v163, (__int64)&v164);
    sub_375E510((__int64)a1, v9, v155, (__int64)&v165, (__int64)&v166);
    v85 = (unsigned __int16 *)(*(_QWORD *)(v163.m128i_i64[0] + 48) + 16LL * v163.m128i_u32[2]);
    v160 = *v85;
    v157 = *((_QWORD *)v85 + 1);
    v86 = sub_38137B0(*a1, a1[1], v160, v157);
    v89 = *(unsigned int *)(a2 + 24);
    v149 = v86;
    if ( (_DWORD)v89 == 182 )
    {
      v90 = 182;
      v91 = 12;
    }
    else if ( (int)v89 > 182 )
    {
      if ( (_DWORD)v89 != 183 )
        goto LABEL_88;
      v90 = 183;
      v91 = 10;
    }
    else if ( (_DWORD)v89 == 180 )
    {
      v90 = 182;
      v91 = 20;
    }
    else
    {
      if ( (_DWORD)v89 != 181 )
        goto LABEL_88;
      v90 = 183;
      v91 = 18;
    }
    v92 = (_QWORD *)a1[1];
    v93 = _mm_loadu_si128((const __m128i *)&v166);
    *((_QWORD *)&v128 + 1) = 2;
    *(_QWORD *)&v128 = &v167;
    v167 = _mm_loadu_si128((const __m128i *)&v164);
    v168 = v93;
    v132 = v90;
    v135 = v87;
    v94 = sub_33FC220(v92, v89, (__int64)&v161, v160, v157, v88, v128);
    v95 = v164;
    v96 = *((_QWORD *)&v166 + 1);
    *(_QWORD *)a4 = v94;
    *(_DWORD *)(a4 + 8) = v97;
    v98 = (_QWORD *)a1[1];
    v152 = __PAIR128__(v96, v166);
    *(_QWORD *)&v99 = sub_33ED040(v98, v91);
    v100 = sub_340F900(v98, 0xD0u, (__int64)&v161, v149, v135, *((__int64 *)&v95 + 1), v95, v152, v99);
    v101 = (_QWORD *)a1[1];
    v102 = v100;
    v153 = (__int128)_mm_loadu_si128((const __m128i *)&v164);
    v104 = v103;
    v141 = v166;
    *(_QWORD *)&v105 = sub_33ED040(v101, 0x11u);
    *(_QWORD *)&v106 = sub_340F900(v101, 0xD0u, (__int64)&v161, v149, v135, *((__int64 *)&v141 + 1), v153, v141, v105);
    v154 = v106;
    v107 = sub_3288B20(a1[1], (int)&v161, v160, v157, v102, v104, *(_OWORD *)&v163, *(_OWORD *)&v165, 0);
    v108 = (_QWORD *)a1[1];
    v109 = v107;
    v111 = v110;
    *((_QWORD *)&v129 + 1) = 2;
    v112 = _mm_loadu_si128(&v165);
    *(_QWORD *)&v129 = &v167;
    v167 = _mm_loadu_si128(&v163);
    v168 = v112;
    *(_QWORD *)&v114 = sub_33FC220(v108, v132, (__int64)&v161, v160, v157, v113, v129);
    *((_QWORD *)&v125 + 1) = v111;
    *(_QWORD *)&v125 = v109;
    *(_QWORD *)a3 = sub_3288B20(a1[1], (int)&v161, v160, v157, v154, *((__int64 *)&v154 + 1), v114, v125, 0);
    *(_DWORD *)(a3 + 8) = v115;
LABEL_33:
    v42 = v161;
    if ( !v161 )
      return;
LABEL_16:
    sub_B91220((__int64)&v161, v42);
    return;
  }
  if ( v13 == 182 )
  {
    v15 = 12;
  }
  else if ( v13 > 182 )
  {
    if ( v13 != 183 )
      goto LABEL_88;
    v15 = 10;
  }
  else
  {
    if ( v13 != 180 )
    {
      if ( v13 == 181 )
      {
        v15 = 18;
        goto LABEL_13;
      }
LABEL_88:
      BUG();
    }
    v15 = 20;
  }
LABEL_13:
  v16 = *(__int16 **)(a2 + 48);
  v17 = *a1;
  v18 = *v16;
  v19 = *((_QWORD *)v16 + 1);
  v20 = a1[1];
  v130 = v18;
  v133 = v18;
  v136 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v142 = *(_QWORD *)(v20 + 64);
  v21 = sub_2E79000(*(__int64 **)(v20 + 40));
  v22 = v136(v17, v21, v142, v133, v19);
  v24 = v23;
  *(_QWORD *)&v137 = v9;
  LODWORD(v133) = v22;
  *((_QWORD *)&v137 + 1) = v155;
  v143 = (_QWORD *)a1[1];
  *(_QWORD *)&v25 = sub_33ED040(v143, v15);
  v27 = sub_340F900(v143, 0xD0u, (__int64)&v161, v133, v24, v26, *(_OWORD *)&v8, v137, v25);
  v28 = (_QWORD *)a1[1];
  v29 = v27;
  v30 = v9;
  v31 = v130;
  v32 = v155;
  v34 = v33;
  v35 = *(_QWORD *)(v27 + 48) + 16LL * v33;
  v36 = *(_WORD *)v35;
  v37 = *(_QWORD *)(v35 + 8);
  v38 = v34;
  v167.m128i_i16[0] = v36;
  v167.m128i_i64[1] = v37;
  if ( v36 )
  {
    v39 = (unsigned __int16)(v36 - 17) <= 0xD3u;
  }
  else
  {
    v145 = v29;
    v39 = sub_30070B0((__int64)&v167);
    v31 = v130;
    v29 = v145;
    v38 = v34;
    v30 = v9;
    v32 = v155;
  }
  *((_QWORD *)&v126 + 1) = v32;
  *(_QWORD *)&v126 = v30;
  v40 = sub_340EC60(v28, 205 - ((unsigned int)!v39 - 1), (__int64)&v161, v31, v19, 0, v29, v38, *(_OWORD *)&v8, v126);
  sub_375BC20(a1, v40, v41, a3, a4, v8);
  v42 = v161;
  if ( v161 )
    goto LABEL_16;
}
