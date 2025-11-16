// Function: sub_342BA70
// Address: 0x342ba70
//
__int64 __fastcall sub_342BA70(__int64 a1)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int); // r13
  __int64 (__fastcall *v7)(__int64, unsigned __int16); // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int32 v20; // edx
  __int32 v21; // r14d
  __int64 (*v22)(); // rax
  __m128i *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r13
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  __int64 *v28; // rbx
  _QWORD *v29; // r15
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v34; // rax
  __int64 v35; // r14
  unsigned __int8 *v36; // rsi
  __int64 v37; // rax
  __int64 *v38; // rbx
  _QWORD *v39; // r13
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  __int64 v45; // rbx
  __int64 v46; // r12
  unsigned int v47; // esi
  __int64 v48; // rdi
  int v49; // r11d
  unsigned int v50; // r10d
  __int64 *v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rcx
  const void *v54; // r8
  __int64 v55; // rcx
  __int64 *v56; // rdi
  __int64 v57; // rax
  __int64 (*v58)(); // rdx
  __int64 (*v59)(); // rax
  int v60; // esi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // r12
  int v65; // eax
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rax
  _QWORD *v70; // rbx
  unsigned int v71; // esi
  __int64 v72; // r9
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rdi
  int v76; // esi
  __int64 v77; // rdx
  __int64 v78; // rdi
  unsigned int v79; // r10d
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  int v83; // esi
  _QWORD *v84; // r11
  __int64 v85; // rax
  __int64 v86; // rdx
  int v87; // ecx
  __int64 v88; // r8
  __int64 v89; // r9
  unsigned __int64 v90; // rax
  int v91; // ecx
  int v92; // ecx
  int v93; // edx
  __int64 v94; // rax
  int v95; // r11d
  int v96; // r11d
  __int64 v97; // r10
  unsigned int v98; // ecx
  __int64 v99; // r8
  int v100; // edi
  __int64 *v101; // rsi
  int v102; // r10d
  int v103; // r10d
  int v104; // esi
  __int64 v105; // r9
  unsigned int v106; // r13d
  __int64 *v107; // rcx
  __int64 v108; // rdi
  int v109; // ecx
  __int64 v110; // rdx
  int v111; // ecx
  int v112; // edx
  int v113; // esi
  int v114; // esi
  __int64 v115; // r8
  unsigned int v116; // edx
  int i; // edi
  __int64 *v118; // rcx
  __int64 v119; // r9
  int v120; // esi
  int v121; // esi
  int v122; // edi
  __int64 v123; // rcx
  unsigned int j; // edx
  __int64 v125; // r8
  unsigned int v126; // edx
  unsigned int v127; // edx
  __int64 v128; // [rsp+0h] [rbp-C0h]
  __int64 v129; // [rsp+8h] [rbp-B8h]
  __int64 v130; // [rsp+10h] [rbp-B0h]
  int v131; // [rsp+1Ch] [rbp-A4h]
  __int64 v132; // [rsp+20h] [rbp-A0h]
  __int32 v133; // [rsp+20h] [rbp-A0h]
  __int64 v134; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v135; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int8 *v136; // [rsp+40h] [rbp-80h] BYREF
  __int64 v137; // [rsp+48h] [rbp-78h]
  __int64 v138; // [rsp+50h] [rbp-70h]
  __m128i v139; // [rsp+60h] [rbp-60h] BYREF
  __int64 v140; // [rsp+70h] [rbp-50h]
  __int64 v141; // [rsp+78h] [rbp-48h]
  __int64 v142; // [rsp+80h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 24);
  v3 = v2[93];
  v134 = v3;
  v4 = sub_B2E500(*v2);
  v5 = *(_QWORD *)(a1 + 808);
  v132 = v4;
  v130 = *(_QWORD *)(v3 + 16);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v5 + 32LL);
  v7 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v5 + 552LL);
  v8 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  if ( v6 == sub_2D42F30 )
  {
    v9 = 2;
    v10 = sub_AE2980(v8, 0)[1];
    if ( v10 != 1 )
    {
      v9 = 3;
      if ( v10 != 2 )
      {
        v9 = 4;
        if ( v10 != 4 )
        {
          v9 = 5;
          if ( v10 != 8 )
          {
            v9 = 6;
            if ( v10 != 16 )
            {
              v9 = 7;
              if ( v10 != 32 )
              {
                v9 = 8;
                if ( v10 != 64 )
                  v9 = 9 * (unsigned int)(v10 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v9 = (unsigned int)v6(v5, v8, 0);
  }
  if ( v7 == sub_2EC09E0 )
    v129 = *(_QWORD *)(v5 + 8LL * (unsigned __int16)v9 + 112);
  else
    v129 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v7)(v5, v9, 0);
  v131 = sub_B2A630(v132);
  if ( (unsigned int)(v131 - 7) <= 3 )
  {
    v14 = sub_AA4FF0(v130);
    v15 = v14;
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)(v14 - 24) != 81 )
      return 1;
    v16 = *(_QWORD *)(v14 - 8);
    if ( !v16 )
      return 1;
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 + 24);
      if ( *(_BYTE *)v17 == 85 )
      {
        v18 = *(_QWORD *)(v17 - 32);
        if ( v18 )
        {
          if ( !*(_BYTE *)v18
            && *(_QWORD *)(v18 + 24) == *(_QWORD *)(v17 + 80)
            && (*(_BYTE *)(v18 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v18 + 36) - 75) <= 1 )
          {
            break;
          }
        }
      }
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        return 1;
    }
    v19 = *(_QWORD *)(a1 + 808);
    v20 = 0;
    v21 = 0;
    v22 = *(__int64 (**)())(*(_QWORD *)v19 + 872LL);
    if ( v22 != sub_2E2F9C0 )
    {
      v21 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v22)(v19, v132, 0);
      v20 = (unsigned __int16)v21;
    }
    v139.m128i_i32[0] = v20;
    v139.m128i_i64[1] = -1;
    v140 = -1;
    v23 = *(__m128i **)(v134 + 192);
    if ( v23 == *(__m128i **)(v134 + 200) )
    {
      sub_2E341F0((unsigned __int64 *)(v134 + 184), v23, &v139);
    }
    else
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(&v139);
        v23[1].m128i_i64[0] = v140;
        v23 = *(__m128i **)(v134 + 192);
      }
      *(_QWORD *)(v134 + 192) = (char *)v23 + 24;
    }
    v133 = sub_3750910(*(_QWORD *)(a1 + 24), v15 - 24, v129);
    v24 = **(_QWORD **)(a1 + 72);
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 800LL;
    if ( v24 )
    {
      v26 = *(unsigned __int8 **)(v24 + 48);
      v135 = v26;
      if ( v26 )
      {
        sub_B96E90((__int64)&v135, (__int64)v26, 1);
        v136 = v135;
        if ( v135 )
        {
          sub_B976B0((__int64)&v135, v135, (__int64)&v136);
          v27 = *(_QWORD *)(a1 + 24);
          v135 = 0;
          v137 = 0;
          v138 = 0;
          v28 = *(__int64 **)(v27 + 752);
          v29 = *(_QWORD **)(v134 + 32);
          v139.m128i_i64[0] = (__int64)v136;
          if ( v136 )
            sub_B96E90((__int64)&v139, (__int64)v136, 1);
          goto LABEL_35;
        }
      }
      else
      {
        v136 = 0;
      }
    }
    else
    {
      v135 = 0;
      v136 = 0;
    }
    v94 = *(_QWORD *)(a1 + 24);
    v137 = 0;
    v138 = 0;
    v28 = *(__int64 **)(v94 + 752);
    v29 = *(_QWORD **)(v134 + 32);
    v139.m128i_i64[0] = 0;
LABEL_35:
    v30 = (__int64)sub_2E7B380(v29, v25, (unsigned __int8 **)&v139, 0);
    if ( v139.m128i_i64[0] )
      sub_B91220((__int64)&v139, v139.m128i_i64[0]);
    sub_2E31040((__int64 *)(v134 + 40), v30);
    v31 = *v28;
    v32 = *(_QWORD *)v30;
    *(_QWORD *)(v30 + 8) = v28;
    v31 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v30 = v31 | v32 & 7;
    *(_QWORD *)(v31 + 8) = v30;
    *v28 = v30 | *v28 & 7;
    if ( v137 )
      sub_2E882B0(v30, (__int64)v29, v137);
    if ( v138 )
      sub_2E88680(v30, (__int64)v29, v138);
    v139.m128i_i64[0] = 0x10000000;
    v139.m128i_i32[2] = v133;
    v140 = 0;
    v141 = 0;
    v142 = 0;
    sub_2E8EAD0(v30, (__int64)v29, &v139);
    v139.m128i_i64[0] = 0x40000000;
    v140 = 0;
    v139.m128i_i32[2] = v21;
    v141 = 0;
    v142 = 0;
    sub_2E8EAD0(v30, (__int64)v29, &v139);
    if ( v136 )
      sub_B91220((__int64)&v136, (__int64)v136);
    if ( v135 )
      sub_B91220((__int64)&v135, (__int64)v135);
    return 1;
  }
  v128 = sub_2E7D350(*(_QWORD *)(a1 + 40), (unsigned __int8 *)v134, v11, v12, v13);
  v34 = **(_QWORD **)(a1 + 72);
  v35 = *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 160LL;
  if ( !v34 )
  {
    v135 = 0;
    v136 = 0;
    goto LABEL_73;
  }
  v36 = *(unsigned __int8 **)(v34 + 48);
  v135 = v36;
  if ( !v36 )
  {
    v136 = 0;
    goto LABEL_73;
  }
  sub_B96E90((__int64)&v135, (__int64)v36, 1);
  v136 = v135;
  if ( !v135 )
  {
LABEL_73:
    v61 = *(_QWORD *)(a1 + 24);
    v137 = 0;
    v138 = 0;
    v38 = *(__int64 **)(v61 + 752);
    v39 = *(_QWORD **)(v134 + 32);
    v139.m128i_i64[0] = 0;
    goto LABEL_51;
  }
  sub_B976B0((__int64)&v135, v135, (__int64)&v136);
  v37 = *(_QWORD *)(a1 + 24);
  v135 = 0;
  v137 = 0;
  v138 = 0;
  v38 = *(__int64 **)(v37 + 752);
  v39 = *(_QWORD **)(v134 + 32);
  v139.m128i_i64[0] = (__int64)v136;
  if ( v136 )
    sub_B96E90((__int64)&v139, (__int64)v136, 1);
LABEL_51:
  v40 = (__int64)sub_2E7B380(v39, v35, (unsigned __int8 **)&v139, 0);
  if ( v139.m128i_i64[0] )
    sub_B91220((__int64)&v139, v139.m128i_i64[0]);
  sub_2E31040((__int64 *)(v134 + 40), v40);
  v41 = *v38;
  v42 = *(_QWORD *)v40;
  *(_QWORD *)(v40 + 8) = v38;
  v41 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v40 = v41 | v42 & 7;
  *(_QWORD *)(v41 + 8) = v40;
  *v38 = v40 | *v38 & 7;
  if ( v137 )
    sub_2E882B0(v40, (__int64)v39, v137);
  if ( v138 )
    sub_2E88680(v40, (__int64)v39, v138);
  v139.m128i_i8[0] = 15;
  v140 = 0;
  v139.m128i_i32[0] &= 0xFFF000FF;
  v141 = v128;
  v139.m128i_i32[2] = 0;
  LODWORD(v142) = 0;
  sub_2E8EAD0(v40, (__int64)v39, &v139);
  if ( v136 )
    sub_B91220((__int64)&v136, (__int64)v136);
  if ( v135 )
    sub_B91220((__int64)&v135, (__int64)v135);
  v43 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 40) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL));
  v44 = *(__int64 (**)())(*(_QWORD *)v43 + 96LL);
  if ( v44 != sub_2E0FEB0 )
  {
    v77 = ((__int64 (__fastcall *)(__int64, _QWORD))v44)(v43, *(_QWORD *)(a1 + 40));
    if ( v77 )
    {
      v78 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
      v79 = (unsigned int)(*(_DWORD *)(v78 + 376) + 31) >> 5;
      if ( (unsigned int)(*(_DWORD *)(v78 + 376) + 31) <= 0x3F )
      {
        LODWORD(v81) = 0;
      }
      else
      {
        v80 = 0;
        v81 = ((v79 - 2) >> 1) + 1;
        v82 = 8 * v81;
        do
        {
          v83 = *(_DWORD *)(v77 + v80);
          v84 = (_QWORD *)(v80 + *(_QWORD *)(v78 + 312));
          v85 = (unsigned int)~*(_DWORD *)(v77 + v80 + 4);
          v80 += 8;
          *v84 |= (unsigned int)~v83 | (unsigned __int64)(v85 << 32);
        }
        while ( v82 != v80 );
        v77 += v82;
        v79 &= 1u;
      }
      if ( v79 )
      {
        v86 = v77 + 4;
        v87 = 0;
        v88 = 8LL * (unsigned int)v81;
        v89 = v86;
        while ( 1 )
        {
          v90 = (unsigned __int64)(unsigned int)~*(_DWORD *)(v86 - 4) << v87;
          v87 += 32;
          *(_QWORD *)(v88 + *(_QWORD *)(v78 + 312)) |= v90;
          if ( v86 == v89 )
            break;
          v86 += 4;
        }
      }
      v91 = *(_DWORD *)(v78 + 376) & 0x3F;
      if ( v91 )
        *(_QWORD *)(*(_QWORD *)(v78 + 312) + 8LL * *(unsigned int *)(v78 + 320) - 8) &= ~(-1LL << v91);
    }
  }
  if ( v131 != 12 )
  {
    v45 = *(_QWORD *)(a1 + 72);
    v46 = *(_QWORD *)(a1 + 40);
    v47 = *(_DWORD *)(v45 + 1008);
    if ( v47 )
    {
      v48 = *(_QWORD *)(v45 + 992);
      v49 = 1;
      v50 = (v47 - 1) & (((unsigned int)v134 >> 4) ^ ((unsigned int)v134 >> 9));
      v51 = (__int64 *)(v48 + 40LL * v50);
      v52 = 0;
      v53 = *v51;
      if ( v134 == *v51 )
        goto LABEL_65;
      while ( v53 != -4096 )
      {
        if ( v52 || v53 != -8192 )
          v51 = v52;
        v50 = (v47 - 1) & (v49 + v50);
        v53 = *(_QWORD *)(v48 + 40LL * v50);
        if ( v134 == v53 )
        {
          v51 = (__int64 *)(v48 + 40LL * v50);
LABEL_65:
          v54 = (const void *)v51[1];
          v55 = *((unsigned int *)v51 + 4);
LABEL_66:
          sub_2E7DA40(v46, v128, v54, v55);
          v56 = *(__int64 **)(a1 + 808);
          v57 = *v56;
          v58 = *(__int64 (**)())(*v56 + 872);
          if ( v58 != sub_2E2F9C0 )
          {
            v76 = ((__int64 (__fastcall *)(__int64 *, __int64))v58)(v56, v132);
            if ( v76 )
              *(_DWORD *)(*(_QWORD *)(a1 + 24) + 884LL) = sub_2E343A0((_QWORD *)v134, v76, v129);
            v56 = *(__int64 **)(a1 + 808);
            v57 = *v56;
          }
          v59 = *(__int64 (**)())(v57 + 880);
          if ( v59 != sub_2E2F9D0 )
          {
            v60 = ((__int64 (__fastcall *)(__int64 *, __int64))v59)(v56, v132);
            if ( v60 )
              *(_DWORD *)(*(_QWORD *)(a1 + 24) + 888LL) = sub_2E343A0((_QWORD *)v134, v60, v129);
          }
          return 1;
        }
        ++v49;
        v52 = v51;
        v51 = (__int64 *)(v48 + 40LL * v50);
      }
      v92 = *(_DWORD *)(v45 + 1000);
      if ( !v52 )
        v52 = v51;
      ++*(_QWORD *)(v45 + 984);
      v93 = v92 + 1;
      if ( 4 * (v92 + 1) < 3 * v47 )
      {
        if ( v47 - *(_DWORD *)(v45 + 1004) - v93 > v47 >> 3 )
        {
LABEL_113:
          *(_DWORD *)(v45 + 1000) = v93;
          if ( *v52 != -4096 )
            --*(_DWORD *)(v45 + 1004);
          v54 = v52 + 3;
          v55 = 0;
          v52[1] = (__int64)(v52 + 3);
          *v52 = v134;
          v52[2] = 0x400000000LL;
          goto LABEL_66;
        }
        sub_3383980(v45 + 984, v47);
        v102 = *(_DWORD *)(v45 + 1008);
        if ( v102 )
        {
          v103 = v102 - 1;
          v104 = 1;
          v105 = *(_QWORD *)(v45 + 992);
          v106 = v103 & (((unsigned int)v134 >> 4) ^ ((unsigned int)v134 >> 9));
          v93 = *(_DWORD *)(v45 + 1000) + 1;
          v107 = 0;
          v52 = (__int64 *)(v105 + 40LL * v106);
          v108 = *v52;
          if ( v134 != *v52 )
          {
            while ( v108 != -4096 )
            {
              if ( v108 == -8192 && !v107 )
                v107 = v52;
              v106 = v103 & (v104 + v106);
              v52 = (__int64 *)(v105 + 40LL * v106);
              v108 = *v52;
              if ( v134 == *v52 )
                goto LABEL_113;
              ++v104;
            }
            if ( v107 )
              v52 = v107;
          }
          goto LABEL_113;
        }
LABEL_193:
        ++*(_DWORD *)(v45 + 1000);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v45 + 984);
    }
    sub_3383980(v45 + 984, 2 * v47);
    v95 = *(_DWORD *)(v45 + 1008);
    if ( v95 )
    {
      v96 = v95 - 1;
      v97 = *(_QWORD *)(v45 + 992);
      v93 = *(_DWORD *)(v45 + 1000) + 1;
      v98 = v96 & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
      v52 = (__int64 *)(v97 + 40LL * v98);
      v99 = *v52;
      if ( v134 != *v52 )
      {
        v100 = 1;
        v101 = 0;
        while ( v99 != -4096 )
        {
          if ( v99 == -8192 && !v101 )
            v101 = v52;
          v98 = v96 & (v100 + v98);
          v52 = (__int64 *)(v97 + 40LL * v98);
          v99 = *v52;
          if ( v134 == *v52 )
            goto LABEL_113;
          ++v100;
        }
        if ( v101 )
          v52 = v101;
      }
      goto LABEL_113;
    }
    goto LABEL_193;
  }
  v62 = sub_AA4FF0(v130);
  v63 = v62;
  if ( !v62 )
    BUG();
  if ( *(_BYTE *)(v62 - 24) == 81 )
  {
    v64 = *(_QWORD *)(v134 + 32);
    v65 = *(_DWORD *)(v62 - 20) & 0x7FFFFFF;
    if ( v65 == 2 )
    {
      if ( sub_AC30F0(*(_QWORD *)(v63 - 88)) )
        return 1;
      v65 = *(_DWORD *)(v63 - 20) & 0x7FFFFFF;
    }
    if ( v65 != 1 )
    {
      v66 = *(_QWORD *)(v63 - 8);
      if ( v66 )
      {
        while ( 1 )
        {
          v67 = *(_QWORD *)(v66 + 24);
          if ( *(_BYTE *)v67 == 85 )
          {
            v68 = *(_QWORD *)(v67 - 32);
            if ( v68 )
            {
              if ( !*(_BYTE *)v68
                && *(_QWORD *)(v68 + 24) == *(_QWORD *)(v67 + 80)
                && (*(_BYTE *)(v68 + 33) & 0x20) != 0
                && *(_DWORD *)(v68 + 36) == 14196 )
              {
                break;
              }
            }
          }
          v66 = *(_QWORD *)(v66 + 8);
          if ( !v66 )
            return 1;
        }
        v69 = *(_QWORD *)(v67 + 32 * (1LL - (*(_DWORD *)(v67 + 4) & 0x7FFFFFF)));
        v70 = *(_QWORD **)(v69 + 24);
        if ( *(_DWORD *)(v69 + 32) > 0x40u )
          v70 = (_QWORD *)*v70;
        v71 = *(_DWORD *)(v64 + 512);
        if ( v71 )
        {
          v72 = *(_QWORD *)(v64 + 496);
          v73 = (v71 - 1) & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
          v74 = v72 + 16 * v73;
          v75 = *(_QWORD *)v74;
          if ( v134 == *(_QWORD *)v74 )
          {
LABEL_91:
            *(_DWORD *)(v74 + 8) = (_DWORD)v70;
            return 1;
          }
          v109 = 1;
          v110 = 0;
          while ( v75 != -4096 )
          {
            if ( v75 == -8192 && !v110 )
              v110 = v74;
            v73 = (v71 - 1) & ((_DWORD)v73 + v109);
            v74 = v72 + 16 * v73;
            v75 = *(_QWORD *)v74;
            if ( v134 == *(_QWORD *)v74 )
              goto LABEL_91;
            ++v109;
          }
          v111 = *(_DWORD *)(v64 + 504);
          if ( v110 )
            v74 = v110;
          ++*(_QWORD *)(v64 + 488);
          v112 = v111 + 1;
          if ( 4 * (v111 + 1) < 3 * v71 )
          {
            if ( v71 - *(_DWORD *)(v64 + 508) - v112 > v71 >> 3 )
              goto LABEL_147;
            sub_2E515B0(v64 + 488, v71);
            v113 = *(_DWORD *)(v64 + 512);
            if ( v113 )
            {
              v114 = v113 - 1;
              v74 = 0;
              v116 = v114 & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
              for ( i = 1; ; ++i )
              {
                v115 = *(_QWORD *)(v64 + 496);
                v118 = (__int64 *)(v115 + 16LL * v116);
                v119 = *v118;
                if ( v134 == *v118 )
                {
                  v112 = *(_DWORD *)(v64 + 504) + 1;
                  v74 = (__int64)v118;
                  goto LABEL_147;
                }
                if ( v119 == -4096 )
                  break;
                if ( v119 != -8192 || v74 )
                  v118 = (__int64 *)v74;
                v126 = i + v116;
                v74 = (__int64)v118;
                v116 = v114 & v126;
              }
              if ( !v74 )
                v74 = v115 + 16LL * v116;
              v112 = *(_DWORD *)(v64 + 504) + 1;
LABEL_147:
              *(_DWORD *)(v64 + 504) = v112;
              if ( *(_QWORD *)v74 != -4096 )
                --*(_DWORD *)(v64 + 508);
              *(_DWORD *)(v74 + 8) = 0;
              *(_QWORD *)v74 = v134;
              goto LABEL_91;
            }
            goto LABEL_192;
          }
        }
        else
        {
          ++*(_QWORD *)(v64 + 488);
        }
        sub_2E515B0(v64 + 488, 2 * v71);
        v120 = *(_DWORD *)(v64 + 512);
        if ( v120 )
        {
          v121 = v120 - 1;
          v122 = 1;
          v123 = 0;
          for ( j = v121 & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4)); ; j = v121 & v127 )
          {
            v74 = *(_QWORD *)(v64 + 496) + 16LL * j;
            v125 = *(_QWORD *)v74;
            if ( v134 == *(_QWORD *)v74 )
            {
              v112 = *(_DWORD *)(v64 + 504) + 1;
              goto LABEL_147;
            }
            if ( v125 == -4096 )
              break;
            if ( v123 || v125 != -8192 )
              v74 = v123;
            v127 = v122 + j;
            v123 = v74;
            ++v122;
          }
          if ( v123 )
            v74 = v123;
          v112 = *(_DWORD *)(v64 + 504) + 1;
          goto LABEL_147;
        }
LABEL_192:
        ++*(_DWORD *)(v64 + 504);
        BUG();
      }
    }
  }
  return 1;
}
