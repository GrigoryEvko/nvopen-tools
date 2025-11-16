// Function: sub_32A6850
// Address: 0x32a6850
//
__int64 __fastcall sub_32A6850(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  unsigned int v5; // r13d
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __int16 *v8; // rax
  __int64 v9; // rsi
  __int16 v10; // bx
  __int64 v11; // rax
  __int64 v12; // rdi
  __m128i si128; // xmm4
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v17; // rcx
  __int64 v18; // r8
  bool v19; // zf
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int16 v22; // cx
  __int64 v23; // r9
  int v24; // r9d
  char v25; // al
  __int64 v26; // rdi
  unsigned int v27; // r8d
  __int64 v28; // r9
  unsigned __int16 v29; // r10
  int v30; // r9d
  int v31; // eax
  int v32; // r9d
  const __m128i *v33; // rax
  __m128i v34; // xmm5
  bool v35; // dl
  unsigned __int64 v36; // rdi
  __int64 v37; // r12
  __int128 v38; // rax
  int v39; // r9d
  int v40; // ecx
  int v41; // esi
  int v42; // r8d
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int16 v46; // cx
  __int64 v47; // r9
  int v48; // r9d
  char v49; // al
  __int64 v50; // rdi
  unsigned int v51; // r8d
  __int64 v52; // r9
  unsigned __int16 v53; // r10
  const __m128i *v54; // rax
  __m128i v55; // xmm6
  unsigned __int64 v56; // rdi
  __int64 v57; // r12
  __int128 v58; // rax
  int v59; // r9d
  bool v60; // bl
  int v61; // r9d
  __int64 v62; // rax
  __int64 v63; // rax
  __int128 v64; // rax
  int v65; // r9d
  __int128 v66; // rax
  __int64 v67; // rax
  __int128 v68; // rax
  int v69; // r9d
  __int64 v70; // rax
  __int128 v71; // rax
  int v72; // r9d
  __int128 v73; // rax
  int v74; // r9d
  __int128 v75; // rax
  __int128 v76; // [rsp-20h] [rbp-260h]
  __int128 v77; // [rsp-10h] [rbp-250h]
  unsigned __int16 v78; // [rsp+0h] [rbp-240h]
  unsigned __int8 v79; // [rsp+Fh] [rbp-231h]
  unsigned __int16 v80; // [rsp+18h] [rbp-228h]
  unsigned __int16 v81; // [rsp+18h] [rbp-228h]
  __int64 v82; // [rsp+20h] [rbp-220h]
  __int64 v83; // [rsp+20h] [rbp-220h]
  int v84; // [rsp+30h] [rbp-210h]
  __int64 v85; // [rsp+30h] [rbp-210h]
  unsigned int v86; // [rsp+30h] [rbp-210h]
  __int64 v87; // [rsp+30h] [rbp-210h]
  int v88; // [rsp+38h] [rbp-208h]
  __int64 v89; // [rsp+38h] [rbp-208h]
  __int64 v90; // [rsp+38h] [rbp-208h]
  __int64 v91; // [rsp+38h] [rbp-208h]
  __int64 v92; // [rsp+38h] [rbp-208h]
  __int64 v93; // [rsp+40h] [rbp-200h]
  __int64 v94; // [rsp+48h] [rbp-1F8h]
  bool v95; // [rsp+48h] [rbp-1F8h]
  bool v96; // [rsp+48h] [rbp-1F8h]
  __int128 v97; // [rsp+50h] [rbp-1F0h] BYREF
  __int128 v98; // [rsp+60h] [rbp-1E0h] BYREF
  __m128i v99; // [rsp+70h] [rbp-1D0h]
  __m128i v100; // [rsp+80h] [rbp-1C0h]
  __m128i v101; // [rsp+90h] [rbp-1B0h]
  __m128i v102; // [rsp+A0h] [rbp-1A0h]
  __int64 v103; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v104; // [rsp+B8h] [rbp-188h]
  __int64 v105; // [rsp+C0h] [rbp-180h] BYREF
  int v106; // [rsp+C8h] [rbp-178h]
  __int128 v107; // [rsp+D0h] [rbp-170h] BYREF
  __int128 v108; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v109; // [rsp+F0h] [rbp-150h] BYREF
  int v110; // [rsp+F8h] [rbp-148h]
  __int64 v111; // [rsp+100h] [rbp-140h] BYREF
  int v112; // [rsp+108h] [rbp-138h]
  int v113; // [rsp+110h] [rbp-130h]
  __int128 *v114; // [rsp+118h] [rbp-128h]
  __int64 v115[2]; // [rsp+120h] [rbp-120h] BYREF
  __int128 *v116; // [rsp+130h] [rbp-110h]
  int v117; // [rsp+140h] [rbp-100h]
  __int128 *v118; // [rsp+148h] [rbp-F8h]
  __int64 v119[2]; // [rsp+150h] [rbp-F0h] BYREF
  __int128 *v120; // [rsp+160h] [rbp-E0h]
  __int64 *v121; // [rsp+168h] [rbp-D8h]
  _DWORD v122[4]; // [rsp+170h] [rbp-D0h] BYREF
  __int128 *v123; // [rsp+180h] [rbp-C0h]
  __int128 *v124; // [rsp+188h] [rbp-B8h]
  char v125; // [rsp+194h] [rbp-ACh]
  __int64 *v126; // [rsp+198h] [rbp-A8h]
  unsigned __int64 v127; // [rsp+1A0h] [rbp-A0h]
  unsigned int v128; // [rsp+1A8h] [rbp-98h]
  char v129; // [rsp+1B4h] [rbp-8Ch]
  __m128i v130; // [rsp+1C0h] [rbp-80h] BYREF
  __m128i v131; // [rsp+1D0h] [rbp-70h] BYREF
  int v132; // [rsp+1E0h] [rbp-60h]
  char v133; // [rsp+1E4h] [rbp-5Ch]
  __int128 *v134; // [rsp+1E8h] [rbp-58h]
  __int64 *v135; // [rsp+1F0h] [rbp-50h]
  __int128 *v136; // [rsp+1F8h] [rbp-48h]
  char v137; // [rsp+204h] [rbp-3Ch]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = _mm_loadu_si128((const __m128i *)v4);
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v94 = *v4;
  v84 = *((_DWORD *)v4 + 2);
  v93 = v4[5];
  v88 = *((_DWORD *)v4 + 12);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 80);
  v98 = (__int128)v6;
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v97 = (__int128)v7;
  v105 = v9;
  LOWORD(v103) = v10;
  v104 = v11;
  if ( v9 )
    sub_B96E90((__int64)&v105, v9, 1);
  v12 = *a1;
  v106 = *(_DWORD *)(a2 + 72);
  si128 = _mm_load_si128((const __m128i *)&v97);
  v130 = _mm_load_si128((const __m128i *)&v98);
  v131 = si128;
  v14 = sub_3402EA0(v12, v5, (unsigned int)&v105, v103, v104, 0, (__int64)&v130, 2);
  if ( v14 )
    goto LABEL_4;
  if ( (unsigned __int8)sub_33E2390(*a1, v98, *((_QWORD *)&v98 + 1), 1)
    && !(unsigned __int8)sub_33E2390(*a1, v97, *((_QWORD *)&v97 + 1), 1) )
  {
    v15 = sub_3411F20(*a1, v5, (unsigned int)&v105, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v32, v97, v98);
    goto LABEL_5;
  }
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
      goto LABEL_11;
  }
  else if ( !sub_30070B0((__int64)&v103) )
  {
    goto LABEL_11;
  }
  v14 = sub_3295970(a1, a2, (__int64)&v105, v17, v18);
  if ( v14 )
  {
LABEL_4:
    v15 = v14;
    goto LABEL_5;
  }
LABEL_11:
  if ( *(_DWORD *)(v94 + 24) == 51 )
  {
    v14 = v97;
    goto LABEL_4;
  }
  if ( *(_DWORD *)(v93 + 24) == 51 || v93 == v94 && v88 == v84 && *((int *)a1 + 6) > 0 )
  {
    v15 = v98;
    goto LABEL_5;
  }
  v133 = 0;
  v19 = *(_DWORD *)(a2 + 24) == 174;
  *(_QWORD *)&v107 = 0;
  DWORD2(v107) = 0;
  *(_QWORD *)&v108 = 0;
  DWORD2(v108) = 0;
  v130.m128i_i32[0] = 174;
  v130.m128i_i64[1] = (__int64)&v107;
  v131.m128i_i32[2] = 64;
  v131.m128i_i64[0] = 0;
  if ( v19 )
  {
    v33 = *(const __m128i **)(a2 + 40);
    v34 = _mm_loadu_si128(v33);
    *(_QWORD *)&v107 = v33->m128i_i64[0];
    v102 = v34;
    DWORD2(v107) = v34.m128i_i32[2];
    if ( !(unsigned __int8)sub_32657E0((__int64)&v131, v33[2].m128i_i64[1]) )
    {
      v63 = v130.m128i_i64[1];
      v101 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
      *(_QWORD *)v130.m128i_i64[1] = v101.m128i_i64[0];
      *(_DWORD *)(v63 + 8) = v101.m128i_i32[2];
      if ( !(unsigned __int8)sub_32657E0((__int64)&v131, **(_QWORD **)(a2 + 40)) )
      {
        if ( v131.m128i_i32[2] > 0x40u && v131.m128i_i64[0] )
          j_j___libc_free_0_0(v131.m128i_u64[0]);
        goto LABEL_16;
      }
    }
    if ( v133 )
    {
      v35 = (v132 & *(_DWORD *)(a2 + 28)) == v132;
      if ( v131.m128i_i32[2] <= 0x40u || (v36 = v131.m128i_i64[0]) == 0 )
      {
LABEL_46:
        if ( !v35 )
          goto LABEL_16;
LABEL_47:
        v37 = *a1;
        *(_QWORD *)&v38 = sub_3400E40(*a1, 1, (unsigned int)v103, v104, &v105);
        v40 = v103;
        v41 = 191;
        v42 = v104;
        v77 = v38;
        v76 = v107;
LABEL_48:
        v43 = sub_3406EB0(v37, v41, (unsigned int)&v105, v40, v42, v39, v76, v77);
LABEL_49:
        v15 = v43;
        goto LABEL_5;
      }
    }
    else
    {
      if ( v131.m128i_i32[2] <= 0x40u )
        goto LABEL_47;
      v36 = v131.m128i_i64[0];
      v35 = 1;
      if ( !v131.m128i_i64[0] )
        goto LABEL_47;
    }
    v95 = v35;
    j_j___libc_free_0_0(v36);
    v35 = v95;
    goto LABEL_46;
  }
LABEL_16:
  v19 = *(_DWORD *)(a2 + 24) == 175;
  v130.m128i_i32[0] = 175;
  v130.m128i_i64[1] = (__int64)&v107;
  v131.m128i_i32[2] = 64;
  v131.m128i_i64[0] = 0;
  v133 = 0;
  if ( !v19 )
    goto LABEL_17;
  v54 = *(const __m128i **)(a2 + 40);
  v55 = _mm_loadu_si128(v54);
  *(_QWORD *)&v107 = v54->m128i_i64[0];
  v100 = v55;
  DWORD2(v107) = v55.m128i_i32[2];
  if ( (unsigned __int8)sub_32657E0((__int64)&v131, v54[2].m128i_i64[1])
    || (v67 = v130.m128i_i64[1],
        v99 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL)),
        *(_QWORD *)v130.m128i_i64[1] = v99.m128i_i64[0],
        *(_DWORD *)(v67 + 8) = v99.m128i_i32[2],
        (unsigned __int8)sub_32657E0((__int64)&v131, **(_QWORD **)(a2 + 40))) )
  {
    if ( v133 )
    {
      v96 = (v132 & *(_DWORD *)(a2 + 28)) == v132;
      if ( v131.m128i_i32[2] <= 0x40u || (v56 = v131.m128i_i64[0]) == 0 )
      {
LABEL_67:
        if ( !v96 )
          goto LABEL_17;
LABEL_68:
        v57 = *a1;
        *(_QWORD *)&v58 = sub_3400E40(*a1, 1, (unsigned int)v103, v104, &v105);
        v43 = sub_3406EB0(v57, 192, (unsigned int)&v105, v103, v104, v59, v107, v58);
        goto LABEL_49;
      }
    }
    else
    {
      if ( v131.m128i_i32[2] <= 0x40u )
        goto LABEL_68;
      v56 = v131.m128i_i64[0];
      v96 = 1;
      if ( !v131.m128i_i64[0] )
        goto LABEL_68;
    }
    j_j___libc_free_0_0(v56);
    goto LABEL_67;
  }
  if ( v131.m128i_i32[2] > 0x40u && v131.m128i_i64[0] )
    j_j___libc_free_0_0(v131.m128i_u64[0]);
LABEL_17:
  v130.m128i_i32[0] = v5;
  if ( ((v5 - 174) & 0xFFFFFFFD) != 0 )
  {
    v131.m128i_i64[0] = (__int64)&v107;
    v130.m128i_i32[2] = 214;
    v131.m128i_i8[12] = 0;
    v132 = 214;
    v134 = &v108;
    BYTE4(v135) = 0;
    BYTE4(v136) = 0;
    if ( !sub_32A63F0(a2, 0, (__int64)&v130)
      || (v44 = *(_QWORD *)(v108 + 48) + 16LL * DWORD2(v108),
          v45 = *(_QWORD *)(v107 + 48) + 16LL * DWORD2(v107),
          v46 = *(_WORD *)v45,
          *(_WORD *)v45 != *(_WORD *)v44)
      || (v47 = *(_QWORD *)(v45 + 8), *(_QWORD *)(v44 + 8) != v47) && !v46
      || (v87 = *(_QWORD *)(v107 + 48),
          v91 = 16LL * DWORD2(v107),
          !(unsigned __int8)sub_328A020(a1[1], v5, v46, v47, *((unsigned __int8 *)a1 + 33))) )
    {
      if ( v5 != 175 )
        goto LABEL_32;
LABEL_56:
      v81 = v103;
      v79 = *((_BYTE *)a1 + 33);
      v78 = v103;
      v83 = v104;
      v92 = a1[1];
      v49 = sub_328A020(v92, 0xAFu, v103, v104, v79);
      v50 = v92;
      v51 = v79;
      v52 = v83;
      v53 = v81;
      if ( v49 )
        goto LABEL_60;
      if ( v79 && !(unsigned __int8)sub_328A020(v92, 0xB1u, v78, v83, 1u) )
      {
        v50 = v92;
        v53 = v81;
        v51 = 1;
        v52 = v83;
        goto LABEL_60;
      }
      if ( (unsigned __int8)sub_33DE9F0(*a1, v97, *((_QWORD *)&v97 + 1), 0) )
      {
        v37 = *a1;
        *(_QWORD *)&v64 = sub_34015B0(*a1, &v105, (unsigned int)v103, v104, 0, 0);
        *(_QWORD *)&v66 = sub_3406EB0(v37, 56, (unsigned int)&v105, v103, v104, v65, v97, v64);
        v77 = v66;
        v76 = v98;
      }
      else
      {
        if ( !(unsigned __int8)sub_33DE9F0(*a1, v98, *((_QWORD *)&v98 + 1), 0) )
        {
          v50 = a1[1];
          v51 = *((unsigned __int8 *)a1 + 33);
          v53 = v103;
          v52 = v104;
LABEL_60:
          if ( !(unsigned __int8)sub_328A020(v50, 0xB1u, v53, v52, v51) )
            goto LABEL_32;
          goto LABEL_26;
        }
        v37 = *a1;
        *(_QWORD *)&v73 = sub_34015B0(*a1, &v105, (unsigned int)v103, v104, 0, 0);
        *(_QWORD *)&v75 = sub_3406EB0(v37, 56, (unsigned int)&v105, v103, v104, v74, v98, v73);
        v77 = v75;
        v76 = v97;
      }
      v40 = v103;
      v41 = 177;
      v42 = v104;
      goto LABEL_48;
    }
    *(_QWORD *)&v71 = sub_3406EB0(
                        *a1,
                        v5,
                        (unsigned int)&v105,
                        *(unsigned __int16 *)(v87 + v91),
                        *(_QWORD *)(v87 + v91 + 8),
                        v48,
                        v107,
                        v108);
    v70 = sub_33FAF80(*a1, 214, (unsigned int)&v105, v103, v104, v72, v71);
  }
  else
  {
    v131.m128i_i64[0] = (__int64)&v107;
    v130.m128i_i32[2] = 213;
    v131.m128i_i8[12] = 0;
    v132 = 213;
    v134 = &v108;
    BYTE4(v135) = 0;
    BYTE4(v136) = 0;
    if ( !sub_32A63F0(a2, 0, (__int64)&v130)
      || (v20 = *(_QWORD *)(v108 + 48) + 16LL * DWORD2(v108),
          v21 = *(_QWORD *)(v107 + 48) + 16LL * DWORD2(v107),
          v22 = *(_WORD *)v21,
          *(_WORD *)v21 != *(_WORD *)v20)
      || (v23 = *(_QWORD *)(v21 + 8), *(_QWORD *)(v20 + 8) != v23) && !v22
      || (v85 = *(_QWORD *)(v107 + 48),
          v89 = 16LL * DWORD2(v107),
          !(unsigned __int8)sub_328A020(a1[1], v5, v22, v23, *((unsigned __int8 *)a1 + 33))) )
    {
      if ( v5 != 175 )
      {
        if ( v5 != 174 )
        {
LABEL_32:
          v15 = 0;
          goto LABEL_5;
        }
        v80 = v103;
        v82 = v104;
        v86 = *((unsigned __int8 *)a1 + 33);
        v90 = a1[1];
        v25 = sub_328A020(v90, 0xB0u, v103, v104, v86);
        v26 = v90;
        v27 = v86;
        v28 = v82;
        v29 = v80;
        if ( !v25 )
          goto LABEL_75;
LABEL_26:
        v109 = 0;
        v124 = &v108;
        v110 = 0;
        v122[0] = v5;
        v122[2] = 56;
        v123 = &v107;
        v125 = 0;
        v126 = &v109;
        v128 = 64;
        v127 = 1;
        v129 = 0;
        if ( sub_32A64A0(a2, 0, (__int64)v122) )
        {
          if ( v128 > 0x40 && v127 )
            j_j___libc_free_0_0(v127);
        }
        else
        {
          v114 = &v107;
          v112 = 64;
          v111 = 1;
          v113 = 56;
          sub_9865C0((__int64)v115, (__int64)&v111);
          v117 = v113;
          v118 = v114;
          BYTE4(v116) = 0;
          sub_9865C0((__int64)v119, (__int64)v115);
          v130.m128i_i32[0] = v5;
          v120 = v116;
          v121 = &v109;
          v130.m128i_i32[2] = v117;
          v131.m128i_i64[0] = (__int64)v118;
          sub_9865C0((__int64)&v131.m128i_i64[1], (__int64)v119);
          v137 = 0;
          v134 = v120;
          v135 = v121;
          v136 = &v108;
          v60 = sub_32A6620(a2, 0, (__int64)&v130);
          sub_969240(&v131.m128i_i64[1]);
          sub_969240(v119);
          sub_969240(v115);
          sub_969240(&v111);
          if ( v128 > 0x40 && v127 )
            j_j___libc_free_0_0(v127);
          if ( !v60 )
          {
LABEL_73:
            if ( v5 != 174 )
              goto LABEL_32;
            v26 = a1[1];
            v27 = *((unsigned __int8 *)a1 + 33);
            v29 = v103;
            v28 = v104;
LABEL_75:
            if ( !(unsigned __int8)sub_328A020(v26, 0xAFu, v29, v28, v27)
              || !(unsigned __int8)sub_33DD2A0(*a1, v98, *((_QWORD *)&v98 + 1), 0)
              || !(unsigned __int8)sub_33DD2A0(*a1, v97, *((_QWORD *)&v97 + 1), 0) )
            {
              goto LABEL_32;
            }
            v62 = sub_3406EB0(*a1, 175, (unsigned int)&v105, v103, v104, v61, v98, v97);
            goto LABEL_79;
          }
        }
        v31 = *(_DWORD *)(v109 + 28);
        if ( ((v5 - 174) & 0xFFFFFFFD) != 0 )
        {
          if ( (v31 & 1) == 0 )
            goto LABEL_32;
          v62 = sub_3406EB0(*a1, 177, (unsigned int)&v105, v103, v104, v30, v107, v108);
          goto LABEL_79;
        }
        if ( (v31 & 2) != 0 )
        {
          v62 = sub_3406EB0(*a1, 176, (unsigned int)&v105, v103, v104, v30, v107, v108);
LABEL_79:
          v15 = v62;
          goto LABEL_5;
        }
        goto LABEL_73;
      }
      goto LABEL_56;
    }
    *(_QWORD *)&v68 = sub_3406EB0(
                        *a1,
                        v5,
                        (unsigned int)&v105,
                        *(unsigned __int16 *)(v85 + v89),
                        *(_QWORD *)(v85 + v89 + 8),
                        v24,
                        v107,
                        v108);
    v70 = sub_33FAF80(*a1, 213, (unsigned int)&v105, v103, v104, v69, v68);
  }
  v15 = v70;
LABEL_5:
  if ( v105 )
    sub_B91220((__int64)&v105, v105);
  return v15;
}
