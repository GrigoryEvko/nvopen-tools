// Function: sub_37A17C0
// Address: 0x37a17c0
//
unsigned __int8 *__fastcall sub_37A17C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  __int64 *v7; // r13
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int128 v16; // xmm1
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int16 *v22; // r12
  int v23; // r14d
  __int64 v24; // rdx
  __int64 v25; // rax
  bool v26; // cc
  unsigned __int64 *v27; // rcx
  unsigned int v28; // esi
  unsigned int v29; // r14d
  unsigned int v30; // eax
  unsigned int j; // ecx
  int v32; // edx
  __int64 *v33; // r12
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // r15d
  __int64 v37; // r9
  unsigned int v38; // r10d
  unsigned int v39; // ebx
  unsigned __int64 v40; // r15
  unsigned int v41; // r13d
  __int128 v42; // rax
  unsigned __int8 *v43; // rax
  unsigned int v44; // r10d
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  unsigned int v50; // ebx
  _QWORD *v51; // rdi
  __int128 v52; // rax
  __int64 v53; // r8
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 *v56; // rax
  unsigned int v57; // eax
  unsigned __int8 *v58; // rax
  unsigned __int8 *v59; // r12
  _QWORD *v61; // rax
  _QWORD *v62; // rdx
  _QWORD *i; // r12
  __int64 v64; // r10
  __int64 v65; // rbx
  unsigned __int64 v66; // r14
  __int128 v67; // rax
  unsigned __int8 *v68; // rax
  int v69; // edx
  int v70; // edi
  unsigned __int8 *v71; // rdx
  __int64 v72; // rax
  _QWORD *v73; // rdi
  int v74; // edx
  _QWORD *v75; // rbx
  __int64 v76; // r9
  int v77; // r8d
  __int64 v78; // rax
  __int64 v79; // rcx
  int v80; // kr30_4
  __int64 v81; // rdx
  unsigned int v82; // edx
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int128 v85; // [rsp-20h] [rbp-270h]
  __int128 v86; // [rsp-20h] [rbp-270h]
  __int128 v87; // [rsp-10h] [rbp-260h]
  __int128 v88; // [rsp-10h] [rbp-260h]
  unsigned int v89; // [rsp+14h] [rbp-23Ch]
  int v90; // [rsp+28h] [rbp-228h]
  __int64 v91; // [rsp+30h] [rbp-220h]
  __int64 v92; // [rsp+38h] [rbp-218h]
  __int64 v93; // [rsp+40h] [rbp-210h]
  __int64 v94; // [rsp+40h] [rbp-210h]
  __int64 v95; // [rsp+50h] [rbp-200h]
  unsigned __int64 v96; // [rsp+50h] [rbp-200h]
  unsigned int v97; // [rsp+58h] [rbp-1F8h]
  unsigned int v98; // [rsp+58h] [rbp-1F8h]
  __int64 v99; // [rsp+58h] [rbp-1F8h]
  _QWORD *v100; // [rsp+60h] [rbp-1F0h]
  __int128 v101; // [rsp+60h] [rbp-1F0h]
  _QWORD *v102; // [rsp+60h] [rbp-1F0h]
  __int64 v103; // [rsp+60h] [rbp-1F0h]
  __int64 v104; // [rsp+60h] [rbp-1F0h]
  __int64 v105; // [rsp+68h] [rbp-1E8h]
  __int64 v106; // [rsp+68h] [rbp-1E8h]
  unsigned int v107; // [rsp+70h] [rbp-1E0h]
  __int16 v108; // [rsp+76h] [rbp-1DAh]
  unsigned __int16 v109; // [rsp+76h] [rbp-1DAh]
  __int64 v110; // [rsp+78h] [rbp-1D8h]
  int v111; // [rsp+78h] [rbp-1D8h]
  __int64 v112; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v113; // [rsp+C8h] [rbp-188h]
  __int64 v114; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v115; // [rsp+D8h] [rbp-178h]
  __int64 v116; // [rsp+E0h] [rbp-170h] BYREF
  int v117; // [rsp+E8h] [rbp-168h]
  __int16 v118; // [rsp+F0h] [rbp-160h] BYREF
  __int64 v119; // [rsp+F8h] [rbp-158h]
  __int64 v120; // [rsp+100h] [rbp-150h] BYREF
  int v121; // [rsp+108h] [rbp-148h]
  _QWORD *v122; // [rsp+110h] [rbp-140h] BYREF
  __int64 v123; // [rsp+118h] [rbp-138h]
  _QWORD v124[38]; // [rsp+120h] [rbp-130h] BYREF

  v7 = a1;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v112) = v9;
  v113 = v10;
  if ( (_WORD)v9 )
  {
    v91 = 0;
    v108 = word_4456580[v9 - 1];
  }
  else
  {
    v80 = sub_3009970((__int64)&v112, a2, v10, a4, a5);
    HIWORD(v5) = HIWORD(v80);
    v108 = v80;
    v91 = v81;
  }
  LOWORD(v5) = v108;
  v11 = a1[1];
  HIWORD(v12) = HIWORD(v5);
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v122, *a1, *(_QWORD *)(v11 + 64), v112, v113);
    LOWORD(v114) = v123;
    v115 = v124[0];
  }
  else
  {
    LODWORD(v114) = v13(*a1, *(_QWORD *)(v11 + 64), v112, v113);
    v115 = v83;
  }
  v14 = *(_QWORD *)(a2 + 40);
  v15 = _mm_loadu_si128((const __m128i *)v14);
  v16 = (__int128)_mm_loadu_si128((const __m128i *)(v14 + 40));
  v110 = *(_QWORD *)v14;
  v17 = *(_QWORD *)(a2 + 80);
  v97 = *(_DWORD *)(v14 + 8);
  v95 = *(_QWORD *)(v14 + 40);
  v116 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v116, v17, 1);
  v18 = a1[1];
  v117 = *(_DWORD *)(a2 + 72);
  v19 = 16LL * v97;
  sub_2FE6CC0(
    (__int64)&v122,
    *a1,
    *(_QWORD *)(v18 + 64),
    *(unsigned __int16 *)(v19 + *(_QWORD *)(v110 + 48)),
    *(_QWORD *)(v19 + *(_QWORD *)(v110 + 48) + 8));
  if ( (_BYTE)v122 == 7 )
  {
    v110 = sub_379AB60((__int64)a1, v15.m128i_u64[0], v15.m128i_i64[1]);
    v97 = v82;
    v19 = 16LL * v82;
  }
  v22 = (unsigned __int16 *)(*(_QWORD *)(v110 + 48) + v19);
  v23 = *v22;
  v24 = *((_QWORD *)v22 + 1);
  v25 = *(_QWORD *)(v95 + 96);
  v118 = *v22;
  v26 = *(_DWORD *)(v25 + 32) <= 0x40u;
  v27 = *(unsigned __int64 **)(v25 + 24);
  v119 = v24;
  if ( v26 )
    v96 = (unsigned __int64)v27;
  else
    v96 = *v27;
  if ( !v96 && (_WORD)v114 == (_WORD)v23 && (v115 == v24 || (_WORD)v23) )
  {
    v59 = (unsigned __int8 *)v110;
    goto LABEL_49;
  }
  if ( (_WORD)v114 )
  {
    v89 = word_4456340[(unsigned __int16)v114 - 1];
    if ( (_WORD)v23 )
      goto LABEL_15;
  }
  else
  {
    v89 = sub_3007240((__int64)&v114);
    if ( (_WORD)v23 )
    {
LABEL_15:
      v28 = word_4456340[v23 - 1];
      goto LABEL_16;
    }
  }
  v28 = sub_3007240((__int64)&v118);
LABEL_16:
  if ( (_WORD)v112 )
  {
    v107 = word_4456340[(unsigned __int16)v112 - 1];
    if ( !(v96 % v89) && v28 > v89 + v96 )
      goto LABEL_75;
    if ( (unsigned __int16)(v112 - 176) <= 0x34u )
      goto LABEL_19;
LABEL_54:
    v61 = v124;
    v62 = v124;
    v122 = v124;
    v123 = 0x1000000000LL;
    if ( v89 )
    {
      if ( v89 > 0x10uLL )
      {
        sub_C8D5F0((__int64)&v122, v124, v89, 0x10u, v20, v21);
        v62 = v122;
        v61 = &v122[2 * (unsigned int)v123];
      }
      for ( i = &v62[2 * v89]; i != v61; v61 += 2 )
      {
        if ( v61 )
        {
          *v61 = 0;
          *((_DWORD *)v61 + 2) = 0;
        }
      }
      LODWORD(v123) = v89;
    }
    if ( v107 )
    {
      v64 = v96;
      v94 = v97;
      v65 = 0;
      v66 = v15.m128i_u64[1];
      do
      {
        v99 = v64;
        v102 = (_QWORD *)v7[1];
        *(_QWORD *)&v67 = sub_3400EE0((__int64)v102, v64, (__int64)&v116, 0, v15);
        LOWORD(v5) = v108;
        v66 = v94 | v66 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v86 + 1) = v66;
        *(_QWORD *)&v86 = v110;
        v68 = sub_3406EB0(v102, 0x9Eu, (__int64)&v116, v5, v91, *((__int64 *)&v67 + 1), v86, v67);
        v70 = v69;
        v71 = v68;
        v72 = (__int64)v122;
        v64 = v99 + 1;
        v122[v65] = v71;
        *(_DWORD *)(v72 + v65 * 8 + 8) = v70;
        v65 += 2;
      }
      while ( 2LL * v107 != v65 );
      HIWORD(v12) = HIWORD(v5);
    }
    LOWORD(v12) = v108;
    v73 = (_QWORD *)v7[1];
    v120 = 0;
    v121 = 0;
    v75 = sub_33F17F0(v73, 51, (__int64)&v120, v12, v91);
    v77 = v74;
    if ( v120 )
    {
      v111 = v74;
      sub_B91220((__int64)&v120, v120);
      v77 = v111;
    }
    if ( v89 > v107 )
    {
      v78 = 2LL * v107;
      do
      {
        v79 = (__int64)v122;
        v122[v78] = v75;
        *(_DWORD *)(v79 + v78 * 8 + 8) = v77;
        v78 += 2;
      }
      while ( 2 * (v107 + (unsigned __int64)(v89 - 1 - v107) + 1) != v78 );
    }
    *((_QWORD *)&v88 + 1) = (unsigned int)v123;
    *(_QWORD *)&v88 = v122;
    v58 = sub_33FC220((_QWORD *)v7[1], 156, (__int64)&v116, v114, v115, v76, v88);
    goto LABEL_42;
  }
  v107 = sub_3007240((__int64)&v112);
  if ( !(v96 % v89) )
  {
    if ( v28 > v89 + v96 )
    {
LABEL_75:
      v59 = sub_3406EB0(
              (_QWORD *)a1[1],
              0xA1u,
              (__int64)&v116,
              (unsigned int)v114,
              v115,
              v21,
              __PAIR128__(v97 | v15.m128i_i64[1] & 0xFFFFFFFF00000000LL, v110),
              v16);
      goto LABEL_49;
    }
    if ( sub_3007100((__int64)&v112) )
      goto LABEL_19;
    goto LABEL_54;
  }
  if ( !sub_3007100((__int64)&v112) )
    goto LABEL_54;
LABEL_19:
  if ( v107 )
  {
    v29 = v107;
    if ( v89 )
    {
      v30 = v89;
      for ( j = v107 % v89; j; j = v32 )
      {
        v32 = v30 % j;
        v30 = j;
      }
      v29 = v30;
    }
  }
  else
  {
    v29 = v89;
  }
  v33 = *(__int64 **)(a1[1] + 64);
  LODWORD(v122) = v29;
  BYTE4(v122) = 1;
  v92 = 0;
  v109 = sub_2D43AD0(v108, v29);
  if ( !v109 )
  {
    v90 = sub_3009450(v33, v5, v91, (__int64)v122, v34, v35);
    v109 = v90;
    v92 = v84;
  }
  HIWORD(v36) = HIWORD(v90);
  sub_2FE6CC0((__int64)&v122, *a1, *(_QWORD *)(a1[1] + 64), v109, v92);
  if ( (_BYTE)v122 == 7 )
    sub_C64ED0("Don't know how to widen the result of EXTRACT_SUBVECTOR for scalable vectors", 1u);
  v122 = v124;
  v123 = 0x300000000LL;
  if ( v29 > v107 )
  {
    v50 = 0;
  }
  else
  {
    v38 = 0;
    HIWORD(v39) = HIWORD(v90);
    v40 = v15.m128i_u64[1];
    v93 = v97;
    v41 = 0;
    do
    {
      v98 = v38;
      v100 = (_QWORD *)a1[1];
      *(_QWORD *)&v42 = sub_3400EE0((__int64)v100, v96 + v41, (__int64)&v116, 0, v15);
      LOWORD(v39) = v109;
      v40 = v93 | v40 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v85 + 1) = v40;
      *(_QWORD *)&v85 = v110;
      v43 = sub_3406EB0(v100, 0xA1u, (__int64)&v116, v39, v92, *((__int64 *)&v42 + 1), v85, v42);
      v44 = v98;
      v45 = (__int64)v43;
      v46 = (unsigned int)v123;
      v37 = v47;
      v48 = (unsigned int)v123 + 1LL;
      if ( v48 > HIDWORD(v123) )
      {
        v104 = v45;
        v106 = v37;
        sub_C8D5F0((__int64)&v122, v124, v48, 0x10u, v45, v37);
        v46 = (unsigned int)v123;
        v44 = v98;
        v37 = v106;
        v45 = v104;
      }
      v49 = &v122[2 * v46];
      v38 = v44 + 1;
      v41 += v29;
      *v49 = v45;
      v49[1] = v37;
      LODWORD(v123) = v123 + 1;
    }
    while ( v38 < v107 / v29 );
    HIWORD(v36) = HIWORD(v90);
    v50 = 1;
    v7 = a1;
    if ( v29 <= v107 )
      v50 = v107 / v29;
  }
  if ( v50 >= v89 / v29 )
  {
    v57 = v123;
  }
  else
  {
    do
    {
      v51 = (_QWORD *)v7[1];
      LOWORD(v36) = v109;
      v120 = 0;
      v121 = 0;
      *(_QWORD *)&v52 = sub_33F17F0(v51, 51, (__int64)&v120, v36, v92);
      v37 = *((_QWORD *)&v52 + 1);
      v53 = v52;
      if ( v120 )
      {
        v101 = v52;
        sub_B91220((__int64)&v120, v120);
        v37 = *((_QWORD *)&v101 + 1);
        v53 = v101;
      }
      v54 = (unsigned int)v123;
      v55 = (unsigned int)v123 + 1LL;
      if ( v55 > HIDWORD(v123) )
      {
        v103 = v53;
        v105 = v37;
        sub_C8D5F0((__int64)&v122, v124, v55, 0x10u, v53, v37);
        v54 = (unsigned int)v123;
        v37 = v105;
        v53 = v103;
      }
      v56 = &v122[2 * v54];
      ++v50;
      *v56 = v53;
      v56[1] = v37;
      v57 = v123 + 1;
      LODWORD(v123) = v123 + 1;
    }
    while ( v50 != v89 / v29 );
  }
  *((_QWORD *)&v87 + 1) = v57;
  *(_QWORD *)&v87 = v122;
  v58 = sub_33FC220((_QWORD *)v7[1], 159, (__int64)&v116, (unsigned int)v114, v115, v37, v87);
LABEL_42:
  v59 = v58;
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
LABEL_49:
  if ( v116 )
    sub_B91220((__int64)&v116, v116);
  return v59;
}
