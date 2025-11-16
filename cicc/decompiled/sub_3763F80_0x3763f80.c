// Function: sub_3763F80
// Address: 0x3763f80
//
void __fastcall sub_3763F80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  unsigned int *v7; // rbx
  __int16 *v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned int v13; // r15d
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // r14
  _QWORD *v19; // rax
  unsigned int v20; // r12d
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // rax
  unsigned int *v31; // rax
  __int64 v32; // rbx
  unsigned int *v33; // r15
  unsigned __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int8 **v37; // rax
  __int64 v38; // r14
  unsigned __int8 *v39; // rsi
  unsigned __int8 *v40; // r12
  __int64 v41; // r13
  unsigned __int16 *v42; // rdx
  unsigned int *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rcx
  _QWORD *v47; // r11
  __int64 v48; // rdx
  __int64 v49; // r8
  unsigned int v50; // edx
  __int64 v51; // rdx
  unsigned __int8 *v52; // rax
  unsigned __int8 *v53; // r15
  __int64 v54; // r8
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // r8
  unsigned __int64 v60; // rdx
  unsigned __int8 **v61; // rax
  _BYTE *v62; // rdi
  _BYTE *v63; // rsi
  __int64 v64; // rcx
  unsigned __int8 *v65; // r12
  unsigned __int8 *v66; // rdx
  unsigned __int8 *v67; // r13
  __int64 v68; // r9
  __int64 v69; // rdx
  unsigned __int8 *v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  unsigned __int8 **v73; // rax
  __int64 v74; // rdx
  unsigned __int8 **v75; // rdx
  _BYTE *v76; // rdi
  __int64 v77; // r13
  __int128 v78; // rax
  unsigned __int8 *v79; // rax
  __int16 *v80; // rsi
  __int64 v81; // r8
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // rcx
  unsigned __int8 *v84; // rdx
  _QWORD *v85; // r10
  __int16 v86; // ax
  unsigned int v87; // esi
  unsigned int v88; // edx
  bool v89; // al
  __int64 v90; // rax
  __int128 v91; // [rsp-20h] [rbp-600h]
  __int128 v92; // [rsp-20h] [rbp-600h]
  __int128 v93; // [rsp-10h] [rbp-5F0h]
  __int128 v94; // [rsp-10h] [rbp-5F0h]
  __int64 v96; // [rsp+38h] [rbp-5A8h]
  __int64 v97; // [rsp+48h] [rbp-598h]
  __int64 v98; // [rsp+50h] [rbp-590h]
  unsigned __int16 v99; // [rsp+5Ah] [rbp-586h]
  unsigned int v100; // [rsp+5Ch] [rbp-584h]
  __int64 v101; // [rsp+60h] [rbp-580h]
  __int64 v102; // [rsp+68h] [rbp-578h]
  unsigned __int64 v103; // [rsp+70h] [rbp-570h]
  unsigned __int64 v104; // [rsp+78h] [rbp-568h]
  __int128 v105; // [rsp+80h] [rbp-560h]
  unsigned __int8 *v106; // [rsp+80h] [rbp-560h]
  unsigned __int64 v107; // [rsp+88h] [rbp-558h]
  __int64 v108; // [rsp+A8h] [rbp-538h]
  __int64 v109; // [rsp+B0h] [rbp-530h]
  __int64 v110; // [rsp+B0h] [rbp-530h]
  __int64 v111; // [rsp+B0h] [rbp-530h]
  __int64 v113; // [rsp+C0h] [rbp-520h]
  __int128 v114; // [rsp+C0h] [rbp-520h]
  unsigned __int64 v115; // [rsp+C0h] [rbp-520h]
  __int64 v116; // [rsp+C0h] [rbp-520h]
  __int64 (__fastcall *v117)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+C0h] [rbp-520h]
  __int64 v118; // [rsp+D8h] [rbp-508h]
  unsigned __int8 *v120; // [rsp+E0h] [rbp-500h]
  unsigned __int8 *v121; // [rsp+E0h] [rbp-500h]
  __int64 v122; // [rsp+E8h] [rbp-4F8h]
  __int64 v123; // [rsp+E8h] [rbp-4F8h]
  __int64 v124; // [rsp+F0h] [rbp-4F0h] BYREF
  __int64 v125; // [rsp+F8h] [rbp-4E8h]
  __int64 v126; // [rsp+100h] [rbp-4E0h] BYREF
  int v127; // [rsp+108h] [rbp-4D8h]
  __int16 v128; // [rsp+110h] [rbp-4D0h] BYREF
  __int64 v129; // [rsp+118h] [rbp-4C8h]
  _QWORD v130[2]; // [rsp+120h] [rbp-4C0h] BYREF
  __int16 v131; // [rsp+130h] [rbp-4B0h]
  __int64 v132; // [rsp+138h] [rbp-4A8h]
  _BYTE *v133; // [rsp+140h] [rbp-4A0h] BYREF
  __int64 v134; // [rsp+148h] [rbp-498h]
  _BYTE v135[64]; // [rsp+150h] [rbp-490h] BYREF
  _BYTE *v136; // [rsp+190h] [rbp-450h] BYREF
  __int64 v137; // [rsp+198h] [rbp-448h]
  _BYTE v138[512]; // [rsp+1A0h] [rbp-440h] BYREF
  _BYTE *v139; // [rsp+3A0h] [rbp-240h] BYREF
  __int64 v140; // [rsp+3A8h] [rbp-238h]
  _BYTE v141[560]; // [rsp+3B0h] [rbp-230h] BYREF

  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v124) = v9;
  v125 = v10;
  if ( v9 )
  {
    v11 = 0;
    v102 = 0;
    v99 = word_4456580[v9 - 1];
    v12 = v99;
    v101 = v99;
  }
  else
  {
    v12 = sub_3009970((__int64)&v124, a2, v10, a4, a5);
    v99 = v12;
    v9 = v124;
    v11 = v51;
    v102 = v51;
    v101 = v12;
    if ( !(_WORD)v124 )
    {
      if ( !sub_3007100((__int64)&v124) )
        goto LABEL_26;
      goto LABEL_25;
    }
  }
  if ( (unsigned __int16)(v9 - 176) > 0x34u )
  {
LABEL_4:
    v13 = word_4456340[(unsigned __int16)v124 - 1];
    goto LABEL_5;
  }
LABEL_25:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v124 )
  {
    if ( (unsigned __int16)(v124 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_4;
  }
LABEL_26:
  v13 = sub_3007130((__int64)&v124, a2);
LABEL_5:
  v100 = *(_DWORD *)(a2 + 64);
  v14 = *a1;
  v15 = *(_QWORD *)(*a1 + 16);
  if ( (unsigned int)(*(_DWORD *)(a2 + 24) - 147) <= 1 )
  {
    v118 = *(_QWORD *)(v14 + 64);
    v117 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v15 + 528LL);
    v90 = sub_2E79000(*(__int64 **)(v14 + 40));
    v17 = v117(v15, v90, v118, (unsigned int)v12, v11);
    v12 = v17;
  }
  else
  {
    v16 = v102;
    LOWORD(v17) = v99;
  }
  LOWORD(v12) = v17;
  v130[1] = v16;
  v130[0] = v12;
  v18 = &v126;
  v19 = *(_QWORD **)(a2 + 40);
  v131 = 1;
  v132 = 0;
  v20 = *((_DWORD *)v19 + 2);
  v104 = v19[1];
  v21 = *(_QWORD *)(a2 + 80);
  v98 = *v19;
  v126 = v21;
  if ( v21 )
    sub_B96E90((__int64)&v126, v21, 1);
  v22 = *(_DWORD *)(a2 + 72);
  v139 = v141;
  v127 = v22;
  v136 = v138;
  v137 = 0x2000000000LL;
  v140 = 0x2000000000LL;
  if ( v13 )
  {
    v108 = 0;
    v97 = v13;
    v96 = v20;
    while ( 1 )
    {
      v133 = v135;
      v134 = 0x400000000LL;
      *(_QWORD *)&v105 = sub_3400EE0(*a1, v108, (__int64)v18, 0, a7);
      v25 = (unsigned int)v134;
      *((_QWORD *)&v105 + 1) = v26;
      v27 = v96 | v104 & 0xFFFFFFFF00000000LL;
      v28 = (unsigned int)v134 + 1LL;
      v104 = v27;
      if ( v28 > HIDWORD(v134) )
      {
        sub_C8D5F0((__int64)&v133, v135, v28, 0x10u, v23, v24);
        v25 = (unsigned int)v134;
      }
      v29 = &v133[16 * v25];
      v29[1] = v27;
      *v29 = v98;
      v30 = (unsigned int)(v134 + 1);
      LODWORD(v134) = v134 + 1;
      if ( v100 > 1 )
        break;
LABEL_29:
      *((_QWORD *)&v93 + 1) = v30;
      *(_QWORD *)&v93 = v133;
      v52 = sub_3411BE0((_QWORD *)*a1, *(_DWORD *)(a2 + 24), (__int64)v18, (unsigned __int16 *)v130, 2, v24, v93);
      v53 = v52;
      if ( (unsigned int)(*(_DWORD *)(a2 + 24) - 147) <= 1 )
      {
        v77 = v101;
        LOWORD(v77) = v99;
        v110 = *a1;
        v101 = v77;
        *(_QWORD *)&v78 = sub_3400BD0(*a1, 0, (__int64)v18, (unsigned int)v77, v102, 0, a7, 0);
        v114 = v78;
        v79 = sub_34015B0(*a1, (__int64)v18, (unsigned int)v77, v102, 0, 0, a7);
        v80 = (__int16 *)*((_QWORD *)v53 + 6);
        v81 = (__int64)v53;
        v83 = v82;
        v84 = v79;
        v85 = (_QWORD *)v110;
        v86 = *v80;
        v129 = *((_QWORD *)v80 + 1);
        v128 = v86;
        if ( v86 )
        {
          v87 = ((unsigned __int16)(v86 - 17) < 0xD4u) + 205;
        }
        else
        {
          v106 = v84;
          v107 = v83;
          v89 = sub_30070B0((__int64)&v128);
          v81 = (__int64)v53;
          v84 = v106;
          v83 = v107;
          v85 = (_QWORD *)v110;
          v87 = 205 - (!v89 - 1);
        }
        v54 = sub_340EC60(v85, v87, (__int64)v18, v77, v102, 0, v81, 0, __PAIR128__(v83, (unsigned __int64)v84), v114);
        a6 = v88;
      }
      else
      {
        v54 = (__int64)v52;
        a6 = 0;
      }
      v55 = (unsigned int)v137;
      v56 = (unsigned int)v137 + 1LL;
      if ( v56 > HIDWORD(v137) )
      {
        v111 = a6;
        v116 = v54;
        sub_C8D5F0((__int64)&v136, v138, v56, 0x10u, v54, a6);
        v55 = (unsigned int)v137;
        a6 = v111;
        v54 = v116;
      }
      v57 = (__int64 *)&v136[16 * v55];
      *v57 = v54;
      v57[1] = a6;
      v58 = (unsigned int)v140;
      v59 = v103 & 0xFFFFFFFF00000000LL | 1;
      LODWORD(v137) = v137 + 1;
      v60 = (unsigned int)v140 + 1LL;
      v103 = v59;
      if ( v60 > HIDWORD(v140) )
      {
        v115 = v59;
        sub_C8D5F0((__int64)&v139, v141, v60, 0x10u, v59, a6);
        v58 = (unsigned int)v140;
        v59 = v115;
      }
      v61 = (unsigned __int8 **)&v139[16 * v58];
      *v61 = v53;
      v62 = v133;
      v61[1] = (unsigned __int8 *)v59;
      LODWORD(v140) = v140 + 1;
      if ( v62 != v135 )
        _libc_free((unsigned __int64)v62);
      if ( v97 == ++v108 )
      {
        v63 = v136;
        v64 = (unsigned int)v137;
        goto LABEL_39;
      }
    }
    v31 = v7;
    v109 = (__int64)v18;
    v32 = 40;
    v33 = v31;
    while ( 1 )
    {
      v43 = (unsigned int *)(v32 + *(_QWORD *)(a2 + 40));
      v38 = v43[2];
      v39 = *(unsigned __int8 **)v43;
      v40 = *(unsigned __int8 **)v43;
      v41 = *((_QWORD *)v43 + 1);
      v42 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v43 + 48LL) + 16 * v38);
      LODWORD(v43) = *v42;
      v44 = *((_QWORD *)v42 + 1);
      v128 = (__int16)v43;
      v129 = v44;
      if ( (_WORD)v43 )
      {
        if ( (unsigned __int16)((_WORD)v43 - 17) <= 0xD3u )
        {
          v49 = 0;
          v47 = (_QWORD *)*a1;
          LOWORD(v43) = word_4456580[(int)v43 - 1];
          goto LABEL_22;
        }
      }
      else if ( sub_30070B0((__int64)&v128) )
      {
        v113 = *a1;
        v43 = (unsigned int *)sub_3009970((__int64)&v128, (__int64)v39, v45, v46, v23);
        v47 = (_QWORD *)v113;
        v33 = v43;
        v49 = v48;
LABEL_22:
        LOWORD(v33) = (_WORD)v43;
        *((_QWORD *)&v91 + 1) = v41;
        *(_QWORD *)&v91 = v40;
        v39 = sub_3406EB0(v47, 0x9Eu, v109, (unsigned int)v33, v49, v24, v91, v105);
        v38 = v50;
      }
      v34 = v38 | v41 & 0xFFFFFFFF00000000LL;
      v35 = (unsigned int)v134;
      v36 = (unsigned int)v134 + 1LL;
      if ( v36 > HIDWORD(v134) )
      {
        sub_C8D5F0((__int64)&v133, v135, v36, 0x10u, v23, v24);
        v35 = (unsigned int)v134;
      }
      v37 = (unsigned __int8 **)&v133[16 * v35];
      v32 += 40;
      *v37 = v39;
      v37[1] = (unsigned __int8 *)v34;
      v30 = (unsigned int)(v134 + 1);
      LODWORD(v134) = v134 + 1;
      if ( v32 == 40LL * (v100 - 2) + 80 )
      {
        v18 = (__int64 *)v109;
        v7 = v33;
        goto LABEL_29;
      }
    }
  }
  v63 = v138;
  v64 = 0;
LABEL_39:
  *((_QWORD *)&v94 + 1) = v64;
  *(_QWORD *)&v94 = v63;
  v65 = sub_33FC220((_QWORD *)*a1, 156, (__int64)v18, v124, v125, a6, v94);
  v67 = v66;
  *((_QWORD *)&v92 + 1) = (unsigned int)v140;
  *(_QWORD *)&v92 = v139;
  v70 = sub_33FC220((_QWORD *)*a1, 2, (__int64)v18, 1, 0, v68, v92);
  v71 = v69;
  v72 = *(unsigned int *)(a3 + 8);
  if ( v72 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v121 = v70;
    v123 = v69;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v72 + 1, 0x10u, (__int64)v70, v69);
    v72 = *(unsigned int *)(a3 + 8);
    v70 = v121;
    v71 = v123;
  }
  v73 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v72);
  v73[1] = v67;
  *v73 = v65;
  v74 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v74;
  if ( v74 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v120 = v70;
    v122 = v71;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v74 + 1, 0x10u, (__int64)v70, v71);
    v74 = *(unsigned int *)(a3 + 8);
    v70 = v120;
    v71 = v122;
  }
  v75 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v74);
  *v75 = v70;
  v76 = v139;
  v75[1] = (unsigned __int8 *)v71;
  ++*(_DWORD *)(a3 + 8);
  if ( v76 != v141 )
    _libc_free((unsigned __int64)v76);
  if ( v136 != v138 )
    _libc_free((unsigned __int64)v136);
  if ( v126 )
    sub_B91220((__int64)v18, v126);
}
