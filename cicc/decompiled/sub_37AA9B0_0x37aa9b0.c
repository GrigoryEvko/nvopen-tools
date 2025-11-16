// Function: sub_37AA9B0
// Address: 0x37aa9b0
//
unsigned __int8 *__fastcall sub_37AA9B0(__int64 a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v3; // rbx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int128 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  unsigned __int16 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // ebx
  __int64 v21; // rcx
  _BYTE *v22; // rax
  __int64 v23; // r13
  _BYTE *v24; // rdx
  _BYTE *v25; // rax
  _BYTE *v26; // rdx
  _BYTE *i; // r13
  unsigned int v28; // r13d
  __int64 v29; // r14
  unsigned int v30; // esi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  _QWORD *v36; // rbx
  __int128 v37; // rax
  __int64 v38; // r9
  __m128i v39; // rax
  _QWORD *v40; // rbx
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned __int8 *v43; // rax
  _QWORD *v44; // rdi
  __int64 v45; // rdx
  unsigned int v46; // esi
  __int64 v47; // r9
  unsigned __int8 *v48; // rax
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rbx
  unsigned __int8 *v52; // rdx
  unsigned __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rbx
  _BYTE *v56; // rax
  __int64 v57; // r9
  __m128i v58; // rax
  __int64 v59; // rdi
  __int64 v60; // r9
  unsigned __int8 *v61; // rax
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rcx
  unsigned __int8 *v64; // rdx
  _QWORD *v65; // r10
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rax
  __int16 v69; // si
  __int64 v70; // rax
  bool v71; // al
  _BYTE *v72; // rsi
  unsigned __int8 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r9
  unsigned __int8 *v76; // r14
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int128 v81; // [rsp-20h] [rbp-2E0h]
  __int128 v82; // [rsp-10h] [rbp-2D0h]
  __int128 v83; // [rsp-10h] [rbp-2D0h]
  __int64 v84; // [rsp+10h] [rbp-2B0h]
  __int64 v85; // [rsp+18h] [rbp-2A8h]
  unsigned __int8 *v86; // [rsp+20h] [rbp-2A0h]
  unsigned __int64 v87; // [rsp+28h] [rbp-298h]
  __int128 v88; // [rsp+30h] [rbp-290h]
  __int64 v89; // [rsp+48h] [rbp-278h]
  __int64 v90; // [rsp+50h] [rbp-270h]
  int v91; // [rsp+5Ch] [rbp-264h]
  __int128 v92; // [rsp+60h] [rbp-260h]
  int v93; // [rsp+70h] [rbp-250h]
  __int16 v94; // [rsp+76h] [rbp-24Ah]
  __int64 v95; // [rsp+78h] [rbp-248h]
  __m128i v97; // [rsp+90h] [rbp-230h] BYREF
  _QWORD *v98; // [rsp+A0h] [rbp-220h]
  __int64 v99; // [rsp+A8h] [rbp-218h]
  __int64 v100; // [rsp+B0h] [rbp-210h]
  __int64 v101; // [rsp+B8h] [rbp-208h]
  __int64 *v102; // [rsp+C0h] [rbp-200h]
  __int64 v103; // [rsp+C8h] [rbp-1F8h]
  __int64 v104; // [rsp+D0h] [rbp-1F0h]
  __int64 v105; // [rsp+D8h] [rbp-1E8h]
  unsigned __int8 *v106; // [rsp+E0h] [rbp-1E0h]
  __int64 v107; // [rsp+E8h] [rbp-1D8h]
  __int64 v108; // [rsp+F0h] [rbp-1D0h] BYREF
  int v109; // [rsp+F8h] [rbp-1C8h]
  __int128 v110; // [rsp+100h] [rbp-1C0h] BYREF
  unsigned __int16 v111; // [rsp+110h] [rbp-1B0h] BYREF
  __int64 v112; // [rsp+118h] [rbp-1A8h]
  __int16 v113; // [rsp+120h] [rbp-1A0h]
  __int64 v114; // [rsp+128h] [rbp-198h]
  __int64 v115; // [rsp+130h] [rbp-190h] BYREF
  __int64 v116; // [rsp+138h] [rbp-188h]
  __m128i v117; // [rsp+140h] [rbp-180h]
  unsigned __int8 *v118; // [rsp+150h] [rbp-170h]
  __int64 v119; // [rsp+158h] [rbp-168h]
  __int64 v120; // [rsp+160h] [rbp-160h]
  int v121; // [rsp+168h] [rbp-158h]
  _BYTE *v122; // [rsp+170h] [rbp-150h] BYREF
  __int64 v123; // [rsp+178h] [rbp-148h]
  _BYTE v124[128]; // [rsp+180h] [rbp-140h] BYREF
  _BYTE *v125; // [rsp+200h] [rbp-C0h] BYREF
  __int64 v126; // [rsp+208h] [rbp-B8h]
  _BYTE v127[176]; // [rsp+210h] [rbp-B0h] BYREF

  v6 = *(__int64 **)(a2 + 40);
  v90 = *v6;
  v91 = *((_DWORD *)v6 + 2);
  *(_QWORD *)&v92 = sub_379AB60(a1, v6[5], v6[6]);
  *((_QWORD *)&v92 + 1) = v7;
  *(_QWORD *)&v8 = sub_379AB60(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v11 = *(_QWORD *)(a2 + 80);
  v88 = v8;
  *(_QWORD *)&v8 = *(_QWORD *)(a2 + 40);
  v12 = *(_QWORD *)(v8 + 120);
  LODWORD(v8) = *(_DWORD *)(v8 + 128);
  v108 = v11;
  v89 = v12;
  v93 = v8;
  if ( v11 )
    sub_B96E90((__int64)&v108, v11, 1);
  v13 = *(unsigned __int16 **)(a2 + 48);
  v109 = *(_DWORD *)(a2 + 72);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  LOWORD(v110) = v14;
  *((_QWORD *)&v110 + 1) = v15;
  if ( (_WORD)v14 )
  {
    v101 = 0;
    v94 = word_4456580[v14 - 1];
  }
  else
  {
    v79 = sub_3009970((__int64)&v110, v11, v15, a2, v9);
    v94 = v79;
    v3 = v79;
    v101 = v80;
  }
  LOWORD(v3) = v94;
  v97.m128i_i64[0] = v3;
  v16 = (unsigned __int16 *)(*(_QWORD *)(v92 + 48) + 16LL * DWORD2(v92));
  LODWORD(v17) = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v125) = v17;
  v126 = v18;
  if ( (_WORD)v17 )
  {
    v99 = 0;
    LOWORD(v17) = word_4456580[(int)v17 - 1];
  }
  else
  {
    v17 = sub_3009970((__int64)&v125, v11, v18, v92, v9);
    v100 = v17;
    v99 = v78;
  }
  v19 = v100;
  LOWORD(v19) = v17;
  v100 = v19;
  if ( (_WORD)v110 )
  {
    if ( (unsigned __int16)(v110 - 176) > 0x34u )
    {
LABEL_9:
      v20 = word_4456340[(unsigned __int16)v110 - 1];
      goto LABEL_12;
    }
  }
  else if ( !sub_3007100((__int64)&v110) )
  {
    goto LABEL_11;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v110 )
  {
    if ( (unsigned __int16)(v110 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_9;
  }
LABEL_11:
  v20 = sub_3007130((__int64)&v110, v11);
LABEL_12:
  v95 = v20;
  v21 = v20;
  v122 = v124;
  v123 = 0x800000000LL;
  if ( !v20 )
  {
    v126 = 0x800000000LL;
    v125 = v127;
    v72 = v127;
    goto LABEL_30;
  }
  v22 = v124;
  v23 = 16LL * v20;
  v24 = &v124[v23];
  if ( v20 > 8uLL
    && (sub_C8D5F0((__int64)&v122, v124, v20, 0x10u, v9, v10),
        v24 = &v122[v23],
        v22 = &v122[16 * (unsigned int)v123],
        v22 == &v122[v23]) )
  {
    LODWORD(v123) = v20;
    v125 = v127;
    v126 = 0x800000000LL;
  }
  else
  {
    do
    {
      if ( v22 )
      {
        *(_QWORD *)v22 = 0;
        *((_DWORD *)v22 + 2) = 0;
      }
      v22 += 16;
    }
    while ( v22 != v24 );
    v25 = v127;
    LODWORD(v123) = v20;
    v26 = v127;
    v125 = v127;
    v126 = 0x800000000LL;
    if ( v20 <= 8uLL )
      goto LABEL_18;
  }
  sub_C8D5F0((__int64)&v125, v127, v20, 0x10u, v9, v10);
  v26 = v125;
  v25 = &v125[16 * (unsigned int)v126];
LABEL_18:
  for ( i = &v26[v23]; i != v25; v25 += 16 )
  {
    if ( v25 )
    {
      *(_QWORD *)v25 = 0;
      *((_DWORD *)v25 + 2) = 0;
    }
  }
  LODWORD(v126) = v20;
  HIWORD(v28) = v97.m128i_i16[1];
  v29 = 0;
  do
  {
    v36 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v37 = sub_3400EE0((__int64)v36, v29, (__int64)&v108, 0, a3);
    v39.m128i_i64[0] = (__int64)sub_3406EB0(v36, 0x9Eu, (__int64)&v108, (unsigned int)v100, v99, v38, v92, v37);
    v40 = *(_QWORD **)(a1 + 8);
    v97 = v39;
    *(_QWORD *)&v41 = sub_3400EE0((__int64)v40, v29, (__int64)&v108, 0, a3);
    v43 = sub_3406EB0(v40, 0x9Eu, (__int64)&v108, (unsigned int)v100, v99, v42, v88, v41);
    v44 = *(_QWORD **)(a1 + 8);
    v118 = v43;
    v115 = v90;
    v120 = v89;
    a3 = _mm_load_si128(&v97);
    LODWORD(v116) = v91;
    v121 = v93;
    v119 = v45;
    v102 = &v115;
    v111 = 2;
    v103 = 4;
    v46 = *(_DWORD *)(a2 + 24);
    *((_QWORD *)&v82 + 1) = 4;
    *(_QWORD *)&v82 = &v115;
    v113 = 1;
    v117 = a3;
    v112 = 0;
    v114 = 0;
    v48 = sub_3411BE0(v44, v46, (__int64)&v108, &v111, 2, v47, v82);
    LOWORD(v28) = v94;
    v49 = v101;
    v51 = v50;
    v52 = v48;
    v53 = (unsigned __int64)v122;
    v54 = v51;
    v106 = v52;
    v55 = 16 * v29;
    v107 = v54;
    *(_QWORD *)&v122[v55] = v52;
    *(_DWORD *)(v53 + v55 + 8) = v107;
    v56 = &v125[16 * v29];
    *(_QWORD *)v56 = *(_QWORD *)&v122[16 * v29];
    *((_DWORD *)v56 + 2) = 1;
    v98 = *(_QWORD **)(a1 + 8);
    v58.m128i_i64[0] = (__int64)sub_3401740((__int64)v98, 0, (__int64)&v108, v28, v49, v57, v110);
    v59 = *(_QWORD *)(a1 + 8);
    v97 = v58;
    v61 = sub_3401740(v59, 1, (__int64)&v108, v28, v101, v60, v110);
    v63 = v62;
    v64 = v61;
    v65 = v98;
    v66 = *(_QWORD *)&v122[16 * v29];
    v67 = *(_QWORD *)&v122[16 * v29 + 8];
    v68 = *(_QWORD *)(*(_QWORD *)&v122[v55] + 48LL) + 16LL * *(unsigned int *)&v122[v55 + 8];
    v69 = *(_WORD *)v68;
    v70 = *(_QWORD *)(v68 + 8);
    LOWORD(v115) = v69;
    v116 = v70;
    if ( v69 )
    {
      v30 = ((unsigned __int16)(v69 - 17) < 0xD4u) + 205;
    }
    else
    {
      v84 = v66;
      v85 = v67;
      v86 = v64;
      v87 = v63;
      v71 = sub_30070B0((__int64)&v115);
      v66 = v84;
      v67 = v85;
      v64 = v86;
      v63 = v87;
      v65 = v98;
      v30 = 205 - (!v71 - 1);
    }
    ++v29;
    v31 = sub_340EC60(
            v65,
            v30,
            (__int64)&v108,
            v28,
            v101,
            0,
            v66,
            v67,
            __PAIR128__(v63, (unsigned __int64)v64),
            *(_OWORD *)&v97);
    v33 = v32;
    v34 = v31;
    v35 = (unsigned __int64)v122;
    v104 = v34;
    v105 = v33;
    *(_QWORD *)&v122[v55] = v34;
    *(_DWORD *)(v35 + v55 + 8) = v105;
  }
  while ( v29 != v95 );
  v72 = v125;
  v21 = (unsigned int)v126;
LABEL_30:
  *((_QWORD *)&v83 + 1) = v21;
  *(_QWORD *)&v83 = v72;
  v73 = sub_33FC220(*(_QWORD **)(a1 + 8), 2, (__int64)&v108, 1, 0, v10, v83);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v73, v74);
  *((_QWORD *)&v81 + 1) = (unsigned int)v123;
  *(_QWORD *)&v81 = v122;
  v76 = sub_33FC220(*(_QWORD **)(a1 + 8), 156, (__int64)&v108, v110, *((__int64 *)&v110 + 1), v75, v81);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
  if ( v108 )
    sub_B91220((__int64)&v108, v108);
  return v76;
}
