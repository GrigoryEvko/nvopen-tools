// Function: sub_3411F50
// Address: 0x3411f50
//
__int64 __fastcall sub_3411F50(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __m128i a6)
{
  int v6; // r12d
  unsigned __int16 *v9; // rcx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned int v17; // edi
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 (__fastcall *v20)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r9
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r8
  unsigned __int8 **v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 *v35; // rax
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // rdx
  unsigned __int8 *v38; // rbx
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned __int8 *v41; // rax
  __int64 v42; // r8
  unsigned __int8 *v43; // r10
  __int64 v44; // rdx
  __int64 v45; // r11
  __int64 v46; // rax
  __int16 v47; // dx
  __int64 v48; // rax
  bool v49; // al
  __int64 v50; // r9
  _QWORD *v51; // r12
  int v52; // edx
  int v53; // ebx
  __int64 v54; // rax
  unsigned __int64 v55; // r8
  int v56; // edx
  _BYTE *v57; // rax
  _BYTE *v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // r8
  __int64 v61; // r9
  _QWORD *v62; // r12
  int v63; // edx
  int v64; // ebx
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  int v67; // ecx
  _BYTE *v68; // rax
  _BYTE *v69; // rdx
  __int64 v70; // r12
  __int64 *v71; // r13
  unsigned __int16 v72; // bx
  __int64 *v73; // r13
  unsigned __int16 v74; // ax
  __int64 v75; // r9
  __int64 v76; // r8
  int v77; // edx
  int v78; // r13d
  __int64 v79; // r9
  int v80; // edx
  int v82; // kr00_4
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rdx
  __int128 v86; // [rsp-20h] [rbp-390h]
  __int128 v87; // [rsp-20h] [rbp-390h]
  __int128 v88; // [rsp-10h] [rbp-380h]
  __int64 v90; // [rsp+30h] [rbp-340h]
  __int64 v91; // [rsp+38h] [rbp-338h]
  __int64 v92; // [rsp+40h] [rbp-330h]
  unsigned int v94; // [rsp+4Ch] [rbp-324h]
  __int64 v95; // [rsp+50h] [rbp-320h]
  unsigned int v96; // [rsp+50h] [rbp-320h]
  unsigned __int8 *v97; // [rsp+70h] [rbp-300h]
  __int64 v98; // [rsp+78h] [rbp-2F8h]
  unsigned int *v99; // [rsp+80h] [rbp-2F0h]
  __int64 v100; // [rsp+88h] [rbp-2E8h]
  __int64 v101; // [rsp+90h] [rbp-2E0h]
  unsigned int v102; // [rsp+98h] [rbp-2D8h]
  __int16 v103; // [rsp+9Eh] [rbp-2D2h]
  __int128 v104; // [rsp+A0h] [rbp-2D0h]
  unsigned int v105; // [rsp+A0h] [rbp-2D0h]
  __int64 v106; // [rsp+A0h] [rbp-2D0h]
  __int64 v107; // [rsp+A0h] [rbp-2D0h]
  __int16 v108; // [rsp+A2h] [rbp-2CEh]
  __int64 v109; // [rsp+A8h] [rbp-2C8h]
  __int64 v110; // [rsp+A8h] [rbp-2C8h]
  __int64 v111; // [rsp+B0h] [rbp-2C0h]
  unsigned __int8 *v112; // [rsp+B0h] [rbp-2C0h]
  __int64 v113; // [rsp+B8h] [rbp-2B8h]
  __int64 v114; // [rsp+B8h] [rbp-2B8h]
  unsigned __int8 *v115; // [rsp+B8h] [rbp-2B8h]
  __int128 v116; // [rsp+C0h] [rbp-2B0h] BYREF
  unsigned __int16 v117; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v118; // [rsp+D8h] [rbp-298h]
  __int64 v119; // [rsp+E0h] [rbp-290h] BYREF
  int v120; // [rsp+E8h] [rbp-288h]
  __int64 v121; // [rsp+F0h] [rbp-280h] BYREF
  __int64 v122; // [rsp+F8h] [rbp-278h]
  unsigned __int64 v123[2]; // [rsp+100h] [rbp-270h] BYREF
  _BYTE v124[128]; // [rsp+110h] [rbp-260h] BYREF
  unsigned __int64 v125[2]; // [rsp+190h] [rbp-1E0h] BYREF
  _BYTE v126[128]; // [rsp+1A0h] [rbp-1D0h] BYREF
  _BYTE *v127; // [rsp+220h] [rbp-150h] BYREF
  __int64 v128; // [rsp+228h] [rbp-148h]
  _BYTE v129[128]; // [rsp+230h] [rbp-140h] BYREF
  _BYTE *v130; // [rsp+2B0h] [rbp-C0h] BYREF
  __int64 v131; // [rsp+2B8h] [rbp-B8h]
  _BYTE v132[176]; // [rsp+2C0h] [rbp-B0h] BYREF

  v9 = *(unsigned __int16 **)(a3 + 48);
  v102 = *(_DWORD *)(a3 + 24);
  v10 = *v9;
  *((_QWORD *)&v116 + 1) = *((_QWORD *)v9 + 1);
  v11 = v9[8];
  v12 = *((_QWORD *)v9 + 3);
  LOWORD(v116) = v10;
  v117 = v11;
  v118 = v12;
  if ( (_WORD)v10 )
  {
    v91 = 0;
    v13 = (int)v10 - 1;
    v14 = (unsigned __int16)word_4456580[(int)v13];
  }
  else
  {
    v95 = sub_3009970((__int64)&v116, a2, v10, v12, a5);
    v14 = (unsigned int)v95;
    v11 = v117;
    v91 = v13;
  }
  v15 = v95;
  LOWORD(v15) = v14;
  v96 = v15;
  if ( (_WORD)v11 )
  {
    v90 = 0;
    v103 = word_4456580[v11 - 1];
  }
  else
  {
    v82 = sub_3009970((__int64)&v117, v15, v13, v14, a5);
    HIWORD(v6) = HIWORD(v82);
    v103 = v82;
    v90 = v83;
  }
  v16 = *(_QWORD *)(a3 + 80);
  v108 = HIWORD(v6);
  v119 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v119, v16, 1);
  v120 = *(_DWORD *)(a3 + 72);
  if ( (_WORD)v116 )
  {
    if ( (unsigned __int16)(v116 - 176) > 0x34u )
    {
LABEL_9:
      v94 = word_4456340[(unsigned __int16)v116 - 1];
      goto LABEL_12;
    }
  }
  else if ( !sub_3007100((__int64)&v116) )
  {
    goto LABEL_11;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v116 )
  {
    if ( (unsigned __int16)(v116 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_9;
  }
LABEL_11:
  v94 = sub_3007130((__int64)&v116, v16);
LABEL_12:
  v17 = a4;
  v18 = v94;
  if ( a4 )
  {
    v94 = a4;
    if ( a4 <= (unsigned int)v18 )
      v18 = a4;
    a4 -= v18;
    v92 = v17 - (unsigned int)v18;
  }
  else
  {
    v92 = 0;
  }
  v123[0] = (unsigned __int64)v124;
  v125[0] = (unsigned __int64)v126;
  v19 = *(__int64 **)(a3 + 40);
  v123[1] = 0x800000000LL;
  v125[1] = 0x800000000LL;
  sub_3408690((_QWORD *)a2, *v19, v19[1], (unsigned __int16 *)v123, 0, v18, a6, 0, 0);
  sub_3408690(
    (_QWORD *)a2,
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL),
    (unsigned __int16 *)v125,
    0,
    v18,
    a6,
    0,
    0);
  v111 = *(_QWORD *)(a2 + 16);
  v113 = *(_QWORD *)(a2 + 64);
  v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v111 + 528LL);
  v21 = sub_2E79000(*(__int64 **)(a2 + 40));
  v22 = v20(v111, v21, v113, v96, v91);
  v24 = sub_33E5110((__int64 *)a2, v96, v91, v22, v23);
  v128 = 0x800000000LL;
  v127 = v129;
  v130 = v132;
  v131 = 0x800000000LL;
  if ( (_DWORD)v18 )
  {
    v99 = (unsigned int *)v24;
    v101 = 16 * v18;
    WORD1(v18) = v108;
    v114 = 0;
    v100 = v25;
    do
    {
      v36 = sub_3411F20(
              (_QWORD *)a2,
              v102,
              (__int64)&v119,
              v99,
              v100,
              v26,
              *(_OWORD *)(v123[0] + v114),
              *(_OWORD *)(v125[0] + v114));
      LOWORD(v18) = v103;
      v112 = v37;
      v38 = v36;
      *(_QWORD *)&v39 = sub_3400BD0(a2, 0, (__int64)&v119, (unsigned int)v18, v90, 0, a6, 0);
      v104 = v39;
      v41 = sub_3401740(a2, 1, (__int64)&v119, (unsigned int)v18, v90, v40, v116);
      v42 = (__int64)v38;
      v43 = v41;
      v45 = v44;
      v46 = *((_QWORD *)v38 + 6);
      v47 = *(_WORD *)(v46 + 16);
      v48 = *(_QWORD *)(v46 + 24);
      LOWORD(v121) = v47;
      v122 = v48;
      if ( v47 )
      {
        v27 = ((unsigned __int16)(v47 - 17) < 0xD4u) + 205;
      }
      else
      {
        v97 = v43;
        v98 = v45;
        v49 = sub_30070B0((__int64)&v121);
        v42 = (__int64)v38;
        v43 = v97;
        v45 = v98;
        v27 = 205 - (!v49 - 1);
      }
      *((_QWORD *)&v86 + 1) = v45;
      *(_QWORD *)&v86 = v43;
      v28 = sub_340EC60((_QWORD *)a2, v27, (__int64)&v119, v18, v90, 0, v42, 1, v86, v104);
      v26 = v29;
      v30 = (unsigned int)v128;
      v31 = v28;
      if ( (unsigned __int64)(unsigned int)v128 + 1 > HIDWORD(v128) )
      {
        v107 = v28;
        v110 = v26;
        sub_C8D5F0((__int64)&v127, v129, (unsigned int)v128 + 1LL, 0x10u, v28, v26);
        v30 = (unsigned int)v128;
        v31 = v107;
        v26 = v110;
      }
      v32 = (unsigned __int8 **)&v127[16 * v30];
      *v32 = v38;
      v32[1] = v112;
      v33 = (unsigned int)v131;
      LODWORD(v128) = v128 + 1;
      v34 = (unsigned int)v131 + 1LL;
      if ( v34 > HIDWORD(v131) )
      {
        v106 = v31;
        v109 = v26;
        sub_C8D5F0((__int64)&v130, v132, v34, 0x10u, v31, v26);
        v33 = (unsigned int)v131;
        v31 = v106;
        v26 = v109;
      }
      v35 = (__int64 *)&v130[16 * v33];
      v114 += 16;
      *v35 = v31;
      v35[1] = v26;
      LODWORD(v131) = v131 + 1;
    }
    while ( v101 != v114 );
    v108 = WORD1(v18);
  }
  v121 = 0;
  LODWORD(v122) = 0;
  v51 = sub_33F17F0((_QWORD *)a2, 51, (__int64)&v121, v96, v91);
  v53 = v52;
  if ( v121 )
    sub_B91220((__int64)&v121, v121);
  v54 = (unsigned int)v128;
  v55 = (unsigned int)v128 + v92;
  v56 = v128;
  if ( v55 > HIDWORD(v128) )
  {
    sub_C8D5F0((__int64)&v127, v129, (unsigned int)v128 + v92, 0x10u, v55, v50);
    v54 = (unsigned int)v128;
    v56 = v128;
  }
  v57 = &v127[16 * v54];
  if ( v92 )
  {
    v58 = &v57[16 * v92];
    do
    {
      if ( v57 )
      {
        *(_QWORD *)v57 = v51;
        *((_DWORD *)v57 + 2) = v53;
      }
      v57 += 16;
    }
    while ( v57 != v58 );
    v56 = v128;
  }
  HIWORD(v59) = v108;
  LOWORD(v59) = v103;
  LODWORD(v128) = a4 + v56;
  v121 = 0;
  v105 = v59;
  LODWORD(v122) = 0;
  v62 = sub_33F17F0((_QWORD *)a2, 51, (__int64)&v121, v59, v90);
  v64 = v63;
  if ( v121 )
    sub_B91220((__int64)&v121, v121);
  v65 = (unsigned int)v131;
  v66 = (unsigned int)v131 + v92;
  v67 = v131;
  if ( v66 > HIDWORD(v131) )
  {
    sub_C8D5F0((__int64)&v130, v132, v66, 0x10u, v60, v61);
    v65 = (unsigned int)v131;
    v67 = v131;
  }
  v68 = &v130[16 * v65];
  if ( v92 )
  {
    v69 = &v68[16 * v92];
    do
    {
      if ( v68 )
      {
        *(_QWORD *)v68 = v62;
        *((_DWORD *)v68 + 2) = v64;
      }
      v68 += 16;
    }
    while ( v69 != v68 );
    v67 = v131;
  }
  v70 = 0;
  v71 = *(__int64 **)(a2 + 64);
  LODWORD(v131) = a4 + v67;
  v72 = sub_2D43050(v96, v94);
  if ( !v72 )
  {
    v72 = sub_3009400(v71, v96, v91, v94, 0);
    v70 = v85;
  }
  v73 = *(__int64 **)(a2 + 64);
  v74 = sub_2D43050(v105, v94);
  v76 = 0;
  if ( !v74 )
  {
    v74 = sub_3009400(v73, v105, v90, v94, 0);
    v76 = v84;
  }
  *((_QWORD *)&v88 + 1) = (unsigned int)v131;
  *(_QWORD *)&v88 = v130;
  v115 = sub_33FC220((_QWORD *)a2, 156, (__int64)&v119, v74, v76, v75, v88);
  v78 = v77;
  *((_QWORD *)&v87 + 1) = (unsigned int)v128;
  *(_QWORD *)&v87 = v127;
  *(_QWORD *)a1 = sub_33FC220((_QWORD *)a2, 156, (__int64)&v119, v72, v70, v79, v87);
  *(_DWORD *)(a1 + 8) = v80;
  *(_QWORD *)(a1 + 16) = v115;
  *(_DWORD *)(a1 + 24) = v78;
  if ( v130 != v132 )
    _libc_free((unsigned __int64)v130);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  if ( (_BYTE *)v125[0] != v126 )
    _libc_free(v125[0]);
  if ( (_BYTE *)v123[0] != v124 )
    _libc_free(v123[0]);
  if ( v119 )
    sub_B91220((__int64)&v119, v119);
  return a1;
}
