// Function: sub_32B51D0
// Address: 0x32b51d0
//
__int64 __fastcall sub_32B51D0(
        _QWORD *a1,
        unsigned __int8 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // r13
  __int64 v12; // rsi
  __int64 v13; // r13
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  bool v16; // zf
  __int16 v17; // cx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // r15
  bool v26; // cl
  __int64 v27; // rdi
  __int64 v28; // r13
  __int64 (*v29)(); // rax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rdi
  __int64 v33; // r15
  __int64 v34; // r13
  __int64 (__fastcall *v35)(__int64, __int64, __int64, __int64, __int64); // r12
  __int64 v36; // rdx
  __int16 v37; // r8
  unsigned __int8 v38; // al
  unsigned __int8 v39; // r13
  char v40; // al
  bool v41; // r8
  unsigned __int8 v42; // dl
  unsigned __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r15
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r13
  __int64 v52; // r12
  __int64 v53; // r14
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // r8
  __int64 v57; // r9
  int v58; // eax
  __m128i v59; // xmm0
  char v60; // al
  __int64 v61; // r12
  int v62; // r9d
  __int64 v63; // r12
  __int64 v64; // rdx
  __int64 v65; // r13
  __int64 v66; // rdx
  int v67; // r9d
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r13
  __int64 v71; // r12
  __int128 v72; // rax
  __int64 v73; // r12
  int v74; // r9d
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r12
  __int64 v78; // r13
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // r12
  int v82; // r9d
  int v83; // r9d
  __int128 v84; // rax
  int v85; // r9d
  __int128 v86; // rax
  __int64 v87; // r15
  __int64 v88; // r14
  int v89; // r9d
  __int128 v90; // rax
  int v91; // r9d
  __int64 v92; // rax
  __int64 v93; // rdx
  __int128 v94; // rax
  int v95; // r9d
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r15
  __int64 v99; // r14
  __int128 v100; // rax
  __int128 v101; // [rsp-20h] [rbp-150h]
  __int128 v102; // [rsp-10h] [rbp-140h]
  __int128 v103; // [rsp-10h] [rbp-140h]
  __int64 v104; // [rsp+0h] [rbp-130h]
  __int128 v105; // [rsp+0h] [rbp-130h]
  __int64 v106; // [rsp+10h] [rbp-120h]
  __int64 v107; // [rsp+10h] [rbp-120h]
  __int128 v108; // [rsp+10h] [rbp-120h]
  __int64 v109; // [rsp+20h] [rbp-110h]
  bool v110; // [rsp+20h] [rbp-110h]
  bool v111; // [rsp+20h] [rbp-110h]
  __int64 v112; // [rsp+20h] [rbp-110h]
  __int128 v113; // [rsp+20h] [rbp-110h]
  __int128 v114; // [rsp+20h] [rbp-110h]
  __int64 v115; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v116; // [rsp+48h] [rbp-E8h]
  __int64 v117[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int128 v118; // [rsp+60h] [rbp-D0h] BYREF
  __int128 v119; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v120; // [rsp+80h] [rbp-B0h] BYREF
  __int128 v121; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v122; // [rsp+A0h] [rbp-90h] BYREF
  int v123; // [rsp+A8h] [rbp-88h]
  __int64 v124; // [rsp+B0h] [rbp-80h] BYREF
  int v125; // [rsp+B8h] [rbp-78h]
  unsigned int v126; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v127; // [rsp+C8h] [rbp-68h]
  __int64 v128; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v129; // [rsp+D8h] [rbp-58h]
  _QWORD v130[10]; // [rsp+E0h] [rbp-50h] BYREF

  v117[1] = a4;
  v10 = (unsigned int)a4;
  v117[0] = a3;
  v115 = a5;
  v116 = a6;
  *(_QWORD *)&v118 = 0;
  DWORD2(v118) = 0;
  *(_QWORD *)&v119 = 0;
  DWORD2(v119) = 0;
  v120.m128i_i64[0] = 0;
  v120.m128i_i32[2] = 0;
  *(_QWORD *)&v121 = 0;
  DWORD2(v121) = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  if ( !(unsigned __int8)sub_3264EF0((__int64)a1, a3, a4, (__int64)&v118, (__int64)&v119, (__int64)&v122, 0) )
    return 0;
  v12 = v115;
  if ( !(unsigned __int8)sub_3264EF0((__int64)a1, v115, v116, (__int64)&v120, (__int64)&v121, (__int64)&v124, 0) )
    return 0;
  v13 = *(_QWORD *)(a3 + 48) + 16 * v10;
  v14 = *(_WORD *)v13;
  v127 = *(_QWORD *)(v13 + 8);
  LOWORD(v126) = v14;
  v15 = *(_QWORD *)(v118 + 48) + 16LL * DWORD2(v118);
  v16 = *((_BYTE *)a1 + 33) == 0;
  v17 = *(_WORD *)v15;
  v18 = *(_QWORD *)(v15 + 8);
  LOWORD(v128) = v17;
  v129 = v18;
  if ( v16 )
  {
    if ( v14 )
    {
      if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
        v14 = word_4456580[v14 - 1];
LABEL_8:
      if ( v14 == 2 )
        goto LABEL_9;
      goto LABEL_30;
    }
    if ( sub_30070B0((__int64)&v126) )
    {
      v14 = sub_3009970((__int64)&v126, v115, v19, v20, v21);
      goto LABEL_8;
    }
  }
LABEL_30:
  v33 = a1[1];
  v34 = v128;
  v109 = v129;
  v35 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v33 + 528LL);
  v106 = *(_QWORD *)(*a1 + 64LL);
  v12 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  v37 = v35(v33, v12, v106, v34, v109);
  if ( (_WORD)v126 != v37 || !(_WORD)v126 && v127 != v36 )
    return 0;
LABEL_9:
  v22 = v120.m128i_i64[0];
  v23 = *(_QWORD *)(v120.m128i_i64[0] + 48) + 16LL * v120.m128i_u32[2];
  if ( (_WORD)v128 != *(_WORD *)v23 )
    return 0;
  if ( (_WORD)v128 )
  {
    v24 = *(unsigned int *)(v122 + 96);
    v25 = *(unsigned int *)(v124 + 96);
    v26 = (unsigned __int16)(v128 - 17) <= 0x6Cu || (unsigned __int16)(v128 - 2) <= 7u;
    if ( !v26 )
      v26 = (unsigned __int16)(v128 - 176) <= 0x1Fu;
  }
  else
  {
    if ( v129 != *(_QWORD *)(v23 + 8) )
      return 0;
    v24 = *(unsigned int *)(v122 + 96);
    v25 = *(unsigned int *)(v124 + 96);
    v26 = sub_3007070((__int64)&v128);
  }
  if ( (_QWORD)v119 != (_QWORD)v121 || DWORD2(v119) != DWORD2(v121) || (_DWORD)v25 != (_DWORD)v24 || !v26 )
  {
    if ( !a2 || v22 != (_QWORD)v118 || DWORD2(v118) != v120.m128i_i32[2] || (_DWORD)v25 != (_DWORD)v24 )
      goto LABEL_16;
    goto LABEL_71;
  }
  v110 = v26;
  v38 = sub_33E0720(v119, *((_QWORD *)&v119 + 1), 0);
  v12 = *((_QWORD *)&v119 + 1);
  v39 = v38;
  v40 = sub_33E07E0(v119, *((_QWORD *)&v119 + 1), 0);
  v26 = v110;
  if ( (a2 & ((_DWORD)v25 == 17)) != 0 && v39 )
    goto LABEL_93;
  v41 = (_DWORD)v25 == 18;
  if ( (a2 & ((_DWORD)v25 == 18)) != 0 )
  {
    if ( v40 )
      goto LABEL_93;
  }
  v42 = a2 ^ 1;
  LOBYTE(v12) = (a2 ^ 1) & ((_DWORD)v25 == 22);
  if ( (_BYTE)v12 )
  {
    if ( v39 )
    {
LABEL_93:
      v73 = *a1;
      sub_3285E70((__int64)v130, v117[0]);
      v75 = sub_3406EB0(v73, 187, (unsigned int)v130, v128, v129, v74, v118, *(_OWORD *)&v120);
LABEL_94:
      v77 = v75;
      v78 = v76;
      sub_9C6650(v130);
      sub_32B3E80((__int64)a1, v77, 1, 0, v79, v80);
      return sub_32889F0(*a1, a7, v126, v127, v77, v78, v119, v25, 0);
    }
    v42 = (a2 ^ 1) & ((_DWORD)v25 == 22);
    v41 = 0;
  }
  if ( (((_DWORD)v25 == 20) & v39) != 0 && v42 )
    goto LABEL_93;
  if ( (a2 & ((_DWORD)v25 == 17)) != 0 && v40
    || (a2 & v39) != 0 && (_DWORD)v25 == 20
    || (_BYTE)v12 && v40
    || (v41 & v42) != 0 && v40 )
  {
    v81 = *a1;
    sub_3285E70((__int64)v130, v117[0]);
    v75 = sub_3406EB0(v81, 186, (unsigned int)v130, v128, v129, v82, v118, *(_OWORD *)&v120);
    goto LABEL_94;
  }
  if ( !a2 || (_QWORD)v118 != v120.m128i_i64[0] || DWORD2(v118) != v120.m128i_i32[2] || (_DWORD)v25 != (_DWORD)v24 )
    goto LABEL_17;
LABEL_71:
  v111 = v26;
  v43 = sub_32844A0((unsigned __int16 *)&v128, v12);
  v26 = v111;
  if ( v43 > 1 && (_DWORD)v24 == 22 )
  {
    if ( !v111 )
      goto LABEL_44;
    if ( (unsigned __int8)sub_33CF170(v119, *((_QWORD *)&v119 + 1))
      && (v44 = *((_QWORD *)&v121 + 1), (unsigned __int8)sub_33CF460(v121, *((_QWORD *)&v121 + 1)))
      || (unsigned __int8)sub_33CF460(v119, *((_QWORD *)&v119 + 1))
      && (v44 = *((_QWORD *)&v121 + 1), (unsigned __int8)sub_33CF170(v121, *((_QWORD *)&v121 + 1))) )
    {
      v45 = sub_3400BD0(*a1, 1, a7, v128, v129, 0, 0, v44);
      LODWORD(v104) = 0;
      v47 = v46;
      v48 = v45;
      v49 = sub_3400BD0(*a1, 2, a7, v128, v129, 0, v104);
      v51 = v50;
      v52 = v49;
      v107 = *a1;
      sub_3285E70((__int64)v130, v117[0]);
      *((_QWORD *)&v102 + 1) = v47;
      *(_QWORD *)&v102 = v48;
      v53 = sub_3406EB0(v107, 56, (unsigned int)v130, v128, v129, (unsigned int)v130, v118, v102);
      v55 = v54;
      sub_9C6650(v130);
      sub_32B3E80((__int64)a1, v53, 1, 0, v56, v57);
      *((_QWORD *)&v101 + 1) = v51;
      *(_QWORD *)&v101 = v52;
      return sub_32889F0(*a1, a7, v126, v127, v53, v55, v101, 0xBu, 0);
    }
    goto LABEL_17;
  }
LABEL_16:
  if ( !v26 )
  {
LABEL_44:
    v28 = v118;
    goto LABEL_18;
  }
LABEL_17:
  v27 = a1[1];
  v28 = v118;
  v29 = *(__int64 (**)())(*(_QWORD *)v27 + 368LL);
  if ( v29 == sub_2FE3010
    || (v60 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v29)(v27, (unsigned int)v128, v129),
        v28 = v118,
        (_DWORD)v25 != (_DWORD)v24)
    || !v60
    || !(unsigned __int8)sub_3286E00(v117)
    || !(unsigned __int8)sub_3286E00(&v115) )
  {
LABEL_18:
    v30 = v120.m128i_i64[0];
    if ( v28 == (_QWORD)v121
      && DWORD2(v118) == DWORD2(v121)
      && (_QWORD)v119 == v120.m128i_i64[0]
      && DWORD2(v119) == v120.m128i_i32[2] )
    {
      v58 = sub_33CBD20((unsigned int)v25);
      v59 = _mm_loadu_si128(&v120);
      LODWORD(v25) = v58;
      v30 = v121;
      v28 = v118;
      v120.m128i_i32[2] = DWORD2(v121);
      v120.m128i_i64[0] = v121;
      DWORD2(v121) = v59.m128i_i32[2];
      *(_QWORD *)&v121 = v59.m128i_i64[0];
    }
    if ( v30 == v28 && DWORD2(v118) == v120.m128i_i32[2] && (_QWORD)v119 == (_QWORD)v121 && DWORD2(v119) == DWORD2(v121) )
    {
      v31 = a2
          ? (unsigned int)sub_33CBF10((unsigned int)v24, (unsigned int)v25, (unsigned int)v128, v129)
          : (unsigned int)sub_33CBDB0((unsigned int)v24, (unsigned int)v25, (unsigned int)v128, v129);
      if ( (_DWORD)v31 != 24 )
      {
        if ( !*((_BYTE *)a1 + 33) )
          return sub_32889F0(*a1, a7, v126, v127, v118, *((__int64 *)&v118 + 1), v119, (unsigned __int64)v31, 0);
        v32 = a1[1];
        if ( ((*(_DWORD *)(v32
                         + 4
                         * (((*(_WORD *)(*(_QWORD *)(v118 + 48) + 16LL * DWORD2(v118)) >> 3) & 0x1FFF)
                          + 35LL * (int)v31
                          + 130384)) >> (4 * (*(_WORD *)(*(_QWORD *)(v118 + 48) + 16LL * DWORD2(v118)) & 7)))
            & 0xF) == 0
          && sub_328D6E0(v32, 0xD0u, v128) )
        {
          return sub_32889F0(*a1, a7, v126, v127, v118, *((__int64 *)&v118 + 1), v119, (unsigned __int64)v31, 0);
        }
      }
    }
    return 0;
  }
  if ( (_DWORD)v25 == 17 && a2 )
    goto LABEL_92;
  if ( a2 != 1 )
  {
    if ( (_DWORD)v25 == 22 )
    {
LABEL_92:
      v61 = *a1;
      sub_3285E70((__int64)v130, v117[0]);
      v63 = sub_3406EB0(v61, 188, (unsigned int)v130, v128, v129, v62, v118, v119);
      v65 = v64;
      sub_9C6650(v130);
      v112 = *a1;
      sub_3285E70((__int64)v130, v115);
      *(_QWORD *)&v113 = sub_3406EB0(v112, 188, (unsigned int)v130, v128, v129, v112, *(_OWORD *)&v120, v121);
      *((_QWORD *)&v113 + 1) = v66;
      sub_9C6650(v130);
      *((_QWORD *)&v103 + 1) = v65;
      *(_QWORD *)&v103 = v63;
      v68 = sub_3406EB0(*a1, 187, a7, v128, v129, v67, v103, v113);
      v70 = v69;
      v71 = v68;
      *(_QWORD *)&v72 = sub_3400BD0(*a1, 0, a7, v128, v129, 0, 0);
      return sub_32889F0(*a1, a7, v126, v127, v71, v70, v72, v25, 0);
    }
  }
  else if ( (_DWORD)v25 == 22 && a2 )
  {
    goto LABEL_40;
  }
  if ( (_DWORD)v25 != 17 || a2 == 1 )
    goto LABEL_44;
LABEL_40:
  v28 = v118;
  if ( (_QWORD)v118 != v120.m128i_i64[0] || DWORD2(v118) != v120.m128i_i32[2] )
    goto LABEL_18;
  v130[3] = sub_3263120;
  v130[2] = sub_325D4E0;
  if ( !(unsigned __int8)sub_33CACD0(v119, DWORD2(v119), v121, DWORD2(v121), (unsigned int)v130, 0, 0) )
  {
    sub_A17130((__int64)v130);
    goto LABEL_44;
  }
  sub_A17130((__int64)v130);
  *(_QWORD *)&v84 = sub_3406EB0(*a1, 183, a7, v128, v129, v83, v119, v121);
  v108 = v84;
  *(_QWORD *)&v86 = sub_3406EB0(*a1, 182, a7, v128, v129, v85, v119, v121);
  v87 = *((_QWORD *)&v86 + 1);
  v88 = v86;
  *(_QWORD *)&v90 = sub_3406EB0(*a1, 57, a7, v128, v129, v89, v118, v86);
  *((_QWORD *)&v105 + 1) = v87;
  *(_QWORD *)&v105 = v88;
  v114 = v90;
  v92 = sub_3406EB0(*a1, 57, a7, v128, v129, v91, v108, v105);
  *(_QWORD *)&v94 = sub_34074A0(*a1, a7, v92, v93, (unsigned int)v128, v129);
  v96 = sub_3406EB0(*a1, 186, a7, v128, v129, v95, v114, v94);
  v98 = v97;
  v99 = v96;
  *(_QWORD *)&v100 = sub_3400BD0(*a1, 0, a7, v128, v129, 0, 0);
  return sub_32889F0(*a1, a7, v126, v127, v99, v98, v100, v24, 0);
}
