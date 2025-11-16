// Function: sub_345AE90
// Address: 0x345ae90
//
__int64 __fastcall sub_345AE90(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, __m128i a7)
{
  __int64 v9; // rsi
  unsigned __int16 *v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // rax
  __m128i v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  int v16; // r9d
  __m128i v17; // rax
  __int64 v18; // r9
  _BOOL4 v19; // r10d
  unsigned int v20; // esi
  unsigned int v21; // r10d
  __int64 v22; // rax
  unsigned __int8 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  unsigned __int8 *v26; // r14
  __int64 v27; // r9
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned __int8 *v30; // rax
  __int64 v31; // r14
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r15
  __int64 v37; // r14
  __int64 v38; // r9
  __int128 v39; // rax
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r10
  __int64 v44; // r11
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r9
  __int64 v48; // rax
  unsigned int v49; // eax
  unsigned int v50; // esi
  unsigned int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // r8
  __int128 v54; // rax
  __int64 v55; // r9
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rdx
  int v60; // eax
  __int128 v61; // rax
  __int64 v62; // r9
  __int64 v63; // rax
  int v64; // r9d
  __int64 v65; // rax
  unsigned int v66; // eax
  __int64 v67; // rdx
  unsigned __int8 *v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r15
  unsigned __int8 *v71; // r14
  __int64 v72; // r9
  unsigned __int8 *v73; // rax
  __m128i v74; // rcx
  __int64 v75; // r10
  __int64 v76; // rdx
  __int64 v77; // r11
  __int64 v78; // rdx
  __int16 v79; // ax
  __int64 v80; // rdx
  bool v81; // al
  __m128i v82; // xmm2
  unsigned int *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r9
  unsigned __int8 *v86; // rbx
  __int128 v87; // rax
  __int64 v88; // r15
  __int64 v89; // r14
  __int128 v90; // rax
  __int64 v91; // r9
  __int64 v92; // rdx
  __int128 v93; // [rsp-20h] [rbp-F0h]
  __int128 v94; // [rsp-20h] [rbp-F0h]
  __int128 v95; // [rsp-10h] [rbp-E0h]
  __int128 v96; // [rsp-10h] [rbp-E0h]
  __int128 v97; // [rsp-10h] [rbp-E0h]
  __int128 v98; // [rsp-10h] [rbp-E0h]
  __int128 v99; // [rsp-10h] [rbp-E0h]
  __int64 v100; // [rsp+0h] [rbp-D0h]
  unsigned int v101; // [rsp+8h] [rbp-C8h]
  __int16 v102; // [rsp+10h] [rbp-C0h]
  __int64 (__fastcall *v103)(_DWORD *, __int64, __int64, _QWORD, __int64); // [rsp+18h] [rbp-B8h]
  __int64 v104; // [rsp+18h] [rbp-B8h]
  __int64 v105; // [rsp+18h] [rbp-B8h]
  __int64 (__fastcall *v106)(_DWORD *, __int64, __int64, _QWORD, __int64); // [rsp+18h] [rbp-B8h]
  unsigned int v107; // [rsp+20h] [rbp-B0h]
  __int64 v108; // [rsp+20h] [rbp-B0h]
  unsigned int v109; // [rsp+20h] [rbp-B0h]
  __int64 v110; // [rsp+20h] [rbp-B0h]
  __int64 v111; // [rsp+28h] [rbp-A8h]
  __int64 v112; // [rsp+30h] [rbp-A0h]
  char v113; // [rsp+30h] [rbp-A0h]
  int v114; // [rsp+3Ch] [rbp-94h]
  __m128i v115; // [rsp+40h] [rbp-90h] BYREF
  __m128i v116; // [rsp+50h] [rbp-80h] BYREF
  __int64 v117; // [rsp+60h] [rbp-70h] BYREF
  int v118; // [rsp+68h] [rbp-68h]
  __m128i v119; // [rsp+70h] [rbp-60h] BYREF
  _OWORD v120[5]; // [rsp+80h] [rbp-50h] BYREF

  v9 = *(_QWORD *)(a2 + 80);
  v117 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v117, v9, 1);
  v118 = *(_DWORD *)(a2 + 72);
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *v10;
  v112 = *((_QWORD *)v10 + 1);
  v119.m128i_i64[1] = v112;
  v12 = *(__int64 **)(a2 + 40);
  v119.m128i_i16[0] = v11;
  v13.m128i_i64[0] = (__int64)sub_33FB960(a3, *v12, v12[1], a7, a4, a5, a6);
  v116 = v13;
  v17.m128i_i64[0] = (__int64)sub_33FB960(
                                a3,
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                a7,
                                v14,
                                v15,
                                v16);
  v115 = v17;
  v114 = *(_DWORD *)(a2 + 24);
  v19 = v114 != 178;
  v20 = 2 * v19 + 181;
  v21 = 2 * v19 + 180;
  if ( (_WORD)v11 == 1 )
  {
    if ( *((_BYTE *)a1 + v20 + 6914) )
      goto LABEL_5;
    v33 = 1;
    goto LABEL_15;
  }
  if ( !(_WORD)v11 )
    goto LABEL_20;
  v33 = (unsigned __int16)v11;
  if ( !*(_QWORD *)&a1[2 * (unsigned __int16)v11 + 28] )
    goto LABEL_20;
  if ( !*((_BYTE *)&a1[125 * (unsigned __int16)v11 + 1603] + v20 + 2) )
  {
LABEL_15:
    v107 = v21;
    if ( !*((_BYTE *)&a1[125 * v33 + 1603] + v21 + 2) )
    {
      v34 = sub_3406EB0(
              (_QWORD *)a3,
              v20,
              (__int64)&v117,
              v119.m128i_u32[0],
              v119.m128i_i64[1],
              v18,
              *(_OWORD *)&v116,
              *(_OWORD *)&v115);
      v36 = v35;
      v37 = (__int64)v34;
      *(_QWORD *)&v39 = sub_3406EB0(
                          (_QWORD *)a3,
                          v107,
                          (__int64)&v117,
                          v119.m128i_u32[0],
                          v119.m128i_i64[1],
                          v38,
                          *(_OWORD *)&v116,
                          *(_OWORD *)&v115);
LABEL_17:
      *((_QWORD *)&v93 + 1) = v36;
      *(_QWORD *)&v93 = v37;
      v30 = sub_3406EB0((_QWORD *)a3, 0x39u, (__int64)&v117, v119.m128i_u32[0], v119.m128i_i64[1], v40, v93, v39);
      goto LABEL_10;
    }
  }
LABEL_5:
  if ( v114 != 178 )
  {
    if ( (_WORD)v11 == 1 )
    {
      v22 = 1;
    }
    else
    {
      v22 = (unsigned __int16)v11;
      if ( !*(_QWORD *)&a1[2 * (unsigned __int16)v11 + 28] )
        goto LABEL_20;
    }
    if ( !HIBYTE(a1[125 * v22 + 1624]) )
    {
      v23 = sub_3406EB0(
              (_QWORD *)a3,
              0x55u,
              (__int64)&v117,
              v119.m128i_u32[0],
              v119.m128i_i64[1],
              v18,
              *(_OWORD *)&v115,
              *(_OWORD *)&v116);
      v25 = v24;
      v26 = v23;
      *(_QWORD *)&v28 = sub_3406EB0(
                          (_QWORD *)a3,
                          0x55u,
                          (__int64)&v117,
                          v119.m128i_u32[0],
                          v119.m128i_i64[1],
                          v27,
                          *(_OWORD *)&v116,
                          *(_OWORD *)&v115);
      *((_QWORD *)&v95 + 1) = v25;
      *(_QWORD *)&v95 = v26;
      v30 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v117, v119.m128i_u32[0], v119.m128i_i64[1], v29, v28, v95);
LABEL_10:
      v31 = (__int64)v30;
      goto LABEL_11;
    }
  }
LABEL_20:
  if ( (unsigned __int8)sub_33DD2A0(
                          a3,
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                          0)
    && (unsigned __int8)sub_33DD2A0(a3, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 0) )
  {
    v63 = *(_QWORD *)(a2 + 40);
    v42 = *(_QWORD *)v63;
    v43 = *(unsigned int *)(v63 + 8);
    v44 = *(_QWORD *)(v63 + 40);
    v45 = *(unsigned int *)(v63 + 48);
  }
  else
  {
    v41 = *(_QWORD *)(a2 + 40);
    v42 = *(_QWORD *)v41;
    v43 = *(unsigned int *)(v41 + 8);
    v44 = *(_QWORD *)(v41 + 40);
    v45 = *(unsigned int *)(v41 + 48);
    if ( v114 != 178 )
    {
      if ( !(unsigned int)sub_33DD890(a3, v42, *(unsigned int *)(v41 + 8), *(_QWORD *)(v41 + 40), v45) )
      {
LABEL_37:
        sub_3406EB0(
          (_QWORD *)a3,
          0x39u,
          (__int64)&v117,
          v119.m128i_u32[0],
          v119.m128i_i64[1],
          v46,
          *(_OWORD *)&v116,
          *(_OWORD *)&v115);
LABEL_38:
        v31 = (__int64)sub_33FAF80(a3, 189, (__int64)&v117, v119.m128i_u32[0], v119.m128i_i64[1], v64, a7);
        goto LABEL_11;
      }
      if ( (unsigned int)sub_33DD890(
                           a3,
                           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                           *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL),
                           **(_QWORD **)(a2 + 40),
                           *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL)) )
      {
        v108 = *(_QWORD *)(a3 + 64);
        v103 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
        v48 = sub_2E79000(*(__int64 **)(a3 + 40));
        v49 = v103(a1, v48, v108, v119.m128i_u32[0], v119.m128i_i64[1]);
        v50 = 10;
        v102 = v49;
        v51 = v49;
        v53 = v52;
        v100 = v52;
        goto LABEL_25;
      }
LABEL_41:
      sub_3406EB0(
        (_QWORD *)a3,
        0x39u,
        (__int64)&v117,
        v119.m128i_u32[0],
        v119.m128i_i64[1],
        v47,
        *(_OWORD *)&v115,
        *(_OWORD *)&v116);
      goto LABEL_38;
    }
  }
  if ( !(unsigned int)sub_33DF620(a3, v42, v43, v44, v45) )
    goto LABEL_37;
  if ( !(unsigned int)sub_33DF620(
                        a3,
                        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                        *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL),
                        **(_QWORD **)(a2 + 40),
                        *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL)) )
    goto LABEL_41;
  v110 = *(_QWORD *)(a3 + 64);
  v106 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v65 = sub_2E79000(*(__int64 **)(a3 + 40));
  v66 = v106(a1, v65, v110, v119.m128i_u32[0], v119.m128i_i64[1]);
  v102 = v66;
  v51 = v66;
  v53 = v67;
  v100 = v67;
  v50 = 8 * (v114 == 178) + 10;
LABEL_25:
  v104 = v53;
  v109 = v51;
  *(_QWORD *)&v54 = sub_33ED040((_QWORD *)a3, v50);
  v105 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v117, v109, v104, v55, *(_OWORD *)&v116, *(_OWORD *)&v115, v54);
  v101 = v59;
  v111 = v59;
  if ( (_WORD)v11 != v102 )
    goto LABEL_42;
  if ( !(_WORD)v11 && v112 != v100 )
  {
    if ( v114 == 178 )
      goto LABEL_61;
    goto LABEL_55;
  }
  a7 = _mm_loadu_si128(&v119);
  v120[0] = a7;
  if ( !v119.m128i_i16[0] )
  {
    v113 = sub_3007030((__int64)v120);
    if ( sub_30070B0((__int64)v120) )
      goto LABEL_71;
    if ( !v113 )
      goto LABEL_31;
LABEL_59:
    v60 = a1[16];
    goto LABEL_32;
  }
  v56 = (unsigned int)v119.m128i_u16[0] - 17;
  if ( (unsigned __int16)(v119.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v119.m128i_i16[0] - 126) > 0x31u )
  {
    if ( (unsigned __int16)(v119.m128i_i16[0] - 17) > 0xD3u )
    {
LABEL_31:
      v60 = a1[15];
      goto LABEL_32;
    }
    goto LABEL_71;
  }
  if ( (unsigned __int16)(v119.m128i_i16[0] - 17) > 0xD3u )
    goto LABEL_59;
LABEL_71:
  v60 = a1[17];
LABEL_32:
  if ( v60 == 2 )
  {
    *(_QWORD *)&v61 = sub_3406EB0(
                        (_QWORD *)a3,
                        0x39u,
                        (__int64)&v117,
                        v119.m128i_u32[0],
                        v119.m128i_i64[1],
                        v58,
                        *(_OWORD *)&v116,
                        *(_OWORD *)&v115);
    v37 = v105;
    v36 = v111;
    *((_QWORD *)&v96 + 1) = v111;
    *(_QWORD *)&v96 = v105;
    *(_QWORD *)&v39 = sub_3406EB0(
                        (_QWORD *)a3,
                        0xBCu,
                        (__int64)&v117,
                        v119.m128i_u32[0],
                        v119.m128i_i64[1],
                        v62,
                        v61,
                        v96);
    goto LABEL_17;
  }
LABEL_42:
  if ( v114 == 178 )
  {
    if ( !(_WORD)v11 )
      goto LABEL_61;
  }
  else
  {
    if ( !(_WORD)v11 )
    {
LABEL_55:
      if ( sub_30070A0((__int64)&v119) )
        goto LABEL_56;
LABEL_61:
      if ( sub_30070B0((__int64)&v119) )
        goto LABEL_62;
LABEL_46:
      v68 = sub_3406EB0(
              (_QWORD *)a3,
              0x39u,
              (__int64)&v117,
              v119.m128i_u32[0],
              v119.m128i_i64[1],
              v58,
              *(_OWORD *)&v115,
              *(_OWORD *)&v116);
      v70 = v69;
      v71 = v68;
      v73 = sub_3406EB0(
              (_QWORD *)a3,
              0x39u,
              (__int64)&v117,
              v119.m128i_u32[0],
              v119.m128i_i64[1],
              v72,
              *(_OWORD *)&v116,
              *(_OWORD *)&v115);
      v74 = v119;
      v75 = (__int64)v73;
      v77 = v76;
      v78 = *(_QWORD *)(v105 + 48) + 16LL * v101;
      v79 = *(_WORD *)v78;
      v80 = *(_QWORD *)(v78 + 8);
      LOWORD(v120[0]) = v79;
      *((_QWORD *)&v120[0] + 1) = v80;
      if ( v79 )
      {
        v81 = (unsigned __int16)(v79 - 17) <= 0xD3u;
      }
      else
      {
        v115.m128i_i64[0] = v119.m128i_i64[0];
        v116.m128i_i64[0] = v75;
        v116.m128i_i64[1] = v77;
        v81 = sub_30070B0((__int64)v120);
        v74.m128i_i32[0] = v115.m128i_i32[0];
        v77 = v116.m128i_i64[1];
        v75 = v116.m128i_i64[0];
      }
      *((_QWORD *)&v97 + 1) = v70;
      *(_QWORD *)&v97 = v71;
      *((_QWORD *)&v94 + 1) = v77;
      *(_QWORD *)&v94 = v75;
      v31 = sub_340EC60(
              (_QWORD *)a3,
              205 - ((unsigned int)!v81 - 1),
              (__int64)&v117,
              v74.m128i_u32[0],
              v74.m128i_i64[1],
              0,
              v105,
              v111,
              v94,
              v97);
      goto LABEL_11;
    }
    if ( (unsigned __int16)(v11 - 2) <= 7u )
    {
      if ( !*(_QWORD *)&a1[2 * v11 + 28] )
      {
LABEL_56:
        v82 = _mm_load_si128(&v115);
        v120[0] = _mm_load_si128(&v116);
        v120[1] = v82;
        v83 = (unsigned int *)sub_33E5110((__int64 *)a3, v119.m128i_u32[0], v119.m128i_i64[1], 2, 0);
        *((_QWORD *)&v98 + 1) = 2;
        *(_QWORD *)&v98 = v120;
        v86 = sub_3411630((_QWORD *)a3, 79, (__int64)&v117, v83, v84, v85, v98);
        *(_QWORD *)&v87 = sub_33FAF80(a3, 213, (__int64)&v117, v119.m128i_u32[0], v119.m128i_i64[1], 0, a7);
        v88 = *((_QWORD *)&v87 + 1);
        v89 = v87;
        *(_QWORD *)&v90 = sub_3406EB0(
                            (_QWORD *)a3,
                            0xBCu,
                            (__int64)&v117,
                            v119.m128i_u32[0],
                            v119.m128i_i64[1],
                            0xFFFFFFFF00000000LL,
                            (unsigned __int64)v86,
                            v87);
        *((_QWORD *)&v99 + 1) = v88;
        *(_QWORD *)&v99 = v89;
        v30 = sub_3406EB0((_QWORD *)a3, 0x39u, (__int64)&v117, v119.m128i_u32[0], v119.m128i_i64[1], v91, v90, v99);
        goto LABEL_10;
      }
      goto LABEL_46;
    }
  }
  if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    goto LABEL_46;
LABEL_62:
  v92 = 1;
  if ( v119.m128i_i16[0] == 1
    || v119.m128i_i16[0] && (v92 = v119.m128i_u16[0], *(_QWORD *)&a1[2 * v119.m128i_u16[0] + 28]) )
  {
    if ( (a1[125 * v92 + 1655] & 0xFB) == 0 )
      goto LABEL_46;
  }
  v31 = (__int64)sub_3412A00((_QWORD *)a3, a2, 0, v56, v57, v58, a7);
LABEL_11:
  if ( v117 )
    sub_B91220((__int64)&v117, v117);
  return v31;
}
