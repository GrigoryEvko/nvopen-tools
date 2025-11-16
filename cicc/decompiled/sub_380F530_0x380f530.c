// Function: sub_380F530
// Address: 0x380f530
//
unsigned __int8 *__fastcall sub_380F530(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // eax
  __int64 v11; // r14
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  __int16 v14; // cx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int128 v27; // rax
  __int128 v28; // rax
  __int64 v29; // r9
  __int128 v30; // rax
  __int64 v31; // r9
  unsigned __int8 *v32; // r12
  unsigned int v33; // edx
  __int64 v34; // r13
  __int64 v35; // rcx
  __int64 v36; // rdx
  char v37; // r15
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  int v42; // r9d
  int v43; // r15d
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int8 *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r13
  unsigned __int8 *v52; // r12
  __int128 v53; // rax
  __int64 v54; // r9
  unsigned __int8 *v55; // rax
  _QWORD *v56; // r15
  __int64 v57; // rdx
  __int64 v58; // r13
  unsigned __int8 *v59; // r12
  __int128 v60; // rax
  __int64 v61; // r9
  unsigned int v62; // edx
  __int64 v63; // r9
  unsigned int v64; // edx
  unsigned __int8 *v65; // r12
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int128 v70; // rax
  __int64 v71; // r9
  unsigned int v72; // edx
  int v73; // r9d
  unsigned int v74; // edx
  __int64 v75; // rax
  unsigned __int8 *v76; // rax
  __int64 v77; // r12
  unsigned int v78; // edx
  __int64 v79; // r13
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int128 v83; // rax
  __int64 v84; // r9
  unsigned int v85; // edx
  __int128 v86; // [rsp-20h] [rbp-140h]
  __int128 v87; // [rsp-10h] [rbp-130h]
  __int128 v88; // [rsp-10h] [rbp-130h]
  __int128 v89; // [rsp-10h] [rbp-130h]
  __int128 v90; // [rsp+0h] [rbp-120h]
  __int128 v91; // [rsp+0h] [rbp-120h]
  __int64 v92; // [rsp+10h] [rbp-110h]
  int v93; // [rsp+18h] [rbp-108h]
  unsigned int v94; // [rsp+20h] [rbp-100h]
  __int64 v95; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v96; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v97; // [rsp+28h] [rbp-F8h]
  __int64 v98; // [rsp+28h] [rbp-F8h]
  __int64 v99; // [rsp+28h] [rbp-F8h]
  __int128 v100; // [rsp+30h] [rbp-F0h]
  unsigned int v101; // [rsp+40h] [rbp-E0h]
  int v102; // [rsp+40h] [rbp-E0h]
  __int128 v103; // [rsp+40h] [rbp-E0h]
  __int128 v104; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v105; // [rsp+50h] [rbp-D0h]
  __int64 v106; // [rsp+70h] [rbp-B0h] BYREF
  int v107; // [rsp+78h] [rbp-A8h]
  unsigned int v108; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v109; // [rsp+88h] [rbp-98h]
  unsigned int v110; // [rsp+90h] [rbp-90h] BYREF
  __int64 v111; // [rsp+98h] [rbp-88h]
  __int64 v112; // [rsp+A0h] [rbp-80h] BYREF
  char v113; // [rsp+A8h] [rbp-78h]
  __int64 v114; // [rsp+B0h] [rbp-70h]
  __int64 v115; // [rsp+B8h] [rbp-68h]
  __int64 v116; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v117; // [rsp+C8h] [rbp-58h]
  __int64 v118; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v119; // [rsp+D8h] [rbp-48h]
  __int64 v120; // [rsp+E0h] [rbp-40h]
  __int64 v121; // [rsp+E8h] [rbp-38h]

  *(_QWORD *)&v100 = sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v100 + 1) = v4;
  v101 = v4;
  v5 = sub_375A6A0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a3);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v5;
  v9 = v8;
  v106 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v106, v6, 1);
  v10 = *(_DWORD *)(a2 + 72);
  v11 = *(_QWORD *)(v100 + 48) + 16LL * v101;
  v107 = v10;
  v12 = *(_WORD *)v11;
  v109 = *(_QWORD *)(v11 + 8);
  v13 = *((_QWORD *)v7 + 6) + 16LL * (unsigned int)v9;
  LOWORD(v108) = v12;
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  LOWORD(v110) = v14;
  v111 = v15;
  if ( v12 )
  {
    if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      goto LABEL_32;
    v75 = 16LL * (v12 - 1);
    v17 = *(_QWORD *)&byte_444C4A0[v75];
    v18 = byte_444C4A0[v75 + 8];
  }
  else
  {
    v120 = sub_3007260((__int64)&v108);
    v121 = v16;
    v17 = v120;
    v18 = v121;
  }
  v118 = v17;
  LOBYTE(v119) = v18;
  v93 = sub_CA1930(&v118);
  if ( (_WORD)v110 )
  {
    if ( (_WORD)v110 == 1 || (unsigned __int16)(v110 - 504) <= 7u )
      goto LABEL_32;
    v20 = 16LL * ((unsigned __int16)v110 - 1);
    v19 = *(_QWORD *)&byte_444C4A0[v20];
    LOBYTE(v20) = byte_444C4A0[v20 + 8];
  }
  else
  {
    v19 = sub_3007260((__int64)&v110);
    v118 = v19;
    v119 = v20;
  }
  LOBYTE(v117) = v20;
  v116 = v19;
  v21 = sub_CA1930(&v116);
  v22 = a1[1];
  v23 = *a1;
  v102 = v21;
  v24 = sub_2E79000(*(__int64 **)(v22 + 40));
  v25 = sub_2FE6750(v23, v110, v111, v24);
  *(_QWORD *)&v27 = sub_3400BD0(v22, (unsigned int)(v102 - 1), (__int64)&v106, v25, v26, 0, a3, 0);
  v103 = v27;
  *(_QWORD *)&v28 = sub_3400BD0(a1[1], 1, (__int64)&v106, v110, v111, 0, a3, 0);
  *(_QWORD *)&v30 = sub_3406EB0((_QWORD *)v22, 0xBEu, (__int64)&v106, v110, v111, v29, v28, v103);
  *((_QWORD *)&v87 + 1) = v9;
  *(_QWORD *)&v87 = v7;
  *((_QWORD *)&v103 + 1) = *((_QWORD *)&v30 + 1);
  v97 = sub_3406EB0((_QWORD *)a1[1], 0xBAu, (__int64)&v106, v110, v111, v31, v87, v30);
  v32 = v97;
  v34 = v33;
  *(_QWORD *)&v104 = v97;
  v94 = v33;
  *((_QWORD *)&v104 + 1) = v33 | *((_QWORD *)&v103 + 1) & 0xFFFFFFFF00000000LL;
  if ( (_WORD)v108 )
  {
    if ( (_WORD)v108 == 1 || (unsigned __int16)(v108 - 504) <= 7u )
      goto LABEL_32;
    v35 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v108 - 16];
    v37 = byte_444C4A0[16 * (unsigned __int16)v108 - 8];
  }
  else
  {
    v116 = sub_3007260((__int64)&v108);
    v35 = v116;
    v117 = v36;
    v37 = v36;
  }
  if ( !(_WORD)v110 )
  {
    v92 = v35;
    v38 = sub_3007260((__int64)&v110);
    v35 = v92;
    v114 = v38;
    v115 = v39;
    goto LABEL_11;
  }
  if ( (_WORD)v110 == 1 || (unsigned __int16)(v110 - 504) <= 7u )
LABEL_32:
    BUG();
  v38 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v110 - 16];
  LOBYTE(v39) = byte_444C4A0[16 * (unsigned __int16)v110 - 8];
LABEL_11:
  v40 = v38 - v35;
  if ( v35 )
    LOBYTE(v39) = v37;
  v112 = v40;
  v113 = v39;
  v41 = sub_CA1930(&v112);
  v43 = v41;
  if ( v41 > 0 )
  {
    v98 = a1[1];
    v95 = *a1;
    v67 = sub_2E79000(*(__int64 **)(v98 + 40));
    v68 = sub_2FE6750(
            v95,
            *(unsigned __int16 *)(*((_QWORD *)v32 + 6) + 16 * v34),
            *(_QWORD *)(*((_QWORD *)v32 + 6) + 16 * v34 + 8),
            v67);
    *(_QWORD *)&v70 = sub_3400BD0(v98, v43, (__int64)&v106, v68, v69, 0, a3, 0);
    sub_3406EB0((_QWORD *)v98, 0xC0u, (__int64)&v106, v110, v111, v71, v104, v70);
    *((_QWORD *)&v104 + 1) = v72 | *((_QWORD *)&v104 + 1) & 0xFFFFFFFF00000000LL;
    v97 = sub_33FAF80(a1[1], 216, (__int64)&v106, v108, v109, v73, a3);
    v94 = v74;
  }
  else if ( v41 )
  {
    v76 = sub_33FAF80(a1[1], 215, (__int64)&v106, v108, v109, v42, a3);
    v77 = a1[1];
    *(_QWORD *)&v104 = v76;
    v96 = v76;
    v99 = *a1;
    v79 = 16LL * v78;
    *((_QWORD *)&v104 + 1) = v78 | *((_QWORD *)&v104 + 1) & 0xFFFFFFFF00000000LL;
    v80 = sub_2E79000(*(__int64 **)(v77 + 40));
    v81 = sub_2FE6750(
            v99,
            *(unsigned __int16 *)(*((_QWORD *)v96 + 6) + v79),
            *(_QWORD *)(*((_QWORD *)v96 + 6) + v79 + 8),
            v80);
    *(_QWORD *)&v83 = sub_3400BD0(v77, -v43, (__int64)&v106, v81, v82, 0, a3, 0);
    v97 = sub_3406EB0((_QWORD *)v77, 0xBEu, (__int64)&v106, v108, v109, v84, v104, v83);
    v94 = v85;
  }
  v44 = a1[1];
  v45 = *a1;
  v46 = sub_2E79000(*(__int64 **)(v44 + 40));
  v47 = sub_2FE6750(v45, v108, v109, v46);
  v49 = sub_3400BD0(v44, (unsigned int)(v93 - 1), (__int64)&v106, v47, v48, 0, a3, 0);
  v51 = v50;
  v52 = v49;
  *(_QWORD *)&v53 = sub_3400BD0(a1[1], 1, (__int64)&v106, v108, v109, 0, a3, 0);
  *((_QWORD *)&v88 + 1) = v51;
  *(_QWORD *)&v88 = v52;
  v55 = sub_3406EB0((_QWORD *)v44, 0xBEu, (__int64)&v106, v108, v109, v54, v53, v88);
  v56 = (_QWORD *)a1[1];
  v58 = v57;
  v59 = v55;
  *(_QWORD *)&v60 = sub_3400BD0((__int64)v56, 1, (__int64)&v106, v108, v109, 0, a3, 0);
  *((_QWORD *)&v86 + 1) = v58;
  *(_QWORD *)&v86 = v59;
  v105 = sub_3406EB0(v56, 0x39u, (__int64)&v106, v108, v109, v61, v86, v60);
  *((_QWORD *)&v90 + 1) = v62 | v58 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v90 = v105;
  *(_QWORD *)&v100 = sub_3406EB0((_QWORD *)a1[1], 0xBAu, (__int64)&v106, v108, v109, v63, v100, v90);
  *((_QWORD *)&v91 + 1) = v94 | *((_QWORD *)&v104 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v91 = v97;
  *((_QWORD *)&v89 + 1) = v64 | *((_QWORD *)&v100 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v89 = v100;
  v65 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v106, v108, v109, v100, v89, v91);
  if ( v106 )
    sub_B91220((__int64)&v106, v106);
  return v65;
}
