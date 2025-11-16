// Function: sub_345C4B0
// Address: 0x345c4b0
//
unsigned __int8 *__fastcall sub_345C4B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  unsigned __int16 *v6; // rax
  int v7; // r15d
  unsigned int v8; // ebx
  __int128 v9; // xmm0
  unsigned __int8 *v10; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  __int128 v15; // rax
  __int64 v16; // r9
  __int128 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  __int128 v20; // rax
  __int64 v21; // r9
  unsigned int v22; // edx
  __int128 v23; // rax
  __int64 v24; // r9
  unsigned int v25; // edx
  __int128 v26; // rax
  __int64 v27; // r9
  unsigned int v28; // edx
  __int128 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // edx
  __int128 v32; // rax
  __int64 v33; // r9
  unsigned int v34; // edx
  __int128 v35; // rax
  __int64 v36; // r9
  unsigned int v37; // edx
  __int128 v38; // rax
  __int64 v39; // r9
  unsigned int v40; // edx
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned int v43; // edx
  __int128 v44; // rax
  __int64 v45; // r9
  unsigned int v46; // edx
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  __int128 v50; // rax
  __int64 v51; // r9
  unsigned int v52; // edx
  __int128 v53; // rax
  __int64 v54; // r9
  unsigned int v55; // edx
  __int128 v56; // rax
  __int64 v57; // r14
  __int64 v58; // r9
  unsigned int v59; // edx
  __int64 v60; // r9
  unsigned int v61; // edx
  __int64 v62; // r9
  unsigned int v63; // edx
  __int64 v64; // r9
  unsigned int v65; // edx
  __int64 v66; // r9
  unsigned int v67; // edx
  __int64 v68; // r9
  unsigned int v69; // edx
  __int64 v70; // r9
  unsigned int v71; // edx
  __int64 v72; // r9
  unsigned __int8 *v73; // rax
  __int128 v74; // rax
  __int64 v75; // r9
  unsigned int v76; // edx
  __int128 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  unsigned int v82; // edx
  __int128 v83; // rax
  __int64 v84; // r9
  unsigned int v85; // edx
  __int128 v86; // rax
  __int64 v87; // r9
  unsigned int v88; // edx
  __int128 v89; // rax
  __int64 v90; // r14
  __int64 v91; // r9
  unsigned int v92; // edx
  __int64 v93; // r9
  unsigned int v94; // edx
  __int64 v95; // r9
  unsigned int v96; // edx
  __int64 v97; // r9
  __int128 v98; // [rsp-10h] [rbp-2C0h]
  __int128 v99; // [rsp-10h] [rbp-2C0h]
  __int128 v100; // [rsp+10h] [rbp-2A0h]
  __int128 v101; // [rsp+20h] [rbp-290h]
  unsigned int v102; // [rsp+38h] [rbp-278h]
  __int128 v103; // [rsp+40h] [rbp-270h]
  __int128 v104; // [rsp+50h] [rbp-260h]
  __int128 v105; // [rsp+60h] [rbp-250h]
  __int128 v106; // [rsp+60h] [rbp-250h]
  __int128 v107; // [rsp+80h] [rbp-230h]
  __int128 v108; // [rsp+80h] [rbp-230h]
  __int128 v109; // [rsp+90h] [rbp-220h]
  __int128 v110; // [rsp+90h] [rbp-220h]
  __int64 v111; // [rsp+A0h] [rbp-210h]
  __int128 v112; // [rsp+A0h] [rbp-210h]
  __int128 v113; // [rsp+A0h] [rbp-210h]
  unsigned __int8 *v114; // [rsp+B0h] [rbp-200h]
  unsigned __int8 *v115; // [rsp+1F0h] [rbp-C0h]
  __int64 v116; // [rsp+270h] [rbp-40h] BYREF
  int v117; // [rsp+278h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v116 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v116, v5, 1);
  v117 = *(_DWORD *)(a2 + 72);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v111 = *((_QWORD *)v6 + 1);
  v8 = (unsigned __int16)v7;
  v9 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( !(_WORD)v7 )
    goto LABEL_4;
  v12 = sub_2E79000(*(__int64 **)(a3 + 40));
  v102 = sub_2FE6750(a1, (unsigned __int16)v7, v111, v12);
  v14 = v13;
  if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
    LOWORD(v7) = word_4456580[v7 - 1];
  if ( (_WORD)v7 == 7 )
  {
    *(_QWORD *)&v74 = sub_3400BD0(a3, 24, (__int64)&v116, v102, v13, 0, (__m128i)v9, 0);
    *(_QWORD *)&v108 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v75, v9, v74);
    *((_QWORD *)&v108 + 1) = v76;
    *(_QWORD *)&v77 = sub_3400BD0(a3, 65280, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v106 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v78, v9, v77);
    *((_QWORD *)&v106 + 1) = v79;
    *(_QWORD *)&v80 = sub_3400BD0(a3, 8, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v106 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v81, v106, v80);
    *((_QWORD *)&v106 + 1) = v82 | *((_QWORD *)&v106 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v83 = sub_3400BD0(a3, 8, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v110 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v84, v9, v83);
    *((_QWORD *)&v110 + 1) = v85;
    *(_QWORD *)&v86 = sub_3400BD0(a3, 65280, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v110 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v87, v110, v86);
    *((_QWORD *)&v110 + 1) = v88 | *((_QWORD *)&v110 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v89 = sub_3400BD0(a3, 24, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    v90 = v111;
    *(_QWORD *)&v113 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v91, v9, v89);
    *((_QWORD *)&v113 + 1) = v92;
    *(_QWORD *)&v108 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v90, v93, v108, v106);
    *((_QWORD *)&v108 + 1) = v94 | *((_QWORD *)&v108 + 1) & 0xFFFFFFFF00000000LL;
    v115 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v90, v95, v110, v113);
    *((_QWORD *)&v99 + 1) = v96 | *((_QWORD *)&v110 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v99 = v115;
    v73 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v90, v97, v108, v99);
    goto LABEL_15;
  }
  if ( (_WORD)v7 == 8 )
  {
    *(_QWORD *)&v17 = sub_3400BD0(a3, 56, (__int64)&v116, v102, v13, 0, (__m128i)v9, 0);
    *(_QWORD *)&v103 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v18, v9, v17);
    *((_QWORD *)&v103 + 1) = v19;
    *(_QWORD *)&v20 = sub_3400BD0(a3, 65280, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v100 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v21, v9, v20);
    *((_QWORD *)&v100 + 1) = v22;
    *(_QWORD *)&v23 = sub_3400BD0(a3, 40, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v100 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v24, v100, v23);
    *((_QWORD *)&v100 + 1) = v25 | *((_QWORD *)&v100 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v26 = sub_3400BD0(a3, (__int64)&loc_FF0000, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v104 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v27, v9, v26);
    *((_QWORD *)&v104 + 1) = v28;
    *(_QWORD *)&v29 = sub_3400BD0(a3, 24, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v104 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v30, v104, v29);
    *((_QWORD *)&v104 + 1) = v31 | *((_QWORD *)&v104 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v32 = sub_3400BD0(a3, 4278190080LL, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v101 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v33, v9, v32);
    *((_QWORD *)&v101 + 1) = v34;
    *(_QWORD *)&v35 = sub_3400BD0(a3, 8, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v101 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v116, v8, v111, v36, v101, v35);
    *((_QWORD *)&v101 + 1) = v37 | *((_QWORD *)&v101 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v38 = sub_3400BD0(a3, 8, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v107 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v39, v9, v38);
    *((_QWORD *)&v107 + 1) = v40;
    *(_QWORD *)&v41 = sub_3400BD0(a3, 4278190080LL, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v107 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v42, v107, v41);
    *((_QWORD *)&v107 + 1) = v43 | *((_QWORD *)&v107 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v44 = sub_3400BD0(a3, 24, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v105 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v45, v9, v44);
    *((_QWORD *)&v105 + 1) = v46;
    *(_QWORD *)&v47 = sub_3400BD0(a3, (__int64)&loc_FF0000, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v105 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v48, v105, v47);
    *((_QWORD *)&v105 + 1) = v49 | *((_QWORD *)&v105 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v50 = sub_3400BD0(a3, 40, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    *(_QWORD *)&v109 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v51, v9, v50);
    *((_QWORD *)&v109 + 1) = v52;
    *(_QWORD *)&v53 = sub_3400BD0(a3, 65280, (__int64)&v116, v8, v111, 0, (__m128i)v9, 0);
    *(_QWORD *)&v109 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v116, v8, v111, v54, v109, v53);
    *((_QWORD *)&v109 + 1) = v55 | *((_QWORD *)&v109 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v56 = sub_3400BD0(a3, 56, (__int64)&v116, v102, v14, 0, (__m128i)v9, 0);
    v57 = v111;
    *(_QWORD *)&v112 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v116, v8, v111, v58, v9, v56);
    *((_QWORD *)&v112 + 1) = v59;
    *(_QWORD *)&v103 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v60, v103, v100);
    *((_QWORD *)&v103 + 1) = v61 | *((_QWORD *)&v103 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v104 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v62, v104, v101);
    *((_QWORD *)&v104 + 1) = v63 | *((_QWORD *)&v104 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v107 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v64, v107, v105);
    *((_QWORD *)&v107 + 1) = v65 | *((_QWORD *)&v107 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v109 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v66, v109, v112);
    *((_QWORD *)&v109 + 1) = v67 | *((_QWORD *)&v109 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v103 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v68, v103, v104);
    *((_QWORD *)&v103 + 1) = v69 | *((_QWORD *)&v103 + 1) & 0xFFFFFFFF00000000LL;
    v114 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v70, v107, v109);
    *((_QWORD *)&v98 + 1) = v71 | *((_QWORD *)&v107 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v98 = v114;
    v73 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v116, v8, v57, v72, v103, v98);
LABEL_15:
    v10 = v73;
    goto LABEL_5;
  }
  if ( (_WORD)v7 != 6 )
  {
LABEL_4:
    v10 = 0;
    goto LABEL_5;
  }
  *(_QWORD *)&v15 = sub_3400BD0(a3, 8, (__int64)&v116, v102, v13, 0, (__m128i)v9, 0);
  v10 = sub_3406EB0((_QWORD *)a3, 0xC1u, (__int64)&v116, v8, v111, v16, v9, v15);
LABEL_5:
  if ( v116 )
    sub_B91220((__int64)&v116, v116);
  return v10;
}
