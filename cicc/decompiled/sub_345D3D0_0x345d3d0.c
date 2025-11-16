// Function: sub_345D3D0
// Address: 0x345d3d0
//
unsigned __int8 *__fastcall sub_345D3D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int16 *v6; // rax
  unsigned __int16 v7; // r8
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // ebx
  __int128 v11; // xmm0
  unsigned __int8 *v12; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int16 v16; // r8
  __int128 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  __int128 v20; // rax
  __int64 v21; // r9
  unsigned int v22; // edx
  __int64 v23; // r9
  unsigned __int8 *v24; // rax
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned int v27; // edx
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned int v30; // edx
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned int v33; // edx
  __int128 v34; // rax
  __int64 v35; // r9
  unsigned int v36; // edx
  __int128 v37; // rax
  __int64 v38; // r9
  unsigned int v39; // edx
  __int128 v40; // rax
  __int64 v41; // r9
  unsigned int v42; // edx
  __int128 v43; // rax
  __int64 v44; // r9
  unsigned int v45; // edx
  __int128 v46; // rax
  __int64 v47; // r9
  unsigned int v48; // edx
  __int128 v49; // rax
  __int64 v50; // r9
  unsigned int v51; // edx
  __int128 v52; // rax
  __int64 v53; // r9
  unsigned int v54; // edx
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned int v57; // edx
  __int128 v58; // rax
  __int64 v59; // r9
  unsigned int v60; // edx
  __int128 v61; // rax
  __int64 v62; // r9
  unsigned int v63; // edx
  __int128 v64; // rax
  __int64 v65; // r9
  unsigned int v66; // edx
  __int64 v67; // r9
  unsigned int v68; // edx
  __int64 v69; // r9
  unsigned int v70; // edx
  __int64 v71; // r9
  unsigned int v72; // edx
  __int64 v73; // r9
  unsigned int v74; // edx
  __int64 v75; // r9
  unsigned int v76; // edx
  __int64 v77; // r9
  unsigned int v78; // edx
  __int64 v79; // r9
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
  __int64 v90; // r9
  unsigned int v91; // edx
  __int128 v92; // rax
  __int64 v93; // r9
  unsigned int v94; // edx
  __int128 v95; // rax
  __int64 v96; // r9
  unsigned int v97; // edx
  __int64 v98; // r9
  unsigned int v99; // edx
  __int64 v100; // r9
  unsigned int v101; // edx
  __int64 v102; // r9
  __int128 v103; // [rsp-50h] [rbp-330h]
  __int128 v104; // [rsp-30h] [rbp-310h]
  __int128 v105; // [rsp-30h] [rbp-310h]
  __int128 v106; // [rsp-30h] [rbp-310h]
  __int128 v107; // [rsp-30h] [rbp-310h]
  unsigned __int16 v108; // [rsp+0h] [rbp-2E0h]
  __int64 v109; // [rsp+8h] [rbp-2D8h]
  unsigned __int64 v110; // [rsp+8h] [rbp-2D8h]
  __int128 v111; // [rsp+10h] [rbp-2D0h]
  unsigned int v112; // [rsp+20h] [rbp-2C0h]
  __int64 v113; // [rsp+28h] [rbp-2B8h]
  __int128 v114; // [rsp+30h] [rbp-2B0h]
  __int128 v115; // [rsp+40h] [rbp-2A0h]
  __int128 v116; // [rsp+50h] [rbp-290h]
  __int128 v117; // [rsp+50h] [rbp-290h]
  __int128 v118; // [rsp+50h] [rbp-290h]
  __int128 v119; // [rsp+60h] [rbp-280h]
  __int128 v120; // [rsp+60h] [rbp-280h]
  __int128 v121; // [rsp+80h] [rbp-260h]
  __int128 v122; // [rsp+80h] [rbp-260h]
  __int128 v123; // [rsp+90h] [rbp-250h]
  __int128 v124; // [rsp+90h] [rbp-250h]
  __int128 v125; // [rsp+A0h] [rbp-240h]
  __int128 v126; // [rsp+B0h] [rbp-230h]
  unsigned __int8 *v127; // [rsp+C0h] [rbp-220h]
  unsigned __int8 *v128; // [rsp+190h] [rbp-150h]
  unsigned __int8 *v129; // [rsp+1A0h] [rbp-140h]
  unsigned __int8 *v130; // [rsp+200h] [rbp-E0h]
  unsigned __int8 *v131; // [rsp+280h] [rbp-60h]
  __int64 v132; // [rsp+2A0h] [rbp-40h] BYREF
  int v133; // [rsp+2A8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v132 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v132, v5, 1);
  v133 = *(_DWORD *)(a2 + 72);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = v7;
  v11 = (__int128)_mm_loadu_si128((const __m128i *)v9);
  v126 = (__int128)_mm_loadu_si128((const __m128i *)(v9 + 40));
  v125 = (__int128)_mm_loadu_si128((const __m128i *)(v9 + 80));
  if ( !v7 )
    goto LABEL_4;
  v108 = v7;
  v14 = sub_2E79000(*(__int64 **)(a3 + 40));
  v112 = sub_2FE6750(a1, v10, v8, v14);
  v16 = v108;
  v113 = v15;
  if ( (unsigned __int16)(v108 - 17) <= 0xD3u )
    v16 = word_4456580[v108 - 1];
  switch ( v16 )
  {
    case 7u:
      *(_QWORD *)&v80 = sub_3400BD0(a3, 24, (__int64)&v132, v112, v15, 0, (__m128i)v11, 0);
      *(_QWORD *)&v122 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v81, v11, v80, v126, v125);
      *((_QWORD *)&v122 + 1) = v82;
      *(_QWORD *)&v83 = sub_3400BD0(a3, 65280, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v120 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v84, v11, v83, v126, v125);
      *((_QWORD *)&v120 + 1) = v85;
      *(_QWORD *)&v86 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v120 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v87, v120, v86, v126, v125);
      *((_QWORD *)&v120 + 1) = v88 | *((_QWORD *)&v120 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v89 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v124 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v90, v11, v89, v126, v125);
      *((_QWORD *)&v124 + 1) = v91;
      *(_QWORD *)&v92 = sub_3400BD0(a3, 65280, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v124 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v93, v124, v92, v126, v125);
      *((_QWORD *)&v124 + 1) = v94 | *((_QWORD *)&v124 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v95 = sub_3400BD0(a3, 24, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v118 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v96, v11, v95, v126, v125);
      *((_QWORD *)&v118 + 1) = v97;
      *(_QWORD *)&v122 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v98, v122, v120, v126, v125);
      *((_QWORD *)&v122 + 1) = v99 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
      v130 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v100, v124, v118, v126, v125);
      *((_QWORD *)&v107 + 1) = v101 | *((_QWORD *)&v124 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v107 = v130;
      v24 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v102, v122, v107, v126, v125);
      break;
    case 8u:
      *(_QWORD *)&v25 = sub_3400BD0(a3, 56, (__int64)&v132, v112, v15, 0, (__m128i)v11, 0);
      *(_QWORD *)&v114 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v26, v11, v25, v126, v125);
      *((_QWORD *)&v114 + 1) = v27;
      *(_QWORD *)&v28 = sub_3400BD0(a3, 65280, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v111 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v29, v11, v28, v126, v125);
      *((_QWORD *)&v111 + 1) = v30;
      *(_QWORD *)&v31 = sub_3400BD0(a3, 40, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v111 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v32, v111, v31, v126, v125);
      *((_QWORD *)&v111 + 1) = v33 | *((_QWORD *)&v111 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v34 = sub_3400BD0(a3, (__int64)&loc_FF0000, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v115 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v35, v11, v34, v126, v125);
      *((_QWORD *)&v115 + 1) = v36;
      *(_QWORD *)&v37 = sub_3400BD0(a3, 24, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v115 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v38, v115, v37, v126, v125);
      *((_QWORD *)&v115 + 1) = v39 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v40 = sub_3400BD0(a3, 4278190080LL, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      v129 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v41, v11, v40, v126, v125);
      v109 = v42;
      *(_QWORD *)&v43 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *((_QWORD *)&v103 + 1) = v109;
      *(_QWORD *)&v103 = v129;
      v128 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v44, v103, v43, v126, v125);
      v110 = v45 | v109 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v46 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v121 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v47, v11, v46, v126, v125);
      *((_QWORD *)&v121 + 1) = v48;
      *(_QWORD *)&v49 = sub_3400BD0(a3, 4278190080LL, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v121 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v50, v121, v49, v126, v125);
      *((_QWORD *)&v121 + 1) = v51 | *((_QWORD *)&v121 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v52 = sub_3400BD0(a3, 24, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v119 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v53, v11, v52, v126, v125);
      *((_QWORD *)&v119 + 1) = v54;
      *(_QWORD *)&v55 = sub_3400BD0(a3, (__int64)&loc_FF0000, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v119 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v56, v119, v55, v126, v125);
      *((_QWORD *)&v119 + 1) = v57 | *((_QWORD *)&v119 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v58 = sub_3400BD0(a3, 40, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v123 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v59, v11, v58, v126, v125);
      *((_QWORD *)&v123 + 1) = v60;
      *(_QWORD *)&v61 = sub_3400BD0(a3, 65280, (__int64)&v132, v10, v8, 0, (__m128i)v11, 0);
      *(_QWORD *)&v123 = sub_33FC130((_QWORD *)a3, 396, (__int64)&v132, v10, v8, v62, v123, v61, v126, v125);
      *((_QWORD *)&v123 + 1) = v63 | *((_QWORD *)&v123 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v64 = sub_3400BD0(a3, 56, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      *(_QWORD *)&v117 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v65, v11, v64, v126, v125);
      *((_QWORD *)&v117 + 1) = v66;
      *(_QWORD *)&v114 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v67, v114, v111, v126, v125);
      *((_QWORD *)&v105 + 1) = v110;
      *(_QWORD *)&v105 = v128;
      *((_QWORD *)&v114 + 1) = v68 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v115 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v69, v115, v105, v126, v125);
      *((_QWORD *)&v115 + 1) = v70 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v121 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v71, v121, v119, v126, v125);
      *((_QWORD *)&v121 + 1) = v72 | *((_QWORD *)&v121 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v123 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v73, v123, v117, v126, v125);
      *((_QWORD *)&v123 + 1) = v74 | *((_QWORD *)&v123 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v114 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v75, v114, v115, v126, v125);
      *((_QWORD *)&v114 + 1) = v76 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
      v127 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v77, v121, v123, v126, v125);
      *((_QWORD *)&v106 + 1) = v78 | *((_QWORD *)&v121 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v106 = v127;
      v24 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v79, v114, v106, v126, v125);
      break;
    case 6u:
      *(_QWORD *)&v17 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v15, 0, (__m128i)v11, 0);
      *(_QWORD *)&v116 = sub_33FC130((_QWORD *)a3, 402, (__int64)&v132, v10, v8, v18, v11, v17, v126, v125);
      *((_QWORD *)&v116 + 1) = v19;
      *(_QWORD *)&v20 = sub_3400BD0(a3, 8, (__int64)&v132, v112, v113, 0, (__m128i)v11, 0);
      v131 = sub_33FC130((_QWORD *)a3, 398, (__int64)&v132, v10, v8, v21, v11, v20, v126, v125);
      *((_QWORD *)&v104 + 1) = v22;
      *(_QWORD *)&v104 = v131;
      v24 = sub_33FC130((_QWORD *)a3, 400, (__int64)&v132, v10, v8, v23, v116, v104, v126, v125);
      break;
    default:
LABEL_4:
      v12 = 0;
      goto LABEL_5;
  }
  v12 = v24;
LABEL_5:
  if ( v132 )
    sub_B91220((__int64)&v132, v132);
  return v12;
}
