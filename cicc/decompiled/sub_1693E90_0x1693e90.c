// Function: sub_1693E90
// Address: 0x1693e90
//
__int64 *__fastcall sub_1693E90(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __m128i v10; // xmm0
  __m128i v11; // xmm0
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __m128i si128; // xmm0
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __m128i v34; // xmm0
  __m128i v35; // xmm0
  __int64 v36; // rax
  __int64 v37; // rdx
  __m128i *v38; // rax
  __int64 v39; // rdx
  __m128i v40; // xmm0
  __m128i v41; // xmm0
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __m128i v46; // xmm0
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __m128i v51; // xmm0
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  __m128i v56; // xmm0
  __m128i v57; // xmm0
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __m128i v62; // xmm0
  __m128i v63; // xmm0
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  __m128i v68; // xmm0
  __m128i v69; // xmm0
  __m128i v70; // xmm0
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdx
  __m128i v74; // xmm0
  __m128i v75; // xmm0
  __m128i *v76; // rax
  __int64 v77; // rdx
  __m128i v78; // xmm0
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rdx
  __m128i v83; // xmm0
  __m128i v84; // xmm0
  __int64 v85; // rax
  __int64 v86; // rdx
  __m128i *v87; // rax
  __int64 v88; // rdx
  __m128i v89; // xmm0
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // rdx
  __m128i v94; // xmm0
  __int64 v95; // rax
  __int64 v96; // rdx
  _QWORD v97[2]; // [rsp+8h] [rbp-18h] BYREF

  switch ( a2 )
  {
    case 0:
      *((_DWORD *)a1 + 4) = 1667462483;
      *a1 = (__int64)(a1 + 2);
      *((_WORD *)a1 + 10) = 29541;
      *((_BYTE *)a1 + 22) = 115;
      a1[1] = 7;
      *((_BYTE *)a1 + 23) = 0;
      return a1;
    case 1:
      *a1 = (__int64)(a1 + 2);
      strcpy((char *)a1 + 16, "End of File");
      a1[1] = 11;
      return a1;
    case 2:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 52;
      v15 = sub_22409D0(a1, v97, 0);
      v16 = v97[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_42AE5D0);
      *a1 = v15;
      a1[2] = v16;
      *(__m128i *)v15 = si128;
      v18 = _mm_load_si128((const __m128i *)&xmmword_42AE5E0);
      *(_DWORD *)(v15 + 48) = 1952542066;
      *(__m128i *)(v15 + 16) = v18;
      *(__m128i *)(v15 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE5F0);
      v19 = v97[0];
      v20 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v20 + v19) = 0;
      return a1;
    case 3:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 48;
      v21 = sub_22409D0(a1, v97, 0);
      v22 = v97[0];
      v23 = _mm_load_si128((const __m128i *)&xmmword_42AE600);
      *a1 = v21;
      a1[2] = v22;
      *(__m128i *)v21 = v23;
      *(__m128i *)(v21 + 16) = _mm_load_si128((const __m128i *)&xmmword_42AE610);
      *(__m128i *)(v21 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE620);
      v24 = v97[0];
      v25 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v25 + v24) = 0;
      return a1;
    case 4:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 61;
      v26 = sub_22409D0(a1, v97, 0);
      v27 = v97[0];
      v28 = _mm_load_si128((const __m128i *)&xmmword_42AE600);
      *a1 = v26;
      a1[2] = v27;
      *(__m128i *)v26 = v28;
      v29 = _mm_load_si128((const __m128i *)&xmmword_42AE610);
      qmemcpy((void *)(v26 + 48), "r is corrupt", 12);
      *(__m128i *)(v26 + 16) = v29;
      *(__m128i *)(v26 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE630);
      goto LABEL_10;
    case 5:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 50;
      v32 = sub_22409D0(a1, v97, 0);
      v33 = v97[0];
      v34 = _mm_load_si128((const __m128i *)&xmmword_42AE640);
      *a1 = v32;
      a1[2] = v33;
      *(__m128i *)v32 = v34;
      v35 = _mm_load_si128((const __m128i *)&xmmword_42AE650);
      *(_WORD *)(v32 + 48) = 28271;
      *(__m128i *)(v32 + 16) = v35;
      *(__m128i *)(v32 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE660);
      v36 = v97[0];
      v37 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v37 + v36) = 0;
      return a1;
    case 6:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 45;
      v38 = (__m128i *)sub_22409D0(a1, v97, 0);
      v39 = v97[0];
      v40 = _mm_load_si128((const __m128i *)&xmmword_42AE640);
      *a1 = (__int64)v38;
      a1[2] = v39;
      *v38 = v40;
      v41 = _mm_load_si128((const __m128i *)&xmmword_42AE650);
      qmemcpy(&v38[2], "ile hash type", 13);
      v38[1] = v41;
      v42 = v97[0];
      v43 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v43 + v42) = 0;
      return a1;
    case 7:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 21;
      v44 = sub_22409D0(a1, v97, 0);
      v45 = v97[0];
      v46 = _mm_load_si128((const __m128i *)&xmmword_3F648A0);
      *a1 = v44;
      a1[2] = v45;
      *(_DWORD *)(v44 + 16) = 1952539680;
      *(_BYTE *)(v44 + 20) = 97;
      *(__m128i *)v44 = v46;
      v47 = v97[0];
      v48 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v48 + v47) = 0;
      return a1;
    case 8:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 22;
      v49 = sub_22409D0(a1, v97, 0);
      v50 = v97[0];
      v51 = _mm_load_si128((const __m128i *)&xmmword_3F648B0);
      *a1 = v49;
      a1[2] = v50;
      *(_DWORD *)(v49 + 16) = 1633951845;
      *(_WORD *)(v49 + 20) = 24948;
      *(__m128i *)v49 = v51;
      v52 = v97[0];
      v53 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v53 + v52) = 0;
      return a1;
    case 9:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 38;
      v54 = sub_22409D0(a1, v97, 0);
      v55 = v97[0];
      v56 = _mm_load_si128((const __m128i *)&xmmword_42AE670);
      *a1 = v54;
      a1[2] = v55;
      *(__m128i *)v54 = v56;
      v57 = _mm_load_si128((const __m128i *)&xmmword_42AE680);
      *(_DWORD *)(v54 + 32) = 1633951845;
      *(_WORD *)(v54 + 36) = 24948;
      *(__m128i *)(v54 + 16) = v57;
      v58 = v97[0];
      v59 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v59 + v58) = 0;
      return a1;
    case 10:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 38;
      v60 = sub_22409D0(a1, v97, 0);
      v61 = v97[0];
      v62 = _mm_load_si128((const __m128i *)&xmmword_42AE690);
      *a1 = v60;
      a1[2] = v61;
      *(__m128i *)v60 = v62;
      v63 = _mm_load_si128((const __m128i *)&xmmword_42AE6A0);
      *(_DWORD *)(v60 + 32) = 1769235310;
      *(_WORD *)(v60 + 36) = 28271;
      *(__m128i *)(v60 + 16) = v63;
      v64 = v97[0];
      v65 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v65 + v64) = 0;
      return a1;
    case 11:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 53;
      v66 = sub_22409D0(a1, v97, 0);
      v67 = v97[0];
      v68 = _mm_load_si128((const __m128i *)&xmmword_42AE6B0);
      *a1 = v66;
      a1[2] = v67;
      *(__m128i *)v66 = v68;
      v69 = _mm_load_si128((const __m128i *)&xmmword_42AE6C0);
      *(_DWORD *)(v66 + 48) = 1751348321;
      *(__m128i *)(v66 + 16) = v69;
      v70 = _mm_load_si128((const __m128i *)&xmmword_42AE6D0);
      *(_BYTE *)(v66 + 52) = 41;
      *(__m128i *)(v66 + 32) = v70;
      v71 = v97[0];
      v72 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v72 + v71) = 0;
      return a1;
    case 12:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 61;
      v26 = sub_22409D0(a1, v97, 0);
      v73 = v97[0];
      v74 = _mm_load_si128((const __m128i *)&xmmword_42AE6E0);
      *a1 = v26;
      a1[2] = v73;
      *(__m128i *)v26 = v74;
      v75 = _mm_load_si128((const __m128i *)&xmmword_42AE6F0);
      qmemcpy((void *)(v26 + 48), "ter mismatch", 12);
      *(__m128i *)(v26 + 16) = v75;
      *(__m128i *)(v26 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE700);
LABEL_10:
      *(_BYTE *)(v26 + 60) = 41;
      v30 = v97[0];
      v31 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v31 + v30) = 0;
      return a1;
    case 13:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 16;
      v76 = (__m128i *)sub_22409D0(a1, v97, 0);
      v77 = v97[0];
      v78 = _mm_load_si128((const __m128i *)&xmmword_3F64940);
      *a1 = (__int64)v76;
      a1[2] = v77;
      *v76 = v78;
      v79 = v97[0];
      v80 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v80 + v79) = 0;
      return a1;
    case 14:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 60;
      v81 = sub_22409D0(a1, v97, 0);
      v82 = v97[0];
      v83 = _mm_load_si128((const __m128i *)&xmmword_42AE710);
      *a1 = v81;
      a1[2] = v82;
      *(__m128i *)v81 = v83;
      v84 = _mm_load_si128((const __m128i *)&xmmword_42AE720);
      qmemcpy((void *)(v81 + 48), "er mismatch)", 12);
      *(__m128i *)(v81 + 16) = v84;
      *(__m128i *)(v81 + 32) = _mm_load_si128((const __m128i *)&xmmword_42AE730);
      v85 = v97[0];
      v86 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v86 + v85) = 0;
      return a1;
    case 15:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 30;
      v87 = (__m128i *)sub_22409D0(a1, v97, 0);
      v88 = v97[0];
      v89 = _mm_load_si128((const __m128i *)&xmmword_42AE740);
      *a1 = (__int64)v87;
      a1[2] = v88;
      qmemcpy(&v87[1], "ss data (zlib)", 14);
      *v87 = v89;
      v90 = v97[0];
      v91 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v91 + v90) = 0;
      return a1;
    case 16:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 32;
      v92 = sub_22409D0(a1, v97, 0);
      v93 = v97[0];
      v94 = _mm_load_si128((const __m128i *)&xmmword_42AE750);
      *a1 = v92;
      a1[2] = v93;
      *(__m128i *)v92 = v94;
      *(__m128i *)(v92 + 16) = _mm_load_si128((const __m128i *)&xmmword_42AE760);
      v95 = v97[0];
      v96 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v96 + v95) = 0;
      return a1;
    case 17:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 22;
      v2 = sub_22409D0(a1, v97, 0);
      v3 = v97[0];
      v4 = _mm_load_si128((const __m128i *)&xmmword_42AE770);
      *a1 = v2;
      a1[2] = v3;
      *(_DWORD *)(v2 + 16) = 1768300645;
      *(_WORD *)(v2 + 20) = 25964;
      *(__m128i *)v2 = v4;
      v5 = v97[0];
      v6 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v6 + v5) = 0;
      return a1;
    case 18:
      *a1 = (__int64)(a1 + 2);
      v97[0] = 83;
      v8 = sub_22409D0(a1, v97, 0);
      v9 = v97[0];
      v10 = _mm_load_si128((const __m128i *)&xmmword_42AE780);
      *a1 = v8;
      a1[2] = v9;
      *(__m128i *)v8 = v10;
      v11 = _mm_load_si128((const __m128i *)&xmmword_42AE790);
      *(_WORD *)(v8 + 80) = 29295;
      *(__m128i *)(v8 + 16) = v11;
      v12 = _mm_load_si128((const __m128i *)&xmmword_42AE7A0);
      *(_BYTE *)(v8 + 82) = 116;
      *(__m128i *)(v8 + 32) = v12;
      *(__m128i *)(v8 + 48) = _mm_load_si128((const __m128i *)&xmmword_42AE7B0);
      *(__m128i *)(v8 + 64) = _mm_load_si128((const __m128i *)&xmmword_42AE7C0);
      v13 = v97[0];
      v14 = *a1;
      a1[1] = v97[0];
      *(_BYTE *)(v14 + v13) = 0;
      return a1;
  }
}
