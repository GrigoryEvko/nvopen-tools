// Function: sub_C1A9F0
// Address: 0xc1a9f0
//
__int64 *__fastcall sub_C1A9F0(__int64 *a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __m128i si128; // xmm0
  __m128i v19; // xmm0
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __m128i v24; // xmm0
  __m128i v25; // xmm0
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __m128i v30; // xmm0
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __m128i v35; // xmm0
  __int64 v36; // rax
  __int64 v37; // rdx
  __m128i *v38; // rax
  __int64 v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rax
  __int64 v42; // rdx
  __m128i *v43; // rax
  __int64 v44; // rdx
  __m128i v45; // xmm0
  __m128i v46; // xmm0
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __m128i v51; // xmm0
  __m128i v52; // xmm0
  __int64 v53; // rax
  __int64 v54; // rdx
  __m128i *v55; // rax
  __int64 v56; // rdx
  __m128i v57; // xmm0
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __m128i v62; // xmm0
  __int64 v63; // rax
  __int64 v64; // rdx
  __m128i *v65; // rax
  __int64 v66; // rdx
  __m128i v67; // xmm0
  __int64 v68; // rax
  __int64 v69; // rdx
  __m128i *v70; // rax
  __int64 v71; // rdx
  __m128i v72; // xmm0
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rdx
  __m128i v77; // xmm0
  __int64 v78; // rax
  __int64 v79; // rdx
  _QWORD v80[4]; // [rsp-20h] [rbp-20h] BYREF

  v80[3] = v3;
  v80[2] = v4;
  switch ( a3 )
  {
    case 0:
      *((_BYTE *)a1 + 22) = 115;
      *a1 = (__int64)(a1 + 2);
      *((_DWORD *)a1 + 4) = 1667462483;
      *((_WORD *)a1 + 10) = 29541;
      a1[1] = 7;
      *((_BYTE *)a1 + 23) = 0;
      break;
    case 1:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 39;
      v16 = sub_22409D0(a1, v80, 0);
      v17 = v80[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F64860);
      *a1 = v16;
      a1[2] = v17;
      *(__m128i *)v16 = si128;
      v19 = _mm_load_si128((const __m128i *)&xmmword_3F64870);
      *(_DWORD *)(v16 + 32) = 1734438176;
      *(_WORD *)(v16 + 36) = 25449;
      *(_BYTE *)(v16 + 38) = 41;
      *(__m128i *)(v16 + 16) = v19;
      v20 = v80[0];
      v21 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v21 + v20) = 0;
      break;
    case 2:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 41;
      v22 = sub_22409D0(a1, v80, 0);
      v23 = v80[0];
      v24 = _mm_load_si128((const __m128i *)&xmmword_3F64880);
      *a1 = v22;
      a1[2] = v23;
      *(__m128i *)v22 = v24;
      v25 = _mm_load_si128((const __m128i *)&xmmword_3F64890);
      *(_QWORD *)(v22 + 32) = 0x6F69737265762074LL;
      *(_BYTE *)(v22 + 40) = 110;
      *(__m128i *)(v22 + 16) = v25;
      v26 = v80[0];
      v27 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v27 + v26) = 0;
      break;
    case 3:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 21;
      v28 = sub_22409D0(a1, v80, 0);
      v29 = v80[0];
      v30 = _mm_load_si128((const __m128i *)&xmmword_3F648A0);
      *a1 = v28;
      a1[2] = v29;
      *(_DWORD *)(v28 + 16) = 1952539680;
      *(_BYTE *)(v28 + 20) = 97;
      *(__m128i *)v28 = v30;
      v31 = v80[0];
      v32 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v32 + v31) = 0;
      break;
    case 4:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 22;
      v33 = sub_22409D0(a1, v80, 0);
      v34 = v80[0];
      v35 = _mm_load_si128((const __m128i *)&xmmword_3F648B0);
      *a1 = v33;
      a1[2] = v34;
      *(_DWORD *)(v33 + 16) = 1633951845;
      *(_WORD *)(v33 + 20) = 24948;
      *(__m128i *)v33 = v35;
      v36 = v80[0];
      v37 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v37 + v36) = 0;
      break;
    case 5:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 29;
      v38 = (__m128i *)sub_22409D0(a1, v80, 0);
      v39 = v80[0];
      v40 = _mm_load_si128((const __m128i *)&xmmword_3F648C0);
      *a1 = (__int64)v38;
      a1[2] = v39;
      qmemcpy(&v38[1], " profile data", 13);
      *v38 = v40;
      v41 = v80[0];
      v42 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v42 + v41) = 0;
      break;
    case 6:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 43;
      v43 = (__m128i *)sub_22409D0(a1, v80, 0);
      v44 = v80[0];
      v45 = _mm_load_si128((const __m128i *)&xmmword_3F648D0);
      *a1 = (__int64)v43;
      a1[2] = v44;
      *v43 = v45;
      v46 = _mm_load_si128((const __m128i *)&xmmword_3F648E0);
      qmemcpy(&v43[2], "ding format", 11);
      v43[1] = v46;
      v47 = v80[0];
      v48 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v48 + v47) = 0;
      break;
    case 7:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 58;
      v49 = sub_22409D0(a1, v80, 0);
      v50 = v80[0];
      v51 = _mm_load_si128((const __m128i *)&xmmword_3F648F0);
      *a1 = v49;
      a1[2] = v50;
      *(__m128i *)v49 = v51;
      v52 = _mm_load_si128((const __m128i *)&xmmword_3F64900);
      qmemcpy((void *)(v49 + 48), "operations", 10);
      *(__m128i *)(v49 + 16) = v52;
      *(__m128i *)(v49 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F64910);
      v53 = v80[0];
      v54 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v54 + v53) = 0;
      break;
    case 8:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 29;
      v55 = (__m128i *)sub_22409D0(a1, v80, 0);
      v56 = v80[0];
      v57 = _mm_load_si128((const __m128i *)&xmmword_3F64920);
      *a1 = (__int64)v55;
      a1[2] = v56;
      qmemcpy(&v55[1], "on name table", 13);
      *v55 = v57;
      v58 = v80[0];
      v59 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v59 + v58) = 0;
      break;
    case 9:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 21;
      v60 = sub_22409D0(a1, v80, 0);
      v61 = v80[0];
      v62 = _mm_load_si128((const __m128i *)&xmmword_3F64930);
      *a1 = v60;
      a1[2] = v61;
      *(_DWORD *)(v60 + 16) = 1920300129;
      *(_BYTE *)(v60 + 20) = 101;
      *(__m128i *)v60 = v62;
      v63 = v80[0];
      v64 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v64 + v63) = 0;
      break;
    case 10:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 16;
      v65 = (__m128i *)sub_22409D0(a1, v80, 0);
      v66 = v80[0];
      v67 = _mm_load_si128((const __m128i *)&xmmword_3F64940);
      *a1 = (__int64)v65;
      a1[2] = v66;
      *v65 = v67;
      v68 = v80[0];
      v69 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v69 + v68) = 0;
      break;
    case 11:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 29;
      v70 = (__m128i *)sub_22409D0(a1, v80, 0);
      v71 = v80[0];
      v72 = _mm_load_si128((const __m128i *)&xmmword_3F64950);
      *a1 = (__int64)v70;
      a1[2] = v71;
      qmemcpy(&v70[1], " support seek", 13);
      *v70 = v72;
      v73 = v80[0];
      v74 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v74 + v73) = 0;
      break;
    case 12:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 18;
      v75 = sub_22409D0(a1, v80, 0);
      v76 = v80[0];
      v77 = _mm_load_si128((const __m128i *)&xmmword_3F64960);
      *a1 = v75;
      a1[2] = v76;
      *(_WORD *)(v75 + 16) = 25970;
      *(__m128i *)v75 = v77;
      v78 = v80[0];
      v79 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v79 + v78) = 0;
      break;
    case 13:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 19;
      v5 = sub_22409D0(a1, v80, 0);
      v6 = v80[0];
      v7 = _mm_load_si128((const __m128i *)&xmmword_3F64970);
      *a1 = v5;
      a1[2] = v6;
      *(_WORD *)(v5 + 16) = 27746;
      *(_BYTE *)(v5 + 18) = 101;
      *(__m128i *)v5 = v7;
      v8 = v80[0];
      v9 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v9 + v8) = 0;
      break;
    case 14:
      *a1 = (__int64)(a1 + 2);
      v80[0] = 22;
      v11 = sub_22409D0(a1, v80, 0);
      v12 = v80[0];
      v13 = _mm_load_si128((const __m128i *)&xmmword_3F64980);
      *a1 = v11;
      a1[2] = v12;
      *(_DWORD *)(v11 + 16) = 1952542067;
      *(_WORD *)(v11 + 20) = 26723;
      *(__m128i *)v11 = v13;
      v14 = v80[0];
      v15 = *a1;
      a1[1] = v80[0];
      *(_BYTE *)(v15 + v14) = 0;
      break;
    default:
      BUG();
  }
  return a1;
}
