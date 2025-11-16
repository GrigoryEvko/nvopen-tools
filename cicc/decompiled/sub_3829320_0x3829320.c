// Function: sub_3829320
// Address: 0x3829320
//
__int64 __fastcall sub_3829320(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  const __m128i *v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rbx
  __m128i v12; // xmm0
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  __int16 v15; // bx
  int v16; // eax
  unsigned __int16 v17; // dx
  __int64 v18; // rcx
  __m128i *v19; // r14
  __int64 v21; // rdx
  __int128 v22; // rax
  bool v23; // zf
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  char v31; // al
  unsigned int v32; // eax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int128 v35; // rax
  unsigned __int64 v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rbx
  unsigned __int8 *v39; // rax
  _QWORD *v40; // rdi
  unsigned int v41; // edx
  __int64 v42; // rax
  unsigned __int8 v43; // cl
  unsigned __int64 v44; // rsi
  __int64 v45; // rbx
  char v46; // r8
  __int64 v47; // r9
  int v48; // edx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // ebx
  __int128 v53; // rax
  unsigned __int64 v54; // rax
  int v55; // ebx
  __int128 v56; // rax
  unsigned int v57; // ebx
  int v58; // eax
  unsigned __int64 v59; // rdx
  __int64 v60; // rdx
  __m128i *v61; // rax
  _QWORD *v62; // rdi
  int v63; // edx
  unsigned int v64; // edx
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 v67; // r9
  unsigned __int64 v68; // rdi
  __int64 v69; // r15
  char v70; // r8
  int v71; // edx
  int v72; // esi
  __int64 v73; // r15
  int v74; // edx
  __int64 v75; // rbx
  __int64 v76; // rax
  int v77; // edx
  unsigned __int16 v78; // ax
  __int128 v79; // rax
  __int64 v80; // rax
  __int128 v81; // rax
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  __int64 v84; // r8
  _QWORD *v85; // r15
  __int64 *v86; // rdi
  int v87; // edx
  __int64 v88; // rax
  int v89; // edx
  unsigned __int16 v90; // ax
  __int128 v91; // rax
  __int64 v92; // r9
  __int128 v93; // rax
  __int64 v94; // r9
  int v95; // edx
  __int64 v96; // rdx
  __int64 v97; // rdx
  int v98; // eax
  __int64 v99; // rdi
  __int64 v100; // [rsp+8h] [rbp-1D8h]
  __int64 (__fastcall *v101)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-1D0h]
  __int64 v102; // [rsp+10h] [rbp-1D0h]
  __int64 v103; // [rsp+18h] [rbp-1C8h]
  __int64 v104; // [rsp+18h] [rbp-1C8h]
  unsigned int v105; // [rsp+20h] [rbp-1C0h]
  unsigned __int8 v106; // [rsp+28h] [rbp-1B8h]
  __int64 (__fastcall *v107)(__int64, __int64, unsigned int); // [rsp+28h] [rbp-1B8h]
  __int64 v108; // [rsp+30h] [rbp-1B0h]
  __int64 v109; // [rsp+30h] [rbp-1B0h]
  __int64 v110; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v111; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v112; // [rsp+38h] [rbp-1A8h]
  __int16 v113; // [rsp+44h] [rbp-19Ch]
  unsigned __int64 v114; // [rsp+48h] [rbp-198h]
  unsigned __int64 v115; // [rsp+50h] [rbp-190h]
  unsigned __int64 v116; // [rsp+60h] [rbp-180h]
  unsigned __int8 *v117; // [rsp+60h] [rbp-180h]
  unsigned __int64 v118; // [rsp+68h] [rbp-178h]
  unsigned __int64 v119; // [rsp+68h] [rbp-178h]
  unsigned int v120; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v121; // [rsp+F8h] [rbp-E8h]
  __int64 v122; // [rsp+100h] [rbp-E0h] BYREF
  int v123; // [rsp+108h] [rbp-D8h]
  __int128 v124; // [rsp+110h] [rbp-D0h] BYREF
  __int128 v125; // [rsp+120h] [rbp-C0h] BYREF
  unsigned __int16 v126; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v127; // [rsp+138h] [rbp-A8h]
  __int64 v128; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v129; // [rsp+148h] [rbp-98h]
  __int128 v130; // [rsp+150h] [rbp-90h]
  __int64 v131; // [rsp+160h] [rbp-80h]
  __int128 v132; // [rsp+170h] [rbp-70h] BYREF
  __int64 v133; // [rsp+180h] [rbp-60h]
  __m128i v134; // [rsp+190h] [rbp-50h] BYREF
  __m128i v135; // [rsp+1A0h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 24) == 299 && (*(_BYTE *)(a2 + 33) & 4) == 0 && (*(_WORD *)(a2 + 32) & 0x380) == 0 )
    return sub_3848BD0();
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL);
  v5 = a1[1];
  v6 = *(_WORD *)v4;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v134, *a1, *(_QWORD *)(v5 + 64), v6, v7);
    LOWORD(v120) = v134.m128i_i16[4];
    v121 = v135.m128i_i64[0];
  }
  else
  {
    v120 = v8(*a1, *(_QWORD *)(v5 + 64), v6, v7);
    v121 = v96;
  }
  v9 = *(const __m128i **)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = v9->m128i_i64[0];
  v12 = _mm_loadu_si128(v9 + 5);
  v122 = v10;
  v115 = v11;
  v13 = v9->m128i_u64[1];
  v14 = *(_QWORD *)(a2 + 112);
  v114 = v13;
  v15 = *(_WORD *)(v14 + 32);
  v134 = _mm_loadu_si128((const __m128i *)(v14 + 40));
  v135 = _mm_loadu_si128((const __m128i *)(v14 + 56));
  if ( v10 )
    sub_B96E90((__int64)&v122, v10, 1);
  v16 = *(_DWORD *)(a2 + 72);
  v17 = *(_WORD *)(a2 + 96);
  *(_QWORD *)&v124 = 0;
  v18 = *(_QWORD *)(a2 + 104);
  DWORD2(v124) = 0;
  v123 = v16;
  v126 = v17;
  *(_QWORD *)&v125 = 0;
  DWORD2(v125) = 0;
  v127 = v18;
  if ( v17 == (_WORD)v120 && ((_WORD)v120 || v18 == v121)
    || ((LOWORD(v128) = v120,
         v129 = v121,
         *(_QWORD *)&v132 = sub_2D5B750((unsigned __int16 *)&v128),
         *((_QWORD *)&v132 + 1) = v21,
         *(_QWORD *)&v22 = sub_2D5B750(&v126),
         v130 = v22,
         !BYTE8(v22))
     || BYTE8(v132))
    && (unsigned __int64)v130 <= (unsigned __int64)v132 )
  {
    sub_375E510(
      (__int64)a1,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
      (__int64)&v124,
      (__int64)&v125);
    v19 = sub_33F5040(
            (_QWORD *)a1[1],
            v115,
            v114,
            (__int64)&v122,
            v124,
            *((unsigned __int64 *)&v124 + 1),
            v12.m128i_u64[0],
            v12.m128i_u64[1],
            *(_OWORD *)*(_QWORD *)(a2 + 112),
            *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
            *(unsigned __int16 *)(a2 + 96),
            *(_QWORD *)(a2 + 104),
            *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
            v15,
            (__int64)&v134);
  }
  else
  {
    v113 = v15;
    v23 = *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) == 0;
    v24 = *(_QWORD *)(a2 + 40);
    v25 = *(_QWORD *)(v24 + 40);
    v26 = *(_QWORD *)(v24 + 48);
    if ( v23 )
    {
      sub_375E510((__int64)a1, v25, v26, (__int64)&v124, (__int64)&v125);
      *(_QWORD *)&v124 = sub_33F4560(
                           (_QWORD *)a1[1],
                           v115,
                           v114,
                           (__int64)&v122,
                           v124,
                           *((unsigned __int64 *)&v124 + 1),
                           v12.m128i_u64[0],
                           v12.m128i_u64[1],
                           *(_OWORD *)*(_QWORD *)(a2 + 112),
                           *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
                           *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
                           v15,
                           (__int64)&v134);
      DWORD2(v124) = v27;
      *(_QWORD *)&v130 = sub_2D5B750((unsigned __int16 *)&v120);
      v28 = *(_QWORD *)(a2 + 104);
      *((_QWORD *)&v130 + 1) = v29;
      LOWORD(v29) = *(_WORD *)(a2 + 96);
      v127 = v28;
      v126 = v29;
      v128 = sub_2D5B750(&v126);
      v129 = v30;
      v31 = v30;
      *(_QWORD *)&v132 = v128 - v130;
      if ( (_QWORD)v130 )
        v31 = BYTE8(v130);
      BYTE8(v132) = v31;
      v32 = sub_CA1930(&v132);
      v33 = sub_327FC40(*(_QWORD **)(a1[1] + 64), v32);
      v111 = v34;
      v108 = v33;
      *(_QWORD *)&v35 = sub_2D5B750((unsigned __int16 *)&v120);
      v132 = v35;
      v36 = sub_CA1930(&v132);
      BYTE8(v132) = 0;
      v37 = (_QWORD *)a1[1];
      v38 = (unsigned int)(v36 >> 3);
      *(_QWORD *)&v132 = v38;
      v39 = sub_3409320(v37, v12.m128i_i64[0], v12.m128i_i64[1], v38, 0, (__int64)&v122, v12, 1);
      v40 = (_QWORD *)a1[1];
      v116 = (unsigned __int64)v39;
      v118 = v41 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v42 = *(_QWORD *)(a2 + 112);
      v43 = *(_BYTE *)(v42 + 34);
      v44 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v44 )
      {
        v45 = *(_QWORD *)(v42 + 8) + v38;
        v46 = *(_BYTE *)(v42 + 20);
        if ( (*(_QWORD *)v42 & 4) != 0 )
        {
          *((_QWORD *)&v130 + 1) = v45;
          BYTE4(v131) = v46;
          *(_QWORD *)&v130 = v44 | 4;
          LODWORD(v131) = *(_DWORD *)(v44 + 12);
        }
        else
        {
          v97 = *(_QWORD *)(v44 + 8);
          *(_QWORD *)&v130 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v130 + 1) = v45;
          v98 = *(unsigned __int8 *)(v97 + 8);
          BYTE4(v131) = v46;
          if ( (unsigned int)(v98 - 17) <= 1 )
            v97 = **(_QWORD **)(v97 + 16);
          LODWORD(v131) = *(_DWORD *)(v97 + 8) >> 8;
        }
      }
      else
      {
        v74 = *(_DWORD *)(v42 + 16);
        v75 = *(_QWORD *)(v42 + 8) + v38;
        BYTE4(v131) = 0;
        *(_QWORD *)&v130 = 0;
        *((_QWORD *)&v130 + 1) = v75;
        LODWORD(v131) = v74;
      }
      *(_QWORD *)&v125 = sub_33F5040(
                           v40,
                           v115,
                           v114,
                           (__int64)&v122,
                           v125,
                           *((unsigned __int64 *)&v125 + 1),
                           v116,
                           v118,
                           v130,
                           v131,
                           v108,
                           v111,
                           v43,
                           v113,
                           (__int64)&v134);
      DWORD2(v125) = v48;
    }
    else
    {
      sub_375E510((__int64)a1, v25, v26, (__int64)&v124, (__int64)&v125);
      v49 = *(_QWORD *)(a2 + 104);
      LOWORD(v128) = *(_WORD *)(a2 + 96);
      v129 = v49;
      v50 = sub_2D5B750((unsigned __int16 *)&v128);
      *((_QWORD *)&v132 + 1) = v51;
      *(_QWORD *)&v132 = (unsigned __int64)(v50 + 7) >> 3;
      v52 = sub_CA1930(&v132);
      *(_QWORD *)&v53 = sub_2D5B750((unsigned __int16 *)&v120);
      v132 = v53;
      v54 = (unsigned __int64)sub_CA1930(&v132) >> 3;
      v55 = v52 - v54;
      v105 = v54;
      *(_QWORD *)&v56 = sub_2D5B750((unsigned __int16 *)&v128);
      v57 = 8 * v55;
      v132 = v56;
      v58 = sub_CA1930(&v132);
      v109 = sub_327FC40(*(_QWORD **)(a1[1] + 64), v58 - v57);
      v112 = v59;
      *(_QWORD *)&v132 = sub_2D5B750((unsigned __int16 *)&v120);
      *((_QWORD *)&v132 + 1) = v60;
      if ( v57 < (unsigned __int64)sub_CA1930(&v132) )
      {
        v103 = a1[1];
        v100 = *a1;
        v101 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
        v76 = sub_2E79000(*(__int64 **)(v103 + 40));
        if ( v101 == sub_2D42F30 )
        {
          v77 = sub_AE2980(v76, 0)[1];
          v78 = 2;
          if ( v77 != 1 )
          {
            v78 = 3;
            if ( v77 != 2 )
            {
              v78 = 4;
              if ( v77 != 4 )
              {
                v78 = 5;
                if ( v77 != 8 )
                {
                  v78 = 6;
                  if ( v77 != 16 )
                  {
                    v78 = 7;
                    if ( v77 != 32 )
                    {
                      v78 = 8;
                      if ( v77 != 64 )
                        v78 = 9 * (v77 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v78 = v101(v100, v76, 0);
        }
        v102 = v78;
        *(_QWORD *)&v79 = sub_2D5B750((unsigned __int16 *)&v120);
        v132 = v79;
        v80 = sub_CA1930(&v132);
        *(_QWORD *)&v81 = sub_3400BD0(v103, v80 - v57, (__int64)&v122, v102, 0, 0, v12, 0);
        v83 = sub_3406EB0((_QWORD *)v103, 0xBEu, (__int64)&v122, v120, v121, v82, v125, v81);
        v84 = *a1;
        v85 = (_QWORD *)a1[1];
        v86 = (__int64 *)v85[5];
        *(_QWORD *)&v125 = v83;
        v104 = v84;
        DWORD2(v125) = v87;
        v107 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v84 + 32LL);
        v88 = sub_2E79000(v86);
        if ( v107 == sub_2D42F30 )
        {
          v89 = sub_AE2980(v88, 0)[1];
          v90 = 2;
          if ( v89 != 1 )
          {
            v90 = 3;
            if ( v89 != 2 )
            {
              v90 = 4;
              if ( v89 != 4 )
              {
                v90 = 5;
                if ( v89 != 8 )
                {
                  v90 = 6;
                  if ( v89 != 16 )
                  {
                    v90 = 7;
                    if ( v89 != 32 )
                    {
                      v90 = 8;
                      if ( v89 != 64 )
                        v90 = 9 * (v89 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v90 = v107(v104, v88, 0);
        }
        *(_QWORD *)&v91 = sub_3400BD0((__int64)v85, v57, (__int64)&v122, v90, 0, 0, v12, 0);
        *(_QWORD *)&v93 = sub_3406EB0(v85, 0xC0u, (__int64)&v122, v120, v121, v92, v124, v91);
        *(_QWORD *)&v125 = sub_3406EB0(v85, 0xBBu, (__int64)&v122, v120, v121, v94, v125, v93);
        DWORD2(v125) = v95;
      }
      v61 = sub_33F5040(
              (_QWORD *)a1[1],
              v115,
              v114,
              (__int64)&v122,
              v125,
              *((unsigned __int64 *)&v125 + 1),
              v12.m128i_u64[0],
              v12.m128i_u64[1],
              *(_OWORD *)*(_QWORD *)(a2 + 112),
              *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
              v109,
              v112,
              *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
              v113,
              (__int64)&v134);
      BYTE8(v132) = 0;
      v62 = (_QWORD *)a1[1];
      *(_QWORD *)&v125 = v61;
      *(_QWORD *)&v132 = v105;
      DWORD2(v125) = v63;
      v117 = sub_3409320(v62, v12.m128i_i64[0], v12.m128i_i64[1], v105, 0, (__int64)&v122, v12, 1);
      v110 = a1[1];
      v119 = v64 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v106 = *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL);
      v65 = sub_327FC40(*(_QWORD **)(v110 + 64), v57);
      v67 = *(_QWORD *)(a2 + 112);
      v68 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v68 )
      {
        v69 = *(_QWORD *)(v67 + 8) + v105;
        v70 = *(_BYTE *)(v67 + 20);
        if ( (*(_QWORD *)v67 & 4) != 0 )
        {
          *((_QWORD *)&v132 + 1) = *(_QWORD *)(v67 + 8) + v105;
          BYTE4(v133) = v70;
          *(_QWORD *)&v132 = v68 | 4;
          LODWORD(v133) = *(_DWORD *)(v68 + 12);
        }
        else
        {
          *(_QWORD *)&v132 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v132 + 1) = v69;
          BYTE4(v133) = v70;
          v99 = *(_QWORD *)(v68 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v99 + 8) - 17 <= 1 )
            v99 = **(_QWORD **)(v99 + 16);
          LODWORD(v133) = *(_DWORD *)(v99 + 8) >> 8;
        }
      }
      else
      {
        v72 = *(_DWORD *)(v67 + 16);
        v73 = *(_QWORD *)(v67 + 8) + v105;
        *(_QWORD *)&v132 = 0;
        *((_QWORD *)&v132 + 1) = v73;
        LODWORD(v133) = v72;
        BYTE4(v133) = 0;
      }
      *(_QWORD *)&v124 = sub_33F5040(
                           (_QWORD *)v110,
                           v115,
                           v114,
                           (__int64)&v122,
                           v124,
                           *((unsigned __int64 *)&v124 + 1),
                           (unsigned __int64)v117,
                           v119,
                           v132,
                           v133,
                           v65,
                           v66,
                           v106,
                           v113,
                           (__int64)&v134);
      DWORD2(v124) = v71;
    }
    v19 = (__m128i *)sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v122, 1, 0, v47, v124, v125);
  }
  if ( v122 )
    sub_B91220((__int64)&v122, v122);
  return (__int64)v19;
}
