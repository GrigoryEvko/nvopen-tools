// Function: sub_202F090
// Address: 0x202f090
//
__int64 *__fastcall sub_202F090(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // r11d
  unsigned int v5; // r15d
  __int64 v8; // rsi
  char v9; // dl
  unsigned int v10; // r11d
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int128 v14; // xmm0
  __m128i v15; // xmm1
  __int64 v16; // rax
  __m128i v17; // xmm2
  unsigned int v18; // eax
  __int64 v19; // rdx
  int v20; // eax
  int v21; // r9d
  __int64 *v22; // r8
  unsigned int v23; // r11d
  __int64 v24; // rdx
  unsigned int v25; // r14d
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 *v28; // rdx
  unsigned int v29; // eax
  unsigned int v30; // r11d
  __int64 v31; // rax
  int v32; // r8d
  __int64 v33; // r9
  unsigned int v34; // r11d
  int v35; // edx
  int v36; // edi
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v42; // r12
  int v43; // ecx
  __int64 v44; // rax
  int v45; // r8d
  int v46; // edx
  int v47; // edi
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // rax
  unsigned int v58; // edx
  unsigned int v59; // r11d
  _QWORD *v60; // rdi
  __int64 v61; // rsi
  unsigned __int64 v62; // r8
  __int64 v63; // r14
  char v64; // r9
  __int64 v65; // rcx
  bool v66; // zf
  unsigned int v67; // r14d
  _QWORD *v68; // rbx
  int v69; // edx
  int v70; // r12d
  __int64 v71; // rax
  __int64 *v72; // rax
  __int64 *v73; // r14
  unsigned int v75; // eax
  unsigned int v76; // eax
  __int128 v77; // [rsp-20h] [rbp-2D0h]
  __int128 v78; // [rsp-10h] [rbp-2C0h]
  unsigned int v79; // [rsp+10h] [rbp-2A0h]
  unsigned int v80; // [rsp+14h] [rbp-29Ch]
  unsigned int v81; // [rsp+18h] [rbp-298h]
  unsigned int v82; // [rsp+20h] [rbp-290h]
  __int64 v83; // [rsp+20h] [rbp-290h]
  unsigned int v84; // [rsp+30h] [rbp-280h]
  unsigned __int16 v85; // [rsp+30h] [rbp-280h]
  unsigned int v86; // [rsp+30h] [rbp-280h]
  unsigned int v87; // [rsp+38h] [rbp-278h]
  unsigned int v88; // [rsp+38h] [rbp-278h]
  unsigned int v89; // [rsp+38h] [rbp-278h]
  int v90; // [rsp+50h] [rbp-260h]
  unsigned int v92; // [rsp+58h] [rbp-258h]
  __int64 v93; // [rsp+60h] [rbp-250h]
  unsigned int v95; // [rsp+70h] [rbp-240h]
  __int64 *v96; // [rsp+70h] [rbp-240h]
  int v97; // [rsp+80h] [rbp-230h]
  __int64 v98; // [rsp+80h] [rbp-230h]
  unsigned __int16 v99; // [rsp+88h] [rbp-228h]
  unsigned int v100; // [rsp+88h] [rbp-228h]
  unsigned int v101; // [rsp+90h] [rbp-220h]
  __int64 v102; // [rsp+90h] [rbp-220h]
  unsigned __int64 v104; // [rsp+A8h] [rbp-208h]
  __int64 v105; // [rsp+E0h] [rbp-1D0h] BYREF
  const void **v106; // [rsp+E8h] [rbp-1C8h]
  char v107[8]; // [rsp+F0h] [rbp-1C0h] BYREF
  __int64 v108; // [rsp+F8h] [rbp-1B8h]
  __int64 v109; // [rsp+100h] [rbp-1B0h] BYREF
  int v110; // [rsp+108h] [rbp-1A8h]
  __int64 v111; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v112; // [rsp+118h] [rbp-198h]
  __int64 v113; // [rsp+120h] [rbp-190h] BYREF
  int v114; // [rsp+128h] [rbp-188h]
  __m128i v115; // [rsp+130h] [rbp-180h] BYREF
  __int64 v116; // [rsp+140h] [rbp-170h]
  __int128 v117; // [rsp+150h] [rbp-160h]
  __int64 v118; // [rsp+160h] [rbp-150h]
  __int64 *v119; // [rsp+170h] [rbp-140h] BYREF
  __int64 v120; // [rsp+178h] [rbp-138h]
  _QWORD v121[38]; // [rsp+180h] [rbp-130h] BYREF

  v101 = v4;
  sub_1F40D10(
    (__int64)&v119,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a3 + 40),
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 8LL));
  v8 = *(_QWORD *)(a3 + 72);
  v9 = *(_BYTE *)(a3 + 88);
  v10 = v101;
  LOBYTE(v105) = v120;
  v107[0] = v9;
  v106 = (const void **)v121[0];
  v11 = *(_QWORD *)(a3 + 96);
  v109 = v8;
  v108 = v11;
  if ( v8 )
  {
    sub_1623A60((__int64)&v109, v8, 2);
    v10 = v101;
  }
  v84 = v10;
  v12 = *(_QWORD *)(a3 + 104);
  v110 = *(_DWORD *)(a3 + 64);
  v13 = *(_QWORD *)(a3 + 32);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)v13);
  v15 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v102 = *(_QWORD *)(v13 + 40);
  v81 = *(_DWORD *)(v13 + 48);
  v97 = sub_1E34390(v12);
  v90 = v97;
  v16 = *(_QWORD *)(a3 + 104);
  v17 = _mm_loadu_si128((const __m128i *)(v16 + 40));
  v99 = *(_WORD *)(v16 + 32);
  v116 = *(_QWORD *)(v16 + 56);
  v115 = v17;
  LOBYTE(v18) = sub_1F7E0F0((__int64)&v105);
  v92 = v18;
  v93 = v19;
  LOBYTE(v20) = sub_1F7E0F0((__int64)v107);
  v22 = &v105;
  v23 = v84;
  LODWORD(v111) = v20;
  v112 = v24;
  if ( v107[0] )
  {
    v25 = word_4305480[(unsigned __int8)(v107[0] - 14)];
  }
  else
  {
    v76 = sub_1F58D30((__int64)v107);
    v23 = v84;
    v22 = &v105;
    v25 = v76;
  }
  if ( (_BYTE)v105 )
  {
    v80 = word_4305480[(unsigned __int8)(v105 - 14)];
  }
  else
  {
    v89 = v23;
    v75 = sub_1F58D30((__int64)&v105);
    v23 = v89;
    v80 = v75;
  }
  v26 = v80;
  v27 = v121;
  v119 = v121;
  v120 = 0x1000000000LL;
  if ( v80 > 0x10 )
  {
    v86 = v23;
    sub_16CD150((__int64)&v119, v121, v80, 16, (int)v22, v21);
    v27 = v119;
    v23 = v86;
    v26 = v80;
  }
  v28 = &v27[2 * v26];
  for ( LODWORD(v120) = v80; v28 != v27; v27 += 2 )
  {
    if ( v27 )
    {
      *v27 = 0;
      *((_DWORD *)v27 + 2) = 0;
    }
  }
  if ( (_BYTE)v111 )
  {
    v29 = sub_2021900(v111);
  }
  else
  {
    v88 = v23;
    v29 = sub_1F58D40((__int64)&v111);
    v30 = v88;
  }
  v87 = v29 >> 3;
  v85 = v99;
  v82 = v30;
  v31 = sub_1D2B810(
          *(_QWORD **)(a1 + 8),
          a4,
          (__int64)&v109,
          v92,
          v93,
          v97,
          v14,
          v15.m128i_i64[0],
          v15.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a3 + 104),
          *(_QWORD *)(*(_QWORD *)(a3 + 104) + 16LL),
          v111,
          v112,
          v99,
          (__int64)&v115);
  v34 = v82;
  v36 = v35;
  v37 = v31;
  v38 = (unsigned __int64)v119;
  *v119 = v37;
  *(_DWORD *)(v38 + 8) = v36;
  v39 = *v119;
  v40 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v40 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v32, v33);
    v40 = *(unsigned int *)(a2 + 8);
    v34 = v82;
  }
  v41 = (__int64 *)(*(_QWORD *)a2 + 16 * v40);
  *v41 = v39;
  v41[1] = 1;
  ++*(_DWORD *)(a2 + 8);
  if ( v25 <= 1 )
  {
    v67 = 1;
  }
  else
  {
    v79 = v25;
    v42 = 16;
    v98 = 16LL * v81;
    v83 = 16LL * v25;
    v100 = v87;
    do
    {
      v53 = *(_QWORD *)(v102 + 40) + v98;
      LOBYTE(v34) = *(_BYTE *)v53;
      v96 = *(__int64 **)(a1 + 8);
      v54 = sub_1D38BB0(
              (__int64)v96,
              v100,
              (__int64)&v109,
              v34,
              *(const void ***)(v53 + 8),
              0,
              (__m128i)v14,
              *(double *)v15.m128i_i64,
              v17,
              0);
      v55 = *(_QWORD *)(v102 + 40) + v98;
      LOBYTE(v5) = *(_BYTE *)v55;
      v104 = v81 | v104 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v77 + 1) = v56;
      *(_QWORD *)&v77 = v54;
      v57 = sub_1D332F0(
              v96,
              52,
              (__int64)&v109,
              v5,
              *(const void ***)(v55 + 8),
              3u,
              *(double *)&v14,
              *(double *)v15.m128i_i64,
              v17,
              v102,
              v104,
              v77);
      v60 = *(_QWORD **)(a1 + 8);
      v61 = *(_QWORD *)(a3 + 104);
      v62 = *(_QWORD *)v61 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v62 )
      {
        v63 = *(_QWORD *)(v61 + 8) + v100;
        v64 = *(_BYTE *)(v61 + 16);
        if ( (*(_QWORD *)v61 & 4) != 0 )
        {
          *((_QWORD *)&v117 + 1) = *(_QWORD *)(v61 + 8) + v100;
          LOBYTE(v118) = v64;
          *(_QWORD *)&v117 = v62 | 4;
          HIDWORD(v118) = *(_DWORD *)(v62 + 12);
        }
        else
        {
          v65 = *(_QWORD *)v62;
          *(_QWORD *)&v117 = *(_QWORD *)v61 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v117 + 1) = v63;
          v66 = *(_BYTE *)(v65 + 8) == 16;
          LOBYTE(v118) = v64;
          if ( v66 )
            v65 = **(_QWORD **)(v65 + 16);
          HIDWORD(v118) = *(_DWORD *)(v65 + 8) >> 8;
        }
      }
      else
      {
        v118 = 0;
        v43 = *(_DWORD *)(v61 + 20);
        v117 = 0u;
        HIDWORD(v118) = v43;
      }
      v95 = v59;
      v44 = sub_1D2B810(
              v60,
              a4,
              (__int64)&v109,
              v92,
              v93,
              v90,
              v14,
              (__int64)v57,
              v58,
              v117,
              v118,
              v111,
              v112,
              v85,
              (__int64)&v115);
      v34 = v95;
      v47 = v46;
      v48 = v44;
      v49 = (unsigned __int64)v119;
      v119[(unsigned __int64)v42 / 8] = v48;
      *(_DWORD *)(v49 + v42 + 8) = v47;
      v50 = v119[(unsigned __int64)v42 / 8];
      v51 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v51 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v45, v33);
        v51 = *(unsigned int *)(a2 + 8);
        v34 = v95;
      }
      v52 = (__int64 *)(*(_QWORD *)a2 + 16 * v51);
      v42 += 16;
      *v52 = v50;
      v52[1] = 1;
      v100 += v87;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v42 != v83 );
    v67 = v79;
  }
  v113 = 0;
  v114 = 0;
  v68 = sub_1D2B300(*(_QWORD **)(a1 + 8), 0x30u, (__int64)&v113, v92, v93, v33);
  v70 = v69;
  if ( v113 )
    sub_161E7C0((__int64)&v113, v113);
  for ( ; v67 != v80; *((_DWORD *)v72 + 2) = v70 )
  {
    v71 = v67++;
    v72 = &v119[2 * v71];
    *v72 = (__int64)v68;
  }
  *((_QWORD *)&v78 + 1) = (unsigned int)v120;
  *(_QWORD *)&v78 = v119;
  v73 = sub_1D359D0(
          *(__int64 **)(a1 + 8),
          104,
          (__int64)&v109,
          v105,
          v106,
          0,
          *(double *)&v14,
          *(double *)v15.m128i_i64,
          v17,
          v78);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
  if ( v109 )
    sub_161E7C0((__int64)&v109, v109);
  return v73;
}
