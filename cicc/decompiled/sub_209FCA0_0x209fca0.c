// Function: sub_209FCA0
// Address: 0x209fca0
//
void __fastcall sub_209FCA0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        char a10)
{
  __int64 v10; // r11
  unsigned __int64 v11; // r15
  __int64 v12; // r14
  __int64 v14; // rdx
  int v15; // r13d
  unsigned __int64 v16; // r12
  char v17; // al
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // r12
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // rax
  int v25; // r14d
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // r14d
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r12
  __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // r12
  __int64 v43; // rax
  unsigned int *v44; // rax
  __int64 v45; // rcx
  int v46; // r8d
  int v47; // r9d
  int v48; // eax
  int v49; // eax
  unsigned __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 *v52; // rax
  unsigned int v53; // esi
  __int64 *v54; // rcx
  int v55; // edx
  int v56; // r14d
  __int64 v57; // r8
  unsigned int v58; // edi
  unsigned __int64 *v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r12
  __int64 v64; // r12
  __int64 v65; // r13
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // rax
  int v69; // r13d
  unsigned __int64 *v70; // r10
  int v71; // edx
  int v72; // r8d
  int v73; // r11d
  int v74; // r11d
  __int64 v75; // rdi
  __int64 v76; // rdx
  unsigned __int64 v77; // rsi
  int v78; // r13d
  unsigned __int64 *v79; // r12
  int v80; // r9d
  int v81; // r9d
  __int64 v82; // rsi
  unsigned __int64 *v83; // r11
  __int64 v84; // r12
  int v85; // r13d
  unsigned __int64 v86; // rdx
  __int64 v87; // [rsp+8h] [rbp-11E8h]
  __int64 v88; // [rsp+8h] [rbp-11E8h]
  __int64 v89; // [rsp+8h] [rbp-11E8h]
  unsigned __int64 v90; // [rsp+10h] [rbp-11E0h]
  unsigned __int64 v91; // [rsp+18h] [rbp-11D8h]
  __int64 v92; // [rsp+20h] [rbp-11D0h]
  __int128 v95; // [rsp+70h] [rbp-1180h]
  __int64 *v96; // [rsp+70h] [rbp-1180h]
  __int64 *v97; // [rsp+70h] [rbp-1180h]
  int v98; // [rsp+A0h] [rbp-1150h] BYREF
  char v99; // [rsp+A4h] [rbp-114Ch]
  int v100; // [rsp+A8h] [rbp-1148h]
  char v101; // [rsp+B0h] [rbp-1140h]
  __m128i v102; // [rsp+C0h] [rbp-1130h] BYREF
  _BYTE v103[128]; // [rsp+D0h] [rbp-1120h] BYREF
  _BYTE *v104; // [rsp+150h] [rbp-10A0h]
  __int64 v105; // [rsp+158h] [rbp-1098h]
  _BYTE v106[128]; // [rsp+160h] [rbp-1090h] BYREF
  _BYTE *v107; // [rsp+1E0h] [rbp-1010h]
  __int64 v108; // [rsp+1E8h] [rbp-1008h]
  _BYTE v109[128]; // [rsp+1F0h] [rbp-1000h] BYREF
  __int64 v110; // [rsp+270h] [rbp-F80h]
  __int64 v111; // [rsp+278h] [rbp-F78h]
  __int64 v112; // [rsp+280h] [rbp-F70h]
  __int64 v113; // [rsp+288h] [rbp-F68h]
  __int64 v114; // [rsp+290h] [rbp-F60h]
  int v115; // [rsp+298h] [rbp-F58h]
  _QWORD v116[3]; // [rsp+2A0h] [rbp-F50h] BYREF
  unsigned __int64 v117; // [rsp+2B8h] [rbp-F38h]
  __int64 v118; // [rsp+2C0h] [rbp-F30h]
  __int64 v119; // [rsp+2C8h] [rbp-F28h]
  __int64 v120; // [rsp+2D0h] [rbp-F20h]
  __int64 v121; // [rsp+2D8h] [rbp-F18h]
  __int64 v122; // [rsp+2E0h] [rbp-F10h]
  __int64 v123; // [rsp+2E8h] [rbp-F08h]
  __int64 v124; // [rsp+2F0h] [rbp-F00h]
  __int64 v125; // [rsp+2F8h] [rbp-EF8h] BYREF
  int v126; // [rsp+300h] [rbp-EF0h]
  __int64 v127; // [rsp+308h] [rbp-EE8h]
  _BYTE *v128; // [rsp+310h] [rbp-EE0h]
  __int64 v129; // [rsp+318h] [rbp-ED8h]
  _BYTE v130[1536]; // [rsp+320h] [rbp-ED0h] BYREF
  _BYTE *v131; // [rsp+920h] [rbp-8D0h]
  __int64 v132; // [rsp+928h] [rbp-8C8h]
  _BYTE v133[512]; // [rsp+930h] [rbp-8C0h] BYREF
  _BYTE *v134; // [rsp+B30h] [rbp-6C0h]
  __int64 v135; // [rsp+B38h] [rbp-6B8h]
  _BYTE v136[1536]; // [rsp+B40h] [rbp-6B0h] BYREF
  _BYTE *v137; // [rsp+1140h] [rbp-B0h]
  __int64 v138; // [rsp+1148h] [rbp-A8h]
  _BYTE v139[64]; // [rsp+1150h] [rbp-A0h] BYREF
  unsigned __int64 v140; // [rsp+1190h] [rbp-60h]
  unsigned __int64 v141; // [rsp+1198h] [rbp-58h]
  __int64 v142; // [rsp+11A0h] [rbp-50h]
  int v143; // [rsp+11A8h] [rbp-48h]
  __int64 v144; // [rsp+11B0h] [rbp-40h]

  v10 = a2;
  v11 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = (a2 >> 2) & 1;
  v102.m128i_i64[0] = (__int64)v103;
  v102.m128i_i64[1] = 0x1000000000LL;
  v105 = 0x1000000000LL;
  v108 = 0x1000000000LL;
  v104 = v106;
  v117 = 0xFFFFFFFF00000020LL;
  v95 = __PAIR128__(a4, a3);
  v14 = *(_QWORD *)(a1 + 552);
  v107 = v109;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = -1;
  memset(v116, 0, sizeof(v116));
  v118 = 0;
  v15 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20);
  v128 = v130;
  v129 = 0x2000000000LL;
  v132 = 0x2000000000LL;
  v135 = 0x2000000000LL;
  v137 = v139;
  v138 = 0x400000000LL;
  v131 = v133;
  v119 = 0;
  v16 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (v15 & 0xFFFFFFF);
  v17 = *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  v120 = 0;
  v121 = 0;
  v18 = v16;
  v122 = 0;
  v123 = 0;
  v124 = v14;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v134 = v136;
  v140 = 0;
  v141 = 0;
  v142 = -1;
  v143 = -1;
  v144 = 0;
  if ( (v17 & 0x40) != 0 )
    v18 = *(_QWORD *)(v11 - 8);
  v19 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v16 - v18) >> 3);
  if ( a10 )
  {
    v89 = v10;
    v68 = sub_1643270(*(_QWORD **)(v14 + 48));
    v15 = *(_DWORD *)(v11 + 20);
    v10 = v89;
    v92 = v68;
    v17 = *(_BYTE *)(v11 + 23);
  }
  else
  {
    v92 = *(_QWORD *)v11;
  }
  v20 = v15 & 0xFFFFFFF;
  if ( (_BYTE)v12 )
  {
    if ( v17 < 0 )
    {
      v87 = v10;
      v21 = sub_1648A40(v11);
      v10 = v87;
      v23 = v21 + v22;
      if ( *(char *)(v11 + 23) >= 0 )
      {
        if ( (unsigned int)(v23 >> 4) )
          goto LABEL_113;
      }
      else
      {
        v24 = sub_1648A40(v11);
        v10 = v87;
        if ( (unsigned int)((v23 - v24) >> 4) )
        {
          if ( *(char *)(v11 + 23) < 0 )
          {
            v25 = *(_DWORD *)(sub_1648A40(v11) + 8);
            if ( *(char *)(v11 + 23) >= 0 )
              BUG();
            v26 = sub_1648A40(v11);
            v10 = v87;
            v28 = *(_DWORD *)(v26 + v27 - 4) - v25;
LABEL_61:
            sub_207E8C0(a1, (__int64)v116, v10, v19, v20 - 1 - v28, v92, a7, a8, a9, v95, 0);
            if ( !a6 )
              LOBYTE(v117) = (4 * (*(_DWORD *)(*(_QWORD *)(v11 + 64) + 8LL) >> 8 != 0)) | v117 & 0xFB;
            if ( *(char *)(v11 + 23) < 0 )
            {
              v61 = sub_1648A40(v11);
              v63 = v61 + v62;
              if ( *(char *)(v11 + 23) < 0 )
                v63 -= sub_1648A40(v11);
              v64 = v63 >> 4;
              if ( (_DWORD)v64 )
              {
                v65 = 0;
                v66 = 16LL * (unsigned int)v64;
                while ( 1 )
                {
                  v67 = 0;
                  if ( *(char *)(v11 + 23) < 0 )
                    v67 = sub_1648A40(v11);
                  v44 = (unsigned int *)(v65 + v67);
                  if ( !*(_DWORD *)(*(_QWORD *)v44 + 8LL) )
                    break;
                  v65 += 16;
                  if ( v66 == v65 )
                    goto LABEL_32;
                }
LABEL_31:
                v90 = v11 + 24LL * v44[2] - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
                v91 = 0xAAAAAAAAAAAAAAABLL * ((24LL * v44[3] - 24LL * v44[2]) >> 3);
                goto LABEL_32;
              }
            }
            goto LABEL_32;
          }
LABEL_113:
          BUG();
        }
      }
    }
    v28 = 0;
    goto LABEL_61;
  }
  if ( v17 >= 0 )
    goto LABEL_19;
  v88 = v10;
  v29 = sub_1648A40(v11);
  v10 = v88;
  v31 = v29 + v30;
  if ( *(char *)(v11 + 23) >= 0 )
  {
    if ( (unsigned int)(v31 >> 4) )
LABEL_117:
      BUG();
LABEL_19:
    v36 = 0;
    goto LABEL_20;
  }
  v32 = sub_1648A40(v11);
  v10 = v88;
  if ( !(unsigned int)((v31 - v32) >> 4) )
    goto LABEL_19;
  if ( *(char *)(v11 + 23) >= 0 )
    goto LABEL_117;
  v33 = *(_DWORD *)(sub_1648A40(v11) + 8);
  if ( *(char *)(v11 + 23) >= 0 )
    BUG();
  v34 = sub_1648A40(v11);
  v10 = v88;
  v36 = *(_DWORD *)(v34 + v35 - 4) - v33;
LABEL_20:
  sub_207E8C0(a1, (__int64)v116, v10, v19, v20 - 3 - v36, v92, a7, a8, a9, v95, 0);
  if ( !a6 )
    LOBYTE(v117) = (4 * (*(_DWORD *)(*(_QWORD *)(v11 + 64) + 8LL) >> 8 != 0)) | v117 & 0xFB;
  if ( *(char *)(v11 + 23) < 0 )
  {
    v37 = sub_1648A40(v11);
    v39 = v37 + v38;
    if ( *(char *)(v11 + 23) < 0 )
      v39 -= sub_1648A40(v11);
    v40 = v39 >> 4;
    if ( (_DWORD)v40 )
    {
      v41 = 0;
      v42 = 16LL * (unsigned int)v40;
      do
      {
        v43 = 0;
        if ( *(char *)(v11 + 23) < 0 )
          v43 = sub_1648A40(v11);
        v44 = (unsigned int *)(v41 + v43);
        if ( !*(_DWORD *)(*(_QWORD *)v44 + 8LL) )
          goto LABEL_31;
        v41 += 16;
      }
      while ( v42 != v41 );
    }
  }
LABEL_32:
  sub_1642E70((__int64)&v98, *(_QWORD *)(v11 + 56));
  v48 = -1412567281;
  if ( v101 )
    v48 = v100;
  v115 = v48;
  v49 = 0;
  if ( v99 )
    v49 = v98;
  v143 = v49;
  v142 = 0;
  v140 = v90;
  v141 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(24 * v91) >> 3);
  v144 = a5;
  v51 = sub_209C180(a1, &v102, 0xAAAAAAAAAAAAAAABLL, v45, v46, v47, a7, a8, a9);
  if ( v51 )
  {
    v52 = sub_2055040(a1, *(_QWORD *)(a1 + 552), v11, v51, v50, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
    v53 = *(_DWORD *)(a1 + 32);
    v54 = v52;
    v56 = v55;
    if ( v53 )
    {
      v57 = *(_QWORD *)(a1 + 16);
      v58 = (v53 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v59 = (unsigned __int64 *)(v57 + 24LL * v58);
      v60 = *v59;
      if ( v11 == *v59 )
      {
LABEL_39:
        v59[1] = (unsigned __int64)v54;
        *((_DWORD *)v59 + 4) = v56;
        goto LABEL_40;
      }
      v69 = 1;
      v70 = 0;
      while ( v60 != -8 )
      {
        if ( v60 == -16 && !v70 )
          v70 = v59;
        v58 = (v53 - 1) & (v69 + v58);
        v59 = (unsigned __int64 *)(v57 + 24LL * v58);
        v60 = *v59;
        if ( v11 == *v59 )
          goto LABEL_39;
        ++v69;
      }
      v71 = *(_DWORD *)(a1 + 24);
      if ( v70 )
        v59 = v70;
      ++*(_QWORD *)(a1 + 8);
      v72 = v71 + 1;
      if ( 4 * (v71 + 1) < 3 * v53 )
      {
        if ( v53 - *(_DWORD *)(a1 + 28) - v72 > v53 >> 3 )
        {
LABEL_80:
          *(_DWORD *)(a1 + 24) = v72;
          if ( *v59 != -8 )
            --*(_DWORD *)(a1 + 28);
          *v59 = v11;
          v59[1] = 0;
          *((_DWORD *)v59 + 4) = 0;
          goto LABEL_39;
        }
        v97 = v54;
        sub_205F3F0(a1 + 8, v53);
        v80 = *(_DWORD *)(a1 + 32);
        if ( v80 )
        {
          v81 = v80 - 1;
          v82 = *(_QWORD *)(a1 + 16);
          v83 = 0;
          LODWORD(v84) = v81 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v85 = 1;
          v72 = *(_DWORD *)(a1 + 24) + 1;
          v54 = v97;
          v59 = (unsigned __int64 *)(v82 + 24LL * (unsigned int)v84);
          v86 = *v59;
          if ( *v59 != v11 )
          {
            while ( v86 != -8 )
            {
              if ( v86 == -16 && !v83 )
                v83 = v59;
              v84 = v81 & (unsigned int)(v84 + v85);
              v59 = (unsigned __int64 *)(v82 + 24 * v84);
              v86 = *v59;
              if ( v11 == *v59 )
                goto LABEL_80;
              ++v85;
            }
            if ( v83 )
              v59 = v83;
          }
          goto LABEL_80;
        }
LABEL_115:
        ++*(_DWORD *)(a1 + 24);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    v96 = v54;
    sub_205F3F0(a1 + 8, 2 * v53);
    v73 = *(_DWORD *)(a1 + 32);
    if ( v73 )
    {
      v74 = v73 - 1;
      v75 = *(_QWORD *)(a1 + 16);
      LODWORD(v76) = v74 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v72 = *(_DWORD *)(a1 + 24) + 1;
      v54 = v96;
      v59 = (unsigned __int64 *)(v75 + 24LL * (unsigned int)v76);
      v77 = *v59;
      if ( v11 != *v59 )
      {
        v78 = 1;
        v79 = 0;
        while ( v77 != -8 )
        {
          if ( !v79 && v77 == -16 )
            v79 = v59;
          v76 = v74 & (unsigned int)(v76 + v78);
          v59 = (unsigned __int64 *)(v75 + 24 * v76);
          v77 = *v59;
          if ( v11 == *v59 )
            goto LABEL_80;
          ++v78;
        }
        if ( v79 )
          v59 = v79;
      }
      goto LABEL_80;
    }
    goto LABEL_115;
  }
LABEL_40:
  if ( v137 != v139 )
    _libc_free((unsigned __int64)v137);
  if ( v134 != v136 )
    _libc_free((unsigned __int64)v134);
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( v125 )
    sub_161E7C0((__int64)&v125, v125);
  if ( v121 )
    j_j___libc_free_0(v121, v123 - v121);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( (_BYTE *)v102.m128i_i64[0] != v103 )
    _libc_free(v102.m128i_u64[0]);
}
