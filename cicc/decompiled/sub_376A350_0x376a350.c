// Function: sub_376A350
// Address: 0x376a350
//
__int64 __fastcall sub_376A350(__int64 *a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  char *v5; // r14
  __int64 v6; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // r13
  size_t v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 *v20; // r13
  __int64 v21; // r9
  _BYTE *v22; // rax
  unsigned int v23; // r14d
  __int64 v24; // rdx
  const void *v25; // rsi
  unsigned __int64 v26; // r13
  unsigned int v27; // r15d
  __int64 v28; // rax
  __int64 v29; // rsi
  char *v30; // r14
  int v31; // edx
  __int64 v32; // r13
  __int64 v33; // r9
  char *v34; // r15
  __int64 v35; // r12
  __m128i *v36; // rsi
  int v37; // eax
  __int64 v38; // rdx
  size_t v39; // rax
  int v41; // edx
  __int64 v42; // rax
  unsigned int v43; // eax
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  __m128i *v52; // rsi
  __int64 v53; // r12
  __int64 v54; // r14
  __int64 (__fastcall *v55)(__int64, __int64, unsigned int); // r13
  __int64 v56; // rax
  int v57; // edx
  unsigned __int16 v58; // ax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // r8
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  int v65; // edx
  void (***v66)(); // rdi
  void (*v67)(); // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __m128i si128; // xmm0
  __int64 v71; // rax
  __int64 v72; // rbx
  unsigned __int64 v73; // rdi
  unsigned __int8 v74; // [rsp+Fh] [rbp-13D1h]
  __int64 v75; // [rsp+10h] [rbp-13D0h]
  __int64 v76; // [rsp+20h] [rbp-13C0h]
  __int64 v77; // [rsp+20h] [rbp-13C0h]
  __int64 (__fastcall *v78)(__int64, __int64, __int64, _QWORD, _QWORD); // [rsp+28h] [rbp-13B8h]
  __int64 v79; // [rsp+28h] [rbp-13B8h]
  __m128i v80; // [rsp+30h] [rbp-13B0h] BYREF
  __int64 v81; // [rsp+40h] [rbp-13A0h]
  __int64 v82; // [rsp+48h] [rbp-1398h]
  _BYTE *v83; // [rsp+50h] [rbp-1390h]
  const char **v84; // [rsp+58h] [rbp-1388h]
  __int64 v85; // [rsp+60h] [rbp-1380h]
  __int64 v86; // [rsp+68h] [rbp-1378h]
  unsigned __int8 *v87; // [rsp+70h] [rbp-1370h]
  __int64 v88; // [rsp+78h] [rbp-1368h]
  int v89; // [rsp+80h] [rbp-1360h] BYREF
  char v90; // [rsp+84h] [rbp-135Ch]
  unsigned __int64 v91; // [rsp+88h] [rbp-1358h]
  __int128 v92; // [rsp+90h] [rbp-1350h] BYREF
  __int64 v93; // [rsp+A0h] [rbp-1340h] BYREF
  int v94; // [rsp+A8h] [rbp-1338h]
  unsigned __int64 v95; // [rsp+B0h] [rbp-1330h] BYREF
  __m128i *v96; // [rsp+B8h] [rbp-1328h]
  const __m128i *v97; // [rsp+C0h] [rbp-1320h]
  char *v98[2]; // [rsp+D0h] [rbp-1310h] BYREF
  __int64 v99; // [rsp+E0h] [rbp-1300h] BYREF
  __m128i v100[2]; // [rsp+F0h] [rbp-12F0h] BYREF
  __m128i v101; // [rsp+110h] [rbp-12D0h] BYREF
  __m128i v102; // [rsp+120h] [rbp-12C0h] BYREF
  __m128i v103; // [rsp+130h] [rbp-12B0h] BYREF
  _BYTE *v104; // [rsp+140h] [rbp-12A0h] BYREF
  __int64 v105; // [rsp+148h] [rbp-1298h]
  _BYTE v106[64]; // [rsp+150h] [rbp-1290h] BYREF
  char v107[8]; // [rsp+190h] [rbp-1250h] BYREF
  char *v108; // [rsp+198h] [rbp-1248h]
  unsigned int v109; // [rsp+1A0h] [rbp-1240h]
  char v110; // [rsp+1A8h] [rbp-1238h] BYREF
  __int64 *v111; // [rsp+228h] [rbp-11B8h]
  __int64 v112; // [rsp+238h] [rbp-11A8h] BYREF
  __int64 *v113; // [rsp+248h] [rbp-1198h]
  __int64 v114; // [rsp+258h] [rbp-1188h] BYREF
  unsigned __int8 v115; // [rsp+270h] [rbp-1170h]
  __int64 v116; // [rsp+280h] [rbp-1160h] BYREF
  __int64 v117; // [rsp+288h] [rbp-1158h]
  __int64 v118; // [rsp+290h] [rbp-1150h]
  unsigned __int64 v119; // [rsp+298h] [rbp-1148h]
  __int64 v120; // [rsp+2A0h] [rbp-1140h]
  __int64 v121; // [rsp+2A8h] [rbp-1138h]
  __int64 v122; // [rsp+2B0h] [rbp-1130h]
  unsigned __int64 v123; // [rsp+2B8h] [rbp-1128h] BYREF
  __m128i *v124; // [rsp+2C0h] [rbp-1120h]
  const __m128i *v125; // [rsp+2C8h] [rbp-1118h]
  __int64 v126; // [rsp+2D0h] [rbp-1110h]
  __int64 v127; // [rsp+2D8h] [rbp-1108h] BYREF
  int v128; // [rsp+2E0h] [rbp-1100h]
  __int64 v129; // [rsp+2E8h] [rbp-10F8h]
  _BYTE *v130; // [rsp+2F0h] [rbp-10F0h]
  __int64 v131; // [rsp+2F8h] [rbp-10E8h]
  _BYTE v132[1792]; // [rsp+300h] [rbp-10E0h] BYREF
  _BYTE *v133; // [rsp+A00h] [rbp-9E0h]
  __int64 v134; // [rsp+A08h] [rbp-9D8h]
  _BYTE v135[512]; // [rsp+A10h] [rbp-9D0h] BYREF
  _BYTE *v136; // [rsp+C10h] [rbp-7D0h]
  __int64 v137; // [rsp+C18h] [rbp-7C8h]
  _BYTE v138[1792]; // [rsp+C20h] [rbp-7C0h] BYREF
  _BYTE *v139; // [rsp+1320h] [rbp-C0h]
  __int64 v140; // [rsp+1328h] [rbp-B8h]
  _BYTE v141[64]; // [rsp+1330h] [rbp-B0h] BYREF
  __int64 v142; // [rsp+1370h] [rbp-70h]
  __int64 v143; // [rsp+1378h] [rbp-68h]
  int v144; // [rsp+1380h] [rbp-60h]
  char v145; // [rsp+13A0h] [rbp-40h]

  v4 = a1[1];
  v82 = a4;
  v5 = *(char **)(v4 + 8LL * a3 + 525288);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(a2 + 48);
  LOWORD(v9) = *(_WORD *)v6;
  v10 = *(_QWORD *)(v6 + 8);
  LOWORD(v92) = v9;
  *((_QWORD *)&v92 + 1) = v10;
  if ( (_WORD)v9 )
  {
    v41 = (unsigned __int16)v9;
    LOBYTE(v9) = (unsigned __int16)(v9 - 176) <= 0x34u;
    LODWORD(v11) = word_4456340[v41 - 1];
  }
  else
  {
    v11 = sub_3007240((__int64)&v92);
    v9 = HIDWORD(v11);
    v91 = v11;
  }
  v90 = v9;
  v12 = *a1;
  v89 = v11;
  v13 = *(__int64 **)(v12 + 24);
  v14 = strlen(v5);
  v84 = (const char **)sub_97F930(*v13, v5, v14, (__int64)&v89, 0);
  if ( !v84 )
  {
    v39 = strlen(v5);
    v84 = (const char **)sub_97F930(*v13, v5, v39, (__int64)&v89, 1);
    if ( !v84 )
      return 0;
  }
  v80.m128i_i64[0] = *(_QWORD *)(*a1 + 64);
  v19 = sub_3007410((__int64)&v92, (__int64 *)v80.m128i_i64[0], v15, v16, v17, v18);
  v81 = v19;
  v20 = (__int64 *)v19;
  if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
    v20 = **(__int64 ***)(v19 + 16);
  v21 = *(unsigned int *)(a2 + 64);
  v22 = v106;
  v83 = v106;
  v104 = v106;
  v105 = 0x800000000LL;
  if ( (_DWORD)v21 )
  {
    v23 = 0;
    v24 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v22[8 * v24] = v20;
      ++v23;
      v24 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      if ( v23 >= *(_DWORD *)(a2 + 64) )
        break;
      if ( v24 + 1 > (unsigned __int64)HIDWORD(v105) )
      {
        sub_C8D5F0((__int64)&v104, v83, v24 + 1, 8u, v24 + 1, v21);
        v24 = (unsigned int)v105;
      }
      v22 = v104;
    }
    v25 = v104;
  }
  else
  {
    v25 = v83;
    v24 = 0;
  }
  v26 = sub_BCF480(v20, v25, v24, 0);
  sub_97F000((__int64 *)v98, (__int64)v84);
  sub_C0A940((__int64)v107, v98[0], v98[1], v26);
  v27 = v115;
  if ( v115 )
  {
    v28 = v109;
    if ( *(_DWORD *)(a2 + 64) + *((unsigned __int8 *)v84 + 40) == v109 )
    {
      v29 = *(_QWORD *)(a2 + 80);
      v93 = v29;
      if ( v29 )
      {
        sub_B96E90((__int64)&v93, v29, 1);
        v28 = v109;
      }
      v30 = v108;
      v31 = *(_DWORD *)(a2 + 72);
      v95 = 0;
      v96 = 0;
      v32 = 0;
      v94 = v31;
      v97 = 0;
      v101 = 0u;
      v102 = 0u;
      v103 = 0u;
      if ( &v108[16 * v28] != v108 )
      {
        v74 = v27;
        v33 = a2;
        v34 = &v108[16 * v28];
        v35 = v81;
        do
        {
          v37 = *((_DWORD *)v30 + 1);
          if ( v37 == 10 )
          {
            v75 = v33;
            v77 = a1[1];
            v78 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v77 + 528LL);
            v42 = sub_2E79000(*(__int64 **)(*a1 + 40));
            v43 = v78(v77, v42, v80.m128i_i64[0], (unsigned int)v92, *((_QWORD *)&v92 + 1));
            v44 = *a1;
            v117 = v45;
            LODWORD(v116) = v43;
            v87 = sub_3401740(v44, 1, (__int64)&v93, v43, v45, v46, v92);
            v88 = v47;
            v101.m128i_i64[1] = (__int64)v87;
            v102.m128i_i32[0] = v47;
            v51 = sub_3007410((__int64)&v116, (__int64 *)v80.m128i_i64[0], v47, v48, v49, v50);
            v52 = v96;
            v33 = v75;
            v102.m128i_i64[1] = v51;
            if ( v96 == v97 )
            {
              sub_332CDC0(&v95, v96, &v101);
              v33 = v75;
            }
            else
            {
              if ( v96 )
              {
                *v96 = _mm_load_si128(&v101);
                v52[1] = _mm_load_si128(&v102);
                v52[2] = _mm_load_si128(&v103);
                v52 = v96;
              }
              v96 = v52 + 3;
            }
          }
          else
          {
            if ( v37 )
            {
              v27 = 0;
              goto LABEL_33;
            }
            v38 = *(_QWORD *)(v33 + 40) + 40 * v32;
            v101.m128i_i64[1] = *(_QWORD *)v38;
            v36 = v96;
            LODWORD(v38) = *(_DWORD *)(v38 + 8);
            v102.m128i_i64[1] = v35;
            v102.m128i_i32[0] = v38;
            if ( v96 == v97 )
            {
              v76 = v33;
              sub_332CDC0(&v95, v96, &v101);
              v33 = v76;
              v32 = (unsigned int)(v32 + 1);
            }
            else
            {
              if ( v96 )
              {
                *v96 = _mm_load_si128(&v101);
                v36[1] = _mm_load_si128(&v102);
                v36[2] = _mm_load_si128(&v103);
                v36 = v96;
              }
              v32 = (unsigned int)(v32 + 1);
              v96 = v36 + 3;
            }
          }
          v30 += 16;
        }
        while ( v30 != v34 );
        v27 = v74;
      }
      v53 = *a1;
      v54 = a1[1];
      v55 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v54 + 32LL);
      v56 = sub_2E79000(*(__int64 **)(*a1 + 40));
      if ( v55 == sub_2D42F30 )
      {
        v57 = sub_AE2980(v56, 0)[1];
        v58 = 2;
        if ( v57 != 1 )
        {
          v58 = 3;
          if ( v57 != 2 )
          {
            v58 = 4;
            if ( v57 != 4 )
            {
              v58 = 5;
              if ( v57 != 8 )
              {
                v58 = 6;
                if ( v57 != 16 )
                {
                  v58 = 7;
                  if ( v57 != 32 )
                  {
                    v58 = 8;
                    if ( v57 != 64 )
                      v58 = 9 * (v57 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v58 = v55(v54, v56, 0);
      }
      v59 = sub_33EED90(v53, v84[2], v58, 0);
      v130 = v132;
      v60 = v59;
      v62 = v61;
      v63 = *a1;
      v119 = 0xFFFFFFFF00000020LL;
      v131 = 0x2000000000LL;
      v134 = 0x2000000000LL;
      v137 = 0x2000000000LL;
      v84 = (const char **)v141;
      v139 = v141;
      v140 = 0x400000000LL;
      v64 = v93;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v120 = 0;
      v121 = 0;
      v122 = 0;
      v123 = 0;
      v124 = 0;
      v125 = 0;
      v126 = v63;
      v128 = 0;
      v129 = 0;
      v133 = v135;
      v136 = v138;
      v142 = 0;
      v143 = 0;
      v144 = 0;
      v145 = 0;
      v127 = v93;
      if ( v93 )
      {
        v79 = v62;
        v80.m128i_i64[0] = v60;
        sub_B96E90((__int64)&v127, v93, 1);
        v64 = v123;
        v63 = *a1;
        v62 = v79;
        v60 = v80.m128i_i64[0];
      }
      v86 = v62;
      v116 = v63 + 288;
      v128 = v94;
      v85 = v60;
      v121 = v60;
      v118 = v81;
      LODWORD(v122) = v62;
      v123 = v95;
      v124 = v96;
      LODWORD(v117) = 0;
      v65 = -1431655765 * ((__int64)((__int64)v96->m128i_i64 - v95) >> 4);
      LODWORD(v120) = 0;
      v95 = 0;
      HIDWORD(v119) = v65;
      v96 = 0;
      v125 = v97;
      v97 = 0;
      if ( v64 )
        j_j___libc_free_0(v64);
      v66 = *(void (****)())(v126 + 16);
      v67 = **v66;
      if ( v67 != nullsub_1688 )
        ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v67)(
          v66,
          *(_QWORD *)(v126 + 40),
          0,
          &v123);
      sub_3377410((__int64)v100, (_WORD *)a1[1], (__int64)&v116);
      si128 = _mm_load_si128(v100);
      v71 = *(unsigned int *)(v82 + 8);
      if ( v71 + 1 > (unsigned __int64)*(unsigned int *)(v82 + 12) )
      {
        v80 = si128;
        sub_C8D5F0(v82, (const void *)(v82 + 16), v71 + 1, 0x10u, v68, v69);
        v71 = *(unsigned int *)(v82 + 8);
        si128 = _mm_load_si128(&v80);
      }
      v72 = v82;
      *(__m128i *)(*(_QWORD *)v82 + 16 * v71) = si128;
      v73 = (unsigned __int64)v139;
      ++*(_DWORD *)(v72 + 8);
      if ( (const char **)v73 != v84 )
        _libc_free(v73);
      if ( v136 != v138 )
        _libc_free((unsigned __int64)v136);
      if ( v133 != v135 )
        _libc_free((unsigned __int64)v133);
      if ( v130 != v132 )
        _libc_free((unsigned __int64)v130);
      if ( v127 )
        sub_B91220((__int64)&v127, v127);
      if ( v123 )
        j_j___libc_free_0(v123);
LABEL_33:
      if ( v95 )
        j_j___libc_free_0(v95);
      if ( v93 )
        sub_B91220((__int64)&v93, v93);
      if ( !v115 )
        goto LABEL_38;
    }
    else
    {
      v27 = 0;
    }
    v115 = 0;
    if ( v113 != &v114 )
      j_j___libc_free_0((unsigned __int64)v113);
    if ( v111 != &v112 )
      j_j___libc_free_0((unsigned __int64)v111);
    if ( v108 != &v110 )
      _libc_free((unsigned __int64)v108);
  }
LABEL_38:
  if ( (__int64 *)v98[0] != &v99 )
    j_j___libc_free_0((unsigned __int64)v98[0]);
  if ( v104 != v83 )
    _libc_free((unsigned __int64)v104);
  return v27;
}
