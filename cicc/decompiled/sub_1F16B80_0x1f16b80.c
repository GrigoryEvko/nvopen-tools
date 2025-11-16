// Function: sub_1F16B80
// Address: 0x1f16b80
//
void __fastcall sub_1F16B80(__int64 a1, char a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  const __m128i *v9; // r13
  unsigned __int64 v10; // rax
  const __m128i *v11; // r14
  __int64 v12; // r8
  unsigned __int64 v13; // rdx
  __int64 i; // rcx
  __int64 v15; // rdi
  __int64 v16; // rcx
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // r11
  signed __int64 v20; // r12
  __int8 v21; // al
  int v22; // edx
  int v23; // r9d
  __int64 *v24; // rdi
  unsigned int v25; // ecx
  unsigned int v26; // eax
  __int64 *v27; // rax
  int v28; // eax
  unsigned __int64 v29; // rcx
  int v30; // r11d
  unsigned int v31; // edx
  __int64 v32; // r10
  __int64 v33; // rax
  __int64 v34; // r15
  __int8 v35; // al
  int v36; // r9d
  __int64 v37; // rdx
  unsigned __int64 v38; // r12
  int v39; // edx
  __int64 v40; // r12
  __int64 *v41; // rax
  __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // r12
  __int64 v47; // r8
  int v48; // r11d
  unsigned __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // r9
  __int64 v52; // r13
  __int64 v53; // r15
  unsigned __int16 v54; // ax
  __int64 v55; // r13
  int v56; // eax
  int v57; // r15d
  __int64 v58; // rdi
  _QWORD *v59; // rcx
  int v60; // edx
  __int64 v61; // r8
  _QWORD *v62; // r13
  _QWORD *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rax
  int *j; // r15
  __int64 v68; // r12
  int v69; // r9d
  __int64 v70; // r14
  unsigned __int64 v71; // rdx
  unsigned int v72; // eax
  __int64 v73; // r13
  __int64 v74; // r10
  unsigned int v75; // eax
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rax
  __m128i *v79; // rax
  __m128i v80; // xmm3
  unsigned int v81; // r15d
  __int64 v82; // rcx
  unsigned int v83; // r15d
  __int64 v84; // rsi
  int v85; // eax
  __int64 v86; // rax
  __int64 v87; // r11
  __int64 v88; // rdx
  __int64 v89; // rsi
  __int64 v90; // rsi
  _QWORD *v91; // rcx
  _QWORD *v92; // rdx
  _QWORD *v93; // rsi
  _QWORD *v94; // rax
  __int64 v95; // rdx
  _QWORD *v96; // rdi
  _QWORD *v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // rsi
  int v100; // r10d
  __int64 v101; // [rsp+8h] [rbp-438h]
  __int64 v102; // [rsp+10h] [rbp-430h]
  int v103; // [rsp+1Ch] [rbp-424h]
  int v104; // [rsp+1Ch] [rbp-424h]
  _BYTE *v105; // [rsp+28h] [rbp-418h]
  __int64 v106; // [rsp+28h] [rbp-418h]
  unsigned int v108; // [rsp+30h] [rbp-410h]
  __int64 v109; // [rsp+38h] [rbp-408h]
  int v110; // [rsp+38h] [rbp-408h]
  int v111; // [rsp+38h] [rbp-408h]
  int v112; // [rsp+40h] [rbp-400h]
  __int64 v113; // [rsp+40h] [rbp-400h]
  __int64 v114; // [rsp+40h] [rbp-400h]
  int v115; // [rsp+40h] [rbp-400h]
  __int64 v116; // [rsp+40h] [rbp-400h]
  __int64 v117; // [rsp+40h] [rbp-400h]
  __int64 v118; // [rsp+40h] [rbp-400h]
  __int64 v119; // [rsp+48h] [rbp-3F8h]
  int v120; // [rsp+48h] [rbp-3F8h]
  __int64 v121; // [rsp+48h] [rbp-3F8h]
  __int64 v122; // [rsp+48h] [rbp-3F8h]
  _QWORD *v123; // [rsp+48h] [rbp-3F8h]
  _QWORD *v124; // [rsp+48h] [rbp-3F8h]
  __int64 *v125; // [rsp+50h] [rbp-3F0h] BYREF
  __int64 v126; // [rsp+58h] [rbp-3E8h]
  _BYTE v127[32]; // [rsp+60h] [rbp-3E0h] BYREF
  _BYTE *v128; // [rsp+80h] [rbp-3C0h] BYREF
  __int64 v129; // [rsp+88h] [rbp-3B8h]
  _BYTE v130[224]; // [rsp+90h] [rbp-3B0h] BYREF
  __m128i v131; // [rsp+170h] [rbp-2D0h] BYREF
  __m128i v132; // [rsp+180h] [rbp-2C0h] BYREF
  __m128i v133; // [rsp+190h] [rbp-2B0h] BYREF
  __int64 v134; // [rsp+1A0h] [rbp-2A0h]
  int v135; // [rsp+1A8h] [rbp-298h]
  __int64 v136; // [rsp+1B0h] [rbp-290h]
  _QWORD *v137; // [rsp+1B8h] [rbp-288h]
  __int64 v138; // [rsp+1C0h] [rbp-280h]
  unsigned int v139; // [rsp+1C8h] [rbp-278h]
  _QWORD *v140; // [rsp+1D0h] [rbp-270h]
  __int64 v141; // [rsp+1D8h] [rbp-268h]
  _QWORD v142[3]; // [rsp+1E0h] [rbp-260h] BYREF
  _BYTE *v143; // [rsp+1F8h] [rbp-248h]
  __int64 v144; // [rsp+200h] [rbp-240h]
  _BYTE v145[568]; // [rsp+208h] [rbp-238h] BYREF

  v6 = *(_QWORD *)(a1 + 72);
  v128 = v130;
  v7 = *(_QWORD *)(a1 + 32);
  v129 = 0x400000000LL;
  v8 = *(unsigned int *)(*(_QWORD *)(v6 + 8) + 112LL);
  if ( (int)v8 >= 0 )
  {
    v9 = *(const __m128i **)(*(_QWORD *)(v7 + 272) + 8 * v8);
    if ( v9 )
      goto LABEL_3;
LABEL_127:
    v99 = *(_QWORD *)(v6 + 16);
    v66 = *(_QWORD *)v99 + 4LL * *(unsigned int *)(v6 + 64);
    v122 = *(_QWORD *)v99 + 4LL * *(unsigned int *)(v99 + 8);
    if ( v66 == v122 )
      return;
    goto LABEL_56;
  }
  v9 = *(const __m128i **)(*(_QWORD *)(v7 + 24) + 16 * (v8 & 0x7FFFFFFF) + 8);
  if ( !v9 )
    goto LABEL_127;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_3:
        while ( 1 )
        {
          v10 = v9[1].m128i_u64[0];
          v11 = v9;
          v9 = (const __m128i *)v9[2].m128i_i64[0];
          if ( **(_WORD **)(v10 + 16) != 12 )
            break;
          sub_1E310D0((__int64)v11, 0);
          if ( !v9 )
            goto LABEL_33;
        }
        v12 = *(_QWORD *)(a1 + 16);
        v13 = v10;
        for ( i = *(_QWORD *)(v12 + 272); (*(_BYTE *)(v13 + 46) & 4) != 0; v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL )
          ;
        v15 = *(_QWORD *)(i + 368);
        v16 = *(unsigned int *)(i + 384);
        if ( !(_DWORD)v16 )
          goto LABEL_78;
        v17 = (v16 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( v13 != *v18 )
        {
          v85 = 1;
          while ( v19 != -8 )
          {
            v100 = v85 + 1;
            v17 = (v16 - 1) & (v85 + v17);
            v18 = (__int64 *)(v15 + 16LL * v17);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_8;
            v85 = v100;
          }
LABEL_78:
          v18 = (__int64 *)(v15 + 16 * v16);
        }
LABEL_8:
        v20 = v18[1];
        v21 = v11->m128i_i8[4];
        if ( (v11->m128i_i8[3] & 0x10) != 0 || (v21 & 1) != 0 )
          v20 = ((v21 & 4) == 0 ? 4LL : 2LL) | v20 & 0xFFFFFFFFFFFFFFF8LL;
        v22 = *(_DWORD *)(a1 + 388);
        if ( !v22 )
        {
LABEL_72:
          v23 = 0;
          goto LABEL_18;
        }
        v23 = *(_DWORD *)(a1 + 384);
        v24 = (__int64 *)(a1 + 200);
        v25 = *(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v20 >> 1) & 3;
        v26 = *(_DWORD *)((*(_QWORD *)(a1 + 200) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a1 + 200) >> 1) & 3;
        if ( v23 )
        {
          if ( v26 > v25 )
            goto LABEL_72;
          v27 = (__int64 *)(a1 + 8LL * (unsigned int)(v22 - 1) + 296);
        }
        else
        {
          if ( v26 > v25 )
            goto LABEL_72;
          v27 = &v24[2 * (unsigned int)(v22 - 1) + 1];
        }
        if ( v25 >= (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) )
          goto LABEL_72;
        if ( v23 )
        {
          v119 = *(_QWORD *)(a1 + 16);
          v28 = sub_1F15FF0((__int64)v24, v20, 0);
          v12 = v119;
          v23 = v28;
        }
        else
        {
          if ( v25 < (*(_DWORD *)((*(_QWORD *)(a1 + 208) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                    | (unsigned int)(*(__int64 *)(a1 + 208) >> 1) & 3) )
          {
            v89 = 0;
          }
          else
          {
            LODWORD(v89) = 0;
            do
              v89 = (unsigned int)(v89 + 1);
            while ( v25 >= (*(_DWORD *)((v24[2 * v89 + 1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                          | (unsigned int)(v24[2 * v89 + 1] >> 1) & 3) );
            v24 += 2 * v89;
          }
          if ( v25 >= (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) )
            v23 = *(_DWORD *)(a1 + 4 * v89 + 344);
        }
LABEL_18:
        v29 = *(unsigned int *)(v12 + 408);
        v30 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                        + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + v23));
        v31 = v30 & 0x7FFFFFFF;
        v32 = v30 & 0x7FFFFFFF;
        v33 = 8 * v32;
        if ( (v30 & 0x7FFFFFFFu) >= (unsigned int)v29 || (v34 = *(_QWORD *)(*(_QWORD *)(v12 + 400) + 8LL * v31)) == 0 )
        {
          v83 = v31 + 1;
          if ( (unsigned int)v29 < v31 + 1 )
          {
            v88 = v83;
            if ( v83 >= v29 )
            {
              if ( v83 > v29 )
              {
                if ( v83 > (unsigned __int64)*(unsigned int *)(v12 + 412) )
                {
                  v101 = v30 & 0x7FFFFFFF;
                  v104 = v23;
                  v111 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                                   + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + v23));
                  v118 = v12;
                  sub_16CD150(v12 + 400, (const void *)(v12 + 416), v83, 8, v12, v23);
                  v12 = v118;
                  v32 = v101;
                  v33 = 8 * v101;
                  v23 = v104;
                  v29 = *(unsigned int *)(v118 + 408);
                  v30 = v111;
                  v88 = v83;
                }
                v84 = *(_QWORD *)(v12 + 400);
                v96 = (_QWORD *)(v84 + 8 * v88);
                v97 = (_QWORD *)(v84 + 8 * v29);
                v98 = *(_QWORD *)(v12 + 416);
                if ( v96 != v97 )
                {
                  do
                    *v97++ = v98;
                  while ( v96 != v97 );
                  v84 = *(_QWORD *)(v12 + 400);
                }
                *(_DWORD *)(v12 + 408) = v83;
                goto LABEL_75;
              }
            }
            else
            {
              *(_DWORD *)(v12 + 408) = v83;
            }
          }
          v84 = *(_QWORD *)(v12 + 400);
LABEL_75:
          v115 = v23;
          v109 = v32;
          v124 = (_QWORD *)v12;
          *(_QWORD *)(v84 + v33) = sub_1DBA290(v30);
          v34 = *(_QWORD *)(v124[50] + 8 * v109);
          sub_1DBB110(v124, v34);
          v23 = v115;
        }
        v120 = v23;
        sub_1E310D0((__int64)v11, *(_DWORD *)(v34 + 112));
        if ( !a2 )
          goto LABEL_32;
        v35 = v11->m128i_i8[4];
        if ( (v35 & 1) != 0 )
          goto LABEL_32;
        v36 = v120;
        if ( (v11->m128i_i8[3] & 0x10) == 0 )
        {
          v44 = 1;
          v43 = v20 & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_30;
        }
        if ( (v11->m128i_i32[0] & 0xFFF00) == 0 && (v35 & 4) == 0 )
          goto LABEL_32;
        v37 = v20;
        v38 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        v39 = (v37 >> 1) & 3;
        v40 = v39 ? (2LL * (v39 - 1)) | v38 : *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v112 = v120;
        v121 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
        v41 = (__int64 *)sub_1DB3C70((__int64 *)v121, v40);
        if ( v41 == (__int64 *)(*(_QWORD *)v121 + 24LL * *(unsigned int *)(v121 + 8)) )
          goto LABEL_32;
        v42 = *v41;
        v36 = v112;
        v43 = v40 & 0xFFFFFFFFFFFFFFF8LL;
        v44 = (v40 >> 1) & 3;
        if ( (*(_DWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v42 >> 1) & 3) > ((unsigned int)v44
                                                                                              | *(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
          goto LABEL_32;
        if ( v44 == 3 )
          break;
LABEL_30:
        v45 = (2 * v44 + 2) | v43;
        if ( *(_QWORD *)(v34 + 104) )
          goto LABEL_31;
LABEL_91:
        sub_1DC5C40(
          (_QWORD *)(a1 + 664LL * ((*(_DWORD *)(a1 + 84) != 0) & (unsigned __int8)(v36 != 0)) + 432),
          v34,
          v45,
          0,
          0,
          0);
        if ( !v9 )
          goto LABEL_33;
      }
      v45 = *(_QWORD *)(v43 + 8) & 0xFFFFFFFFFFFFFFF9LL;
      if ( !*(_QWORD *)(v34 + 104) )
        goto LABEL_91;
LABEL_31:
      if ( (v11->m128i_i8[3] & 0x10) == 0 )
        break;
LABEL_32:
      if ( !v9 )
        goto LABEL_33;
    }
    v131 = _mm_loadu_si128(v11);
    v132 = _mm_loadu_si128(v11 + 1);
    v77 = v11[2].m128i_i64[0];
    v133.m128i_i32[2] = v36;
    v133.m128i_i64[0] = v77;
    v78 = (unsigned int)v129;
    v134 = v45;
    if ( (unsigned int)v129 >= HIDWORD(v129) )
    {
      sub_16CD150((__int64)&v128, v130, 0, 56, a5, v36);
      v78 = (unsigned int)v129;
    }
    v79 = (__m128i *)&v128[56 * v78];
    *v79 = _mm_loadu_si128(&v131);
    v80 = _mm_loadu_si128(&v132);
    LODWORD(v129) = v129 + 1;
    v79[1] = v80;
    v79[2] = _mm_loadu_si128(&v133);
    v79[3].m128i_i64[0] = v134;
  }
  while ( v9 );
LABEL_33:
  v46 = (unsigned __int64)v128;
  v105 = &v128[56 * (unsigned int)v129];
  if ( v128 != v105 )
  {
    while ( 1 )
    {
      v47 = *(_QWORD *)(a1 + 16);
      v48 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                      + 4LL * (unsigned int)(*(_DWORD *)(v46 + 40) + *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)));
      v49 = *(unsigned int *)(v47 + 408);
      v50 = v48 & 0x7FFFFFFF;
      v51 = v48 & 0x7FFFFFFF;
      v52 = 8 * v51;
      if ( (v48 & 0x7FFFFFFFu) >= (unsigned int)v49 )
        break;
      v53 = *(_QWORD *)(*(_QWORD *)(v47 + 400) + 8LL * v50);
      if ( !v53 )
        break;
LABEL_36:
      v131 = 0u;
      v140 = v142;
      v132 = 0u;
      v143 = v145;
      v133 = 0u;
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v137 = 0;
      v138 = 0;
      v139 = 0;
      v141 = 0;
      v142[0] = 0;
      v142[1] = 0;
      v144 = 0x1000000000LL;
      v54 = (*(_DWORD *)v46 >> 8) & 0xFFF;
      if ( !v54 )
      {
        v56 = sub_1E69F40(*(_QWORD *)(a1 + 32), *(_DWORD *)(v46 + 8));
        v55 = *(_QWORD *)(v53 + 104);
        if ( v55 )
        {
LABEL_38:
          v113 = v53;
          v57 = v56;
          do
          {
            if ( (*(_DWORD *)(v55 + 112) & v57) != 0 )
            {
              if ( *(_DWORD *)(v55 + 8) )
              {
                sub_1DC3BD0(
                  &v131,
                  *(_QWORD *)(*(_QWORD *)(a1 + 24) + 256LL),
                  *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL),
                  *(_QWORD *)(a1 + 40),
                  *(_QWORD *)(a1 + 16) + 296LL,
                  *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL));
                v58 = *(_QWORD *)(a1 + 16);
                v59 = *(_QWORD **)(a1 + 32);
                v60 = *(_DWORD *)(v55 + 112);
                v125 = (__int64 *)v127;
                v61 = *(_QWORD *)(v58 + 272);
                v126 = 0x400000000LL;
                sub_1DB4D80(v113, (__int64)&v125, v60, v59, v61);
                sub_1DC5C40(&v131, v55, *(_QWORD *)(v46 + 48), 0, v125, (unsigned int)v126);
                if ( v125 != (__int64 *)v127 )
                  _libc_free((unsigned __int64)v125);
              }
            }
            v55 = *(_QWORD *)(v55 + 104);
          }
          while ( v55 );
        }
        if ( v143 != v145 )
          _libc_free((unsigned __int64)v143);
        if ( v140 != v142 )
          _libc_free((unsigned __int64)v140);
        goto LABEL_48;
      }
      v55 = *(_QWORD *)(v53 + 104);
      v56 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 248LL) + 4LL * v54);
      if ( v55 )
        goto LABEL_38;
LABEL_48:
      if ( v139 )
      {
        v62 = v137;
        v63 = &v137[7 * v139];
        do
        {
          if ( *v62 != -16 && *v62 != -8 )
          {
            _libc_free(v62[4]);
            _libc_free(v62[1]);
          }
          v62 += 7;
        }
        while ( v63 != v62 );
      }
      v46 += 56LL;
      j___libc_free_0(v137);
      _libc_free(v133.m128i_u64[1]);
      if ( (_BYTE *)v46 == v105 )
        goto LABEL_55;
    }
    v81 = v50 + 1;
    if ( (unsigned int)v49 >= v50 + 1 )
      goto LABEL_70;
    v86 = v81;
    if ( v81 < v49 )
    {
      *(_DWORD *)(v47 + 408) = v81;
      goto LABEL_70;
    }
    if ( v81 <= v49 )
    {
LABEL_70:
      v82 = *(_QWORD *)(v47 + 400);
    }
    else
    {
      if ( v81 > (unsigned __int64)*(unsigned int *)(v47 + 412) )
      {
        v102 = v48 & 0x7FFFFFFF;
        v103 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                         + 4LL * (unsigned int)(*(_DWORD *)(v46 + 40) + *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)));
        v117 = *(_QWORD *)(a1 + 16);
        sub_16CD150(v47 + 400, (const void *)(v47 + 416), v81, 8, v47, v51);
        v47 = v117;
        v51 = v102;
        v48 = v103;
        v86 = v81;
        v49 = *(unsigned int *)(v117 + 408);
      }
      v82 = *(_QWORD *)(v47 + 400);
      v93 = (_QWORD *)(v82 + 8 * v86);
      v94 = (_QWORD *)(v82 + 8 * v49);
      v95 = *(_QWORD *)(v47 + 416);
      if ( v93 != v94 )
      {
        do
          *v94++ = v95;
        while ( v93 != v94 );
        v82 = *(_QWORD *)(v47 + 400);
      }
      *(_DWORD *)(v47 + 408) = v81;
    }
    v114 = v51;
    v123 = (_QWORD *)v47;
    *(_QWORD *)(v82 + v52) = sub_1DBA290(v48);
    v53 = *(_QWORD *)(v123[50] + 8 * v114);
    sub_1DBB110(v123, v53);
    goto LABEL_36;
  }
LABEL_55:
  v64 = *(_QWORD *)(a1 + 72);
  v65 = *(_QWORD *)(v64 + 16);
  v66 = *(_QWORD *)v65 + 4LL * *(unsigned int *)(v64 + 64);
  v122 = *(_QWORD *)v65 + 4LL * *(unsigned int *)(v65 + 8);
  if ( v122 != v66 )
  {
LABEL_56:
    for ( j = (int *)v66; (int *)v122 != j; ++j )
    {
      v69 = *j;
      v70 = *(_QWORD *)(a1 + 16);
      v71 = *(unsigned int *)(v70 + 408);
      v72 = *j & 0x7FFFFFFF;
      v73 = v72;
      v74 = 8LL * v72;
      if ( v72 < (unsigned int)v71 )
      {
        v68 = *(_QWORD *)(*(_QWORD *)(v70 + 400) + 8LL * v72);
        if ( v68 )
          goto LABEL_58;
      }
      v75 = v72 + 1;
      if ( (unsigned int)v71 < v75 )
      {
        v87 = v75;
        if ( v75 >= v71 )
        {
          if ( v75 > v71 )
          {
            if ( v75 > (unsigned __int64)*(unsigned int *)(v70 + 412) )
            {
              v106 = v74;
              v108 = v75;
              v110 = *j;
              v116 = v75;
              sub_16CD150(v70 + 400, (const void *)(v70 + 416), v75, 8, a5, v69);
              v71 = *(unsigned int *)(v70 + 408);
              v74 = v106;
              v75 = v108;
              v69 = v110;
              v87 = v116;
            }
            v76 = *(_QWORD *)(v70 + 400);
            v90 = *(_QWORD *)(v70 + 416);
            v91 = (_QWORD *)(v76 + 8 * v87);
            v92 = (_QWORD *)(v76 + 8 * v71);
            if ( v91 != v92 )
            {
              do
                *v92++ = v90;
              while ( v91 != v92 );
              v76 = *(_QWORD *)(v70 + 400);
            }
            *(_DWORD *)(v70 + 408) = v75;
            goto LABEL_64;
          }
        }
        else
        {
          *(_DWORD *)(v70 + 408) = v75;
        }
      }
      v76 = *(_QWORD *)(v70 + 400);
LABEL_64:
      *(_QWORD *)(v74 + v76) = sub_1DBA290(v69);
      v68 = *(_QWORD *)(*(_QWORD *)(v70 + 400) + 8 * v73);
      sub_1DBB110((_QWORD *)v70, v68);
LABEL_58:
      if ( *(_QWORD *)(v68 + 104) )
      {
        *(_DWORD *)(v68 + 72) = 0;
        *(_DWORD *)(v68 + 8) = 0;
        sub_1DB4C70(v68);
        sub_1DBEDA0(*(_QWORD **)(a1 + 16), v68);
      }
    }
  }
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
}
