// Function: sub_D0B5F0
// Address: 0xd0b5f0
//
__int64 __fastcall sub_D0B5F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned __int64 v10; // r15
  unsigned __int8 *v11; // r11
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // r11
  char v14; // al
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  char v19; // al
  unsigned __int8 *v20; // r11
  char v21; // al
  unsigned __int8 *v22; // r11
  unsigned __int8 *v23; // rdx
  char v24; // al
  unsigned __int8 *v25; // rdx
  char v26; // al
  char v27; // al
  __int64 v28; // r12
  char v29; // bl
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rdx
  unsigned __int64 v33; // r11
  __int64 v34; // rax
  char v35; // r12
  __m128i v36; // xmm1
  __m128i *v37; // rbx
  __int64 v38; // rax
  int v39; // edx
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rdx
  __int64 v46; // r12
  __int64 v47; // r13
  unsigned int v48; // esi
  __int64 v49; // rax
  _DWORD *v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int8 *v52; // rax
  __int64 *v53; // r8
  unsigned __int64 v54; // rcx
  int v55; // eax
  int v56; // r15d
  __int64 v57; // r14
  int v58; // ecx
  __int64 v59; // rdi
  __int64 v60; // rdx
  int v61; // eax
  unsigned __int64 *v62; // rdx
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // r8
  unsigned __int64 v65; // rsi
  unsigned __int64 v66; // r9
  __int64 v67; // r10
  int v68; // ecx
  int v69; // ecx
  int v70; // r11d
  unsigned int i; // edx
  _QWORD *v72; // rax
  unsigned int v73; // edx
  char v74; // dl
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // rdx
  __m128i *v80; // rax
  unsigned __int64 *v81; // r14
  unsigned __int64 *v82; // rbx
  __int64 v83; // rdx
  __int64 v84; // rax
  unsigned __int64 v85; // r12
  __int64 v86; // rdi
  const void *v87; // rsi
  unsigned int v88; // eax
  char v89; // [rsp+4h] [rbp-12Ch]
  unsigned int v90; // [rsp+8h] [rbp-128h]
  unsigned __int8 *v91; // [rsp+8h] [rbp-128h]
  unsigned int v92; // [rsp+8h] [rbp-128h]
  unsigned __int64 v93; // [rsp+10h] [rbp-120h]
  unsigned int v94; // [rsp+10h] [rbp-120h]
  unsigned __int8 *v95; // [rsp+10h] [rbp-120h]
  int v96; // [rsp+10h] [rbp-120h]
  __int64 v97; // [rsp+18h] [rbp-118h]
  unsigned __int8 *v98; // [rsp+20h] [rbp-110h]
  __int64 *v99; // [rsp+20h] [rbp-110h]
  __int64 v100; // [rsp+20h] [rbp-110h]
  unsigned __int8 *v101; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v102; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v103; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v104; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v105; // [rsp+28h] [rbp-108h]
  __int64 v106; // [rsp+28h] [rbp-108h]
  __int64 v107; // [rsp+28h] [rbp-108h]
  char v108; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v109; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v110; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v111; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v112; // [rsp+30h] [rbp-100h]
  unsigned __int8 v113; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v114; // [rsp+38h] [rbp-F8h]
  unsigned __int8 *v115; // [rsp+38h] [rbp-F8h]
  int v116; // [rsp+38h] [rbp-F8h]
  __int64 v117; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v118; // [rsp+48h] [rbp-E8h] BYREF
  unsigned __int64 v119; // [rsp+50h] [rbp-E0h]
  __int64 v120; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v121; // [rsp+60h] [rbp-D0h]
  __int64 v122; // [rsp+68h] [rbp-C8h]
  __m128i v123; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v124; // [rsp+80h] [rbp-B0h] BYREF
  __int64 *v125[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v126; // [rsp+A0h] [rbp-90h]
  __int64 *v127[2]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v128; // [rsp+C0h] [rbp-70h]
  __int64 *v129[2]; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v130; // [rsp+E0h] [rbp-50h]
  char v131; // [rsp+F0h] [rbp-40h]

  v118 = a3;
  v117 = a5;
  if ( a3 != -1 && a3 != 0xBFFFFFFFFFFFFFFELL && (a3 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
    return 0;
  if ( v117 != -1 && v117 != 0xBFFFFFFFFFFFFFFELL && (v117 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
    return 0;
  v10 = (unsigned __int64)sub_BD42C0((unsigned __int8 *)a2, a2);
  v11 = sub_BD42C0((unsigned __int8 *)a4, a2);
  if ( (unsigned int)*(unsigned __int8 *)v10 - 12 <= 1 || (unsigned int)*v11 - 12 <= 1 )
    return 0;
  v114 = v11;
  if ( (unsigned __int8)sub_D04110((__int64)a1, v10, (__int64)v11, a6) )
    return 3;
  if ( *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL) != 14 )
    return 0;
  if ( *(_BYTE *)(*((_QWORD *)v114 + 1) + 8LL) != 14 )
    return 0;
  v111 = v114;
  v115 = sub_98ACB0((unsigned __int8 *)v10, 6u);
  v101 = v111;
  v12 = sub_98ACB0(v111, 6u);
  v13 = v111;
  v112 = v12;
  if ( *v115 == 20 )
  {
    v14 = sub_B2F070(a1[1], *(_DWORD *)(*((_QWORD *)v115 + 1) + 8LL) >> 8);
    v13 = v101;
    if ( !v14 )
      return 0;
  }
  if ( *v112 == 20 )
  {
    v102 = v13;
    v15 = sub_B2F070(a1[1], *(_DWORD *)(*((_QWORD *)v112 + 1) + 8LL) >> 8);
    v13 = v102;
    if ( !v15 )
      return 0;
  }
  if ( v115 == v112 )
    goto LABEL_37;
  if ( *v115 == 3 && *v112 == 3 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( *(_BYTE *)(v16 + 8) == 14 )
    {
      v17 = *(_QWORD *)(a4 + 8);
      if ( *(_BYTE *)(v17 + 8) == 14 )
      {
        v18 = *(_DWORD *)(v16 + 8) >> 8;
        if ( *(_DWORD *)(v17 + 8) >> 8 == v18 && v18 == 3 )
        {
          v41 = *((_QWORD *)v115 + 3);
          if ( *(_BYTE *)(v41 + 8) == 16 && !*(_QWORD *)(v41 + 32) )
          {
            v42 = *((_QWORD *)v112 + 3);
            if ( *(_BYTE *)(v42 + 8) == 16 && !*(_QWORD *)(v42 + 32) )
              return 1;
          }
        }
      }
    }
  }
  v103 = v13;
  v19 = sub_CF7060(v115);
  v20 = v103;
  if ( v19 )
  {
    if ( (unsigned __int8)sub_CF7060(v112) )
      return 0;
    v20 = v103;
  }
  if ( *v115 == 22 )
  {
    v110 = v20;
    if ( (unsigned __int8)sub_CF70D0(v112) )
      return 0;
    v20 = v110;
  }
  if ( *v112 == 22 )
  {
    v109 = v20;
    v40 = sub_CF70D0(v115);
    v20 = v109;
    if ( v40 )
      return 0;
  }
  v104 = v20;
  v21 = sub_CF74F0(v115);
  v22 = v104;
  if ( v21 )
  {
    v23 = 0;
    if ( *v115 >= 0x1Du )
      v23 = v115;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int8 *, unsigned __int8 *, __int64))(**(_QWORD **)(a6 + 344)
                                                                                                  + 16LL))(
           *(_QWORD *)(a6 + 344),
           v112,
           v23,
           1) )
    {
      return 0;
    }
    v22 = v104;
  }
  v105 = v22;
  v24 = sub_CF74F0(v112);
  v13 = v105;
  if ( v24 )
  {
    v25 = 0;
    if ( *v112 >= 0x1Du )
      v25 = v112;
    v26 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *, unsigned __int8 *, __int64))(**(_QWORD **)(a6 + 344)
                                                                                           + 16LL))(
            *(_QWORD *)(a6 + 344),
            v115,
            v25,
            1);
    v13 = v105;
    if ( v26 )
      return 0;
  }
LABEL_37:
  v98 = v13;
  v27 = sub_B2F070(a1[1], 0);
  v28 = *a1;
  v29 = v27;
  v106 = a1[2];
  v119 = sub_D00120((unsigned __int8 *)v10, &v118, *a1, v27);
  v120 = v30;
  if ( sub_D00760(v112, v119, (unsigned __int8)v30, v28, v106, v29) )
    return 0;
  v31 = *a1;
  v107 = a1[2];
  v121 = sub_D00120(v98, &v117, *a1, v29);
  v122 = v32;
  v108 = sub_D00760(v115, v121, (unsigned __int8)v32, v31, v107, v29);
  if ( v108 )
    return 0;
  v33 = (unsigned __int64)v98;
  if ( (_BYTE)qword_4F867A8 )
  {
    v43 = sub_988050(a1[3], (__int64)v115);
    v33 = (unsigned __int64)v98;
    v44 = v43;
    v97 = v43 + 32 * v45;
    if ( v43 != v97 )
    {
      v100 = a6;
      v46 = v33;
      do
      {
        v47 = *(_QWORD *)(v44 + 16);
        if ( v47 )
        {
          v48 = *(_DWORD *)(v44 + 24);
          if ( v48 != -1 )
          {
            v125[0] = *(__int64 **)(v44 + 16);
            v49 = 0;
            if ( *(char *)(v47 + 7) < 0 )
              v49 = sub_BD2BC0(v47);
            v50 = (_DWORD *)(16LL * v48 + v49);
            v94 = v50[2];
            v90 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
            if ( **(_QWORD **)v50 == 16 && !memcmp((const void *)(*(_QWORD *)v50 + 16LL), "separate_storage", 0x10u) )
            {
              v51 = 32 * (v94 - (unsigned __int64)v90);
              v91 = *(unsigned __int8 **)(v47 + v51 + 32);
              v95 = sub_98ACB0(*(unsigned __int8 **)(v47 + v51), 6u);
              v52 = sub_98ACB0(v91, 6u);
              v53 = *(_BYTE *)(v100 + 513) ? (__int64 *)a1[4] : 0LL;
              if ( (v127[0] = v53, v129[0] = (__int64 *)v125, v129[1] = (__int64 *)v127, v112 == v52) && v115 == v95
                || v112 == v95 && v115 == v52 )
              {
                if ( a7 && (unsigned __int8)sub_98CF40((__int64)v125[0], a7, (__int64)v53, 1)
                  || (unsigned __int8)sub_D00320(v129, v10)
                  || (unsigned __int8)sub_D00320(v129, v46) )
                {
                  return 0;
                }
              }
            }
          }
        }
        v44 += 32;
      }
      while ( v44 != v97 );
      a6 = v100;
      v33 = v46;
    }
  }
  if ( v118 == -1 || v117 == -1 )
  {
    v118 = 0xBFFFFFFFFFFFFFFELL;
    v117 = 0xBFFFFFFFFFFFFFFELL;
  }
  if ( *(_DWORD *)(a6 + 352) > 0x1FFu )
    return 1;
  v34 = 4LL * *(unsigned __int8 *)(a6 + 512);
  v123.m128i_i64[0] = v34 | v10 & 0xFFFFFFFFFFFFFFFBLL;
  v123.m128i_i64[1] = v118;
  v124.m128i_i64[0] = v33 & 0xFFFFFFFFFFFFFFFBLL | v34;
  v35 = v10 > v33;
  v124.m128i_i64[1] = v117;
  if ( v10 > v33 )
  {
    v36 = _mm_loadu_si128(&v124);
    v124 = _mm_loadu_si128(&v123);
    v123 = v36;
  }
  v37 = &v123;
  v93 = v33;
  v99 = (__int64 *)(a6 + 8);
  v127[0] = 0;
  sub_D0B410((__int64)v129, a6 + 8, &v123, (__int64 *)v127);
  if ( !v131 )
  {
    v38 = v130;
    if ( *(_DWORD *)(v130 + 36) != -2 )
    {
      ++*(_DWORD *)(a6 + 356);
      v39 = *(_DWORD *)(v38 + 36);
      if ( v39 >= 0 )
        *(_DWORD *)(v38 + 36) = v39 + 1;
    }
    LODWORD(v127[0]) = *(_DWORD *)(v38 + 32);
    sub_D036D0((__int64)v127, v35);
    return LODWORD(v127[0]);
  }
  v54 = v93;
  v96 = *(_DWORD *)(a6 + 356);
  v92 = *(_DWORD *)(a6 + 376);
  v55 = sub_D09E10(a1, v10, v118, v54, v117, a6, (char *)v115, (char *)v112);
  v113 = v55;
  v89 = v55;
  v116 = v55 >> 9;
  v56 = ((unsigned int)v55 >> 8) & 1;
  sub_D091D0(v125, v99, (unsigned __int64 *)&v123);
  v57 = v126;
  v58 = *(_DWORD *)(v126 + 36);
  v59 = v126 + 32;
  if ( v58 > 0 )
  {
    v59 = v126 + 32;
    if ( v89 )
    {
      *(_DWORD *)(a6 + 356) -= v58;
      *(_DWORD *)(v57 + 32) = 1;
      sub_D036D0(v59, v35);
      while ( 1 )
      {
LABEL_87:
        v60 = *(unsigned int *)(a6 + 376);
        if ( v92 >= (unsigned int)v60 )
        {
          v116 = 0;
          v113 = 1;
          goto LABEL_97;
        }
        v61 = v60 - 1;
        v62 = (unsigned __int64 *)(*(_QWORD *)(a6 + 368) + 32 * v60 - 32);
        v63 = *v62;
        v64 = v62[1];
        v65 = v62[2];
        v66 = v62[3];
        *(_DWORD *)(a6 + 376) = v61;
        if ( (*(_BYTE *)(a6 + 16) & 1) != 0 )
        {
          v67 = a6 + 24;
          v68 = 8;
LABEL_90:
          v69 = v68 - 1;
          v70 = 1;
          for ( i = v69
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)((0xBF58476D1CE4E5B9LL * v66) >> 31)
                      ^ (484763065 * (_DWORD)v66)
                      ^ (unsigned int)v65
                      ^ (unsigned int)(v65 >> 9)
                      | ((unsigned __int64)((unsigned int)v63
                                          ^ (unsigned int)(v63 >> 9)
                                          ^ (unsigned int)((0xBF58476D1CE4E5B9LL * v64) >> 31)
                                          ^ (484763065 * (_DWORD)v64)) << 32))) >> 31)
                   ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v66) >> 31) ^ (484763065 * v66) ^ v65 ^ (v65 >> 9))));
                ;
                i = v69 & v73 )
          {
            v72 = (_QWORD *)(v67 + 40LL * i);
            if ( v63 == *v72 && v64 == v72[1] && v65 == v72[2] && v66 == v72[3] )
              break;
            if ( *v72 == -4 && v72[1] == -3 && v72[2] == -4 && v72[3] == -3 )
              goto LABEL_87;
            v73 = v70 + i;
            ++v70;
          }
          *v72 = -16;
          v72[1] = -4;
          v72[2] = -16;
          v72[3] = -4;
          v88 = *(_DWORD *)(a6 + 16);
          ++*(_DWORD *)(a6 + 20);
          *(_DWORD *)(a6 + 16) = (2 * (v88 >> 1) - 2) | v88 & 1;
        }
        else
        {
          v68 = *(_DWORD *)(a6 + 32);
          v67 = *(_QWORD *)(a6 + 24);
          if ( v68 )
            goto LABEL_90;
        }
      }
    }
  }
  *(_DWORD *)(a6 + 356) -= v58;
  v74 = *(_BYTE *)(v57 + 33);
  *(_BYTE *)(v57 + 32) = v89;
  *(_BYTE *)(v57 + 33) = v56 & 1 | v74 & 0xFE;
  *(_DWORD *)(v57 + 32) = (v116 << 9) | *(_DWORD *)(v57 + 32) & 0x1FF;
  sub_D036D0(v59, v35);
  if ( *(_DWORD *)(a6 + 356) == v96 || v89 == 1 )
  {
    v108 = v56;
LABEL_97:
    *(_DWORD *)(v57 + 36) = -2;
    LOBYTE(v56) = v108;
  }
  else
  {
    v77 = *(unsigned int *)(a6 + 376);
    v78 = v77 + 1;
    if ( v77 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 380) )
    {
      v85 = *(_QWORD *)(a6 + 368);
      v86 = a6 + 368;
      v87 = (const void *)(a6 + 384);
      if ( v85 > (unsigned __int64)&v123 || (unsigned __int64)&v123 >= v85 + 32 * v77 )
      {
        sub_C8D5F0(v86, v87, v78, 0x20u, v75, v76);
        v79 = *(_QWORD *)(a6 + 368);
        v77 = *(unsigned int *)(a6 + 376);
      }
      else
      {
        sub_C8D5F0(v86, v87, v78, 0x20u, v75, v76);
        v79 = *(_QWORD *)(a6 + 368);
        v77 = *(unsigned int *)(a6 + 376);
        v37 = (__m128i *)((char *)&v123 + v79 - v85);
      }
    }
    else
    {
      v79 = *(_QWORD *)(a6 + 368);
    }
    v80 = (__m128i *)(v79 + 32 * v77);
    *v80 = _mm_loadu_si128(v37);
    v80[1] = _mm_loadu_si128(v37 + 1);
    ++*(_DWORD *)(a6 + 376);
    *(_DWORD *)(v57 + 36) = -1;
  }
  if ( *(_DWORD *)(a6 + 352) == 1 )
  {
    v81 = *(unsigned __int64 **)(a6 + 368);
    v82 = &v81[4 * *(unsigned int *)(a6 + 376)];
    while ( v81 != v82 )
    {
      sub_D091D0(v127, v99, v81);
      if ( (*(_BYTE *)(a6 + 16) & 1) != 0 )
      {
        v83 = a6 + 24;
        v84 = 8;
      }
      else
      {
        v83 = *(_QWORD *)(a6 + 24);
        v84 = *(unsigned int *)(a6 + 32);
      }
      if ( v128 != v83 + 40 * v84 )
        *(_DWORD *)(v128 + 36) = -2;
      v81 += 4;
    }
    *(_DWORD *)(a6 + 376) = 0;
    *(_DWORD *)(a6 + 356) = 0;
  }
  return v113 | ((unsigned __int8)v56 << 8) | (unsigned int)(v116 << 9);
}
