// Function: sub_2863070
// Address: 0x2863070
//
void __fastcall sub_2863070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 v7; // r12
  unsigned __int64 v8; // rdx
  __int64 v9; // r8
  size_t v10; // r13
  int v11; // ebx
  _BYTE *v12; // rdi
  bool v13; // zf
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rsi
  char *v18; // rax
  char v19; // dl
  __int64 v20; // rbx
  __int16 v21; // ax
  __int64 v22; // rax
  __int64 v23; // rbx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rsi
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _QWORD *v33; // r14
  __int64 *v34; // rbx
  __int64 v35; // r12
  unsigned __int8 v36; // al
  unsigned int v37; // ecx
  __int64 v38; // r15
  __int64 v39; // rsi
  unsigned __int64 v40; // rax
  int v41; // ecx
  unsigned int v42; // ecx
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  int v45; // eax
  __int64 *v46; // rax
  __int64 v47; // rsi
  __int64 *v48; // rbx
  __int64 v49; // r8
  __m128i v50; // xmm1
  __int64 v51; // r10
  __int64 v52; // r9
  __int64 v53; // rcx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rsi
  int v56; // edx
  __int64 v57; // r13
  const __m128i *v58; // rax
  __m128i *v59; // r13
  __int64 v60; // r13
  char v61; // al
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rsi
  unsigned __int64 v65; // r14
  unsigned __int64 v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r13
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 *v71; // r13
  __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // r14
  __int64 v75; // rsi
  __int64 *v76; // rax
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rsi
  __int64 v83; // rax
  __int64 v84; // r8
  unsigned __int64 v85; // rcx
  int v86; // ecx
  unsigned int v87; // ecx
  unsigned __int64 v88; // rdx
  __int64 v89; // rdi
  __int64 v90; // [rsp+0h] [rbp-280h]
  __int64 v91; // [rsp+0h] [rbp-280h]
  __int64 v92; // [rsp+8h] [rbp-278h]
  __int64 v93; // [rsp+8h] [rbp-278h]
  __int64 v94; // [rsp+8h] [rbp-278h]
  __int64 v95; // [rsp+8h] [rbp-278h]
  __int64 v96; // [rsp+8h] [rbp-278h]
  __int64 v97; // [rsp+10h] [rbp-270h]
  __int64 v98; // [rsp+10h] [rbp-270h]
  const __m128i *v99; // [rsp+10h] [rbp-270h]
  __int64 v100; // [rsp+10h] [rbp-270h]
  __int64 v101; // [rsp+10h] [rbp-270h]
  __int64 *v102; // [rsp+18h] [rbp-268h]
  unsigned int v103; // [rsp+18h] [rbp-268h]
  __int64 v104; // [rsp+20h] [rbp-260h]
  __int64 v105; // [rsp+20h] [rbp-260h]
  __int64 *v106; // [rsp+20h] [rbp-260h]
  __int64 v107; // [rsp+28h] [rbp-258h]
  __int64 v108; // [rsp+38h] [rbp-248h] BYREF
  __m128i v109; // [rsp+40h] [rbp-240h]
  __int64 v110; // [rsp+50h] [rbp-230h] BYREF
  __m128i v111; // [rsp+58h] [rbp-228h] BYREF
  _BYTE *v112; // [rsp+70h] [rbp-210h] BYREF
  __int64 v113; // [rsp+78h] [rbp-208h]
  _BYTE v114[64]; // [rsp+80h] [rbp-200h] BYREF
  __int128 v115; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v116; // [rsp+D0h] [rbp-1B0h]
  _OWORD *v117; // [rsp+D8h] [rbp-1A8h]
  __int128 v118; // [rsp+E0h] [rbp-1A0h]
  _OWORD v119[2]; // [rsp+F0h] [rbp-190h] BYREF
  __int64 v120; // [rsp+118h] [rbp-168h]
  __int64 v121; // [rsp+120h] [rbp-160h]
  char v122; // [rsp+128h] [rbp-158h]
  __int64 v123; // [rsp+130h] [rbp-150h] BYREF
  char *v124; // [rsp+138h] [rbp-148h]
  __int64 v125; // [rsp+140h] [rbp-140h]
  int v126; // [rsp+148h] [rbp-138h]
  unsigned __int8 v127; // [rsp+14Ch] [rbp-134h]
  char v128; // [rsp+150h] [rbp-130h] BYREF

  v6 = &v112;
  v7 = a1;
  v8 = *(unsigned int *)(a1 + 36320);
  v112 = v114;
  v113 = 0x800000000LL;
  v9 = *(_QWORD *)(a1 + 36312);
  v10 = 8 * v8;
  v11 = v8;
  if ( v8 > 8 )
  {
    v107 = *(_QWORD *)(a1 + 36312);
    sub_C8D5F0((__int64)&v112, v114, v8, 8u, v9, a6);
    v9 = v107;
    v12 = &v112[8 * (unsigned int)v113];
  }
  else
  {
    v12 = v114;
    if ( !v10 )
      goto LABEL_3;
  }
  memcpy(v12, (const void *)v9, v10);
  v12 = v112;
  LODWORD(v10) = v113;
LABEL_3:
  v13 = *(_DWORD *)(v7 + 72) == 1;
  v123 = 0;
  LODWORD(v113) = v11 + v10;
  v14 = v11 + v10;
  v124 = &v128;
  v125 = 32;
  v126 = 0;
  v127 = 1;
  if ( v13 )
    goto LABEL_21;
  v15 = 1;
  if ( !(v11 + (_DWORD)v10) )
    goto LABEL_21;
  while ( 2 )
  {
    v16 = v14;
    v17 = *(_QWORD *)&v12[8 * v14 - 8];
    LODWORD(v113) = v14 - 1;
    v108 = v17;
    if ( !(_BYTE)v15 )
      goto LABEL_15;
    v18 = v124;
    v16 = (__int64)&v124[8 * HIDWORD(v125)];
    if ( v124 != (char *)v16 )
    {
      while ( v17 != *(_QWORD *)v18 )
      {
        v18 += 8;
        if ( (char *)v16 == v18 )
          goto LABEL_28;
      }
      goto LABEL_10;
    }
LABEL_28:
    if ( HIDWORD(v125) < (unsigned int)v125 )
    {
      ++HIDWORD(v125);
      *(_QWORD *)v16 = v17;
      v15 = v127;
      ++v123;
    }
    else
    {
LABEL_15:
      sub_C8CC70((__int64)&v123, v17, v16, v15, v9, a6);
      v15 = v127;
      if ( !v19 )
        goto LABEL_10;
    }
    v20 = v108;
    v21 = *(_WORD *)(v108 + 24);
    if ( (unsigned __int16)(v21 - 5) <= 1u || (unsigned __int16)(v21 - 8) <= 5u )
    {
      sub_28555C0(
        (__int64)v6,
        &v112[8 * (unsigned int)v113],
        *(char **)(v108 + 32),
        (char *)(*(_QWORD *)(v108 + 32) + 8LL * *(_QWORD *)(v108 + 40)));
      v14 = v113;
      v15 = v127;
      if ( (_DWORD)v113 )
        goto LABEL_12;
      break;
    }
    if ( (unsigned __int16)(v21 - 2) <= 2u )
    {
      v22 = (unsigned int)v113;
      v23 = *(_QWORD *)(v108 + 32);
      v24 = (unsigned int)v113 + 1LL;
      if ( v24 <= HIDWORD(v113) )
      {
LABEL_27:
        *(_QWORD *)&v112[8 * v22] = v23;
        v15 = v127;
        v14 = v113 + 1;
        LODWORD(v113) = v113 + 1;
        goto LABEL_11;
      }
LABEL_34:
      sub_C8D5F0((__int64)v6, v114, v24, 8u, v9, a6);
      v22 = (unsigned int)v113;
      goto LABEL_27;
    }
    if ( v21 == 7 )
    {
      v25 = (unsigned int)v113;
      v26 = *(_QWORD *)(v108 + 32);
      v27 = (unsigned int)v113 + 1LL;
      if ( v27 > HIDWORD(v113) )
      {
        sub_C8D5F0((__int64)v6, v114, v27, 8u, v9, a6);
        v25 = (unsigned int)v113;
      }
      *(_QWORD *)&v112[8 * v25] = v26;
      v23 = *(_QWORD *)(v20 + 40);
      v22 = (unsigned int)(v113 + 1);
      v24 = v22 + 1;
      LODWORD(v113) = v113 + 1;
      if ( v22 + 1 <= (unsigned __int64)HIDWORD(v113) )
        goto LABEL_27;
      goto LABEL_34;
    }
    if ( v21 != 15 )
    {
LABEL_10:
      v14 = v113;
      goto LABEL_11;
    }
    v28 = *(_QWORD *)(v108 - 8);
    if ( *(_BYTE *)v28 <= 0x1Cu )
    {
      if ( *(_BYTE *)v28 <= 0x15u )
        goto LABEL_10;
    }
    else
    {
      v29 = *(_QWORD *)(v7 + 56);
      v30 = *(_QWORD *)(v28 + 40);
      if ( *(_BYTE *)(v29 + 84) )
      {
        v31 = *(_QWORD **)(v29 + 64);
        v32 = &v31[*(unsigned int *)(v29 + 76)];
        if ( v31 == v32 )
          goto LABEL_44;
        while ( v30 != *v31 )
        {
          if ( v32 == ++v31 )
            goto LABEL_44;
        }
        goto LABEL_10;
      }
      if ( sub_C8CA60(v29 + 56, v30) )
      {
        v15 = v127;
        goto LABEL_10;
      }
    }
LABEL_44:
    v33 = *(_QWORD **)(v28 + 16);
    if ( !v33 )
      goto LABEL_84;
    v102 = (__int64 *)v20;
    v34 = (__int64 *)v7;
    v104 = (__int64)v6;
    while ( 1 )
    {
      v35 = v33[3];
      v36 = *(_BYTE *)v35;
      if ( *(_BYTE *)v35 > 0x1Cu )
      {
        v37 = v36 - 39;
        if ( v37 > 0x38 || ((1LL << v37) & 0x100060000000001LL) == 0 )
        {
          v38 = *(_QWORD *)(v35 + 40);
          v39 = **(_QWORD **)(v34[7] + 32);
          if ( *(_QWORD *)(v39 + 72) == *(_QWORD *)(v38 + 72) )
          {
            if ( v36 == 84 )
            {
              v38 = *(_QWORD *)(*(_QWORD *)(v35 - 8)
                              + 32LL * *(unsigned int *)(v35 + 72)
                              + 8LL * (unsigned int)sub_BD2910((__int64)v33));
              v39 = **(_QWORD **)(v34[7] + 32);
            }
            if ( (unsigned __int8)sub_B19720(v34[2], v39, v38) )
            {
              v40 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v40 == v38 + 48 )
                goto LABEL_108;
              if ( !v40 )
                BUG();
              v41 = *(unsigned __int8 *)(v40 - 24);
              if ( (unsigned int)(v41 - 30) > 0xA )
LABEL_108:
                BUG();
              v42 = v41 - 39;
              if ( v42 > 0x38 || ((1LL << v42) & 0x100060000000001LL) == 0 )
              {
                if ( *(_BYTE *)v35 == 84 && (*(_DWORD *)(v35 + 4) & 0x7FFFFFF) != 0 )
                {
                  v82 = *(_QWORD *)(v35 - 8);
                  v83 = 0;
                  do
                  {
                    if ( *v33 == *(_QWORD *)(v82 + 4 * v83) )
                    {
                      v84 = *(_QWORD *)(v82 + 32LL * *(unsigned int *)(v35 + 72) + v83);
                      v85 = *(_QWORD *)(v84 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v85 == v84 + 48 )
                        goto LABEL_110;
                      if ( !v85 )
                        BUG();
                      v86 = *(unsigned __int8 *)(v85 - 24);
                      v9 = (unsigned int)(v86 - 30);
                      if ( (unsigned int)v9 > 0xA )
LABEL_110:
                        BUG();
                      v87 = v86 - 39;
                      if ( v87 <= 0x38 && ((1LL << v87) & 0x100060000000001LL) != 0 )
                        goto LABEL_46;
                    }
                    v83 += 8;
                  }
                  while ( v83 != 8LL * (*(_DWORD *)(v35 + 4) & 0x7FFFFFF) );
                }
                v43 = *(_QWORD *)(v35 + 40);
                v44 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v44 == v43 + 48 )
                  goto LABEL_105;
                if ( !v44 )
                  BUG();
                if ( (unsigned int)*(unsigned __int8 *)(v44 - 24) - 30 > 0xA )
LABEL_105:
                  BUG();
                if ( *(_BYTE *)(v44 - 24) != 39 )
                {
                  if ( !sub_D97040(v34[1], *(_QWORD *)(v35 + 8)) )
                    goto LABEL_65;
                  v76 = sub_DD8400(v34[1], v35);
                  if ( *((_WORD *)v76 + 12) == 15 )
                    break;
                }
              }
            }
          }
        }
      }
LABEL_46:
      v33 = (_QWORD *)v33[1];
      if ( !v33 )
      {
        v6 = (_QWORD *)v104;
        v7 = (__int64)v34;
        goto LABEL_84;
      }
    }
    if ( v76 == v102 )
    {
      v77 = sub_DA3860((_QWORD *)v34[1], v35);
      sub_D9B3A0(v104, (__int64)v77, v78, v79, v80, v81);
      goto LABEL_46;
    }
LABEL_65:
    if ( *(_BYTE *)v35 == 82 )
    {
      v45 = sub_BD2910((__int64)v33);
      v92 = v34[7];
      v97 = v34[1];
      v46 = sub_DD8400(v97, *(_QWORD *)(v35 + 32LL * (v45 == 0) - 64));
      if ( sub_DAE0A0(v97, (__int64)v46, v92) )
        goto LABEL_46;
    }
    v98 = v35;
    v7 = (__int64)v34;
    v47 = (__int64)v34;
    v48 = v102;
    v6 = (_QWORD *)v104;
    sub_2857EE0((__int64)&v110, v47, &v108, 0, 0, 0xFFFFFFFFLL);
    v50 = _mm_loadu_si128(&v111);
    v118 = 0;
    v51 = v98;
    v103 = v110;
    v52 = *(_QWORD *)(v7 + 1320) + 2184 * v110;
    v116 = 0;
    *(_QWORD *)&v118 = 2;
    v117 = v119;
    DWORD2(v118) = 0;
    BYTE12(v118) = 1;
    v115 = 0;
    memset(v119, 0, sizeof(v119));
    v53 = *(unsigned int *)(v52 + 64);
    v54 = *(unsigned int *)(v52 + 68);
    v109 = v50;
    v55 = v53 + 1;
    v56 = v53;
    if ( v53 + 1 > v54 )
    {
      v88 = *(_QWORD *)(v52 + 56);
      v89 = v52 + 56;
      if ( v88 > (unsigned __int64)&v115 || (v91 = *(_QWORD *)(v52 + 56), (unsigned __int64)&v115 >= v88 + 80 * v53) )
      {
        v96 = v52;
        sub_2851200(v89, v55, v88, v53, v49, v52);
        v52 = v96;
        v58 = (const __m128i *)&v115;
        v51 = v98;
        v53 = *(unsigned int *)(v96 + 64);
        v57 = *(_QWORD *)(v96 + 56);
        v56 = *(_DWORD *)(v96 + 64);
      }
      else
      {
        v95 = v52;
        sub_2851200(v89, v55, v88, v53, v49, v52);
        v52 = v95;
        v51 = v98;
        v57 = *(_QWORD *)(v95 + 56);
        v53 = *(unsigned int *)(v95 + 64);
        v58 = (const __m128i *)((char *)&v115 + v57 - v91);
        v56 = *(_DWORD *)(v95 + 64);
      }
    }
    else
    {
      v57 = *(_QWORD *)(v52 + 56);
      v58 = (const __m128i *)&v115;
    }
    v59 = (__m128i *)(80 * v53 + v57);
    if ( v59 )
    {
      v90 = v52;
      v93 = v51;
      v59->m128i_i64[0] = v58->m128i_i64[0];
      v99 = v58;
      v59->m128i_i64[1] = v58->m128i_i64[1];
      sub_C8CF70((__int64)v59[1].m128i_i64, &v59[3], 2, (__int64)v58[3].m128i_i64, (__int64)v58[1].m128i_i64);
      v52 = v90;
      v51 = v93;
      v59[4] = _mm_loadu_si128(v99 + 4);
      v56 = *(_DWORD *)(v90 + 64);
    }
    *(_DWORD *)(v52 + 64) = v56 + 1;
    if ( !BYTE12(v118) )
    {
      v94 = v52;
      v101 = v51;
      _libc_free((unsigned __int64)v117);
      v52 = v94;
      v51 = v101;
    }
    v100 = v52;
    v60 = *(_QWORD *)(v52 + 56) + 80LL * *(unsigned int *)(v52 + 64) - 80;
    *(_QWORD *)v60 = v51;
    *(_QWORD *)(v60 + 8) = *v33;
    *(_QWORD *)(v60 + 64) = v109.m128i_i64[0];
    *(_BYTE *)(v60 + 72) = v109.m128i_i8[8];
    v61 = sub_2855480((_QWORD *)v60, *(_QWORD *)(v7 + 56));
    v63 = v100;
    v64 = *(_QWORD *)(v100 + 752);
    *(_BYTE *)(v100 + 744) &= v61;
    if ( !v64
      || (v65 = sub_D97050(*(_QWORD *)(v7 + 8), v64),
          v66 = sub_D97050(*(_QWORD *)(v7 + 8), *(_QWORD *)(*(_QWORD *)(v60 + 8) + 8LL)),
          v63 = v100,
          v65 < v66) )
    {
      *(_QWORD *)(v63 + 752) = *(_QWORD *)(*(_QWORD *)(v60 + 8) + 8LL);
    }
    v105 = v63;
    v115 = 0u;
    LOBYTE(v116) = 0;
    *(_QWORD *)&v118 = 0;
    *((_QWORD *)&v118 + 1) = (char *)v119 + 8;
    v120 = 0;
    v121 = 0;
    v122 = 0;
    *((_QWORD *)&v119[0] + 1) = v48;
    *(_QWORD *)&v119[0] = 0x400000001LL;
    LOBYTE(v117) = 1;
    sub_2862B30(v7, v63, v103, (unsigned __int64)&v115, v62, v63);
    a6 = v105;
    if ( *((_OWORD **)&v118 + 1) != (_OWORD *)((char *)v119 + 8) )
    {
      _libc_free(*((unsigned __int64 *)&v118 + 1));
      a6 = v105;
    }
    v67 = *(unsigned int *)(v7 + 1328) - 1LL;
    v68 = *(_QWORD *)(a6 + 760) + 112LL * *(unsigned int *)(a6 + 768) - 112;
    v69 = *(_QWORD *)(v68 + 88);
    if ( v69 )
      sub_285AF50(v7 + 36280, v69, *(unsigned int *)(v7 + 1328) - 1LL);
    v70 = *(unsigned int *)(v68 + 48);
    v71 = *(__int64 **)(v68 + 40);
    v106 = &v71[v70];
    if ( v71 != v106 )
    {
      v72 = v7;
      v73 = v7 + 36280;
      v74 = v72;
      do
      {
        v75 = *v71++;
        sub_285AF50(v73, v75, v67);
      }
      while ( v106 != v71 );
      v7 = v74;
    }
LABEL_84:
    v14 = v113;
    v15 = v127;
LABEL_11:
    if ( v14 )
    {
LABEL_12:
      v12 = v112;
      continue;
    }
    break;
  }
  if ( !(_BYTE)v15 )
    _libc_free((unsigned __int64)v124);
  v12 = v112;
LABEL_21:
  if ( v12 != v114 )
    _libc_free((unsigned __int64)v12);
}
