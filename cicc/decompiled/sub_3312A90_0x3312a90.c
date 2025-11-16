// Function: sub_3312A90
// Address: 0x3312a90
//
__int64 __fastcall sub_3312A90(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  unsigned int v7; // r13d
  __m128i v9; // xmm0
  __int128 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r9
  int v19; // eax
  __int64 i; // rax
  __m128i v21; // xmm0
  unsigned __int8 v22; // r13
  __int64 v23; // r12
  __int64 v24; // r14
  unsigned __int8 v25; // al
  __int64 (__fastcall **v26)(); // rax
  __int64 v27; // r14
  int v28; // r15d
  __int64 v29; // rax
  bool v30; // zf
  __int64 v31; // rdx
  __int64 v32; // rdi
  __m128i v33; // xmm0
  __int64 v34; // r12
  int v35; // r14d
  int v36; // r15d
  __int64 v37; // rcx
  const void *v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r10
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  _QWORD *v47; // rdx
  int v48; // ecx
  __int64 *v49; // r8
  __int64 v50; // rdx
  const void **v51; // r10
  int v52; // ecx
  unsigned int v53; // eax
  unsigned int v54; // eax
  __int64 v55; // r15
  __int64 v56; // rdi
  __int64 v57; // r8
  unsigned int *v58; // rsi
  int v59; // eax
  int v60; // r9d
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  const void *v63; // rax
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  __int64 v70; // rax
  __int128 v71; // [rsp-28h] [rbp-3B0h]
  __int128 v72; // [rsp-18h] [rbp-3A0h]
  __int64 v73; // [rsp-8h] [rbp-390h]
  int v74; // [rsp+14h] [rbp-374h]
  int v75; // [rsp+28h] [rbp-360h]
  const void *v76; // [rsp+28h] [rbp-360h]
  const void *v77; // [rsp+30h] [rbp-358h]
  int v78; // [rsp+30h] [rbp-358h]
  int v79; // [rsp+30h] [rbp-358h]
  unsigned int v80; // [rsp+30h] [rbp-358h]
  __int16 v81; // [rsp+32h] [rbp-356h]
  unsigned int v82; // [rsp+38h] [rbp-350h]
  __int64 v83; // [rsp+38h] [rbp-350h]
  __int64 *v84; // [rsp+38h] [rbp-350h]
  __int64 *v85; // [rsp+38h] [rbp-350h]
  __int64 *v86; // [rsp+38h] [rbp-350h]
  __int64 *v87; // [rsp+38h] [rbp-350h]
  __int16 v88; // [rsp+3Ah] [rbp-34Eh]
  __int64 v89; // [rsp+40h] [rbp-348h]
  const void *v90; // [rsp+48h] [rbp-340h]
  unsigned __int8 v91; // [rsp+50h] [rbp-338h]
  unsigned int v92; // [rsp+58h] [rbp-330h]
  int v93; // [rsp+58h] [rbp-330h]
  unsigned __int8 v94; // [rsp+5Eh] [rbp-32Ah]
  bool v95; // [rsp+5Fh] [rbp-329h]
  __int64 v96; // [rsp+60h] [rbp-328h]
  __int64 v97; // [rsp+60h] [rbp-328h]
  __int64 v98; // [rsp+60h] [rbp-328h]
  unsigned __int8 v99; // [rsp+A2h] [rbp-2E6h] BYREF
  char v100; // [rsp+A3h] [rbp-2E5h] BYREF
  int v101; // [rsp+A4h] [rbp-2E4h] BYREF
  __int64 v102; // [rsp+A8h] [rbp-2E0h] BYREF
  __int64 v103; // [rsp+B0h] [rbp-2D8h]
  __m128i v104; // [rsp+B8h] [rbp-2D0h] BYREF
  __int128 v105; // [rsp+C8h] [rbp-2C0h] BYREF
  const void *v106; // [rsp+D8h] [rbp-2B0h] BYREF
  unsigned int v107; // [rsp+E0h] [rbp-2A8h]
  const void *v108; // [rsp+E8h] [rbp-2A0h] BYREF
  unsigned int v109; // [rsp+F0h] [rbp-298h]
  __int64 (__fastcall **v110)(); // [rsp+F8h] [rbp-290h] BYREF
  __int64 v111; // [rsp+100h] [rbp-288h]
  __int64 v112; // [rsp+108h] [rbp-280h]
  __int64 *v113; // [rsp+110h] [rbp-278h]
  unsigned __int64 v114[2]; // [rsp+118h] [rbp-270h] BYREF
  _QWORD v115[16]; // [rsp+128h] [rbp-260h] BYREF
  _BYTE *v116; // [rsp+1A8h] [rbp-1E0h] BYREF
  __int64 v117; // [rsp+1B0h] [rbp-1D8h]
  _BYTE v118[128]; // [rsp+1B8h] [rbp-1D0h] BYREF
  __int64 v119; // [rsp+238h] [rbp-150h] BYREF
  char *v120; // [rsp+240h] [rbp-148h]
  __int64 v121; // [rsp+248h] [rbp-140h]
  int v122; // [rsp+250h] [rbp-138h]
  char v123; // [rsp+254h] [rbp-134h]
  char v124; // [rsp+258h] [rbp-130h] BYREF

  v73 = a1[1];
  v99 = 1;
  v100 = 0;
  v102 = 0;
  LODWORD(v103) = 0;
  if ( !(unsigned __int8)sub_325E1D0(a2, 1u, 2u, &v99, &v100, (__int64)&v102, v73) )
    return 0;
  if ( (unsigned int)(*(_DWORD *)(v102 + 24) - 56) > 1 )
    return 0;
  v4 = *(_QWORD *)(v102 + 56);
  if ( v4 )
  {
    if ( !*(_QWORD *)(v4 + 32) )
      return 0;
  }
  v104.m128i_i64[0] = 0;
  v5 = a1[1];
  v104.m128i_i32[2] = 0;
  *(_QWORD *)&v105 = 0;
  DWORD2(v105) = 0;
  v101 = 0;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 1888LL);
  if ( v6 == sub_302E060 )
    return 0;
  v91 = ((__int64 (__fastcall *)(__int64, __int64, __m128i *, __int128 *, int *, __int64))v6)(
          v5,
          a2,
          &v104,
          &v105,
          &v101,
          *a1);
  if ( !v91 )
    return 0;
  v95 = *(_DWORD *)(v104.m128i_i64[0] + 24) == 35 || *(_DWORD *)(v104.m128i_i64[0] + 24) == 11;
  if ( v95 )
  {
    v9 = _mm_loadu_si128(&v104);
    v104.m128i_i64[0] = v105;
    v104.m128i_i32[2] = DWORD2(v105);
    *(_QWORD *)&v105 = v9.m128i_i64[0];
    DWORD2(v105) = v9.m128i_i32[2];
  }
  v10 = v105;
  v7 = sub_33CF170(v105, *((_QWORD *)&v105 + 1));
  if ( (_BYTE)v7 )
    return 0;
  v14 = v104.m128i_i64[0];
  v15 = *(unsigned int *)(v104.m128i_i64[0] + 24);
  if ( (unsigned int)v15 <= 0x27 )
  {
    v11 = 0x8000008200LL;
    if ( _bittest64(&v11, v15) )
      return 0;
  }
  if ( !v99
    && ((v16 = *(_QWORD *)(a2 + 40),
         *(_QWORD *)&v10 = *(_QWORD *)(v16 + 40),
         v17 = *(_DWORD *)(v16 + 48),
         v104.m128i_i64[0] == (_QWORD)v10)
     && v17 == v104.m128i_i32[2]
     || (*((_QWORD *)&v10 + 1) = v102, (_QWORD)v10 == v102) && v17 == (_DWORD)v103
     || (unsigned __int8)sub_33CFFC0(v10, v102)) )
  {
    return 0;
  }
  else
  {
    v119 = 0;
    v120 = &v124;
    v114[0] = (unsigned __int64)v115;
    v114[1] = 0x1000000001LL;
    v116 = v118;
    v121 = 32;
    v122 = 0;
    v123 = 1;
    v115[0] = a2;
    v117 = 0x1000000000LL;
    v92 = sub_33CA560(v10, *((_QWORD *)&v10 + 1), v14, v11, v12, v13);
    v19 = *(_DWORD *)(v105 + 24);
    if ( v19 != 35 && v19 != 11 )
    {
      i = v102;
      goto LABEL_19;
    }
    v55 = *(_QWORD *)(v104.m128i_i64[0] + 56);
    for ( i = v102; v55; v55 = *(_QWORD *)(v55 + 32) )
    {
      v56 = *(_QWORD *)(v55 + 16);
      if ( v56 != i && *(_QWORD *)v55 == v104.m128i_i64[0] && *(_DWORD *)(v55 + 8) == v104.m128i_i32[2] )
      {
        if ( !(unsigned __int8)sub_3285B00(v56, (__int64)&v119, (__int64)v114, v92, 0, v18) )
        {
          v57 = *(_QWORD *)(v55 + 16);
          if ( (unsigned int)(*(_DWORD *)(v57 + 24) - 56) > 1
            || (v58 = (unsigned int *)(*(_QWORD *)(v57 + 40)
                                     + 40LL * ((-51 * (unsigned __int8)((v55 - *(_QWORD *)(v57 + 40)) >> 3) + 1) & 1)),
                v59 = *(_DWORD *)(*(_QWORD *)v58 + 24LL),
                v59 != 35)
            && v59 != 11
            || (v66 = *(_QWORD *)(v105 + 48) + 16LL * DWORD2(v105),
                v67 = *(_QWORD *)(*(_QWORD *)v58 + 48LL) + 16LL * v58[2],
                *(_WORD *)v66 != *(_WORD *)v67)
            || *(_QWORD *)(v66 + 8) != *(_QWORD *)(v67 + 8) && !*(_WORD *)v67 )
          {
            LODWORD(v117) = 0;
            i = v102;
            break;
          }
          v68 = (unsigned int)v117;
          v69 = (unsigned int)v117 + 1LL;
          if ( v69 > HIDWORD(v117) )
          {
            v98 = *(_QWORD *)(v55 + 16);
            sub_C8D5F0((__int64)&v116, v118, v69, 8u, v57, v18);
            v68 = (unsigned int)v117;
            v57 = v98;
          }
          *(_QWORD *)&v116[8 * v68] = v57;
          LODWORD(v117) = v117 + 1;
        }
        i = v102;
      }
    }
LABEL_19:
    if ( v95 )
    {
      v21 = _mm_loadu_si128(&v104);
      v104.m128i_i64[0] = v105;
      v104.m128i_i32[2] = DWORD2(v105);
      *(_QWORD *)&v105 = v21.m128i_i64[0];
      DWORD2(v105) = v21.m128i_i32[2];
    }
    if ( *(_QWORD *)(i + 56) )
    {
      v22 = 0;
      v96 = a2;
      v23 = *(_QWORD *)(i + 56);
      do
      {
        v24 = *(_QWORD *)(v23 + 16);
        if ( v96 != v24 )
        {
          if ( (unsigned __int8)sub_3285B00(*(_QWORD *)(v23 + 16), (__int64)&v119, (__int64)v114, v92, 0, v18) )
          {
            v7 = 0;
            goto LABEL_79;
          }
          if ( !(unsigned __int8)sub_3264B60(v102, v24, *a1, a1[1]) )
            v22 = v91;
        }
        v23 = *(_QWORD *)(v23 + 32);
      }
      while ( v23 );
      v25 = v22;
      v94 = v22;
      v7 = 0;
      if ( v25 )
      {
        v26 = *(__int64 (__fastcall ***)())(v96 + 80);
        v27 = *a1;
        v110 = v26;
        v28 = v101;
        if ( v100 )
        {
          if ( v99 )
          {
            if ( v26 )
              sub_325F5D0((__int64 *)&v110);
            LODWORD(v111) = *(_DWORD *)(v96 + 72);
            v29 = sub_33E95C0(
                    v27,
                    v96,
                    0,
                    (unsigned int)&v110,
                    v104.m128i_i32[0],
                    v104.m128i_i32[2],
                    v105,
                    *((__int64 *)&v105 + 1),
                    v28);
          }
          else
          {
            if ( v26 )
              sub_325F5D0((__int64 *)&v110);
            LODWORD(v111) = *(_DWORD *)(v96 + 72);
            v29 = sub_33F6C50(
                    v27,
                    v96,
                    0,
                    (unsigned int)&v110,
                    v104.m128i_i32[0],
                    v104.m128i_i32[2],
                    v105,
                    *((__int64 *)&v105 + 1),
                    v28);
          }
        }
        else if ( v99 )
        {
          if ( v26 )
            sub_325F5D0((__int64 *)&v110);
          LODWORD(v111) = *(_DWORD *)(v96 + 72);
          v29 = sub_33EA400(
                  v27,
                  v96,
                  0,
                  (unsigned int)&v110,
                  v104.m128i_i32[0],
                  v104.m128i_i32[2],
                  v105,
                  *((__int64 *)&v105 + 1),
                  v28);
        }
        else
        {
          if ( v26 )
            sub_325F5D0((__int64 *)&v110);
          LODWORD(v111) = *(_DWORD *)(v96 + 72);
          v29 = sub_33EA4E0(v27, v96, 0, (unsigned int)&v110, v104.m128i_i32[0], v104.m128i_i32[2], v105, v28);
        }
        v90 = (const void *)v29;
        if ( v110 )
          sub_B91220((__int64)&v110, (__int64)v110);
        v30 = v99 == 0;
        v31 = *(_QWORD *)(*a1 + 768);
        v112 = *a1;
        v111 = v31;
        *(_QWORD *)(v112 + 768) = &v110;
        v32 = *a1;
        v110 = off_4A360B8;
        v113 = a1;
        if ( v30 )
        {
          sub_34161C0(v32, v96, 0, v90, 1);
        }
        else
        {
          sub_34161C0(v32, v96, 0, v90, 0);
          sub_34161C0(*a1, v96, 1, v90, 2);
        }
        sub_32EB240((__int64)a1, v96);
        if ( v95 )
        {
          v33 = _mm_loadu_si128(&v104);
          v104.m128i_i64[0] = v105;
          v104.m128i_i32[2] = DWORD2(v105);
          *(_QWORD *)&v105 = v33.m128i_i64[0];
          DWORD2(v105) = v33.m128i_i32[2];
        }
        if ( (_DWORD)v117 )
        {
          v34 = 0;
          HIWORD(v35) = v88;
          HIWORD(v36) = v81;
          v89 = 8LL * (unsigned int)v117;
          while ( 1 )
          {
            v46 = *(_QWORD *)&v116[v34];
            v47 = *(_QWORD **)(v46 + 40);
            v48 = *(_DWORD *)(v46 + 24);
            v97 = v47[5];
            if ( v97 != v104.m128i_i64[0] )
              break;
            v97 = *v47;
            v50 = *(_QWORD *)(*v47 + 96LL);
            v49 = (__int64 *)(*(_QWORD *)(v105 + 96) + 24LL);
            v51 = (const void **)(v50 + 24);
            if ( v48 != 57 )
            {
              v52 = 1;
LABEL_56:
              if ( v101 == 2 )
              {
                if ( v95 )
                  v93 = 57;
                else
                  v93 = 56;
                v60 = -1;
              }
              else
              {
                v93 = 56;
                v60 = 1;
              }
              goto LABEL_61;
            }
            if ( v101 == 2 )
            {
              if ( v95 )
                v93 = 56;
              else
                v93 = 57;
              v60 = 1;
              v52 = 1;
            }
            else
            {
              v93 = 57;
              v60 = -1;
              v52 = 1;
            }
LABEL_61:
            v53 = *(_DWORD *)(v50 + 32);
            v107 = v53;
            if ( v53 <= 0x40 )
            {
              v106 = *(const void **)(v50 + 24);
              if ( v52 != -1 )
              {
                if ( v60 != -1 )
                  goto LABEL_64;
                v109 = v53;
LABEL_92:
                v108 = v106;
                goto LABEL_77;
              }
              v109 = v53;
LABEL_107:
              v61 = (unsigned __int64)v106;
LABEL_108:
              v62 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v53) & ~v61;
              v30 = v53 == 0;
              v63 = 0;
              if ( !v30 )
                v63 = (const void *)v62;
              v108 = v63;
              goto LABEL_111;
            }
            v75 = v52;
            v78 = v60;
            v84 = v49;
            sub_C43780((__int64)&v106, v51);
            v53 = v107;
            v49 = v84;
            v60 = v78;
            if ( v75 != -1 )
              goto LABEL_74;
            v109 = v107;
            if ( v107 <= 0x40 )
              goto LABEL_107;
            sub_C43780((__int64)&v108, &v106);
            v53 = v109;
            v49 = v84;
            v60 = v78;
            if ( v109 <= 0x40 )
            {
              v61 = (unsigned __int64)v108;
              goto LABEL_108;
            }
            sub_C43D10((__int64)&v108);
            v60 = v78;
            v49 = v84;
LABEL_111:
            v79 = v60;
            v87 = v49;
            sub_C46250((__int64)&v108);
            v53 = v109;
            v109 = 0;
            v49 = v87;
            v60 = v79;
            if ( v107 > 0x40 && v106 )
            {
              v74 = v79;
              v76 = v108;
              v80 = v53;
              j_j___libc_free_0_0((unsigned __int64)v106);
              v53 = v80;
              v49 = v87;
              v106 = v76;
              v60 = v74;
              v107 = v80;
              if ( v109 > 0x40 && v108 )
              {
                j_j___libc_free_0_0((unsigned __int64)v108);
                v53 = v107;
                v60 = v74;
                v49 = v87;
              }
            }
            else
            {
              v106 = v108;
              v107 = v53;
            }
LABEL_74:
            if ( v60 != -1 )
            {
LABEL_64:
              v109 = v53;
              if ( v53 > 0x40 )
              {
                v86 = v49;
                sub_C43780((__int64)&v108, &v106);
                v49 = v86;
              }
              else
              {
                v108 = v106;
              }
              sub_C46B40((__int64)&v108, v49);
              goto LABEL_67;
            }
            v109 = v53;
            if ( v53 <= 0x40 )
              goto LABEL_92;
            v85 = v49;
            sub_C43780((__int64)&v108, &v106);
            v49 = v85;
LABEL_77:
            sub_C45EE0((__int64)&v108, v49);
LABEL_67:
            v54 = v109;
            v109 = 0;
            if ( v107 > 0x40 && v106 )
            {
              v77 = v108;
              v82 = v54;
              j_j___libc_free_0_0((unsigned __int64)v106);
              v106 = v77;
              v107 = v82;
              if ( v109 > 0x40 && v108 )
                j_j___libc_free_0_0((unsigned __int64)v108);
            }
            else
            {
              v106 = v108;
              v107 = v54;
            }
            v37 = *(_QWORD *)&v116[v34];
            v38 = *(const void **)(v37 + 80);
            v108 = v38;
            if ( v38 )
            {
              v83 = v37;
              sub_B96E90((__int64)&v108, (__int64)v38, 1);
              v37 = v83;
            }
            v39 = *a1;
            v109 = *(_DWORD *)(v37 + 72);
            v40 = *(_QWORD *)(v97 + 48);
            LOWORD(v35) = *(_WORD *)v40;
            v41 = sub_34007B0(v39, (unsigned int)&v106, (unsigned int)&v108, v35, *(_QWORD *)(v40 + 8), 0, 0);
            v42 = *(_QWORD *)(*(_QWORD *)&v116[v34] + 48LL);
            LOWORD(v36) = *(_WORD *)v42;
            *((_QWORD *)&v72 + 1) = v99;
            *(_QWORD *)&v72 = v90;
            *((_QWORD *)&v71 + 1) = v43;
            *(_QWORD *)&v71 = v41;
            v44 = sub_3406EB0(*a1, v93, (unsigned int)&v108, v36, *(_QWORD *)(v42 + 8), *a1, v71, v72);
            sub_34161C0(*a1, *(_QWORD *)&v116[v34], 0, v44, v45);
            sub_32EB240((__int64)a1, *(_QWORD *)&v116[v34]);
            if ( v108 )
              sub_B91220((__int64)&v108, (__int64)v108);
            if ( v107 > 0x40 && v106 )
              j_j___libc_free_0_0((unsigned __int64)v106);
            v34 += 8;
            if ( v89 == v34 )
              goto LABEL_117;
          }
          v49 = (__int64 *)(*(_QWORD *)(v105 + 96) + 24LL);
          v50 = *(_QWORD *)(v97 + 96);
          v51 = (const void **)(v50 + 24);
          v52 = 2 * (v48 != 57) - 1;
          goto LABEL_56;
        }
LABEL_117:
        sub_34161C0(*a1, v102, v103, v90, v99);
        sub_32EB240((__int64)a1, v102);
        if ( *((_DWORD *)v90 + 6) != 328 )
        {
          v108 = v90;
          sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v108);
          if ( *((int *)v90 + 22) < 0 )
          {
            *((_DWORD *)v90 + 22) = *((_DWORD *)a1 + 12);
            v70 = *((unsigned int *)a1 + 12);
            if ( v70 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
            {
              sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v70 + 1, 8u, v64, v65);
              v70 = *((unsigned int *)a1 + 12);
            }
            *(_QWORD *)(a1[5] + 8 * v70) = v90;
            ++*((_DWORD *)a1 + 12);
          }
        }
        v7 = v94;
        *(_QWORD *)(v112 + 768) = v111;
      }
    }
LABEL_79:
    if ( v116 != v118 )
      _libc_free((unsigned __int64)v116);
    if ( (_QWORD *)v114[0] != v115 )
      _libc_free(v114[0]);
    if ( !v123 )
      _libc_free((unsigned __int64)v120);
  }
  return v7;
}
