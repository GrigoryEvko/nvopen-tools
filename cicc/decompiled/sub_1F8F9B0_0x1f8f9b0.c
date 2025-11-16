// Function: sub_1F8F9B0
// Address: 0x1f8f9b0
//
__int64 __fastcall sub_1F8F9B0(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int16 v5; // ax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 (*v13)(); // rax
  unsigned int v14; // r14d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  int v23; // r9d
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  int v27; // eax
  unsigned __int8 v28; // r14
  __int64 v29; // r12
  __int64 v30; // r15
  unsigned __int8 v31; // al
  __int64 v33; // rsi
  _QWORD *v34; // r14
  unsigned int v35; // ecx
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // r12
  unsigned int v39; // r13d
  unsigned int v40; // r15d
  __int64 v41; // rcx
  const void *v42; // rsi
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 *v48; // rax
  unsigned int v49; // edx
  __int64 v50; // r9
  _BYTE *v51; // rcx
  __int64 v52; // rax
  __int64 v53; // r8
  bool v54; // r10
  __int64 v55; // rdx
  __int64 v56; // rax
  __int16 v57; // dx
  int v58; // r10d
  int v59; // edx
  unsigned int v60; // ecx
  unsigned int v61; // eax
  __int64 i; // r15
  __int64 v63; // rdi
  __int64 v64; // r8
  unsigned int *v65; // rdx
  int v66; // eax
  unsigned __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rdi
  __int128 v73; // [rsp-20h] [rbp-3C0h]
  __int64 v74; // [rsp+0h] [rbp-3A0h]
  unsigned int v75; // [rsp+18h] [rbp-388h]
  const void *v76; // [rsp+18h] [rbp-388h]
  int v77; // [rsp+18h] [rbp-388h]
  __int64 v78; // [rsp+18h] [rbp-388h]
  const void *v79; // [rsp+18h] [rbp-388h]
  unsigned int v80; // [rsp+20h] [rbp-380h]
  unsigned int v81; // [rsp+20h] [rbp-380h]
  __int64 v82; // [rsp+20h] [rbp-380h]
  int v83; // [rsp+20h] [rbp-380h]
  __int64 v84; // [rsp+20h] [rbp-380h]
  int v85; // [rsp+20h] [rbp-380h]
  unsigned int v86; // [rsp+20h] [rbp-380h]
  __int64 v87; // [rsp+28h] [rbp-378h]
  __int64 v88; // [rsp+30h] [rbp-370h]
  unsigned __int8 v89; // [rsp+38h] [rbp-368h]
  __int64 v90; // [rsp+38h] [rbp-368h]
  unsigned __int8 v91; // [rsp+48h] [rbp-358h]
  unsigned int v93; // [rsp+50h] [rbp-350h]
  __int64 v94; // [rsp+50h] [rbp-350h]
  char v95; // [rsp+50h] [rbp-350h]
  __int64 v96; // [rsp+50h] [rbp-350h]
  bool v97; // [rsp+58h] [rbp-348h]
  int v98; // [rsp+58h] [rbp-348h]
  unsigned __int8 v99; // [rsp+5Eh] [rbp-342h]
  bool v100; // [rsp+5Fh] [rbp-341h]
  int v101; // [rsp+60h] [rbp-340h]
  unsigned __int8 v102; // [rsp+68h] [rbp-338h]
  unsigned int v103; // [rsp+68h] [rbp-338h]
  __int64 v104; // [rsp+68h] [rbp-338h]
  unsigned int v105; // [rsp+ACh] [rbp-2F4h] BYREF
  __m128i v106; // [rsp+B0h] [rbp-2F0h] BYREF
  __int128 v107; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v108; // [rsp+D0h] [rbp-2D0h] BYREF
  unsigned int v109; // [rsp+D8h] [rbp-2C8h]
  const void *v110; // [rsp+E0h] [rbp-2C0h] BYREF
  unsigned int v111; // [rsp+E8h] [rbp-2B8h]
  const void *v112; // [rsp+F0h] [rbp-2B0h] BYREF
  unsigned int v113; // [rsp+F8h] [rbp-2A8h]
  __int64 (__fastcall **v114)(); // [rsp+100h] [rbp-2A0h] BYREF
  __int64 v115; // [rsp+108h] [rbp-298h]
  __int64 v116; // [rsp+110h] [rbp-290h]
  __int64 *v117; // [rsp+118h] [rbp-288h]
  unsigned __int64 v118[2]; // [rsp+120h] [rbp-280h] BYREF
  _QWORD v119[16]; // [rsp+130h] [rbp-270h] BYREF
  _BYTE *v120; // [rsp+1B0h] [rbp-1F0h] BYREF
  __int64 v121; // [rsp+1B8h] [rbp-1E8h]
  _BYTE v122[128]; // [rsp+1C0h] [rbp-1E0h] BYREF
  __int64 v123; // [rsp+240h] [rbp-160h] BYREF
  _BYTE *v124; // [rsp+248h] [rbp-158h]
  _BYTE *v125; // [rsp+250h] [rbp-150h]
  __int64 v126; // [rsp+258h] [rbp-148h]
  int v127; // [rsp+260h] [rbp-140h]
  _BYTE v128[312]; // [rsp+268h] [rbp-138h] BYREF

  if ( *((int *)a1 + 4) <= 2 )
    return 0;
  v5 = *(_WORD *)(a2 + 24);
  if ( v5 == 185 )
  {
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
      return 0;
    v7 = *(unsigned __int8 *)(a2 + 88);
    if ( !(_BYTE)v7 )
      return 0;
    v8 = a1[1];
    v9 = v8 + 5 * v7;
    if ( ((*(_BYTE *)(v9 + 71884) >> 4) & 0xB) != 0 && ((*(_BYTE *)(v9 + 71885) >> 4) & 0xB) != 0 )
      return 0;
    v10 = *(_QWORD *)(a2 + 32);
    v102 = 1;
    v11 = *(_QWORD *)(v10 + 40);
    v101 = *(_DWORD *)(v10 + 48);
  }
  else
  {
    if ( v5 != 186 )
      return 0;
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 )
      return 0;
    v16 = *(unsigned __int8 *)(a2 + 88);
    if ( !(_BYTE)v16 )
      return 0;
    v8 = a1[1];
    v17 = v8 + 5 * v16;
    if ( (*(_BYTE *)(v17 + 71884) & 0xB) != 0 && (*(_BYTE *)(v17 + 71885) & 0xB) != 0 )
      return 0;
    v18 = *(_QWORD *)(a2 + 32);
    v102 = 0;
    v11 = *(_QWORD *)(v18 + 80);
    v101 = *(_DWORD *)(v18 + 88);
  }
  if ( (unsigned int)*(unsigned __int16 *)(v11 + 24) - 52 > 1 )
    return 0;
  v12 = *(_QWORD *)(v11 + 48);
  if ( v12 )
  {
    if ( !*(_QWORD *)(v12 + 32) )
      return 0;
  }
  v106.m128i_i64[0] = 0;
  v106.m128i_i32[2] = 0;
  *(_QWORD *)&v107 = 0;
  DWORD2(v107) = 0;
  v105 = 0;
  v13 = *(__int64 (**)())(*(_QWORD *)v8 + 1000LL);
  if ( v13 == sub_1F6BB40 )
    return 0;
  v91 = ((__int64 (__fastcall *)(__int64, __int64, __m128i *, __int128 *, unsigned int *, __int64))v13)(
          v8,
          a2,
          &v106,
          &v107,
          &v105,
          *a1);
  if ( !v91 )
    return 0;
  v100 = *(_WORD *)(v106.m128i_i64[0] + 24) == 32 || *(_WORD *)(v106.m128i_i64[0] + 24) == 10;
  if ( v100 )
  {
    a3 = _mm_loadu_si128(&v106);
    v106.m128i_i64[0] = v107;
    v106.m128i_i32[2] = DWORD2(v107);
    *(_QWORD *)&v107 = a3.m128i_i64[0];
    DWORD2(v107) = a3.m128i_i32[2];
  }
  LOBYTE(v19) = sub_1D185B0(v107);
  v14 = v19;
  if ( (_BYTE)v19 )
    return 0;
  v24 = *(unsigned __int16 *)(v106.m128i_i64[0] + 24);
  if ( (unsigned __int16)v24 <= 0x24u )
  {
    v20 = 0x1000004100LL;
    if ( _bittest64(&v20, v24) )
      return 0;
  }
  if ( !v102 )
  {
    v25 = *(_QWORD *)(a2 + 32);
    v26 = *(_QWORD *)(v25 + 40);
    if ( v26 == v106.m128i_i64[0] )
    {
      v21 = v106.m128i_u32[2];
      if ( *(_DWORD *)(v25 + 48) == v106.m128i_i32[2] )
        return 0;
    }
    if ( (unsigned __int8)sub_1D19270(v26, v106.m128i_i64[0], v20, v21, v22, v23) )
      return 0;
  }
  v126 = 32;
  v124 = v128;
  v125 = v128;
  v118[0] = (unsigned __int64)v119;
  v118[1] = 0x1000000001LL;
  v120 = v122;
  v121 = 0x1000000000LL;
  v127 = 0;
  v27 = *(unsigned __int16 *)(v107 + 24);
  v119[0] = a2;
  v123 = 0;
  if ( v27 == 32 || v27 == 10 )
  {
    for ( i = *(_QWORD *)(v106.m128i_i64[0] + 48); i; i = *(_QWORD *)(i + 32) )
    {
      v63 = *(_QWORD *)(i + 16);
      if ( v11 != v63
        && *(_QWORD *)i == v106.m128i_i64[0]
        && *(_DWORD *)(i + 8) == v106.m128i_i32[2]
        && !(unsigned __int8)sub_1D15B50(v63, (__int64)&v123, (__int64)v118, 0, 0, v23) )
      {
        v64 = *(_QWORD *)(i + 16);
        if ( (unsigned int)*(unsigned __int16 *)(v64 + 24) - 52 > 1
          || (v65 = (unsigned int *)(*(_QWORD *)(v64 + 32)
                                   + 40LL * ((-51 * (unsigned __int8)((i - *(_QWORD *)(v64 + 32)) >> 3) + 1) & 1)),
              v66 = *(unsigned __int16 *)(*(_QWORD *)v65 + 24LL),
              v66 != 32)
          && v66 != 10
          || (v68 = *(_QWORD *)(v107 + 40) + 16LL * DWORD2(v107),
              v69 = *(_QWORD *)(*(_QWORD *)v65 + 40LL) + 16LL * v65[2],
              *(_BYTE *)v69 != *(_BYTE *)v68)
          || *(_QWORD *)(v69 + 8) != *(_QWORD *)(v68 + 8) && !*(_BYTE *)v69 )
        {
          LODWORD(v121) = 0;
          break;
        }
        v70 = (unsigned int)v121;
        if ( (unsigned int)v121 >= HIDWORD(v121) )
        {
          v96 = *(_QWORD *)(i + 16);
          sub_16CD150((__int64)&v120, v122, 0, 8, v64, v23);
          v70 = (unsigned int)v121;
          v64 = v96;
        }
        *(_QWORD *)&v120[8 * v70] = v64;
        LODWORD(v121) = v121 + 1;
      }
    }
  }
  if ( v100 )
  {
    a3 = _mm_loadu_si128(&v106);
    v106.m128i_i64[0] = v107;
    v106.m128i_i32[2] = DWORD2(v107);
    *(_QWORD *)&v107 = a3.m128i_i64[0];
    DWORD2(v107) = a3.m128i_i32[2];
  }
  if ( *(_QWORD *)(v11 + 48) )
  {
    v89 = v14;
    v28 = 0;
    v29 = *(_QWORD *)(v11 + 48);
    do
    {
      v30 = *(_QWORD *)(v29 + 16);
      if ( v30 != a2 )
      {
        if ( (unsigned __int8)sub_1D15B50(*(_QWORD *)(v29 + 16), (__int64)&v123, (__int64)v118, 0, 0, v23) )
        {
          v14 = v89;
          goto LABEL_102;
        }
        if ( !(unsigned __int8)sub_1F6D430(v11, v30, *a1, a1[1]) )
          v28 = v91;
      }
      v29 = *(_QWORD *)(v29 + 32);
    }
    while ( v29 );
    v31 = v28;
    v99 = v28;
    v14 = v89;
    if ( v31 )
    {
      v33 = *(_QWORD *)(a2 + 72);
      v34 = (_QWORD *)*a1;
      v35 = v105;
      v114 = *(__int64 (__fastcall ***)())(a2 + 72);
      if ( v102 )
      {
        if ( v33 )
        {
          v93 = v105;
          sub_1623A60((__int64)&v114, v33, 2);
          v35 = v93;
        }
        LODWORD(v115) = *(_DWORD *)(a2 + 64);
        v90 = sub_1D26680(v34, a2, 0, (__int64)&v114, v106.m128i_i64[0], v106.m128i_i64[1], v107, v35);
        if ( v114 )
          sub_161E7C0((__int64)&v114, (__int64)v114);
        v36 = *(_QWORD *)(*a1 + 664);
        v116 = *a1;
        v115 = v36;
        *(_QWORD *)(v116 + 664) = &v114;
        v37 = *a1;
        v114 = off_49FFF30;
        v117 = a1;
        sub_1D44C70(v37, a2, 0, v90, 0);
        sub_1D44C70(*a1, a2, 1, v90, 2u);
      }
      else
      {
        if ( v33 )
        {
          v95 = v105;
          sub_1623A60((__int64)&v114, v33, 2);
          LOBYTE(v35) = v95;
        }
        LODWORD(v115) = *(_DWORD *)(a2 + 64);
        v90 = (__int64)sub_1D25500(v34, a2, 0, (__int64)&v114, v106.m128i_i64[0], v106.m128i_i64[1], v107, v35);
        if ( v114 )
          sub_161E7C0((__int64)&v114, (__int64)v114);
        v71 = *(_QWORD *)(*a1 + 664);
        v116 = *a1;
        v115 = v71;
        *(_QWORD *)(v116 + 664) = &v114;
        v72 = *a1;
        v114 = off_49FFF30;
        v117 = a1;
        sub_1D44C70(v72, a2, 0, v90, 1u);
      }
      sub_1F81E80(a1, a2);
      if ( v100 )
      {
        a3 = _mm_loadu_si128(&v106);
        v106.m128i_i64[0] = v107;
        v106.m128i_i32[2] = DWORD2(v107);
        *(_QWORD *)&v107 = a3.m128i_i64[0];
        DWORD2(v107) = a3.m128i_i32[2];
      }
      if ( (_DWORD)v121 )
      {
        v74 = v11;
        v38 = 0;
        v39 = v80;
        v87 = 8LL * (unsigned int)v121;
        v40 = v75;
        v88 = v102;
        while ( 1 )
        {
          v50 = v106.m128i_i64[0];
          v51 = &v120[v38];
          v52 = *(_QWORD *)(*(_QWORD *)&v120[v38] + 32LL);
          v53 = *(_QWORD *)(v52 + 40);
          v54 = v106.m128i_i64[0] != v53;
          v55 = *(_QWORD *)(v107 + 88);
          v94 = *(_QWORD *)(v52 + 40LL * (v106.m128i_i64[0] != v53));
          v56 = *(_QWORD *)(v94 + 88);
          v109 = *(_DWORD *)(v55 + 32);
          if ( v109 > 0x40 )
          {
            v97 = v106.m128i_i64[0] != v53;
            v78 = v56;
            v84 = v53;
            v104 = v106.m128i_i64[0];
            sub_16A4FD0((__int64)&v108, (const void **)(v55 + 24));
            v54 = v97;
            v56 = v78;
            v53 = v84;
            v50 = v104;
            v51 = &v120[v38];
          }
          else
          {
            v108 = *(_QWORD *)(v55 + 24);
          }
          v57 = *(_WORD *)(*(_QWORD *)v51 + 24LL);
          if ( v57 == 53 && v54 )
          {
            v58 = -1;
          }
          else
          {
            v58 = 1;
            if ( v57 == 53 && v50 == v53 )
            {
              if ( !v100 && v105 == 2 )
              {
                v103 = 53;
                v59 = 1;
              }
              else
              {
                if ( v105 != 2 || !v100 )
                {
                  v58 = 1;
LABEL_100:
                  v103 = 53;
                  v59 = -1;
                  goto LABEL_74;
                }
                v103 = 52;
                v59 = 1;
                v58 = 1;
              }
              goto LABEL_74;
            }
          }
          if ( v100 || v105 != 2 )
          {
            if ( v105 != 2 || !v100 )
            {
              v103 = 52;
              v59 = 1;
              goto LABEL_74;
            }
            goto LABEL_100;
          }
          v103 = 52;
          v59 = -1;
LABEL_74:
          v60 = *(_DWORD *)(v56 + 32);
          v111 = v60;
          if ( v60 <= 0x40 )
          {
            v110 = *(const void **)(v56 + 24);
            if ( v58 != -1 )
            {
              if ( v59 != -1 )
                goto LABEL_77;
              v113 = v60;
LABEL_109:
              v112 = v110;
              goto LABEL_92;
            }
            v113 = v60;
LABEL_124:
            v67 = (unsigned __int64)v110;
            goto LABEL_125;
          }
          v77 = v59;
          v83 = v58;
          sub_16A4FD0((__int64)&v110, (const void **)(v56 + 24));
          v60 = v111;
          v59 = v77;
          if ( v83 != -1 )
            goto LABEL_89;
          v113 = v111;
          if ( v111 <= 0x40 )
            goto LABEL_124;
          sub_16A4FD0((__int64)&v112, &v110);
          LOBYTE(v60) = v113;
          v59 = v77;
          if ( v113 > 0x40 )
          {
            sub_16A8F40((__int64 *)&v112);
            v59 = v77;
            goto LABEL_126;
          }
          v67 = (unsigned __int64)v112;
LABEL_125:
          v112 = (const void *)(~v67 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v60));
LABEL_126:
          v85 = v59;
          sub_16A7400((__int64)&v112);
          v60 = v113;
          v113 = 0;
          v59 = v85;
          if ( v111 > 0x40 && v110 )
          {
            v98 = v85;
            v79 = v112;
            v86 = v60;
            j_j___libc_free_0_0(v110);
            v60 = v86;
            v59 = v98;
            v110 = v79;
            v111 = v86;
            if ( v113 > 0x40 && v112 )
            {
              j_j___libc_free_0_0(v112);
              v60 = v111;
              v59 = v98;
            }
          }
          else
          {
            v110 = v112;
            v111 = v60;
          }
LABEL_89:
          if ( v59 != -1 )
          {
LABEL_77:
            v113 = v60;
            if ( v60 > 0x40 )
              sub_16A4FD0((__int64)&v112, &v110);
            else
              v112 = v110;
            sub_16A7590((__int64)&v112, &v108);
            goto LABEL_80;
          }
          v113 = v60;
          if ( v60 <= 0x40 )
            goto LABEL_109;
          sub_16A4FD0((__int64)&v112, &v110);
LABEL_92:
          sub_16A7200((__int64)&v112, &v108);
LABEL_80:
          v61 = v113;
          v113 = 0;
          if ( v111 > 0x40 && v110 )
          {
            v76 = v112;
            v81 = v61;
            j_j___libc_free_0_0(v110);
            v110 = v76;
            v111 = v81;
            if ( v113 > 0x40 && v112 )
              j_j___libc_free_0_0(v112);
          }
          else
          {
            v110 = v112;
            v111 = v61;
          }
          v41 = *(_QWORD *)&v120[v38];
          v42 = *(const void **)(v41 + 72);
          v112 = v42;
          if ( v42 )
          {
            v82 = v41;
            sub_1623A60((__int64)&v112, (__int64)v42, 2);
            v41 = v82;
          }
          v43 = *a1;
          v113 = *(_DWORD *)(v41 + 64);
          v44 = *(_QWORD *)(v94 + 40);
          LOBYTE(v40) = *(_BYTE *)v44;
          v45 = sub_1D38970(v43, (__int64)&v110, (__int64)&v112, v40, *(const void ***)(v44 + 8), 0, a3, a4, a5, 0);
          v46 = *(_QWORD *)(*(_QWORD *)&v120[v38] + 40LL);
          LOBYTE(v39) = *(_BYTE *)v46;
          *((_QWORD *)&v73 + 1) = v88;
          *(_QWORD *)&v73 = v90;
          v48 = sub_1D332F0(
                  (__int64 *)*a1,
                  v103,
                  (__int64)&v112,
                  v39,
                  *(const void ***)(v46 + 8),
                  0,
                  *(double *)a3.m128i_i64,
                  a4,
                  a5,
                  v45,
                  v47,
                  v73);
          sub_1D44C70(*a1, *(_QWORD *)&v120[v38], 0, (__int64)v48, v49);
          sub_1F81E80(a1, *(_QWORD *)&v120[v38]);
          if ( v112 )
            sub_161E7C0((__int64)&v112, (__int64)v112);
          if ( v111 > 0x40 && v110 )
            j_j___libc_free_0_0(v110);
          if ( v109 > 0x40 && v108 )
            j_j___libc_free_0_0(v108);
          v38 += 8;
          if ( v87 == v38 )
          {
            v11 = v74;
            goto LABEL_134;
          }
        }
      }
      LODWORD(v88) = v102;
LABEL_134:
      sub_1D44C70(*a1, v11, v101, v90, v88);
      sub_1F81E80(a1, v11);
      sub_1F81BC0((__int64)a1, v90);
      v14 = v99;
      *(_QWORD *)(v116 + 664) = v115;
    }
  }
LABEL_102:
  if ( v120 != v122 )
    _libc_free((unsigned __int64)v120);
  if ( (_QWORD *)v118[0] != v119 )
    _libc_free(v118[0]);
  if ( v125 != v124 )
    _libc_free((unsigned __int64)v125);
  return v14;
}
