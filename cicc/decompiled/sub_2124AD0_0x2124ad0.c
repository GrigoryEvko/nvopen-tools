// Function: sub_2124AD0
// Address: 0x2124ad0
//
__int64 __fastcall sub_2124AD0(__int64 a1, unsigned __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v5; // r14d
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r15
  int v11; // edx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // rax
  const void **v20; // rdx
  const void **v21; // r8
  __int64 v22; // rcx
  __int64 *v23; // r14
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r15
  char v27; // r9
  __int64 v28; // rsi
  const void **v29; // r8
  __int64 *v30; // r13
  __int64 v31; // rcx
  char *v32; // rdx
  __int64 v33; // r8
  unsigned __int8 v34; // al
  unsigned int v35; // eax
  const void **v36; // rdx
  char v37; // cl
  __int64 v38; // rdx
  _QWORD *v39; // rcx
  _DWORD *v40; // r15
  char v41; // cl
  const __m128i *v42; // r9
  int v43; // esi
  int v44; // edi
  unsigned int v45; // eax
  __int8 *v46; // rdx
  int v47; // r12d
  __int64 *v48; // rcx
  unsigned __int64 v49; // r8
  __int64 v50; // r14
  __int64 v51; // rsi
  __int64 v53; // rax
  char v54; // dl
  __int64 v55; // rax
  unsigned int v56; // eax
  __int64 *v57; // r12
  unsigned int v58; // edx
  _DWORD *v59; // r15
  char v60; // r12
  __int64 v61; // r8
  int v62; // esi
  int v63; // eax
  unsigned int v64; // edx
  __int64 v65; // rcx
  int v66; // r9d
  __int64 v67; // rdx
  __int64 v68; // rax
  __m128i *v69; // rdx
  unsigned int v70; // esi
  unsigned int v71; // esi
  const void ***v72; // rdx
  __int128 v73; // rax
  unsigned int v74; // eax
  __int64 v75; // rdi
  int v76; // edx
  unsigned int v77; // ecx
  int v78; // eax
  unsigned int v79; // eax
  __m128i *v80; // r8
  int v81; // edx
  unsigned int v82; // edi
  __int32 v83; // eax
  int v84; // r10d
  int v85; // r10d
  __int64 v86; // rdi
  int v87; // edx
  int v88; // eax
  unsigned int v89; // ecx
  __int32 v90; // esi
  __int64 v91; // rsi
  int v92; // edx
  int v93; // eax
  __int64 v94; // r9
  int v95; // ecx
  __int64 v96; // rcx
  int v97; // eax
  int v98; // edx
  __int64 v99; // r9
  int v100; // esi
  int v101; // r10d
  __int64 v102; // r12
  __int64 v103; // rdi
  int v104; // edx
  int v105; // ecx
  unsigned int v106; // esi
  __int32 v107; // eax
  int v108; // r12d
  int v109; // edx
  int v110; // edx
  int v111; // eax
  int v112; // eax
  int v113; // r12d
  int v114; // r10d
  __int128 v115; // [rsp-10h] [rbp-110h]
  unsigned int v116; // [rsp+4h] [rbp-FCh]
  unsigned __int64 v117; // [rsp+8h] [rbp-F8h]
  __int64 v118; // [rsp+10h] [rbp-F0h]
  char v119; // [rsp+20h] [rbp-E0h]
  const void **v120; // [rsp+20h] [rbp-E0h]
  __int64 v121; // [rsp+28h] [rbp-D8h]
  const void **v122; // [rsp+28h] [rbp-D8h]
  char v123; // [rsp+30h] [rbp-D0h]
  __int64 v124; // [rsp+70h] [rbp-90h] BYREF
  int v125; // [rsp+78h] [rbp-88h]
  char v126[8]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v127; // [rsp+88h] [rbp-78h]
  __int64 v128; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v129; // [rsp+98h] [rbp-68h]
  __int64 v130; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v131; // [rsp+A8h] [rbp-58h]
  __int64 v132; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v133; // [rsp+B8h] [rbp-48h]
  const void **v134; // [rsp+C0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 72);
  v124 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v124, v8, 2);
  v125 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_QWORD *)(v9 + 40);
  v11 = *(unsigned __int16 *)(v10 + 24);
  if ( v11 != 32 && v11 != 10 )
    goto LABEL_5;
  a3 = _mm_loadu_si128((const __m128i *)v9);
  a4 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v32 = *(char **)(*(_QWORD *)v9 + 40LL);
  v116 = *(_DWORD *)(v9 + 48);
  v33 = *((_QWORD *)v32 + 1);
  v34 = *v32;
  v126[0] = v34;
  v127 = v33;
  if ( v34 )
  {
    switch ( v34 )
    {
      case 0xEu:
      case 0xFu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x3Du:
        v37 = 2;
        break;
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x3Eu:
      case 0x3Fu:
      case 0x40u:
      case 0x41u:
      case 0x42u:
      case 0x43u:
        v37 = 3;
        break;
      case 0x21u:
      case 0x22u:
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x44u:
      case 0x45u:
      case 0x46u:
      case 0x47u:
      case 0x48u:
      case 0x49u:
        v37 = 4;
        break;
      case 0x29u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Eu:
      case 0x2Fu:
      case 0x30u:
      case 0x4Au:
      case 0x4Bu:
      case 0x4Cu:
      case 0x4Du:
      case 0x4Eu:
      case 0x4Fu:
        v37 = 5;
        break;
      case 0x31u:
      case 0x32u:
      case 0x33u:
      case 0x34u:
      case 0x35u:
      case 0x36u:
      case 0x50u:
      case 0x51u:
      case 0x52u:
      case 0x53u:
      case 0x54u:
      case 0x55u:
        v37 = 6;
        break;
      case 0x37u:
        v37 = 7;
        break;
      case 0x56u:
      case 0x57u:
      case 0x58u:
      case 0x62u:
      case 0x63u:
      case 0x64u:
        v37 = 8;
        break;
      case 0x59u:
      case 0x5Au:
      case 0x5Bu:
      case 0x5Cu:
      case 0x5Du:
      case 0x65u:
      case 0x66u:
      case 0x67u:
      case 0x68u:
      case 0x69u:
        v37 = 9;
        break;
      case 0x5Eu:
      case 0x5Fu:
      case 0x60u:
      case 0x61u:
      case 0x6Au:
      case 0x6Bu:
      case 0x6Cu:
      case 0x6Du:
        v37 = 10;
        break;
    }
    v120 = 0;
  }
  else
  {
    LOBYTE(v35) = sub_1F596B0((__int64)v126);
    v33 = v127;
    v120 = v36;
    v5 = v35;
    v37 = v35;
    v34 = v126[0];
  }
  v38 = *(_QWORD *)(v10 + 88);
  LOBYTE(v5) = v37;
  v39 = *(_QWORD **)(v38 + 24);
  if ( *(_DWORD *)(v38 + 32) > 0x40u )
    v39 = (_QWORD *)*v39;
  v117 = (unsigned __int64)v39;
  sub_1F40D10((__int64)&v132, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v34, v33);
  if ( (_BYTE)v132 == 6 )
  {
    v128 = 0;
    LODWORD(v129) = 0;
    v130 = 0;
    LODWORD(v131) = 0;
    sub_2017DE0(a1, a3.m128i_u64[0], a3.m128i_i64[1], &v128, &v130);
    v53 = *(_QWORD *)(v128 + 40) + 16LL * (unsigned int)v129;
    v54 = *(_BYTE *)v53;
    v55 = *(_QWORD *)(v53 + 8);
    LOBYTE(v132) = v54;
    v133 = v55;
    if ( v54 )
      v56 = word_430C840[(unsigned __int8)(v54 - 14)];
    else
      v56 = sub_1F58D30((__int64)&v132);
    v57 = *(__int64 **)(a1 + 8);
    if ( v56 <= v117 )
    {
      v72 = (const void ***)(*(_QWORD *)(v10 + 40) + 16LL * v116);
      *(_QWORD *)&v73 = sub_1D38BB0(
                          *(_QWORD *)(a1 + 8),
                          v117 - v56,
                          (__int64)&v124,
                          *(unsigned __int8 *)v72,
                          v72[1],
                          0,
                          a3,
                          *(double *)a4.m128i_i64,
                          a5,
                          0);
      v48 = sub_1D332F0(
              v57,
              *(unsigned __int16 *)(a2 + 24),
              (__int64)&v124,
              v5,
              v120,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v130,
              v131,
              v73);
    }
    else
    {
      v48 = sub_1D332F0(
              *(__int64 **)(a1 + 8),
              *(unsigned __int16 *)(a2 + 24),
              (__int64)&v124,
              v5,
              v120,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v128,
              v129,
              *(_OWORD *)&a4);
    }
    v49 = v58;
    goto LABEL_23;
  }
  if ( (_BYTE)v132 == 7 )
  {
    LODWORD(v132) = sub_200F8F0(a1, a3.m128i_u64[0], a3.m128i_i64[1]);
    v59 = sub_20322E0(a1 + 1208, &v132);
    sub_200D1B0(a1, v59 + 1);
    v60 = *(_BYTE *)(a1 + 352) & 1;
    if ( v60 )
    {
      v61 = a1 + 360;
      v62 = 7;
    }
    else
    {
      v71 = *(_DWORD *)(a1 + 368);
      v61 = *(_QWORD *)(a1 + 360);
      if ( !v71 )
      {
        v74 = *(_DWORD *)(a1 + 352);
        ++*(_QWORD *)(a1 + 344);
        v75 = 0;
        v76 = (v74 >> 1) + 1;
        goto LABEL_70;
      }
      v62 = v71 - 1;
    }
    v63 = v59[1];
    v64 = v62 & (37 * v63);
    v65 = v61 + 24LL * v64;
    v66 = *(_DWORD *)v65;
    if ( v63 == *(_DWORD *)v65 )
    {
LABEL_41:
      v67 = *(_QWORD *)(v65 + 8);
      v68 = *(unsigned int *)(v65 + 16);
LABEL_42:
      v48 = sub_1D332F0(
              *(__int64 **)(a1 + 8),
              *(unsigned __int16 *)(a2 + 24),
              (__int64)&v124,
              v5,
              v120,
              0,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v67,
              v68 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL,
              *(_OWORD *)&a4);
      v49 = (unsigned __int64)v69;
      goto LABEL_23;
    }
    v84 = 1;
    v75 = 0;
    while ( v66 != -1 )
    {
      if ( !v75 && v66 == -2 )
        v75 = v65;
      v64 = v62 & (v84 + v64);
      v65 = v61 + 24LL * v64;
      v66 = *(_DWORD *)v65;
      if ( v63 == *(_DWORD *)v65 )
        goto LABEL_41;
      ++v84;
    }
    v74 = *(_DWORD *)(a1 + 352);
    v71 = 8;
    if ( !v75 )
      v75 = v65;
    ++*(_QWORD *)(a1 + 344);
    v77 = 24;
    v76 = (v74 >> 1) + 1;
    if ( v60 )
    {
LABEL_71:
      if ( v77 > 4 * v76 )
      {
        if ( v71 - *(_DWORD *)(a1 + 356) - v76 > v71 >> 3 )
        {
LABEL_73:
          *(_DWORD *)(a1 + 352) = (2 * (v74 >> 1) + 2) | v74 & 1;
          if ( *(_DWORD *)v75 != -1 )
            --*(_DWORD *)(a1 + 356);
          v78 = v59[1];
          v67 = 0;
          *(_QWORD *)(v75 + 8) = 0;
          *(_DWORD *)(v75 + 16) = 0;
          *(_DWORD *)v75 = v78;
          v68 = 0;
          goto LABEL_42;
        }
        sub_200F500(a1 + 344, v71);
        if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
        {
          v96 = a1 + 360;
          v97 = 7;
          goto LABEL_105;
        }
        v112 = *(_DWORD *)(a1 + 368);
        v96 = *(_QWORD *)(a1 + 360);
        if ( v112 )
        {
          v97 = v112 - 1;
LABEL_105:
          v98 = v59[1];
          LODWORD(v99) = v97 & (37 * v98);
          v75 = v96 + 24LL * (unsigned int)v99;
          v100 = *(_DWORD *)v75;
          if ( v98 != *(_DWORD *)v75 )
          {
            v101 = 1;
            v102 = 0;
            while ( v100 != -1 )
            {
              if ( !v102 && v100 == -2 )
                v102 = v75;
              v99 = v97 & (unsigned int)(v99 + v101);
              v75 = v96 + 24 * v99;
              v100 = *(_DWORD *)v75;
              if ( v98 == *(_DWORD *)v75 )
                goto LABEL_102;
              ++v101;
            }
LABEL_108:
            if ( v102 )
              v75 = v102;
            goto LABEL_102;
          }
          goto LABEL_102;
        }
LABEL_160:
        *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
        BUG();
      }
      sub_200F500(a1 + 344, 2 * v71);
      if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
      {
        v91 = a1 + 360;
        v92 = 7;
      }
      else
      {
        v110 = *(_DWORD *)(a1 + 368);
        v91 = *(_QWORD *)(a1 + 360);
        if ( !v110 )
          goto LABEL_160;
        v92 = v110 - 1;
      }
      v93 = v59[1];
      LODWORD(v94) = v92 & (37 * v93);
      v75 = v91 + 24LL * (unsigned int)v94;
      v95 = *(_DWORD *)v75;
      if ( *(_DWORD *)v75 != v93 )
      {
        v114 = 1;
        v102 = 0;
        while ( v95 != -1 )
        {
          if ( !v102 && v95 == -2 )
            v102 = v75;
          v94 = v92 & (unsigned int)(v94 + v114);
          v75 = v91 + 24 * v94;
          v95 = *(_DWORD *)v75;
          if ( v93 == *(_DWORD *)v75 )
            goto LABEL_102;
          ++v114;
        }
        goto LABEL_108;
      }
LABEL_102:
      v74 = *(_DWORD *)(a1 + 352);
      goto LABEL_73;
    }
    v71 = *(_DWORD *)(a1 + 368);
LABEL_70:
    v77 = 3 * v71;
    goto LABEL_71;
  }
  v9 = *(_QWORD *)(a2 + 32);
  if ( (_BYTE)v132 == 5 )
  {
    LODWORD(v132) = sub_200F8F0(a1, *(_QWORD *)v9, *(_QWORD *)(v9 + 8));
    v40 = sub_20322E0(a1 + 1016, &v132);
    sub_200D1B0(a1, v40 + 1);
    v41 = *(_BYTE *)(a1 + 352) & 1;
    if ( v41 )
    {
      v42 = (const __m128i *)(a1 + 360);
      v43 = 7;
    }
    else
    {
      v70 = *(_DWORD *)(a1 + 368);
      v42 = *(const __m128i **)(a1 + 360);
      if ( !v70 )
      {
        v79 = *(_DWORD *)(a1 + 352);
        ++*(_QWORD *)(a1 + 344);
        v80 = 0;
        v81 = (v79 >> 1) + 1;
        goto LABEL_77;
      }
      v43 = v70 - 1;
    }
    v44 = v40[1];
    v45 = v43 & (37 * v44);
    v46 = &v42->m128i_i8[24 * v45];
    v47 = *(_DWORD *)v46;
    if ( v44 == *(_DWORD *)v46 )
    {
LABEL_22:
      v48 = (__int64 *)*((_QWORD *)v46 + 1);
      v49 = *((unsigned int *)v46 + 4);
LABEL_23:
      v50 = 0;
      sub_2013400(a1, a2, 0, (__int64)v48, (__m128i *)v49, v42);
      goto LABEL_29;
    }
    v85 = 1;
    v80 = 0;
    while ( v47 != -1 )
    {
      if ( !v80 && v47 == -2 )
        v80 = (__m128i *)v46;
      v45 = v43 & (v85 + v45);
      v46 = &v42->m128i_i8[24 * v45];
      v47 = *(_DWORD *)v46;
      if ( v44 == *(_DWORD *)v46 )
        goto LABEL_22;
      ++v85;
    }
    v79 = *(_DWORD *)(a1 + 352);
    v82 = 24;
    v70 = 8;
    if ( !v80 )
      v80 = (__m128i *)v46;
    ++*(_QWORD *)(a1 + 344);
    v81 = (v79 >> 1) + 1;
    if ( v41 )
    {
LABEL_78:
      v42 = (const __m128i *)(a1 + 344);
      if ( 4 * v81 < v82 )
      {
        if ( v70 - *(_DWORD *)(a1 + 356) - v81 > v70 >> 3 )
        {
LABEL_80:
          *(_DWORD *)(a1 + 352) = (2 * (v79 >> 1) + 2) | v79 & 1;
          if ( v80->m128i_i32[0] != -1 )
            --*(_DWORD *)(a1 + 356);
          v83 = v40[1];
          v48 = 0;
          v80->m128i_i64[1] = 0;
          v80[1].m128i_i32[0] = 0;
          v80->m128i_i32[0] = v83;
          v49 = 0;
          goto LABEL_23;
        }
        sub_200F500(a1 + 344, v70);
        if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
        {
          v103 = a1 + 360;
          v104 = 7;
          goto LABEL_112;
        }
        v111 = *(_DWORD *)(a1 + 368);
        v103 = *(_QWORD *)(a1 + 360);
        if ( v111 )
        {
          v104 = v111 - 1;
LABEL_112:
          v105 = v40[1];
          v106 = v104 & (37 * v105);
          v80 = (__m128i *)(v103 + 24LL * v106);
          v107 = v80->m128i_i32[0];
          if ( v105 != v80->m128i_i32[0] )
          {
            v108 = 1;
            v42 = 0;
            while ( v107 != -1 )
            {
              if ( !v42 && v107 == -2 )
                v42 = v80;
              v106 = v104 & (v108 + v106);
              v80 = (__m128i *)(v103 + 24LL * v106);
              v107 = v80->m128i_i32[0];
              if ( v105 == v80->m128i_i32[0] )
                goto LABEL_98;
              ++v108;
            }
LABEL_115:
            if ( v42 )
              v80 = (__m128i *)v42;
            goto LABEL_98;
          }
          goto LABEL_98;
        }
LABEL_159:
        *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
        BUG();
      }
      sub_200F500(a1 + 344, 2 * v70);
      if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
      {
        v86 = a1 + 360;
        v87 = 7;
      }
      else
      {
        v109 = *(_DWORD *)(a1 + 368);
        v86 = *(_QWORD *)(a1 + 360);
        if ( !v109 )
          goto LABEL_159;
        v87 = v109 - 1;
      }
      v88 = v40[1];
      v89 = v87 & (37 * v88);
      v80 = (__m128i *)(v86 + 24LL * v89);
      v90 = v80->m128i_i32[0];
      if ( v88 != v80->m128i_i32[0] )
      {
        v113 = 1;
        v42 = 0;
        while ( v90 != -1 )
        {
          if ( v90 == -2 && !v42 )
            v42 = v80;
          v89 = v87 & (v113 + v89);
          v80 = (__m128i *)(v86 + 24LL * v89);
          v90 = v80->m128i_i32[0];
          if ( v88 == v80->m128i_i32[0] )
            goto LABEL_98;
          ++v113;
        }
        goto LABEL_115;
      }
LABEL_98:
      v79 = *(_DWORD *)(a1 + 352);
      goto LABEL_80;
    }
    v70 = *(_DWORD *)(a1 + 368);
LABEL_77:
    v82 = 3 * v70;
    goto LABEL_78;
  }
LABEL_5:
  v12 = sub_200D430(
          a1,
          *(_QWORD *)v9,
          *(_QWORD *)(v9 + 8),
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64);
  v14 = v13;
  v15 = v12;
  v16 = *(_QWORD *)(v12 + 40) + 16LL * (unsigned int)v13;
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOBYTE(v132) = v17;
  v133 = v18;
  if ( v17 )
  {
    switch ( v17 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        LOBYTE(v19) = 2;
        goto LABEL_50;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        LOBYTE(v19) = 3;
        goto LABEL_50;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        LOBYTE(v19) = 4;
        goto LABEL_50;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        LOBYTE(v19) = 5;
        goto LABEL_50;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        LOBYTE(v19) = 6;
        goto LABEL_50;
      case 55:
        LOBYTE(v19) = 7;
        goto LABEL_50;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v19) = 8;
        goto LABEL_50;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        LOBYTE(v19) = 9;
        goto LABEL_50;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v19) = 10;
LABEL_50:
        v21 = 0;
        goto LABEL_7;
      default:
        goto LABEL_159;
    }
  }
  LOBYTE(v19) = sub_1F596B0((__int64)&v132);
  v121 = v19;
  v21 = v20;
LABEL_7:
  v22 = v121;
  LOBYTE(v22) = v19;
  v23 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          106,
          (__int64)&v124,
          v22,
          v21,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v15,
          v14,
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  v24 = *(unsigned __int8 **)(a2 + 40);
  v26 = v25;
  v123 = *v24;
  sub_1F40D10((__int64)&v132, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v24, *((_QWORD *)v24 + 1));
  v27 = v133;
  v28 = *(_QWORD *)(a2 + 72);
  v29 = v134;
  v30 = *(__int64 **)(a1 + 8);
  v132 = v28;
  v31 = (unsigned __int8)v133;
  if ( v28 )
  {
    v118 = (unsigned __int8)v133;
    v119 = v133;
    v122 = v134;
    sub_1623A60((__int64)&v132, v28, 2);
    v31 = v118;
    v27 = v119;
    v29 = v122;
  }
  LODWORD(v133) = *(_DWORD *)(a2 + 64);
  if ( v123 == 8 )
  {
    v51 = 160;
  }
  else
  {
    if ( v27 != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v51 = 161;
  }
  *((_QWORD *)&v115 + 1) = v26;
  *(_QWORD *)&v115 = v23;
  v50 = sub_1D309E0(
          v30,
          v51,
          (__int64)&v132,
          v31,
          v29,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          v115);
  if ( v132 )
    sub_161E7C0((__int64)&v132, v132);
LABEL_29:
  if ( v124 )
    sub_161E7C0((__int64)&v124, v124);
  return v50;
}
