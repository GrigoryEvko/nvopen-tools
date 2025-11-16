// Function: sub_1AAC5F0
// Address: 0x1aac5f0
//
__int64 __fastcall sub_1AAC5F0(
        __int64 a1,
        unsigned int a2,
        __int128 *a3,
        __m128 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // r13
  __m128 *v20; // rax
  __m128 *v21; // rcx
  __int64 v22; // rsi
  __m128 *v23; // r8
  __int64 *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned int v27; // edx
  __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // rax
  unsigned __int64 v32; // rdi
  int v33; // r8d
  int v34; // r9d
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r14
  int v37; // r15d
  unsigned int v38; // ebx
  unsigned int v39; // r13d
  unsigned int v40; // r15d
  unsigned __int64 v41; // rax
  int v42; // edx
  __int64 v43; // rsi
  int v44; // edx
  unsigned int v45; // ecx
  __int64 *v46; // rax
  __int64 v47; // r8
  __int64 v48; // rbx
  unsigned int v49; // ecx
  __int64 *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rdi
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  _QWORD *v55; // rax
  _QWORD *v56; // r15
  __int64 v57; // rax
  __int64 v58; // r15
  _QWORD *v59; // rax
  int v60; // r9d
  double v61; // xmm4_8
  double v62; // xmm5_8
  __int64 v63; // r14
  int v64; // esi
  __int64 v65; // rdi
  unsigned int v66; // ecx
  __int64 *v67; // rdx
  __int64 v68; // r8
  __int64 v69; // rdx
  int v70; // esi
  _QWORD *v71; // rax
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  _QWORD **v75; // rdx
  char v76; // al
  __m128i *v77; // rcx
  char v78; // dl
  const char **v79; // rsi
  unsigned int v80; // r14d
  int i; // ebx
  _QWORD *v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rcx
  _QWORD *v86; // rax
  int v87; // eax
  __int64 v88; // rax
  int v89; // eax
  int v90; // r9d
  int v91; // edx
  int v92; // eax
  int v93; // r8d
  __int64 v94; // [rsp+8h] [rbp-158h]
  int v95; // [rsp+10h] [rbp-150h]
  int v96; // [rsp+14h] [rbp-14Ch]
  __m128 *v97; // [rsp+20h] [rbp-140h]
  __int128 v98; // [rsp+20h] [rbp-140h]
  __m128 *v99; // [rsp+20h] [rbp-140h]
  __m128 *v100; // [rsp+28h] [rbp-138h]
  __m128 *v101; // [rsp+28h] [rbp-138h]
  __m128 *v102; // [rsp+28h] [rbp-138h]
  __int64 v103; // [rsp+28h] [rbp-138h]
  __int64 v105; // [rsp+48h] [rbp-118h] BYREF
  _QWORD v106[2]; // [rsp+50h] [rbp-110h] BYREF
  _QWORD v107[2]; // [rsp+60h] [rbp-100h] BYREF
  __m128i v108; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v109; // [rsp+80h] [rbp-E0h]
  _QWORD *v110; // [rsp+90h] [rbp-D0h] BYREF
  __int16 v111; // [rsp+A0h] [rbp-C0h]
  __m128i v112; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v113; // [rsp+C0h] [rbp-A0h]
  const char *v114; // [rsp+D0h] [rbp-90h] BYREF
  char v115; // [rsp+E0h] [rbp-80h]
  char v116; // [rsp+E1h] [rbp-7Fh]
  __m128 v117; // [rsp+F0h] [rbp-70h] BYREF
  _QWORD v118[12]; // [rsp+100h] [rbp-60h] BYREF

  if ( !(unsigned __int8)sub_137E040(a1, a2, *((unsigned __int8 *)a3 + 16)) )
    return 0;
  v105 = *(_QWORD *)(a1 + 40);
  v12 = sub_15F4DF0(a1, a2);
  v13 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v12) + 16) - 34;
  if ( (unsigned int)v13 <= 0x36 )
  {
    v14 = 0x40018000000001LL;
    if ( _bittest64(&v14, v13) )
      return 0;
  }
  v116 = 1;
  v114 = "_crit_edge";
  v115 = 3;
  v107[0] = sub_1649960(v12);
  v107[1] = v15;
  v111 = 261;
  v110 = v107;
  v106[0] = sub_1649960(v105);
  v108.m128i_i64[0] = (__int64)v106;
  v108.m128i_i64[1] = (__int64)".";
  v16 = v111;
  v106[1] = v17;
  LOWORD(v109) = 773;
  if ( !(_BYTE)v111 )
  {
    LOWORD(v113) = 256;
LABEL_6:
    LOWORD(v118[0]) = 256;
    goto LABEL_7;
  }
  if ( (_BYTE)v111 != 1 )
  {
    v75 = (_QWORD **)v110;
    if ( HIBYTE(v111) != 1 )
    {
      v75 = &v110;
      v16 = 2;
    }
    BYTE1(v113) = v16;
    v76 = v115;
    v112.m128i_i64[0] = (__int64)&v108;
    v112.m128i_i64[1] = (__int64)v75;
    LOBYTE(v113) = 2;
    if ( !v115 )
      goto LABEL_6;
    if ( v115 != 1 )
      goto LABEL_91;
LABEL_109:
    a5 = _mm_load_si128(&v112);
    v117 = (__m128)a5;
    v118[0] = v113;
    goto LABEL_7;
  }
  a4 = (__m128)_mm_load_si128(&v108);
  v113 = v109;
  v76 = v115;
  v112 = (__m128i)a4;
  if ( !v115 )
    goto LABEL_6;
  if ( v115 == 1 )
    goto LABEL_109;
  if ( BYTE1(v113) == 1 )
  {
    v77 = (__m128i *)v112.m128i_i64[0];
    v78 = 5;
    goto LABEL_98;
  }
LABEL_91:
  v77 = &v112;
  v78 = 2;
LABEL_98:
  v79 = (const char **)v114;
  if ( v116 != 1 )
  {
    v79 = &v114;
    v76 = 2;
  }
  v117.m128_u64[0] = (unsigned __int64)v77;
  v117.m128_u64[1] = (unsigned __int64)v79;
  LOBYTE(v118[0]) = v78;
  BYTE1(v118[0]) = v76;
LABEL_7:
  v18 = sub_16498A0(a1);
  v19 = sub_22077B0(64);
  if ( v19 )
    sub_157FB60((_QWORD *)v19, v18, (__int64)&v117, 0, 0);
  v20 = (__m128 *)sub_1648A60(56, 1u);
  v21 = v20;
  if ( v20 )
  {
    v100 = v20;
    sub_15F8590((__int64)v20, v12, v19);
    v21 = v100;
  }
  v22 = *(_QWORD *)(a1 + 48);
  v23 = v21 + 3;
  v117.m128_u64[0] = v22;
  if ( v22 )
  {
    v101 = v21 + 3;
    v97 = v21;
    sub_1623A60((__int64)&v117, v22, 2);
    v23 = v101;
    if ( v101 == &v117 )
    {
      if ( v117.m128_u64[0] )
        sub_161E7C0((__int64)&v117, v117.m128_i64[0]);
      goto LABEL_15;
    }
    v21 = v97;
    v73 = v97[3].m128_i64[0];
    if ( !v73 )
    {
LABEL_84:
      v74 = (unsigned __int8 *)v117.m128_u64[0];
      v21[3].m128_u64[0] = v117.m128_u64[0];
      if ( v74 )
        sub_1623210((__int64)&v117, v74, (__int64)v23);
      goto LABEL_15;
    }
LABEL_83:
    v99 = v21;
    v102 = v23;
    sub_161E7C0((__int64)v23, v73);
    v21 = v99;
    v23 = v102;
    goto LABEL_84;
  }
  if ( v23 != &v117 )
  {
    v73 = v21[3].m128_i64[0];
    if ( v73 )
      goto LABEL_83;
  }
LABEL_15:
  sub_15F4ED0(a1, a2, v19);
  v24 = *(__int64 **)(v105 + 32);
  sub_15E01D0(*(_QWORD *)(v105 + 56) + 72LL, v19);
  v25 = *v24;
  v26 = *(_QWORD *)(v19 + 24);
  *(_QWORD *)(v19 + 32) = v24;
  v27 = 0;
  v25 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v19 + 24) = v25 | v26 & 7;
  *(_QWORD *)(v25 + 8) = v19 + 24;
  *v24 = *v24 & 7 | (v19 + 24);
  v28 = *(_QWORD *)(v12 + 48);
  while ( 1 )
  {
    if ( !v28 )
      BUG();
    if ( *(_BYTE *)(v28 - 8) != 77 )
      break;
    if ( (*(_BYTE *)(v28 - 1) & 0x40) != 0 )
      v29 = *(_QWORD *)(v28 - 32);
    else
      v29 = v28 - 24 - 24LL * (*(_DWORD *)(v28 - 4) & 0xFFFFFFF);
    v30 = 24LL * *(unsigned int *)(v28 + 32);
    v31 = (__int64 *)(v29 + v30 + 8 + 8LL * v27);
    if ( v105 == *v31 )
      goto LABEL_25;
    if ( (*(_DWORD *)(v28 - 4) & 0xFFFFFFF) != 0 )
    {
      v31 = (__int64 *)(v29 + v30 + 8);
      v27 = 0;
      while ( v105 != *v31 )
      {
        ++v27;
        ++v31;
        if ( (*(_DWORD *)(v28 - 4) & 0xFFFFFFF) == v27 )
          goto LABEL_27;
      }
LABEL_25:
      *v31 = v19;
      v28 = *(_QWORD *)(v28 + 8);
    }
    else
    {
LABEL_27:
      v27 = -1;
      *(_QWORD *)(v29 + v30 + 0x800000000LL) = v19;
      v28 = *(_QWORD *)(v28 + 8);
    }
  }
  if ( *((_BYTE *)a3 + 16) )
  {
    v80 = a2 + 1;
    for ( i = sub_15F4D60(a1); i != v80; ++v80 )
    {
      if ( v12 == sub_15F4DF0(a1, v80) )
      {
        sub_157F2D0(v12, v105, *((_BYTE *)a3 + 17));
        sub_15F4ED0(a1, v80, v19);
      }
    }
  }
  v98 = *a3;
  if ( *a3 == 0 )
    return v19;
  if ( !(_QWORD)v98 )
    goto LABEL_45;
  v118[2] = v19;
  v117.m128_u64[0] = (unsigned __int64)v118;
  v118[0] = v105;
  v118[1] = v19 & 0xFFFFFFFFFFFFFFFBLL;
  v118[3] = v12 & 0xFFFFFFFFFFFFFFFBLL;
  v117.m128_u64[1] = 0x300000002LL;
  v32 = sub_157EBA0(v105);
  if ( !v32 )
  {
    v96 = 0;
    goto LABEL_107;
  }
  v96 = sub_15F4D60(v32);
  v35 = sub_157EBA0(v105);
  if ( !v35 )
  {
LABEL_107:
    v95 = 0;
    goto LABEL_108;
  }
  v95 = sub_15F4D60(v35);
  v36 = sub_157EBA0(v105);
  v37 = v95 >> 2;
  if ( v95 >> 2 <= 0 )
  {
    v87 = v95;
    v38 = 0;
  }
  else
  {
    v94 = v19;
    v38 = 0;
    do
    {
      if ( v12 == sub_15F4DF0(v36, v38) )
      {
        v19 = v94;
        goto LABEL_42;
      }
      v39 = v38 + 1;
      if ( v12 == sub_15F4DF0(v36, v38 + 1)
        || (v39 = v38 + 2, v12 == sub_15F4DF0(v36, v38 + 2))
        || (v39 = v38 + 3, v12 == sub_15F4DF0(v36, v38 + 3)) )
      {
        v40 = v39;
        v19 = v94;
        v38 = v40;
        goto LABEL_42;
      }
      v38 += 4;
      --v37;
    }
    while ( v37 );
    v19 = v94;
    v87 = v95 - v38;
  }
  switch ( v87 )
  {
    case 2:
LABEL_146:
      if ( v12 != sub_15F4DF0(v36, v38) )
      {
        ++v38;
LABEL_140:
        if ( v12 != sub_15F4DF0(v36, v38) )
LABEL_108:
          v38 = v95;
      }
      break;
    case 3:
      if ( v12 != sub_15F4DF0(v36, v38) )
      {
        ++v38;
        goto LABEL_146;
      }
      break;
    case 1:
      goto LABEL_140;
    default:
      goto LABEL_108;
  }
LABEL_42:
  v41 = v117.m128_u32[2];
  if ( v38 == v96 )
  {
    if ( v117.m128_i32[2] >= (unsigned __int32)v117.m128_i32[3] )
    {
      sub_16CD150((__int64)&v117, v118, 0, 16, v33, v34);
      v41 = v117.m128_u32[2];
    }
    v41 = v117.m128_u64[0] + 16 * v41;
    *(_QWORD *)v41 = v105;
    *(_QWORD *)(v41 + 8) = v12 & 0xFFFFFFFFFFFFFFFBLL | 4;
    LODWORD(v41) = ++v117.m128_i32[2];
  }
  sub_15DC140(v98, (__int64 *)v117.m128_u64[0], (unsigned int)v41);
  if ( (_QWORD *)v117.m128_u64[0] != v118 )
    _libc_free(v117.m128_u64[0]);
LABEL_45:
  if ( !*((_QWORD *)&v98 + 1) )
    return v19;
  v42 = *(_DWORD *)(*((_QWORD *)&v98 + 1) + 24LL);
  if ( !v42 )
    return v19;
  v43 = *(_QWORD *)(*((_QWORD *)&v98 + 1) + 8LL);
  v44 = v42 - 1;
  v45 = v44 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
  v46 = (__int64 *)(v43 + 16LL * v45);
  v47 = *v46;
  if ( v105 != *v46 )
  {
    v89 = 1;
    while ( v47 != -8 )
    {
      v90 = v89 + 1;
      v45 = v44 & (v89 + v45);
      v46 = (__int64 *)(v43 + 16LL * v45);
      v47 = *v46;
      if ( v105 == *v46 )
        goto LABEL_48;
      v89 = v90;
    }
    return v19;
  }
LABEL_48:
  v48 = v46[1];
  if ( !v48 )
    return v19;
  v49 = v44 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v50 = (__int64 *)(v43 + 16LL * v49);
  v51 = *v50;
  if ( v12 == *v50 )
  {
LABEL_50:
    v52 = (_QWORD *)v50[1];
    if ( v52 )
    {
      v53 = (_QWORD *)v50[1];
      if ( (_QWORD *)v48 != v52 )
      {
        while ( 1 )
        {
          v53 = (_QWORD *)*v53;
          if ( (_QWORD *)v48 == v53 )
            break;
          if ( !v53 )
          {
            if ( v52 != (_QWORD *)v48 )
            {
              v86 = (_QWORD *)v48;
              while ( 1 )
              {
                v86 = (_QWORD *)*v86;
                if ( v52 == v86 )
                  break;
                if ( !v86 )
                {
                  v52 = (_QWORD *)*v52;
                  if ( v52 )
                    break;
                  goto LABEL_56;
                }
              }
            }
            sub_1400330((__int64)v52, v19, *((__int64 *)&v98 + 1));
            goto LABEL_56;
          }
        }
      }
      sub_1400330(v48, v19, *((__int64 *)&v98 + 1));
    }
  }
  else
  {
    v92 = 1;
    while ( v51 != -8 )
    {
      v93 = v92 + 1;
      v49 = v44 & (v92 + v49);
      v50 = (__int64 *)(v43 + 16LL * v49);
      v51 = *v50;
      if ( v12 == *v50 )
        goto LABEL_50;
      v92 = v93;
    }
  }
LABEL_56:
  v54 = *(_QWORD **)(v48 + 72);
  v55 = *(_QWORD **)(v48 + 64);
  if ( v54 == v55 )
  {
    v82 = &v55[*(unsigned int *)(v48 + 84)];
    if ( v55 == v82 )
    {
      v56 = *(_QWORD **)(v48 + 64);
    }
    else
    {
      do
      {
        if ( v12 == *v55 )
          break;
        ++v55;
      }
      while ( v82 != v55 );
      v56 = v82;
    }
    goto LABEL_113;
  }
  v56 = &v54[*(unsigned int *)(v48 + 80)];
  v55 = sub_16CC9F0(v48 + 56, v12);
  if ( v12 == *v55 )
  {
    v84 = *(_QWORD *)(v48 + 72);
    if ( v84 == *(_QWORD *)(v48 + 64) )
      v85 = *(unsigned int *)(v48 + 84);
    else
      v85 = *(unsigned int *)(v48 + 80);
    v82 = (_QWORD *)(v84 + 8 * v85);
    goto LABEL_113;
  }
  v57 = *(_QWORD *)(v48 + 72);
  if ( v57 == *(_QWORD *)(v48 + 64) )
  {
    v55 = (_QWORD *)(v57 + 8LL * *(unsigned int *)(v48 + 84));
    v82 = v55;
LABEL_113:
    while ( v82 != v55 && *v55 >= 0xFFFFFFFFFFFFFFFELL )
      ++v55;
    goto LABEL_60;
  }
  v55 = (_QWORD *)(v57 + 8LL * *(unsigned int *)(v48 + 80));
LABEL_60:
  if ( v56 == v55 && *((_BYTE *)a3 + 19) )
  {
    if ( *((_BYTE *)a3 + 18) )
      sub_1AABF30(&v105, 1, v19, v12);
    v117.m128_u64[0] = (unsigned __int64)v118;
    v117.m128_u64[1] = 0x400000000LL;
    v58 = *(_QWORD *)(v12 + 8);
    if ( v58 )
    {
      while ( 1 )
      {
        v59 = sub_1648700(v58);
        if ( (unsigned __int8)(*((_BYTE *)v59 + 16) - 25) <= 9u )
          break;
        v58 = *(_QWORD *)(v58 + 8);
        if ( !v58 )
          return v19;
      }
      v63 = *((_QWORD *)&v98 + 1);
LABEL_74:
      v88 = v59[5];
      if ( v88 == v19 )
        goto LABEL_72;
      v70 = *(_DWORD *)(v63 + 24);
      if ( v70 )
      {
        v64 = v70 - 1;
        v65 = *(_QWORD *)(v63 + 8);
        v66 = v64 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
        v67 = (__int64 *)(v65 + 16LL * v66);
        v68 = *v67;
        if ( v88 != *v67 )
        {
          v91 = 1;
          while ( v68 != -8 )
          {
            v60 = v91 + 1;
            v66 = v64 & (v66 + v91);
            v67 = (__int64 *)(v65 + 16LL * v66);
            v68 = *v67;
            if ( v88 == *v67 )
              goto LABEL_68;
            v91 = v60;
          }
          goto LABEL_76;
        }
LABEL_68:
        if ( v48 != v67[1] )
          goto LABEL_76;
        v69 = v117.m128_u32[2];
        if ( v117.m128_i32[2] >= (unsigned __int32)v117.m128_i32[3] )
        {
          v103 = v88;
          sub_16CD150((__int64)&v117, v118, 0, 8, v68, v60);
          v69 = v117.m128_u32[2];
          v88 = v103;
        }
        *(_QWORD *)(v117.m128_u64[0] + 8 * v69) = v88;
        ++v117.m128_i32[2];
LABEL_72:
        while ( 1 )
        {
          v58 = *(_QWORD *)(v58 + 8);
          if ( !v58 )
            break;
          v59 = sub_1648700(v58);
          if ( (unsigned __int8)(*((_BYTE *)v59 + 16) - 25) <= 9u )
            goto LABEL_74;
        }
        v71 = (_QWORD *)v117.m128_u64[0];
        if ( v117.m128_i32[2] )
        {
          v83 = sub_1AAB350(
                  v12,
                  (__int64 *)v117.m128_u64[0],
                  v117.m128_u32[2],
                  "split",
                  v98,
                  v63,
                  a4,
                  *(double *)a5.m128i_i64,
                  a6,
                  a7,
                  v61,
                  v62,
                  a10,
                  a11,
                  *((_BYTE *)a3 + 18));
          if ( *((_BYTE *)a3 + 18) )
            sub_1AABF30((__int64 *)v117.m128_u64[0], v117.m128_i32[2], v83, v12);
          goto LABEL_76;
        }
      }
      else
      {
LABEL_76:
        v71 = (_QWORD *)v117.m128_u64[0];
      }
      if ( v71 != v118 )
        _libc_free((unsigned __int64)v71);
    }
  }
  return v19;
}
