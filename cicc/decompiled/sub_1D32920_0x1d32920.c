// Function: sub_1D32920
// Address: 0x1d32920
//
__int64 *__fastcall sub_1D32920(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        unsigned __int64 a10)
{
  __int64 *result; // rax
  __int64 v12; // r15
  __int64 v15; // r9
  int v16; // edx
  __int16 v17; // ax
  __int64 v18; // r9
  __int64 v19; // r8
  int v20; // edx
  _QWORD *v21; // r12
  __int64 v22; // rdi
  bool (__fastcall *v23)(__int64, unsigned int); // rdx
  __int32 v24; // eax
  __int64 v25; // r9
  bool v26; // zf
  __int64 v27; // rdx
  __m128i v28; // xmm1
  char v29; // al
  __int64 v30; // rax
  int v31; // r14d
  __int64 v32; // rsi
  unsigned __int64 v33; // rcx
  __int128 v34; // xmm0
  __int64 v35; // rbx
  __int64 *v36; // rax
  __int64 v37; // r12
  unsigned __int64 v38; // r13
  __int64 v39; // r15
  __int64 v40; // rdx
  char v41; // al
  __int64 v42; // rax
  char v43; // di
  __int64 v44; // rax
  __int64 v45; // rax
  char v46; // di
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __m128i v50; // rax
  __int64 v51; // rcx
  int v52; // r8d
  int v53; // r9d
  int v54; // eax
  unsigned __int32 v55; // eax
  int v56; // r8d
  int v57; // r9d
  unsigned __int64 v58; // rcx
  __m128i *v59; // r10
  unsigned __int64 v60; // rbx
  __int32 v61; // r13d
  __int64 v62; // rdx
  __int16 v63; // ax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  unsigned __int32 v68; // eax
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  unsigned int v74; // eax
  __int64 v75; // rax
  __m128i v76; // xmm5
  __int64 v77; // rdx
  __int32 v78; // eax
  unsigned int v79; // eax
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rcx
  char v83; // di
  __m128i v84; // xmm6
  int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r9
  __int64 v89; // r8
  unsigned int v90; // eax
  const __m128i *v91; // r12
  unsigned __int32 v92; // eax
  __m128i *v93; // rax
  __m128i *i; // rdx
  __int128 v95; // [rsp-10h] [rbp-190h]
  __int128 v96; // [rsp-10h] [rbp-190h]
  __int128 v97; // [rsp-10h] [rbp-190h]
  int v98; // [rsp-10h] [rbp-190h]
  __int64 v99; // [rsp-8h] [rbp-188h]
  int v100; // [rsp-8h] [rbp-188h]
  unsigned int v101; // [rsp+10h] [rbp-170h]
  unsigned __int8 v102; // [rsp+10h] [rbp-170h]
  unsigned int v103; // [rsp+14h] [rbp-16Ch]
  __int64 v104; // [rsp+20h] [rbp-160h]
  __int64 v105; // [rsp+30h] [rbp-150h]
  __int64 *v106; // [rsp+38h] [rbp-148h]
  __int64 v107; // [rsp+40h] [rbp-140h]
  __int64 v108; // [rsp+50h] [rbp-130h]
  unsigned __int64 v109; // [rsp+58h] [rbp-128h]
  __m128i v110; // [rsp+60h] [rbp-120h] BYREF
  __int64 v111; // [rsp+70h] [rbp-110h]
  unsigned __int64 v112; // [rsp+78h] [rbp-108h]
  __m128i v113; // [rsp+80h] [rbp-100h]
  __int64 v114; // [rsp+90h] [rbp-F0h]
  __int64 v115; // [rsp+98h] [rbp-E8h]
  __int64 v116; // [rsp+A0h] [rbp-E0h]
  __int64 v117; // [rsp+A8h] [rbp-D8h]
  __int64 v118; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v119; // [rsp+B8h] [rbp-C8h]
  __m128i v120; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v121; // [rsp+D0h] [rbp-B0h] BYREF
  char v122[8]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v123; // [rsp+E8h] [rbp-98h]
  __m128i v124; // [rsp+F0h] [rbp-90h] BYREF
  __m128i v125; // [rsp+100h] [rbp-80h] BYREF
  __int64 v126; // [rsp+110h] [rbp-70h] BYREF
  int v127; // [rsp+118h] [rbp-68h]

  v118 = a4;
  v119 = a5;
  if ( a2 > 0x102 )
    return 0;
  v125.m128i_i64[0] = a6;
  v12 = (__int64)a1;
  v125.m128i_i32[2] = 0;
  v126 = a10;
  v127 = 0;
  if ( sub_1D18610((__int64)a1, a2, (__int64)&v125) )
  {
    v125.m128i_i64[0] = 0;
    v125.m128i_i32[2] = 0;
    v21 = sub_1D2B300(a1, 0x30u, (__int64)&v125, v118, v119, v15);
    if ( v125.m128i_i64[0] )
      sub_161E7C0((__int64)&v125, v125.m128i_i64[0]);
    return v21;
  }
  v16 = *(unsigned __int16 *)(a6 + 24);
  v17 = *(_WORD *)(a6 + 24);
  if ( (_WORD)v16 == 32 || v16 == 10 )
  {
    v20 = *(unsigned __int16 *)(a10 + 24);
    if ( v20 == 10 || v20 == 32 )
      return (__int64 *)sub_1D392A0(a1, a2, a3, (unsigned int)v118, v119, a6, a10);
  }
  if ( (unsigned __int16)(v17 - 34) <= 1u || (unsigned __int16)(v17 - 12) <= 1u )
  {
    v18 = a10;
    v19 = a6;
    return sub_1D29890(v12, a2, v118, v119, v19, v18);
  }
  v22 = a1[2];
  v23 = *(bool (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v22 + 776LL);
  if ( v23 != sub_1D12DA0 )
  {
    if ( !v23(v22, a2) )
    {
LABEL_59:
      v17 = *(_WORD *)(a6 + 24);
      goto LABEL_22;
    }
LABEL_54:
    v63 = *(_WORD *)(a10 + 24);
    if ( (unsigned __int16)(v63 - 12) <= 1u || (unsigned __int16)(v63 - 34) <= 1u )
    {
      v19 = a10;
      v18 = a6;
      return sub_1D29890(v12, a2, v118, v119, v19, v18);
    }
    goto LABEL_59;
  }
  if ( a2 > 0x78 )
  {
    if ( a2 - 180 > 3 )
      goto LABEL_22;
    goto LABEL_54;
  }
  if ( a2 > 0x33 )
  {
    switch ( a2 )
    {
      case '4':
      case '6':
      case ';':
      case '<':
      case '@':
      case 'B':
      case 'F':
      case 'G':
      case 'L':
      case 'N':
      case 'p':
      case 'q':
      case 'r':
      case 's':
      case 't':
      case 'u':
      case 'v':
      case 'w':
      case 'x':
        goto LABEL_54;
      default:
        break;
    }
  }
LABEL_22:
  if ( v17 != 104 || *(_WORD *)(a10 + 24) != 104 )
    return 0;
  LOBYTE(v24) = sub_1D15870((char *)&v118);
  v26 = *(_BYTE *)(v12 + 658) == 0;
  v120.m128i_i32[0] = v24;
  v120.m128i_i64[1] = v27;
  v28 = _mm_loadu_si128(&v120);
  v121 = v28;
  if ( v26 )
    goto LABEL_28;
  v29 = v121.m128i_i8[0]
      ? (unsigned __int8)(v121.m128i_i8[0] - 2) <= 5u || (unsigned __int8)(v121.m128i_i8[0] - 14) <= 0x47u
      : sub_1F58CF0(&v121);
  if ( !v29 )
    goto LABEL_28;
  v80 = *(_QWORD *)(v12 + 16);
  sub_1F40D10(&v125, v80, *(_QWORD *)(v12 + 48), v121.m128i_i64[0], v121.m128i_i64[1]);
  v83 = v125.m128i_i8[8];
  v84 = _mm_loadu_si128(&v120);
  v121.m128i_i8[0] = v125.m128i_i8[8];
  v121.m128i_i64[1] = v126;
  v125 = v84;
  if ( v121.m128i_i8[0] != v120.m128i_i8[0] )
  {
    if ( v83 )
    {
      LODWORD(v112) = sub_1D13440(v83);
LABEL_86:
      if ( (_BYTE)v89 )
        v90 = sub_1D13440(v89);
      else
        v90 = sub_1F58D40(&v125, v80, v86, v87, v89, v88);
      if ( v90 > (unsigned int)v112 )
        return 0;
      goto LABEL_28;
    }
LABEL_85:
    v110.m128i_i8[0] = v120.m128i_i8[0];
    v85 = sub_1F58D40(&v121, v80, v81, v82, v120.m128i_u8[0], v25);
    v89 = v110.m128i_u8[0];
    LODWORD(v112) = v85;
    goto LABEL_86;
  }
  v81 = v125.m128i_i64[1];
  if ( !v120.m128i_i8[0] && v126 != v125.m128i_i64[1] )
    goto LABEL_85;
LABEL_28:
  v125.m128i_i64[0] = (__int64)&v126;
  v125.m128i_i64[1] = 0x400000000LL;
  v30 = *(unsigned int *)(a6 + 56);
  if ( (_DWORD)v30 )
  {
    v112 = 0;
    v106 = (__int64 *)v12;
    v107 = v120.m128i_i64[1];
    v104 = 40 * v30;
    v103 = a2;
    v105 = a3;
    v111 = a6;
    v31 = v120.m128i_u8[0];
    while ( 1 )
    {
      v32 = v112;
      v33 = a10;
      v34 = (__int128)_mm_loadu_si128((const __m128i *)(v112 + *(_QWORD *)(v111 + 32)));
      v35 = *(_QWORD *)(v112 + *(_QWORD *)(v111 + 32));
      v109 = *((_QWORD *)&v34 + 1);
      v36 = (__int64 *)(v112 + *(_QWORD *)(a10 + 32));
      v37 = *v36;
      v38 = v36[1];
      v39 = *v36;
      if ( (_BYTE)v31 )
      {
        v40 = (unsigned int)(v31 - 14);
        LOBYTE(v40) = (unsigned __int8)(v31 - 14) <= 0x47u;
        v41 = v40 | ((unsigned __int8)(v31 - 2) <= 5u);
      }
      else
      {
        v41 = sub_1F58CF0(&v120);
      }
      if ( !v41 )
        goto LABEL_37;
      v42 = *(_QWORD *)(v35 + 40);
      a9 = _mm_loadu_si128(&v120);
      v43 = *(_BYTE *)v42;
      v110 = a9;
      v44 = *(_QWORD *)(v42 + 8);
      v124 = a9;
      v122[0] = v43;
      v123 = v44;
      if ( a9.m128i_i8[0] != v43 )
        break;
      if ( !a9.m128i_i8[0] && v44 != v124.m128i_i64[1] )
        goto LABEL_79;
LABEL_35:
      v45 = *(_QWORD *)(v39 + 40);
      v46 = *(_BYTE *)v45;
      v47 = *(_QWORD *)(v45 + 8);
      v124 = _mm_load_si128(&v110);
      v122[0] = v46;
      v123 = v47;
      if ( v46 != v110.m128i_i8[0] )
      {
        if ( v46 )
        {
          v110.m128i_i32[0] = sub_1D13440(v46);
        }
        else
        {
LABEL_76:
          v102 = v110.m128i_i8[0];
          v78 = sub_1F58D40(v122, v32, v40, v33, v110.m128i_u8[0], v25);
          v66 = v102;
          v110.m128i_i32[0] = v78;
        }
        if ( (_BYTE)v66 )
          v68 = sub_1D13440(v66);
        else
          v68 = sub_1F58D40(&v124, v32, v64, v65, v66, v67);
        if ( v68 < v110.m128i_i32[0] )
        {
          *((_QWORD *)&v97 + 1) = v38;
          *(_QWORD *)&v97 = v37;
          v114 = sub_1D309E0(
                   v106,
                   145,
                   v105,
                   v120.m128i_u32[0],
                   (const void **)v120.m128i_i64[1],
                   0,
                   *(double *)&v34,
                   *(double *)v28.m128i_i64,
                   *(double *)a9.m128i_i64,
                   v97);
          v39 = v114;
          v115 = v69;
          v38 = (unsigned int)v69 | v38 & 0xFFFFFFFF00000000LL;
        }
        goto LABEL_37;
      }
      if ( !v110.m128i_i8[0] && v47 != v124.m128i_i64[1] )
        goto LABEL_76;
LABEL_37:
      v48 = *(_QWORD *)(v35 + 40);
      if ( *(_BYTE *)v48 != (_BYTE)v31 || *(_QWORD *)(v48 + 8) != v107 && !(_BYTE)v31 )
        goto LABEL_57;
      v49 = *(_QWORD *)(v39 + 40);
      if ( *(_BYTE *)v49 != (_BYTE)v31 || *(_QWORD *)(v49 + 8) != v107 && !(_BYTE)v31 )
        goto LABEL_57;
      *((_QWORD *)&v95 + 1) = v38;
      *(_QWORD *)&v95 = v39;
      v50.m128i_i64[0] = sub_1D332F0((_DWORD)v106, v103, v105, v120.m128i_i32[0], v120.m128i_i32[2], 0, v35, v109, v95);
      v124 = v50;
      if ( v121.m128i_i8[0] != (_BYTE)v31 || !(_BYTE)v31 && v121.m128i_i64[1] != v107 )
      {
        v50.m128i_i64[0] = sub_1D309E0(
                             v106,
                             142,
                             v105,
                             v121.m128i_u32[0],
                             (const void **)v121.m128i_i64[1],
                             0,
                             *(double *)&v34,
                             *(double *)v28.m128i_i64,
                             *(double *)a9.m128i_i64,
                             *(_OWORD *)&v124);
        v52 = v98;
        v53 = v100;
        v113 = v50;
        v124.m128i_i64[0] = v50.m128i_i64[0];
        v124.m128i_i32[2] = v50.m128i_i32[2];
      }
      v54 = *(unsigned __int16 *)(v124.m128i_i64[0] + 24);
      if ( (_WORD)v54 != 48 && (unsigned int)(v54 - 10) > 1 )
      {
LABEL_57:
        result = 0;
        v62 = 0;
        goto LABEL_52;
      }
      sub_1D23890((__int64)&v125, &v124, v50.m128i_i64[1], v51, v52, v53);
      v112 += 40LL;
      if ( v104 == v112 )
      {
        v110.m128i_i64[0] = (__int64)&v124;
        LODWORD(v12) = (_DWORD)v106;
        v112 = v125.m128i_u32[2];
        v55 = sub_1D15970(&v118);
        v58 = v112;
        v59 = (__m128i *)v110.m128i_i64[0];
        v60 = v55;
        v61 = v55;
        if ( v55 < v112 )
          goto LABEL_49;
        v91 = (const __m128i *)(v125.m128i_i64[0] + 16 * v112 - 16);
        goto LABEL_94;
      }
    }
    if ( v43 )
    {
      v101 = sub_1D13440(v43);
    }
    else
    {
LABEL_79:
      v79 = sub_1F58D40(v122, v32, v40, v33, a9.m128i_u8[0], v25);
      v72 = a9.m128i_u8[0];
      v101 = v79;
    }
    if ( (_BYTE)v72 )
      v74 = sub_1D13440(v72);
    else
      v74 = sub_1F58D40(&v124, v32, v70, v71, v72, v73);
    if ( v74 < v101 )
    {
      v32 = 145;
      v75 = sub_1D309E0(
              v106,
              145,
              v105,
              v120.m128i_u32[0],
              (const void **)v120.m128i_i64[1],
              0,
              *(double *)&v34,
              *(double *)v28.m128i_i64,
              *(double *)a9.m128i_i64,
              v34);
      v76 = _mm_loadu_si128(&v120);
      v33 = 0xFFFFFFFF00000000LL;
      v116 = v75;
      v35 = v75;
      v117 = v77;
      v110 = v76;
      v109 = (unsigned int)v77 | *((_QWORD *)&v34 + 1) & 0xFFFFFFFF00000000LL;
      v40 = v99;
    }
    goto LABEL_35;
  }
  v91 = &v125;
  v92 = sub_1D15970(&v118);
  v58 = 0;
  v59 = &v124;
  v60 = v92;
  v61 = v92;
LABEL_94:
  if ( v60 > v58 )
  {
    if ( v60 > v125.m128i_u32[3] )
    {
      v112 = (unsigned __int64)v59;
      sub_16CD150((__int64)&v125, &v126, v60, 16, v56, v57);
      v59 = (__m128i *)v112;
    }
    v93 = (__m128i *)(v125.m128i_i64[0] + 16 * v60);
    for ( i = (__m128i *)(v125.m128i_i64[0] + 16LL * v125.m128i_u32[2]); v93 != i; ++i )
    {
      if ( i )
        *i = _mm_loadu_si128(v91);
    }
LABEL_49:
    v125.m128i_i32[2] = v61;
  }
  v124.m128i_i64[0] = 0;
  *((_QWORD *)&v96 + 1) = v125.m128i_u32[2];
  *(_QWORD *)&v96 = v125.m128i_i64[0];
  v124.m128i_i32[2] = 0;
  v108 = (__int64)v59;
  result = (__int64 *)sub_1D359D0(v12, 104, (_DWORD)v59, v118, v119, 0, v96);
  if ( v124.m128i_i64[0] )
  {
    v110.m128i_i64[0] = v62;
    v112 = (unsigned __int64)result;
    sub_161E7C0(v108, v124.m128i_i64[0]);
    v62 = v110.m128i_i64[0];
    result = (__int64 *)v112;
  }
LABEL_52:
  if ( (__int64 *)v125.m128i_i64[0] != &v126 )
  {
    v110.m128i_i64[0] = v62;
    v112 = (unsigned __int64)result;
    _libc_free(v125.m128i_u64[0]);
    return (__int64 *)v112;
  }
  return result;
}
