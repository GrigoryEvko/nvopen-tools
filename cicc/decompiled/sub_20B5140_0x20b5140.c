// Function: sub_20B5140
// Address: 0x20b5140
//
__int64 *__fastcall sub_20B5140(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        char a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9)
{
  char *v12; // rax
  __int64 v13; // rsi
  unsigned __int8 v14; // r14
  const void **v15; // rax
  __int64 v16; // rdi
  const __m128i *v17; // roff
  __m128i v18; // xmm0
  unsigned int v19; // eax
  __int64 v21; // rax
  __int64 *v22; // r13
  __int64 v24; // rax
  __int128 v25; // rax
  int v26; // r9d
  __int64 *v27; // r10
  unsigned int v28; // edx
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  const void **v33; // rdx
  __int128 v34; // rax
  __int64 *v35; // rax
  int v36; // r9d
  __int64 *v37; // r8
  unsigned __int64 v38; // rdx
  unsigned int v39; // r15d
  __int64 v40; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // rax
  const void **v43; // rdx
  __int128 v44; // rax
  __int64 *v45; // r8
  int v46; // r9d
  unsigned int v47; // edx
  unsigned int v48; // r15d
  __int64 v49; // rax
  int v50; // edx
  int v51; // r9d
  __int64 *v52; // r15
  unsigned int v53; // r8d
  __int64 v54; // rax
  __int64 v55; // rbx
  __int64 v56; // rax
  const void **v57; // rdx
  __int128 v58; // rax
  __int128 v59; // rax
  const void ***v60; // rax
  int v61; // edx
  __int64 v62; // r9
  __int64 *v63; // rax
  bool v65; // zf
  unsigned int v66; // edx
  unsigned __int8 *v67; // rax
  __int64 v68; // rax
  const void **v69; // rdx
  __int128 v70; // rax
  __int64 *v71; // rax
  int v72; // r8d
  int v73; // r9d
  unsigned __int32 v74; // edx
  unsigned __int64 v75; // rcx
  __int64 v76; // rdx
  unsigned int v77; // eax
  __int64 *v78; // [rsp+8h] [rbp-F8h]
  unsigned __int32 v79; // [rsp+1Ch] [rbp-E4h]
  __int64 v80; // [rsp+20h] [rbp-E0h]
  unsigned int v81; // [rsp+28h] [rbp-D8h]
  __int64 v82; // [rsp+30h] [rbp-D0h]
  unsigned int v84; // [rsp+38h] [rbp-C8h]
  __int64 *v85; // [rsp+38h] [rbp-C8h]
  __int64 v87; // [rsp+40h] [rbp-C0h]
  __int64 *v88; // [rsp+40h] [rbp-C0h]
  __int128 v89; // [rsp+40h] [rbp-C0h]
  __int64 *v90; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v91; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v92; // [rsp+48h] [rbp-B8h]
  const void **v93; // [rsp+50h] [rbp-B0h]
  __int128 v95; // [rsp+60h] [rbp-A0h]
  int v96; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v97; // [rsp+68h] [rbp-98h]
  unsigned __int64 v98; // [rsp+68h] [rbp-98h]
  unsigned __int64 v99; // [rsp+68h] [rbp-98h]
  __int64 v100; // [rsp+70h] [rbp-90h] BYREF
  int v101; // [rsp+78h] [rbp-88h]
  unsigned __int64 v102; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v103; // [rsp+88h] [rbp-78h]
  __int64 v104; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v105; // [rsp+98h] [rbp-68h]
  char v106; // [rsp+A0h] [rbp-60h]
  unsigned int v107; // [rsp+A4h] [rbp-5Ch]
  __int64 v108; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v109; // [rsp+B8h] [rbp-48h]
  char v110; // [rsp+C0h] [rbp-40h]
  unsigned int v111; // [rsp+C4h] [rbp-3Ch]

  v12 = *(char **)(a2 + 40);
  v13 = *(_QWORD *)(a2 + 72);
  v14 = *v12;
  v15 = (const void **)*((_QWORD *)v12 + 1);
  v100 = v13;
  v93 = v15;
  if ( v13 )
    sub_1623A60((__int64)&v100, v13, 2);
  v16 = a4[4];
  v101 = *(_DWORD *)(a2 + 64);
  v82 = sub_1E0A0C0(v16);
  if ( !v14 || !*(_QWORD *)(a1 + 8LL * v14 + 120) )
  {
    v22 = 0;
    goto LABEL_14;
  }
  sub_16AC9F0((__int64)&v104, a3, 0);
  v17 = *(const __m128i **)(a2 + 32);
  v18 = _mm_loadu_si128(v17);
  v80 = v17->m128i_i64[0];
  v79 = v17->m128i_u32[2];
  v97 = v18.m128i_u64[1];
  if ( v106 )
  {
    v19 = *(_DWORD *)(a3 + 8);
    _RDX = *(_BYTE **)a3;
    if ( v19 > 0x40 )
    {
      if ( (*_RDX & 1) != 0 )
        goto LABEL_8;
      v81 = sub_16A58A0(a3);
    }
    else
    {
      if ( ((unsigned __int8)_RDX & 1) != 0 )
        goto LABEL_8;
      __asm { tzcnt   rcx, rdx }
      v65 = _RDX == 0;
      v66 = 64;
      if ( !v65 )
        v66 = _RCX;
      if ( v19 <= v66 )
        v66 = *(_DWORD *)(a3 + 8);
      v81 = v66;
    }
    v67 = (unsigned __int8 *)(*(_QWORD *)(v80 + 40) + 16LL * v79);
    v68 = sub_1F40B60(a1, *v67, *((_QWORD *)v67 + 1), v82, 1);
    *(_QWORD *)&v70 = sub_1D38BB0((__int64)a4, v81, (__int64)&v100, v68, v69, 0, v18, a8, a9, 0);
    v71 = sub_1D332F0(
            a4,
            124,
            (__int64)&v100,
            v14,
            v93,
            0,
            *(double *)v18.m128i_i64,
            a8,
            a9,
            v18.m128i_i64[0],
            v18.m128i_u64[1],
            v70);
    v79 = v74;
    v80 = (__int64)v71;
    v75 = v74 | v18.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v76 = *(unsigned int *)(a6 + 8);
    v97 = v75;
    if ( (unsigned int)v76 >= *(_DWORD *)(a6 + 12) )
    {
      v78 = v71;
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v72, v73);
      v76 = *(unsigned int *)(a6 + 8);
      v71 = v78;
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v76) = v71;
    ++*(_DWORD *)(a6 + 8);
    v77 = *(_DWORD *)(a3 + 8);
    v103 = v77;
    if ( v77 > 0x40 )
    {
      sub_16A4FD0((__int64)&v102, (const void **)a3);
      v77 = v103;
      if ( v103 > 0x40 )
      {
        sub_16A8110((__int64)&v102, v81);
LABEL_59:
        sub_16AC9F0((__int64)&v108, (__int64)&v102, v81);
        if ( v105 > 0x40 && v104 )
          j_j___libc_free_0_0(v104);
        v104 = v108;
        v105 = v109;
        v106 = v110;
        v107 = v111;
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        goto LABEL_8;
      }
    }
    else
    {
      v102 = *(_QWORD *)a3;
    }
    if ( v81 == v77 )
      v102 = 0;
    else
      v102 >>= v81;
    goto LABEL_59;
  }
LABEL_8:
  if ( !a5 )
  {
    if ( v14 == 1 )
    {
      if ( (*(_BYTE *)(a1 + 2793) & 0xFB) == 0 )
        goto LABEL_30;
      v24 = 1;
    }
    else
    {
      if ( !*(_QWORD *)(a1 + 8LL * v14 + 120) )
        goto LABEL_25;
      if ( (*(_BYTE *)(a1 + 259LL * v14 + 2534) & 0xFB) == 0 )
        goto LABEL_30;
      if ( !*(_QWORD *)(a1 + 8 * (v14 + 14LL) + 8) )
        goto LABEL_25;
      v24 = v14;
    }
    if ( (*(_BYTE *)(a1 + 259 * v24 + 2482) & 0xFB) != 0 )
      goto LABEL_25;
LABEL_47:
    *(_QWORD *)&v59 = sub_1D38970((__int64)a4, (__int64)&v104, (__int64)&v100, v14, v93, 0, v18, a8, a9, 0);
    v89 = v59;
    v60 = (const void ***)sub_1D252B0((__int64)a4, v14, (__int64)v93, v14, (__int64)v93);
    v99 = v79 | v97 & 0xFFFFFFFF00000000LL;
    v63 = sub_1D37440(
            a4,
            60,
            (__int64)&v100,
            v60,
            v61,
            v62,
            *(double *)v18.m128i_i64,
            a8,
            a9,
            __PAIR128__(v99, v80),
            v89);
    v29 = 1;
    v27 = v63;
    goto LABEL_31;
  }
  if ( v14 == 1 )
  {
    if ( *(_BYTE *)(a1 + 2793) )
    {
      v21 = 1;
LABEL_24:
      if ( *(_BYTE *)(a1 + 259 * v21 + 2482) )
        goto LABEL_25;
      goto LABEL_47;
    }
LABEL_30:
    *(_QWORD *)&v25 = sub_1D38970((__int64)a4, (__int64)&v104, (__int64)&v100, v14, v93, 0, v18, a8, a9, 0);
    v98 = v79 | v97 & 0xFFFFFFFF00000000LL;
    v27 = sub_1D332F0(a4, 112, (__int64)&v100, v14, v93, 0, *(double *)v18.m128i_i64, a8, a9, v80, v98, v25);
    v29 = v28;
    v99 = v28 | v98 & 0xFFFFFFFF00000000LL;
LABEL_31:
    v30 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v30 >= *(_DWORD *)(a6 + 12) )
    {
      v84 = v29;
      v90 = v27;
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v29, v26);
      v30 = *(unsigned int *)(a6 + 8);
      v29 = v84;
      v27 = v90;
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v30) = v27;
    ++*(_DWORD *)(a6 + 8);
    if ( v106 )
    {
      *(_QWORD *)&v95 = v27;
      *((_QWORD *)&v95 + 1) = v29 | v99 & 0xFFFFFFFF00000000LL;
      v88 = sub_1D332F0(
              a4,
              53,
              (__int64)&v100,
              v14,
              v93,
              0,
              *(double *)v18.m128i_i64,
              a8,
              a9,
              **(_QWORD **)(a2 + 32),
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
              __PAIR128__(*((unsigned __int64 *)&v95 + 1), (unsigned __int64)v27));
      v37 = v88;
      v39 = v38;
      v40 = *(unsigned int *)(a6 + 8);
      v91 = v38;
      if ( (unsigned int)v40 >= *(_DWORD *)(a6 + 12) )
      {
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, (int)v88, v36);
        v40 = *(unsigned int *)(a6 + 8);
        v37 = v88;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v40) = v37;
      ++*(_DWORD *)(a6 + 8);
      v41 = (unsigned __int8 *)(v37[5] + 16LL * v39);
      v42 = sub_1F40B60(a1, *v41, *((_QWORD *)v41 + 1), v82, 1);
      *(_QWORD *)&v44 = sub_1D38BB0((__int64)a4, 1, (__int64)&v100, v42, v43, 0, v18, a8, a9, 0);
      v45 = sub_1D332F0(a4, 124, (__int64)&v100, v14, v93, 0, *(double *)v18.m128i_i64, a8, a9, (__int64)v88, v91, v44);
      v48 = v47;
      v49 = *(unsigned int *)(a6 + 8);
      if ( (unsigned int)v49 >= *(_DWORD *)(a6 + 12) )
      {
        v85 = v45;
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, (int)v45, v46);
        v49 = *(unsigned int *)(a6 + 8);
        v45 = v85;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v49) = v45;
      ++*(_DWORD *)(a6 + 8);
      v92 = v48 | v91 & 0xFFFFFFFF00000000LL;
      v52 = sub_1D332F0(a4, 52, (__int64)&v100, v14, v93, 0, *(double *)v18.m128i_i64, a8, a9, (__int64)v45, v92, v95);
      v53 = v50;
      v54 = *(unsigned int *)(a6 + 8);
      if ( (unsigned int)v54 >= *(_DWORD *)(a6 + 12) )
      {
        v96 = v50;
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v50, v51);
        v54 = *(unsigned int *)(a6 + 8);
        v53 = v96;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v54) = v52;
      ++*(_DWORD *)(a6 + 8);
      v55 = v53;
      v56 = sub_1F40B60(a1, *(unsigned __int8 *)(v52[5] + 16 * v55), *(_QWORD *)(v52[5] + 16 * v55 + 8), v82, 1);
      *(_QWORD *)&v58 = sub_1D38BB0((__int64)a4, v107 - 1, (__int64)&v100, v56, v57, 0, v18, a8, a9, 0);
      v35 = sub_1D332F0(
              a4,
              124,
              (__int64)&v100,
              v14,
              v93,
              0,
              *(double *)v18.m128i_i64,
              a8,
              a9,
              (__int64)v52,
              v55 | v92 & 0xFFFFFFFF00000000LL,
              v58);
    }
    else
    {
      v31 = (unsigned int)v29;
      v87 = (__int64)v27;
      v32 = sub_1F40B60(a1, *(unsigned __int8 *)(v27[5] + 16 * v31), *(_QWORD *)(v27[5] + 16 * v31 + 8), v82, 1);
      *(_QWORD *)&v34 = sub_1D38BB0((__int64)a4, v107, (__int64)&v100, v32, v33, 0, v18, a8, a9, 0);
      v35 = sub_1D332F0(
              a4,
              124,
              (__int64)&v100,
              v14,
              v93,
              0,
              *(double *)v18.m128i_i64,
              a8,
              a9,
              v87,
              v31 | v99 & 0xFFFFFFFF00000000LL,
              v34);
    }
    v22 = v35;
    goto LABEL_26;
  }
  if ( *(_QWORD *)(a1 + 8LL * v14 + 120) )
  {
    if ( *(_BYTE *)(a1 + 259LL * v14 + 2534) )
    {
      v21 = v14;
      goto LABEL_24;
    }
    goto LABEL_30;
  }
LABEL_25:
  v22 = 0;
LABEL_26:
  if ( v105 > 0x40 && v104 )
    j_j___libc_free_0_0(v104);
LABEL_14:
  if ( v100 )
    sub_161E7C0((__int64)&v100, v100);
  return v22;
}
