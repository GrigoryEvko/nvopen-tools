// Function: sub_1F729C0
// Address: 0x1f729c0
//
__int64 __fastcall sub_1F729C0(
        int a1,
        unsigned int a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  unsigned int v9; // r15d
  __int64 v10; // rbx
  int v11; // edi
  __int64 *v12; // r14
  unsigned int v13; // r12d
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int64 v21; // rdx
  char v22; // r8
  __int64 result; // rax
  unsigned int v24; // r12d
  char v25; // al
  __int64 v26; // r12
  const __m128i *v27; // rax
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rcx
  unsigned int v31; // edx
  __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  int v34; // eax
  __int64 v35; // rsi
  __int64 *v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rcx
  char v39; // r8
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rax
  const void **v43; // rdx
  __int128 v44; // rax
  __int64 *v45; // rax
  unsigned int v46; // edx
  unsigned int v47; // r12d
  const __m128i *v48; // rax
  char v49; // di
  __int64 v50; // r12
  __int64 v51; // rax
  bool v52; // zf
  __int64 v53; // rdx
  char v54; // di
  __int64 v55; // rax
  int v56; // eax
  const __m128i *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r15
  __int128 v60; // rax
  unsigned int v61; // edx
  __int128 v62; // [rsp-10h] [rbp-120h]
  unsigned int v63; // [rsp+8h] [rbp-108h]
  unsigned int v64; // [rsp+10h] [rbp-100h]
  __int64 v65; // [rsp+10h] [rbp-100h]
  __int64 v66; // [rsp+18h] [rbp-F8h]
  __int64 v67; // [rsp+18h] [rbp-F8h]
  __int64 v68; // [rsp+20h] [rbp-F0h]
  __int64 *v69; // [rsp+20h] [rbp-F0h]
  int v70; // [rsp+28h] [rbp-E8h]
  unsigned int v71; // [rsp+2Ch] [rbp-E4h]
  __m128i v73; // [rsp+30h] [rbp-E0h]
  char v74; // [rsp+30h] [rbp-E0h]
  unsigned int v75; // [rsp+30h] [rbp-E0h]
  unsigned __int8 v77; // [rsp+40h] [rbp-D0h]
  __int64 v80; // [rsp+50h] [rbp-C0h]
  __int64 v81; // [rsp+50h] [rbp-C0h]
  __int64 v82; // [rsp+50h] [rbp-C0h]
  unsigned __int64 v83; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v84; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v85; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v86; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v87; // [rsp+88h] [rbp-88h]
  __int64 v88; // [rsp+90h] [rbp-80h] BYREF
  int v89; // [rsp+98h] [rbp-78h]
  __int128 v90; // [rsp+A0h] [rbp-70h]
  __int64 v91; // [rsp+B0h] [rbp-60h]
  __int64 v92; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v93; // [rsp+C8h] [rbp-48h]
  __int64 v94; // [rsp+D0h] [rbp-40h]

  v9 = 8 * a2;
  v10 = a3;
  v11 = a2 + a1;
  v12 = *(__int64 **)a6;
  v66 = (unsigned int)a4;
  v13 = 8 * v11;
  v68 = 16LL * (unsigned int)a4;
  v14 = *(_QWORD *)(a3 + 40) + v68;
  v70 = v11;
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v71 = a2;
  v83 = a4;
  LOBYTE(v92) = v15;
  v93 = v16;
  if ( v15 )
    v17 = sub_1F6C8D0(v15);
  else
    v17 = sub_1F58D40((__int64)&v92);
  LODWORD(v93) = v17;
  if ( v17 <= 0x40 )
  {
    v92 = 0;
    if ( v13 == v9 )
    {
      v20 = -1;
      goto LABEL_9;
    }
    if ( v13 <= 0x40 && v9 <= 0x3F )
    {
      v18 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 + 64 - (unsigned __int8)v13) << v9;
      v19 = 0;
LABEL_8:
      v20 = ~(v19 | v18);
LABEL_9:
      v21 = v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
      goto LABEL_17;
    }
LABEL_67:
    sub_16A5260(&v92, v9, v13);
    v17 = v93;
    if ( (unsigned int)v93 <= 0x40 )
      goto LABEL_74;
    goto LABEL_16;
  }
  sub_16A4EF0((__int64)&v92, 0, 0);
  if ( v13 != v9 )
  {
    if ( v9 > 0x3F || v13 > 0x40 )
      goto LABEL_67;
    v17 = v93;
    v18 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 + 64 - (unsigned __int8)v13) << v9;
    v19 = v92;
    if ( (unsigned int)v93 <= 0x40 )
      goto LABEL_8;
    *(_QWORD *)v92 |= v18;
  }
  v17 = v93;
  if ( (unsigned int)v93 <= 0x40 )
  {
LABEL_74:
    v20 = ~v92;
    goto LABEL_9;
  }
LABEL_16:
  sub_16A8F40(&v92);
  v17 = v93;
  v21 = v92;
LABEL_17:
  v86 = v21;
  v87 = v17;
  v22 = sub_1D1F940((__int64)v12, a3, v83, (__int64)&v86, 0);
  result = 0;
  if ( !v22 )
    goto LABEL_36;
  v24 = 8 * a1;
  v25 = *(_BYTE *)(a6 + 25);
  if ( 8 * a1 == 32 )
  {
    v77 = 5;
    goto LABEL_22;
  }
  if ( v24 > 0x20 )
  {
    if ( v24 != 64 )
    {
      if ( v24 == 128 )
      {
        v77 = 7;
        if ( !v25 )
          goto LABEL_23;
        goto LABEL_45;
      }
      goto LABEL_70;
    }
    v77 = 6;
LABEL_22:
    if ( !v25 )
      goto LABEL_23;
LABEL_45:
    if ( !*(_QWORD *)(*(_QWORD *)(a6 + 8) + 8LL * (v77 & 7) + 120) )
      goto LABEL_46;
    goto LABEL_23;
  }
  if ( v24 == 8 )
  {
    v77 = 3;
    goto LABEL_22;
  }
  v77 = 4;
  if ( v24 == 16 )
    goto LABEL_22;
LABEL_70:
  if ( v25 )
  {
LABEL_46:
    result = 0;
    if ( v87 <= 0x40 )
      return result;
    goto LABEL_37;
  }
  v77 = 0;
LABEL_23:
  if ( v71 )
  {
    v37 = *(_QWORD *)(v10 + 72);
    v92 = v37;
    if ( v37 )
      sub_1623A60((__int64)&v92, v37, 2);
    v38 = a6;
    v39 = *(_BYTE *)(a6 + 25);
    v40 = *(_QWORD *)(a6 + 8);
    LODWORD(v93) = *(_DWORD *)(v10 + 64);
    v74 = v39;
    v65 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + v68 + 8);
    v67 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + v68);
    v41 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)v38 + 32LL));
    v42 = sub_1F40B60(v40, v67, v65, v41, v74);
    *(_QWORD *)&v44 = sub_1D38BB0((__int64)v12, v9, (__int64)&v92, v42, v43, 0, a7, *(double *)a8.m128i_i64, a9, 0);
    v45 = sub_1D332F0(
            v12,
            124,
            (__int64)&v92,
            *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + v68),
            *(const void ***)(*(_QWORD *)(v10 + 40) + v68 + 8),
            0,
            *(double *)a7.m128i_i64,
            *(double *)a8.m128i_i64,
            a9,
            a3,
            v83,
            v44);
    v47 = v46;
    v10 = (__int64)v45;
    v83 = v46 | v83 & 0xFFFFFFFF00000000LL;
    if ( v92 )
    {
      sub_161E7C0((__int64)&v92, v92);
      v46 = v47;
    }
    v75 = v46;
    v63 = sub_1E34390(*(_QWORD *)(a5 + 104));
    if ( !*(_BYTE *)sub_1E0A0C0(v12[4]) )
    {
      v48 = *(const __m128i **)(a5 + 32);
      a9 = _mm_loadu_si128(v48 + 5);
      v28 = v48[5].m128i_u32[2];
      v69 = (__int64 *)v48[5].m128i_i64[0];
      v66 = v75;
      v73 = a9;
      goto LABEL_63;
    }
    v64 = v63;
    v66 = v75;
    v68 = 16LL * v47;
LABEL_60:
    v53 = *(_QWORD *)(v10 + 40) + v68;
    v54 = *(_BYTE *)v53;
    v55 = *(_QWORD *)(v53 + 8);
    LOBYTE(v92) = v54;
    v93 = v55;
    if ( v54 )
      v56 = sub_1F6C8D0(v54);
    else
      v56 = sub_1F58D40((__int64)&v92);
    v26 = 0;
    v71 = ((unsigned int)(v56 + 7) >> 3) - v70;
    v57 = *(const __m128i **)(a5 + 32);
    a7 = _mm_loadu_si128(v57 + 5);
    v28 = v57[5].m128i_u32[2];
    v69 = (__int64 *)v57[5].m128i_i64[0];
    v73 = a7;
    if ( !v71 )
      goto LABEL_26;
LABEL_63:
    v58 = *(_QWORD *)(v10 + 72);
    v92 = v58;
    if ( v58 )
      sub_1623A60((__int64)&v92, v58, 2);
    v59 = 16 * v28;
    v26 = v71;
    LODWORD(v93) = *(_DWORD *)(v10 + 64);
    *(_QWORD *)&v60 = sub_1D38BB0(
                        (__int64)v12,
                        v71,
                        (__int64)&v92,
                        *(unsigned __int8 *)(v59 + v69[5]),
                        *(const void ***)(v59 + v69[5] + 8),
                        0,
                        a7,
                        *(double *)a8.m128i_i64,
                        a9,
                        0);
    v69 = sub_1D332F0(
            v12,
            52,
            (__int64)&v92,
            *(unsigned __int8 *)(v69[5] + v59),
            *(const void ***)(v69[5] + v59 + 8),
            0,
            *(double *)a7.m128i_i64,
            *(double *)a8.m128i_i64,
            a9,
            v73.m128i_i64[0],
            v73.m128i_u64[1],
            v60);
    v28 = v61;
    v64 = (v26 | v63) & -(v26 | v63);
    if ( v92 )
      sub_161E7C0((__int64)&v92, v92);
    goto LABEL_26;
  }
  v63 = sub_1E34390(*(_QWORD *)(a5 + 104));
  v64 = v63;
  if ( *(_BYTE *)sub_1E0A0C0(v12[4]) )
    goto LABEL_60;
  v26 = 0;
  v27 = *(const __m128i **)(a5 + 32);
  a8 = _mm_loadu_si128(v27 + 5);
  v28 = v27[5].m128i_u32[2];
  v69 = (__int64 *)v27[5].m128i_i64[0];
  v73.m128i_i64[1] = a8.m128i_i64[1];
LABEL_26:
  v29 = *(_QWORD *)(v10 + 72);
  v30 = v77;
  v92 = v29;
  if ( v29 )
  {
    sub_1623A60((__int64)&v92, v29, 2);
    v30 = v77;
  }
  LODWORD(v93) = *(_DWORD *)(v10 + 64);
  v84 = v66 | v83 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v62 + 1) = v84;
  *(_QWORD *)&v62 = v10;
  v80 = sub_1D309E0(
          v12,
          145,
          (__int64)&v92,
          v30,
          0,
          0,
          *(double *)a7.m128i_i64,
          *(double *)a8.m128i_i64,
          *(double *)a9.m128i_i64,
          v62);
  v85 = v31 | v84 & 0xFFFFFFFF00000000LL;
  if ( v92 )
    sub_161E7C0((__int64)&v92, v92);
  v92 = 0;
  v93 = 0;
  v32 = *(_QWORD *)(a5 + 104);
  v94 = 0;
  v33 = *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 )
  {
    v49 = *(_BYTE *)(v32 + 16);
    v50 = *(_QWORD *)(v32 + 8) + v26;
    if ( (*(_QWORD *)v32 & 4) != 0 )
    {
      *((_QWORD *)&v90 + 1) = v50;
      LOBYTE(v91) = v49;
      *(_QWORD *)&v90 = v33 | 4;
      HIDWORD(v91) = *(_DWORD *)(v33 + 12);
    }
    else
    {
      v51 = *(_QWORD *)v33;
      *(_QWORD *)&v90 = *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v90 + 1) = v50;
      v52 = *(_BYTE *)(v51 + 8) == 16;
      LOBYTE(v91) = v49;
      if ( v52 )
        v51 = **(_QWORD **)(v51 + 16);
      HIDWORD(v91) = *(_DWORD *)(v51 + 8) >> 8;
    }
  }
  else
  {
    v34 = *(_DWORD *)(v32 + 20);
    LODWORD(v91) = 0;
    v90 = 0u;
    HIDWORD(v91) = v34;
  }
  v35 = *(_QWORD *)(a5 + 72);
  v88 = v35;
  if ( v35 )
    sub_1623A60((__int64)&v88, v35, 2);
  v36 = *(__int64 **)(a5 + 32);
  v89 = *(_DWORD *)(a5 + 64);
  result = sub_1D2BF40(
             v12,
             *v36,
             v36[1],
             (__int64)&v88,
             v80,
             v85,
             (__int64)v69,
             v28 | v73.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             v90,
             v91,
             v64,
             0,
             (__int64)&v92);
  if ( v88 )
  {
    v81 = result;
    sub_161E7C0((__int64)&v88, v88);
    result = v81;
  }
LABEL_36:
  if ( v87 <= 0x40 )
    return result;
LABEL_37:
  if ( v86 )
  {
    v82 = result;
    j_j___libc_free_0_0(v86);
    return v82;
  }
  return result;
}
