// Function: sub_3297B40
// Address: 0x3297b40
//
__int64 __fastcall sub_3297B40(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  int v5; // r13d
  __int64 v6; // r14
  __m128i v7; // xmm1
  __int128 *v8; // r15
  __int16 *v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // cx
  __int64 v12; // rax
  __int64 v13; // rdi
  __m128i si128; // xmm2
  __int64 v15; // rax
  __int64 *v16; // r11
  __int64 v17; // r12
  char v19; // al
  __int64 v20; // rcx
  __int64 v21; // r8
  bool v22; // zf
  bool v23; // al
  __int64 v24; // rdi
  __int64 v25; // rax
  char v26; // al
  int v27; // r9d
  __int64 v28; // rcx
  __int64 v29; // rdi
  int v30; // r8d
  __int64 v31; // rax
  const __m128i *v32; // rax
  __m128i v33; // xmm3
  char v34; // al
  int v35; // r9d
  char v36; // r8
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 v41; // rax
  const __m128i *v42; // rax
  __m128i v43; // xmm4
  char v44; // al
  bool v45; // r12
  unsigned __int64 v46; // rdi
  char v47; // al
  char v48; // al
  char v49; // al
  int v50; // r9d
  __int64 v51; // rdi
  __int64 v52; // rax
  __int128 *v53; // rax
  char v54; // al
  __int128 *v55; // rax
  char v56; // al
  __int128 v57; // [rsp-20h] [rbp-130h]
  __int128 v58; // [rsp-20h] [rbp-130h]
  __int128 v59; // [rsp-10h] [rbp-120h]
  __int128 v60; // [rsp-10h] [rbp-120h]
  int v61; // [rsp+4h] [rbp-10Ch]
  int v62; // [rsp+8h] [rbp-108h]
  __int64 v63; // [rsp+10h] [rbp-100h]
  unsigned __int16 v64; // [rsp+18h] [rbp-F8h]
  __int64 v65; // [rsp+20h] [rbp-F0h]
  bool v66; // [rsp+20h] [rbp-F0h]
  char v67; // [rsp+28h] [rbp-E8h]
  __int128 v68; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v69; // [rsp+40h] [rbp-D0h]
  __m128i v70; // [rsp+50h] [rbp-C0h]
  __m128i v71; // [rsp+60h] [rbp-B0h]
  __m128i v72; // [rsp+70h] [rbp-A0h]
  __int64 v73; // [rsp+80h] [rbp-90h] BYREF
  __int64 v74; // [rsp+88h] [rbp-88h]
  __int64 v75; // [rsp+90h] [rbp-80h] BYREF
  int v76; // [rsp+98h] [rbp-78h]
  __int128 v77; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v78; // [rsp+B0h] [rbp-60h] BYREF
  __int128 *v79; // [rsp+B8h] [rbp-58h]
  __m128i v80; // [rsp+C0h] [rbp-50h] BYREF
  int v81; // [rsp+D0h] [rbp-40h]
  char v82; // [rsp+D4h] [rbp-3Ch]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *v4;
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v8 = (__int128 *)v4[1];
  v61 = *((_DWORD *)v4 + 2);
  v65 = *v4;
  v63 = v4[5];
  v62 = *((_DWORD *)v4 + 12);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v68 = (__int128)v7;
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v75 = v10;
  v64 = v11;
  LOWORD(v73) = v11;
  v74 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v75, v10, 1);
  v13 = *a1;
  v76 = *(_DWORD *)(a2 + 72);
  si128 = _mm_load_si128((const __m128i *)&v68);
  v78 = v6;
  v79 = v8;
  v80 = si128;
  v15 = sub_3402EA0(v13, v5, (unsigned int)&v75, v73, v74, 0, (__int64)&v78, 2);
  v16 = &v75;
  if ( v15 )
    goto LABEL_4;
  v19 = sub_33E2390(*a1, v6, v8, 1);
  v16 = &v75;
  if ( v19 )
  {
    v26 = sub_33E2390(*a1, v68, *((_QWORD *)&v68 + 1), 1);
    v16 = &v75;
    if ( !v26 )
    {
      v28 = *(_QWORD *)(a2 + 48);
      *((_QWORD *)&v59 + 1) = v8;
      *(_QWORD *)&v59 = v6;
      v29 = *a1;
      v57 = v68;
      v30 = *(_DWORD *)(a2 + 68);
      *(_QWORD *)&v68 = &v75;
      v31 = sub_3411F20(v29, v5, (unsigned int)&v75, v28, v30, v27, v57, v59);
      v16 = (__int64 *)v68;
      v17 = v31;
      goto LABEL_5;
    }
  }
  if ( v64 )
  {
    if ( (unsigned __int16)(v64 - 17) > 0xD3u )
      goto LABEL_11;
  }
  else
  {
    v23 = sub_30070B0((__int64)&v73);
    v16 = &v75;
    if ( !v23 )
      goto LABEL_11;
  }
  v15 = sub_3295970(a1, a2, (__int64)&v75, v20, v21);
  v16 = &v75;
  if ( v15 )
  {
LABEL_4:
    v17 = v15;
    goto LABEL_5;
  }
LABEL_11:
  if ( *(_DWORD *)(v65 + 24) == 51 || *(_DWORD *)(v63 + 24) == 51 || v62 == v61 && v65 == v63 )
  {
    v24 = *a1;
    *(_QWORD *)&v68 = &v75;
    v25 = sub_3400BD0(v24, 0, (unsigned int)&v75, v73, v74, 0, 0);
    v16 = (__int64 *)v68;
    v17 = v25;
    goto LABEL_5;
  }
  *(_QWORD *)&v77 = 0;
  v22 = *(_DWORD *)(a2 + 24) == 178;
  DWORD2(v77) = 0;
  LODWORD(v78) = 178;
  v79 = &v77;
  v80.m128i_i32[2] = 64;
  v80.m128i_i64[0] = 0;
  v82 = 0;
  if ( v22 )
  {
    v32 = *(const __m128i **)(a2 + 40);
    v66 = v62 == v61 && v65 == v63;
    v33 = _mm_loadu_si128(v32);
    *(_QWORD *)&v77 = v32->m128i_i64[0];
    v72 = v33;
    DWORD2(v77) = v33.m128i_i32[2];
    v34 = sub_32657E0((__int64)&v80, v32[2].m128i_i64[1]);
    v36 = v66;
    v16 = &v75;
    if ( !v34 )
    {
      v53 = v79;
      v71 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
      *(_QWORD *)v79 = v71.m128i_i64[0];
      *((_DWORD *)v53 + 2) = v71.m128i_i32[2];
      v54 = sub_32657E0((__int64)&v80, **(_QWORD **)(a2 + 40));
      v36 = v66;
      v16 = &v75;
      if ( !v54 )
        goto LABEL_61;
    }
    if ( (!v82 || v81 == (v81 & *(_DWORD *)(a2 + 28)))
      && (!*((_BYTE *)a1 + 33)
       || ((v37 = a1[1], v38 = 1, v64 == 1) || v64 && (v38 = v64, *(_QWORD *)(v37 + 8LL * v64 + 112)))
       && !*(_BYTE *)(v37 + 500 * v38 + 6603)) )
    {
      if ( v80.m128i_i32[2] <= 0x40u )
        goto LABEL_35;
      v39 = v80.m128i_i64[0];
      v36 = 1;
      if ( !v80.m128i_i64[0] )
        goto LABEL_35;
    }
    else
    {
LABEL_61:
      if ( v80.m128i_i32[2] <= 0x40u )
        goto LABEL_15;
      v39 = v80.m128i_i64[0];
      if ( !v80.m128i_i64[0] )
        goto LABEL_15;
    }
    v67 = v36;
    j_j___libc_free_0_0(v39);
    v16 = &v75;
    if ( !v67 )
      goto LABEL_15;
LABEL_35:
    v40 = *a1;
    *(_QWORD *)&v68 = &v75;
    v41 = sub_33FAF80(v40, 189, (unsigned int)&v75, v73, v74, v35, v77);
    v16 = (__int64 *)v68;
    v17 = v41;
    goto LABEL_5;
  }
LABEL_15:
  v22 = *(_DWORD *)(a2 + 24) == 179;
  LODWORD(v78) = 179;
  v79 = &v77;
  v80.m128i_i32[2] = 64;
  v80.m128i_i64[0] = 0;
  v82 = 0;
  if ( !v22 )
  {
LABEL_16:
    if ( v5 == 178
      && (v47 = sub_328A020(a1[1], 0xB3u, v73, v74, *((unsigned __int8 *)a1 + 33)), v16 = &v75, v47)
      && (v48 = sub_33DD2A0(*a1, v6, v8, 0), v16 = &v75, v48)
      && (v49 = sub_33DD2A0(*a1, v68, *((_QWORD *)&v68 + 1), 0), v16 = &v75, v49) )
    {
      *((_QWORD *)&v60 + 1) = v8;
      v51 = *a1;
      *(_QWORD *)&v60 = v6;
      v58 = v68;
      *(_QWORD *)&v68 = &v75;
      v52 = sub_3406EB0(v51, 179, (unsigned int)&v75, v73, v74, v50, v58, v60);
      v16 = (__int64 *)v68;
      v17 = v52;
    }
    else
    {
      v17 = 0;
    }
    goto LABEL_5;
  }
  v42 = *(const __m128i **)(a2 + 40);
  v43 = _mm_loadu_si128(v42);
  *(_QWORD *)&v77 = v42->m128i_i64[0];
  v70 = v43;
  DWORD2(v77) = v43.m128i_i32[2];
  v44 = sub_32657E0((__int64)&v80, v42[2].m128i_i64[1]);
  v16 = &v75;
  if ( !v44 )
  {
    v55 = v79;
    v69 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
    *(_QWORD *)v79 = v69.m128i_i64[0];
    *((_DWORD *)v55 + 2) = v69.m128i_i32[2];
    v56 = sub_32657E0((__int64)&v80, **(_QWORD **)(a2 + 40));
    v16 = &v75;
    if ( !v56 )
    {
      if ( v80.m128i_i32[2] > 0x40u && v80.m128i_i64[0] )
      {
        j_j___libc_free_0_0(v80.m128i_u64[0]);
        v16 = &v75;
      }
      goto LABEL_16;
    }
  }
  if ( v82 )
  {
    v45 = (v81 & *(_DWORD *)(a2 + 28)) == v81;
    if ( v80.m128i_i32[2] <= 0x40u || (v46 = v80.m128i_i64[0]) == 0 )
    {
LABEL_44:
      if ( !v45 )
        goto LABEL_16;
      goto LABEL_45;
    }
LABEL_43:
    j_j___libc_free_0_0(v46);
    v16 = &v75;
    goto LABEL_44;
  }
  if ( v80.m128i_i32[2] > 0x40u )
  {
    v46 = v80.m128i_i64[0];
    v45 = 1;
    if ( v80.m128i_i64[0] )
      goto LABEL_43;
  }
LABEL_45:
  v17 = v77;
LABEL_5:
  if ( v75 )
    sub_B91220((__int64)v16, v75);
  return v17;
}
