// Function: sub_3782790
// Address: 0x3782790
//
void __fastcall sub_3782790(__int64 *a1, __int64 a2, __int64 a3, __m128i *a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int16 v10; // dx
  __int16 *v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int16 v16; // bx
  __m128i v17; // kr00_16
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rbx
  unsigned __int16 v27; // dx
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  _BYTE *v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int16 v35; // ax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int16 v44; // cx
  bool v45; // di
  unsigned int v46; // esi
  int v47; // esi
  unsigned __int16 v48; // ax
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned __int16 v51; // bx
  __int64 v52; // rcx
  __int64 v53; // r9
  __int128 v54; // rax
  _QWORD *v55; // rsi
  int v56; // r9d
  int v57; // edx
  int v58; // r9d
  __int32 v59; // edx
  bool v60; // al
  unsigned __int16 v61; // ax
  __int64 v62; // rdx
  int v63; // eax
  __int64 v64; // rdx
  unsigned __int64 v65; // rax
  __int128 v66; // rax
  _QWORD *v67; // rsi
  __m128i v68; // xmm0
  __int64 v69; // rbx
  _QWORD *v70; // rdi
  __int64 v71; // r9
  __m128i v72; // xmm2
  __int64 v73; // rsi
  int v74; // edx
  _QWORD *v75; // rdi
  __int64 v76; // rsi
  __int128 v77; // [rsp-20h] [rbp-1C0h]
  __int128 v78; // [rsp-10h] [rbp-1B0h]
  __int64 v79; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 v80; // [rsp+8h] [rbp-198h]
  __int64 v81; // [rsp+18h] [rbp-188h]
  __int128 v82; // [rsp+20h] [rbp-180h]
  __int64 v83; // [rsp+30h] [rbp-170h]
  __int64 v84; // [rsp+38h] [rbp-168h]
  __int64 v85; // [rsp+40h] [rbp-160h]
  __int128 v87; // [rsp+60h] [rbp-140h]
  __int64 v88; // [rsp+D0h] [rbp-D0h] BYREF
  int v89; // [rsp+D8h] [rbp-C8h]
  unsigned __int16 v90; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+E8h] [rbp-B8h]
  __int64 v92; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v93; // [rsp+F8h] [rbp-A8h]
  __int64 v94; // [rsp+100h] [rbp-A0h]
  __int64 v95; // [rsp+108h] [rbp-98h]
  unsigned __int64 v96; // [rsp+110h] [rbp-90h]
  _BYTE *v97; // [rsp+118h] [rbp-88h]
  __int64 v98; // [rsp+120h] [rbp-80h] BYREF
  __int64 v99; // [rsp+128h] [rbp-78h]
  __int128 v100; // [rsp+130h] [rbp-70h] BYREF
  __m128i v101; // [rsp+140h] [rbp-60h] BYREF
  __int64 v102; // [rsp+150h] [rbp-50h]
  __int64 v103; // [rsp+158h] [rbp-48h]
  __int64 v104; // [rsp+160h] [rbp-40h]
  __int64 v105; // [rsp+168h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v87 = a3;
  v88 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v88, v7, 1);
  v8 = a1[1];
  v89 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v10 = *(_WORD *)v9;
  v91 = *(_QWORD *)(v9 + 8);
  v11 = *(__int16 **)(a2 + 48);
  v90 = v10;
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v92) = v12;
  v93 = v13;
  sub_33D0340((__int64)&v101, v8, &v92);
  v16 = v90;
  v17 = v101;
  v85 = v102;
  v84 = v103;
  if ( v90 )
  {
    v18 = v90 - 1;
    if ( (word_4456340[v18] & 1) != 0 )
      goto LABEL_5;
    if ( (unsigned __int16)(v90 - 17) <= 0xD3u )
    {
      v16 = word_4456580[v18];
      v20 = 0;
      goto LABEL_11;
    }
LABEL_10:
    v20 = v91;
    goto LABEL_11;
  }
  if ( (sub_3007240((__int64)&v90) & 1) != 0 )
    goto LABEL_5;
  if ( !sub_30070B0((__int64)&v90) )
    goto LABEL_10;
  v16 = sub_3009970((__int64)&v90, v8, v19, v14, v15);
LABEL_11:
  v101.m128i_i16[0] = v16;
  v101.m128i_i64[1] = v20;
  if ( v16 )
  {
    if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
      goto LABEL_71;
    v26 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
  }
  else
  {
    v21 = sub_3007260((__int64)&v101);
    v23 = v22;
    v24 = v21;
    v25 = v23;
    v94 = v24;
    v26 = v24;
    v95 = v25;
  }
  v27 = v92;
  v28 = 2 * v26;
  if ( (_WORD)v92 )
  {
    if ( (unsigned __int16)(v92 - 17) > 0xD3u )
    {
LABEL_15:
      v29 = v93;
      goto LABEL_16;
    }
    v27 = word_4456580[(unsigned __int16)v92 - 1];
    v29 = 0;
  }
  else
  {
    v60 = sub_30070B0((__int64)&v92);
    v27 = 0;
    if ( !v60 )
      goto LABEL_15;
    v61 = sub_3009970((__int64)&v92, v8, 0, v14, v15);
    v15 = v62;
    v27 = v61;
    v29 = v15;
  }
LABEL_16:
  LOWORD(v100) = v27;
  *((_QWORD *)&v100 + 1) = v29;
  if ( !v27 )
  {
    v30 = sub_3007260((__int64)&v100);
    v96 = v30;
    v97 = v31;
    goto LABEL_18;
  }
  v63 = v27;
  if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
LABEL_71:
    BUG();
  v31 = byte_444C4A0;
  v30 = *(_QWORD *)&byte_444C4A0[16 * v63 - 16];
LABEL_18:
  if ( v28 >= v30 )
  {
LABEL_5:
    sub_3781AE0(a1, a2, v87, a4, a5);
    if ( v88 )
      sub_B91220((__int64)&v88, v88);
    return;
  }
  *(_QWORD *)&v82 = *(_QWORD *)(a1[1] + 64);
  LODWORD(v98) = sub_3281170(&v90, v8, (__int64)v31, v14, v15);
  v99 = v32;
  *(_QWORD *)&v100 = sub_2D5B750((unsigned __int16 *)&v98);
  *((_QWORD *)&v100 + 1) = v33;
  v101.m128i_i64[0] = 2 * v100;
  v101.m128i_i8[8] = v33;
  v34 = sub_CA1930(&v101);
  switch ( v34 )
  {
    case 1u:
      v35 = 2;
      v36 = 0;
      break;
    case 2u:
      v35 = 3;
      v36 = 0;
      break;
    case 4u:
      v35 = 4;
      v36 = 0;
      break;
    case 8u:
      v35 = 5;
      v36 = 0;
      break;
    case 0x10u:
      v35 = 6;
      v36 = 0;
      break;
    case 0x20u:
      v35 = 7;
      v36 = 0;
      break;
    case 0x40u:
      v35 = 8;
      v36 = 0;
      break;
    case 0x80u:
      v35 = 9;
      v36 = 0;
      break;
    default:
      v35 = sub_3007020((_QWORD *)v82, v34);
      break;
  }
  v99 = v36;
  LOWORD(v98) = v35;
  v37 = sub_3281590((__int64)&v90);
  v38 = (unsigned int)v98;
  LOWORD(v98) = sub_327FD70((__int64 *)v82, v98, v99, v37);
  LOWORD(v42) = v90;
  v99 = v39;
  if ( v90 )
  {
    v83 = 0;
    v43 = v90 - 1;
    v44 = word_4456580[v43];
LABEL_30:
    v45 = (unsigned __int16)(v42 - 176) <= 0x34u;
    v46 = word_4456340[v43];
    LOBYTE(v42) = v45;
    goto LABEL_31;
  }
  v44 = sub_3009970((__int64)&v90, v38, v39, v40, v41);
  LOWORD(v42) = v90;
  v83 = v64;
  if ( v90 )
  {
    v43 = v90 - 1;
    goto LABEL_30;
  }
  LOWORD(v81) = v44;
  v65 = sub_3007240((__int64)&v90);
  v44 = v81;
  v46 = v65;
  v42 = HIDWORD(v65);
  v45 = v42;
LABEL_31:
  v47 = v46 >> 1;
  v101.m128i_i8[4] = v42;
  v101.m128i_i32[0] = v47;
  LODWORD(v81) = v44;
  if ( v45 )
    v48 = sub_2D43AD0(v44, v47);
  else
    v48 = sub_2D43050(v44, v47);
  v51 = v48;
  if ( !v48 )
    v51 = sub_3009450((__int64 *)v82, (unsigned int)v81, v83, v101.m128i_i64[0], v49, v50);
  sub_33D0340((__int64)&v101, a1[1], &v98);
  if ( !v90 )
    goto LABEL_5;
  v52 = *a1;
  if ( !*(_QWORD *)(*a1 + 8LL * v90 + 112) || v51 && *(_QWORD *)(v52 + 8LL * v51 + 112) )
    goto LABEL_5;
  if ( !(_WORD)v98
    || !*(_QWORD *)(v52 + 8LL * (unsigned __int16)v98 + 112)
    || !v101.m128i_i16[0]
    || !*(_QWORD *)(v52 + 8LL * v101.m128i_u16[0] + 112) )
  {
    goto LABEL_5;
  }
  if ( sub_33CB110(*(_DWORD *)(a2 + 24)) )
  {
    *(_QWORD *)&v66 = sub_340F900(
                        (_QWORD *)a1[1],
                        *(_DWORD *)(a2 + 24),
                        (__int64)&v88,
                        v98,
                        v99,
                        v53,
                        *(_OWORD *)*(_QWORD *)(a2 + 40),
                        *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                        *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    v67 = (_QWORD *)a1[1];
    v100 = v66;
    sub_3776C40((__int64)&v101, v67, &v100, (__int64)&v88);
    *(_QWORD *)v87 = v101.m128i_i64[0];
    *(_DWORD *)(v87 + 8) = v101.m128i_i32[2];
    a4->m128i_i64[0] = v102;
    a4->m128i_i32[2] = v103;
    sub_3777990(&v101, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a5);
    v68 = _mm_cvtsi32_si128(v103);
    v69 = v102;
    v79 = v101.m128i_i64[0];
    v80 = _mm_cvtsi32_si128(v101.m128i_u32[2]).m128i_u64[0];
    sub_3408380(
      &v101,
      (_QWORD *)a1[1],
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
      **(unsigned __int16 **)(a2 + 48),
      *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
      v68,
      (__int64)&v88);
    v70 = (_QWORD *)a1[1];
    v71 = v102;
    v105 = v101.m128i_u32[2];
    v72 = _mm_loadu_si128((const __m128i *)v87);
    *((_QWORD *)&v77 + 1) = 3;
    *(_QWORD *)&v77 = &v101;
    v73 = *(unsigned int *)(a2 + 24);
    v104 = v101.m128i_i64[0];
    v81 = v102;
    v83 = (unsigned int)v103;
    v101 = v72;
    v103 = v80;
    v102 = v79;
    *(_QWORD *)v87 = sub_33FC220(v70, v73, (__int64)&v88, v17.m128i_i64[0], v17.m128i_i64[1], v71, v77);
    v102 = v69;
    *(_DWORD *)(v87 + 8) = v74;
    v75 = (_QWORD *)a1[1];
    v104 = v81;
    v76 = *(unsigned int *)(a2 + 24);
    v105 = v83;
    *((_QWORD *)&v78 + 1) = 3;
    *(_QWORD *)&v78 = &v101;
    v101 = _mm_loadu_si128(a4);
    v103 = v68.m128i_i64[0];
    a4->m128i_i64[0] = (__int64)sub_33FC220(v75, v76, (__int64)&v88, v85, v84, v81, v78);
  }
  else
  {
    *(_QWORD *)&v54 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v88, (unsigned int)v98, v99, v53, a5);
    v55 = (_QWORD *)a1[1];
    v100 = v54;
    sub_3776C40((__int64)&v101, v55, &v100, (__int64)&v88);
    *(_QWORD *)v87 = v101.m128i_i64[0];
    *(_DWORD *)(v87 + 8) = v101.m128i_i32[2];
    a4->m128i_i64[0] = v102;
    a4->m128i_i32[2] = v103;
    *(_QWORD *)v87 = sub_33FAF80(
                       a1[1],
                       *(unsigned int *)(a2 + 24),
                       (__int64)&v88,
                       v17.m128i_i64[0],
                       v17.m128i_i64[1],
                       v56,
                       a5);
    *(_DWORD *)(v87 + 8) = v57;
    a4->m128i_i64[0] = (__int64)sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v88, v85, v84, v58, a5);
  }
  a4->m128i_i32[2] = v59;
  sub_9C6650(&v88);
}
