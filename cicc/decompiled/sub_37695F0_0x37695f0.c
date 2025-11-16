// Function: sub_37695F0
// Address: 0x37695f0
//
unsigned __int8 *__fastcall sub_37695F0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int16 v6; // dx
  __int64 v7; // rax
  __int64 *v8; // rax
  __m128i v9; // xmm0
  _BYTE *v10; // rax
  unsigned __int8 *v11; // r13
  __int32 v13; // eax
  __int64 v14; // r10
  __int64 v15; // r11
  unsigned __int16 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r14
  __int128 v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v21; // r11
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r9
  unsigned __int8 *v25; // r8
  __int64 v26; // rdx
  __int16 v27; // ax
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rax
  _QWORD *v31; // r13
  unsigned int v32; // edx
  unsigned int v33; // edi
  __int64 v34; // rsi
  unsigned int v35; // r9d
  int v36; // r14d
  unsigned __int64 v37; // r11
  __int64 v38; // rax
  unsigned int v39; // r9d
  unsigned __int64 v40; // r11
  unsigned int v41; // edx
  unsigned int v42; // ecx
  __int64 v43; // rdi
  unsigned int v44; // edx
  int v45; // r9d
  unsigned int v46; // edx
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  __int64 v50; // r9
  unsigned int v51; // edx
  __int64 v52; // r9
  int v53; // r9d
  bool v54; // al
  unsigned __int8 *v55; // rax
  unsigned __int8 *v56; // r8
  unsigned int v57; // edx
  bool v58; // al
  bool v59; // al
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  unsigned __int16 v63; // ax
  __int64 v64; // rdx
  unsigned __int8 *v65; // rax
  unsigned int v66; // edx
  __int128 v67; // [rsp-20h] [rbp-160h]
  __int128 v68; // [rsp-10h] [rbp-150h]
  __int128 v69; // [rsp-10h] [rbp-150h]
  __int128 v70; // [rsp-10h] [rbp-150h]
  __int64 v71; // [rsp+0h] [rbp-140h]
  __int64 v72; // [rsp+8h] [rbp-138h]
  __int64 v73; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v74; // [rsp+10h] [rbp-130h]
  __int64 v75; // [rsp+18h] [rbp-128h]
  __int64 v76; // [rsp+18h] [rbp-128h]
  __int128 v77; // [rsp+20h] [rbp-120h]
  __int64 v78; // [rsp+20h] [rbp-120h]
  __int64 v79; // [rsp+28h] [rbp-118h]
  unsigned __int64 v80; // [rsp+28h] [rbp-118h]
  unsigned __int64 v81; // [rsp+28h] [rbp-118h]
  __int64 v82; // [rsp+30h] [rbp-110h]
  _QWORD *v83; // [rsp+30h] [rbp-110h]
  __int128 v84; // [rsp+30h] [rbp-110h]
  unsigned __int8 *v85; // [rsp+30h] [rbp-110h]
  __int64 v86; // [rsp+38h] [rbp-108h]
  unsigned __int64 v87; // [rsp+38h] [rbp-108h]
  unsigned __int64 v88; // [rsp+38h] [rbp-108h]
  unsigned __int64 v89; // [rsp+38h] [rbp-108h]
  unsigned int v90; // [rsp+38h] [rbp-108h]
  __int128 v91; // [rsp+40h] [rbp-100h]
  __int128 v92; // [rsp+50h] [rbp-F0h]
  unsigned __int8 *v93; // [rsp+60h] [rbp-E0h]
  unsigned __int16 v94; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v95; // [rsp+C8h] [rbp-78h]
  __int64 v96; // [rsp+D0h] [rbp-70h] BYREF
  int v97; // [rsp+D8h] [rbp-68h]
  __m128i v98; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v99; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v100; // [rsp+100h] [rbp-40h] BYREF
  __int64 v101; // [rsp+108h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *(_WORD *)v4;
  v7 = *(_QWORD *)(v4 + 8);
  v96 = v5;
  v94 = v6;
  v95 = v7;
  if ( v5 )
  {
    sub_B96E90((__int64)&v96, v5, 1);
    v6 = v94;
  }
  v97 = *(_DWORD *)(a2 + 72);
  v8 = *(__int64 **)(a2 + 40);
  v9 = _mm_loadu_si128((const __m128i *)(v8 + 5));
  v82 = *v8;
  v86 = v8[1];
  v91 = (__int128)_mm_loadu_si128((const __m128i *)v8 + 5);
  if ( !v6
    || (v10 = (_BYTE *)(a1[1] + 500LL * v6), v10[6600] == 2)
    || v10[6602] == 2
    || v10[6601] == 2
    || v10[((unsigned __int16)(v6 - 17) < 0x9Fu ? 156 : 168) + 6414] == 2 )
  {
    v11 = 0;
    goto LABEL_9;
  }
  v13 = sub_327FDF0(&v94, v5);
  v14 = v82;
  v15 = v86;
  v98.m128i_i32[0] = v13;
  v16 = v13;
  v98.m128i_i64[1] = v17;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
    {
LABEL_14:
      v18 = v98.m128i_i64[1];
      goto LABEL_15;
    }
    v18 = 0;
    v16 = word_4456580[(unsigned __int16)v13 - 1];
  }
  else
  {
    v59 = sub_30070B0((__int64)&v98);
    v14 = v82;
    v15 = v86;
    if ( !v59 )
      goto LABEL_14;
    v63 = sub_3009970((__int64)&v98, v5, v60, v61, v62);
    v14 = v82;
    v15 = v86;
    v16 = v63;
    v18 = v64;
  }
LABEL_15:
  v73 = v14;
  v75 = v15;
  v83 = (_QWORD *)*a1;
  *(_QWORD *)&v19 = sub_3400BD0(*a1, 0, (__int64)&v96, v16, v18, 0, v9, 0);
  v77 = v19;
  v20 = sub_34015B0(*a1, (__int64)&v96, v16, v18, 0, 0, v9);
  v21 = v75;
  v22 = v73;
  v24 = v23;
  v25 = v20;
  v26 = *(_QWORD *)(v73 + 48) + 16LL * (unsigned int)v75;
  v27 = *(_WORD *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  LOWORD(v100) = v27;
  v101 = v28;
  if ( v27 )
  {
    v29 = ((unsigned __int16)(v27 - 17) < 0xD4u) + 205;
  }
  else
  {
    v71 = v73;
    v72 = v75;
    v74 = v25;
    v76 = v24;
    v58 = sub_30070B0((__int64)&v100);
    v22 = v71;
    v21 = v72;
    v25 = v74;
    v24 = v76;
    v29 = 205 - (!v58 - 1);
  }
  v68 = v77;
  *((_QWORD *)&v67 + 1) = v24;
  *(_QWORD *)&v67 = v25;
  v79 = v21;
  v30 = sub_340EC60(v83, v29, (__int64)&v96, v16, v18, 0, v22, v21, v67, v68);
  v31 = (_QWORD *)*a1;
  v33 = v32;
  v34 = v30;
  v35 = v32;
  v99 = _mm_loadu_si128(&v98);
  v36 = v32;
  v37 = v32 | v79 & 0xFFFFFFFF00000000LL;
  if ( !v98.m128i_i16[0] )
  {
    v78 = v30;
    v54 = sub_3007100((__int64)&v99);
    v37 = v33 | v79 & 0xFFFFFFFF00000000LL;
    v34 = v78;
    if ( !v54 )
      goto LABEL_19;
LABEL_22:
    v89 = v37;
    if ( *(_DWORD *)(v34 + 24) == 51 )
    {
      v100 = 0;
      LODWORD(v101) = 0;
      v65 = (unsigned __int8 *)sub_33F17F0(v31, 51, (__int64)&v100, v99.m128i_u32[0], v99.m128i_i64[1]);
      v40 = v89;
      v56 = v65;
      v39 = v66;
      if ( v100 )
      {
        v81 = v89;
        v85 = v65;
        v90 = v66;
        sub_B91220((__int64)&v100, v100);
        v56 = v85;
        v39 = v90;
        v40 = v81;
      }
    }
    else
    {
      v55 = sub_33FAF80((__int64)v31, 168, (__int64)&v96, v99.m128i_i64[0], v99.m128i_i64[1], v35, v9);
      v40 = v89;
      v56 = v55;
      v39 = v57;
    }
    v43 = (__int64)v56;
    v42 = v39;
    goto LABEL_20;
  }
  if ( (unsigned __int16)(v98.m128i_i16[0] - 176) <= 0x34u )
    goto LABEL_22;
LABEL_19:
  v87 = v37;
  v38 = sub_32886A0((__int64)v31, v99.m128i_u32[0], v99.m128i_i64[1], (int)&v96, v34, v36);
  v40 = v87;
  v42 = v41;
  v43 = v38;
LABEL_20:
  v88 = v42 | v40 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v92 = sub_33FAF80(*a1, 234, (__int64)&v96, v98.m128i_u32[0], v98.m128i_i64[1], v39, v9);
  *((_QWORD *)&v92 + 1) = v44 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v91 = sub_33FAF80(*a1, 234, (__int64)&v96, v98.m128i_u32[0], v98.m128i_i64[1], v45, v9);
  v80 = v88;
  *((_QWORD *)&v91 + 1) = v46 | *((_QWORD *)&v91 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v47 = sub_34074A0((_QWORD *)*a1, (__int64)&v96, v43, v88, v98.m128i_u32[0], v98.m128i_i64[1], v9);
  v84 = v47;
  *((_QWORD *)&v69 + 1) = v80;
  *(_QWORD *)&v69 = v43;
  *(_QWORD *)&v92 = sub_3406EB0((_QWORD *)*a1, 0xBAu, (__int64)&v96, v98.m128i_u32[0], v98.m128i_i64[1], v48, v92, v69);
  *((_QWORD *)&v92 + 1) = v49 | *((_QWORD *)&v92 + 1) & 0xFFFFFFFF00000000LL;
  v93 = sub_3406EB0((_QWORD *)*a1, 0xBAu, (__int64)&v96, v98.m128i_u32[0], v98.m128i_i64[1], v50, v91, v84);
  *((_QWORD *)&v70 + 1) = v51 | *((_QWORD *)&v91 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v70 = v93;
  sub_3406EB0((_QWORD *)*a1, 0xBBu, (__int64)&v96, v98.m128i_u32[0], v98.m128i_i64[1], v52, v92, v70);
  v11 = sub_33FAF80(
          *a1,
          234,
          (__int64)&v96,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v53,
          v9);
LABEL_9:
  if ( v96 )
    sub_B91220((__int64)&v96, v96);
  return v11;
}
