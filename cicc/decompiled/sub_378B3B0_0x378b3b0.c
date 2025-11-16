// Function: sub_378B3B0
// Address: 0x378b3b0
//
unsigned __int8 *__fastcall sub_378B3B0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  unsigned int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // rdx
  char v12; // dl
  unsigned __int64 v13; // r9
  int v14; // r8d
  __int64 *v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // r9
  int v18; // esi
  unsigned int v19; // r14d
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r11
  __int64 v24; // r9
  __int64 v25; // r9
  unsigned int v26; // edx
  __int64 v27; // r9
  unsigned __int8 *v28; // rax
  __int64 v29; // r11
  __int64 v30; // r9
  unsigned __int8 *v31; // r14
  unsigned int v32; // edx
  int v33; // r9d
  _DWORD *v34; // r15
  __int64 v35; // rdx
  __int16 v36; // ax
  __int64 v37; // rdx
  unsigned __int16 v38; // cx
  unsigned int v39; // eax
  unsigned __int8 *v40; // r14
  unsigned __int16 v42; // ax
  __int64 v43; // r9
  int v44; // esi
  __int64 v45; // rdx
  char v46; // r14
  unsigned int *v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned int *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r9
  unsigned int v55; // edx
  unsigned __int8 *v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // edx
  __int64 v59; // r9
  __int64 v60; // rax
  unsigned int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // rdx
  unsigned int v64; // eax
  __int64 v65; // rdx
  __int128 v66; // [rsp-20h] [rbp-1A0h]
  __int128 v67; // [rsp-20h] [rbp-1A0h]
  __int128 v68; // [rsp-20h] [rbp-1A0h]
  __int128 v69; // [rsp-10h] [rbp-190h]
  __int128 v70; // [rsp-10h] [rbp-190h]
  __int8 v71; // [rsp+Fh] [rbp-171h]
  __int8 v72; // [rsp+Fh] [rbp-171h]
  __int64 *v73; // [rsp+10h] [rbp-170h]
  __int128 v74; // [rsp+10h] [rbp-170h]
  __int128 *v75; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v76; // [rsp+10h] [rbp-170h]
  unsigned int v77; // [rsp+20h] [rbp-160h]
  __int128 v78; // [rsp+20h] [rbp-160h]
  __int64 v79; // [rsp+20h] [rbp-160h]
  unsigned int v80; // [rsp+30h] [rbp-150h]
  __int128 v81; // [rsp+30h] [rbp-150h]
  __int64 *v82; // [rsp+30h] [rbp-150h]
  __int128 *v83; // [rsp+30h] [rbp-150h]
  __int64 v84; // [rsp+30h] [rbp-150h]
  unsigned __int8 v85; // [rsp+40h] [rbp-140h]
  __int64 v86; // [rsp+40h] [rbp-140h]
  unsigned int v87; // [rsp+40h] [rbp-140h]
  unsigned int v88; // [rsp+40h] [rbp-140h]
  unsigned int v89; // [rsp+48h] [rbp-138h]
  __int64 v90; // [rsp+48h] [rbp-138h]
  __int64 *v91; // [rsp+48h] [rbp-138h]
  __int64 v92; // [rsp+58h] [rbp-128h]
  __int64 v93; // [rsp+68h] [rbp-118h]
  unsigned __int64 v94; // [rsp+D8h] [rbp-A8h]
  __int128 v95; // [rsp+E0h] [rbp-A0h] BYREF
  __int128 v96; // [rsp+F0h] [rbp-90h] BYREF
  __int128 v97; // [rsp+100h] [rbp-80h] BYREF
  __int128 v98; // [rsp+110h] [rbp-70h] BYREF
  __int64 v99; // [rsp+120h] [rbp-60h] BYREF
  int v100; // [rsp+128h] [rbp-58h]
  __m128i v101; // [rsp+130h] [rbp-50h] BYREF
  __int64 v102; // [rsp+140h] [rbp-40h]
  unsigned int v103; // [rsp+148h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)&v95 = 0;
  v6 = *(_QWORD *)(a2 + 80);
  v77 = v5;
  v80 = v5 - 147;
  v99 = v6;
  DWORD2(v95) = 0;
  *(_QWORD *)&v96 = 0;
  DWORD2(v96) = 0;
  *(_QWORD *)&v97 = 0;
  DWORD2(v97) = 0;
  *(_QWORD *)&v98 = 0;
  DWORD2(v98) = 0;
  if ( v6 )
    sub_B96E90((__int64)&v99, v6, 1);
  v100 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  if ( v80 <= 1 )
  {
    sub_375E8D0((__int64)a1, *(_QWORD *)(v7 + 40), *(_QWORD *)(v7 + 48), (__int64)&v95, (__int64)&v96);
    v8 = 80;
  }
  else
  {
    sub_375E8D0((__int64)a1, *(_QWORD *)v7, *(_QWORD *)(v7 + 8), (__int64)&v95, (__int64)&v96);
    v8 = 40;
  }
  sub_375E8D0(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + v8),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + v8 + 8),
    (__int64)&v97,
    (__int64)&v98);
  v9 = (unsigned __int16 *)(*(_QWORD *)(v95 + 48) + 16LL * DWORD2(v95));
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v101.m128i_i16[0] = v10;
  v101.m128i_i64[1] = v11;
  if ( (_WORD)v10 )
  {
    v12 = (unsigned __int16)(v10 - 176) <= 0x34u;
    LOBYTE(v13) = v12;
    v14 = word_4456340[v10 - 1];
  }
  else
  {
    v94 = sub_3007240((__int64)&v101);
    v14 = v94;
    v13 = HIDWORD(v94);
    v12 = BYTE4(v94);
  }
  v85 = v13;
  v89 = v14;
  v15 = *(__int64 **)(a1[1] + 64);
  v101.m128i_i32[0] = v14;
  v101.m128i_i8[4] = v13;
  v73 = v15;
  if ( v12 )
  {
    v16 = sub_2D43AD0(2, v14);
    v17 = v85;
    if ( v16 )
    {
      v18 = 2 * v89;
      v101.m128i_i8[4] = v85;
      v101.m128i_i32[0] = 2 * v89;
      v19 = v16;
      v90 = 0;
    }
    else
    {
      v71 = v85;
      v87 = v89;
      v62 = sub_3009450(v73, 2, 0, v101.m128i_i64[0], v89, v17);
      v90 = v63;
      v19 = v62;
      v18 = 2 * v87;
      v101.m128i_i8[4] = v71;
      v101.m128i_i32[0] = 2 * v87;
    }
    LOWORD(v20) = sub_2D43AD0(2, v18);
    v23 = 0;
    if ( (_WORD)v20 )
      goto LABEL_11;
LABEL_29:
    v20 = sub_3009450(v73, 2, 0, v101.m128i_i64[0], v21, v22);
    HIWORD(v3) = HIWORD(v20);
    v23 = v45;
    goto LABEL_11;
  }
  v42 = sub_2D43050(2, v14);
  v43 = v85;
  if ( v42 )
  {
    v44 = 2 * v89;
    v101.m128i_i8[4] = v85;
    v101.m128i_i32[0] = 2 * v89;
    v19 = v42;
    v90 = 0;
  }
  else
  {
    v72 = v85;
    v88 = v89;
    v64 = sub_3009450(v73, 2, 0, v101.m128i_i64[0], v89, v43);
    v90 = v65;
    v19 = v64;
    v44 = 2 * v88;
    v101.m128i_i8[4] = v72;
    v101.m128i_i32[0] = 2 * v88;
  }
  LOWORD(v20) = sub_2D43050(2, v44);
  v23 = 0;
  if ( !(_WORD)v20 )
    goto LABEL_29;
LABEL_11:
  v24 = *(_QWORD *)(a2 + 40);
  LOWORD(v3) = v20;
  if ( v77 == 208 )
  {
    v79 = v23;
    v84 = sub_340F900((_QWORD *)a1[1], 0xD0u, (__int64)&v99, v19, v90, v24, v95, v97, *(_OWORD *)(v24 + 80));
    v93 = v58;
    v60 = sub_340F900(
            (_QWORD *)a1[1],
            0xD0u,
            (__int64)&v99,
            v19,
            v90,
            v59,
            v96,
            v98,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    v30 = v84;
    v29 = v79;
    v31 = (unsigned __int8 *)v60;
    v92 = v61;
  }
  else
  {
    v86 = v23;
    if ( v80 <= 1 )
    {
      v75 = *(__int128 **)(a2 + 40);
      v82 = (__int64 *)a1[1];
      v47 = (unsigned int *)sub_33E5110(
                              v82,
                              v19,
                              v90,
                              *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
                              *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL));
      v76 = sub_34129B0(
              v82,
              v77,
              (__int64)&v99,
              v47,
              v48,
              (__int64)v75,
              *v75,
              v95,
              v97,
              *(__int128 *)((char *)v75 + 120));
      v50 = v49;
      v51 = v90;
      v91 = (__int64 *)a1[1];
      v93 = v50;
      v83 = *(__int128 **)(a2 + 40);
      v52 = (unsigned int *)sub_33E5110(
                              v91,
                              v19,
                              v51,
                              *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
                              *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL));
      v31 = sub_34129B0(v91, v77, (__int64)&v99, v52, v53, v54, *v83, v96, v98, *(__int128 *)((char *)v83 + 120));
      v92 = v55;
      *((_QWORD *)&v70 + 1) = 1;
      *(_QWORD *)&v70 = v31;
      *((_QWORD *)&v68 + 1) = 1;
      *(_QWORD *)&v68 = v76;
      v56 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v99, 1, 0, (__int64)v76, v68, v70);
      sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v56, v57);
      v30 = (__int64)v76;
      v29 = v86;
    }
    else
    {
      sub_3777990(&v101, a1, *(_QWORD *)(v24 + 120), *(_QWORD *)(v24 + 128), a3);
      *(_QWORD *)&v81 = v101.m128i_i64[0];
      *((_QWORD *)&v81 + 1) = v101.m128i_u32[2];
      *(_QWORD *)&v78 = v102;
      *((_QWORD *)&v78 + 1) = v103;
      sub_3408380(
        &v101,
        (_QWORD *)a1[1],
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL),
        **(unsigned __int16 **)(a2 + 48),
        *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
        a3,
        (__int64)&v99);
      *((_QWORD *)&v66 + 1) = v101.m128i_u32[2];
      *(_QWORD *)&v66 = v101.m128i_i64[0];
      *(_QWORD *)&v74 = v102;
      *((_QWORD *)&v74 + 1) = v103;
      *(_QWORD *)&v81 = sub_33FC1D0(
                          (_QWORD *)a1[1],
                          463,
                          (__int64)&v99,
                          v19,
                          v90,
                          v25,
                          v95,
                          v97,
                          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                          v81,
                          v66);
      v93 = v26;
      v28 = sub_33FC1D0(
              (_QWORD *)a1[1],
              463,
              (__int64)&v99,
              v19,
              v90,
              v27,
              v96,
              v98,
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
              v78,
              v74);
      v29 = v86;
      v30 = v81;
      v31 = v28;
      v92 = v32;
    }
  }
  *((_QWORD *)&v69 + 1) = v92;
  *(_QWORD *)&v69 = v31;
  *((_QWORD *)&v67 + 1) = v93;
  *(_QWORD *)&v67 = v30;
  sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v99, v3, v29, v30, v67, v69);
  v34 = (_DWORD *)*a1;
  v35 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v36 = *(_WORD *)v35;
  v37 = *(_QWORD *)(v35 + 8);
  v101.m128i_i16[0] = v36;
  v101.m128i_i64[1] = v37;
  if ( v36 )
  {
    v38 = v36 - 17;
    if ( (unsigned __int16)(v36 - 10) > 6u && (unsigned __int16)(v36 - 126) > 0x31u )
    {
      if ( v38 > 0xD3u )
      {
LABEL_18:
        v39 = v34[15];
        goto LABEL_19;
      }
      goto LABEL_24;
    }
    if ( v38 <= 0xD3u )
    {
LABEL_24:
      v39 = v34[17];
      goto LABEL_19;
    }
  }
  else
  {
    v46 = sub_3007030((__int64)&v101);
    if ( sub_30070B0((__int64)&v101) )
      goto LABEL_24;
    if ( !v46 )
      goto LABEL_18;
  }
  v39 = v34[16];
LABEL_19:
  if ( v39 > 2 )
    BUG();
  v40 = sub_33FAF80(
          a1[1],
          215 - v39,
          (__int64)&v99,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v33,
          a3);
  if ( v99 )
    sub_B91220((__int64)&v99, v99);
  return v40;
}
