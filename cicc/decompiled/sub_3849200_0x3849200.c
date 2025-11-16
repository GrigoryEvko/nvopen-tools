// Function: sub_3849200
// Address: 0x3849200
//
void __fastcall sub_3849200(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int16 v13; // cx
  __int64 v14; // rax
  bool v15; // al
  bool v16; // al
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int16 v21; // cx
  __int64 v22; // rax
  __int64 v23; // r9
  __m128i v24; // xmm0
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  _QWORD *v28; // rdi
  unsigned __int16 *v29; // rax
  unsigned __int8 *v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  unsigned __int16 *v33; // rax
  __int64 v34; // r9
  unsigned __int8 *v35; // rax
  __int64 v36; // rsi
  int v37; // edx
  unsigned __int8 *v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // rsi
  __int64 v41; // rax
  __int16 v42; // dx
  __int64 v43; // rax
  __m128i v44; // xmm2
  __int128 *v45; // rdx
  unsigned __int16 *v46; // rax
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rdx
  unsigned __int16 *v50; // rax
  __int64 v51; // r9
  __int64 v52; // rax
  int v53; // edx
  bool v54; // al
  bool v55; // al
  unsigned __int16 *v56; // rax
  unsigned __int16 *v57; // rax
  __int64 v58; // rsi
  __int16 v59; // dx
  __int64 v60; // rax
  __m128i v61; // xmm4
  unsigned int *v62; // rdx
  __int64 v63; // rcx
  unsigned __int16 *v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // r8
  __int64 v67; // rdx
  __int64 v68; // rax
  __int16 v69; // ax
  __int64 v70; // rax
  __int16 v71; // r10
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rsi
  __int64 v76; // rax
  __int16 v77; // dx
  __int64 v78; // rax
  __m128i v79; // xmm6
  __int128 v80; // [rsp-20h] [rbp-1C0h]
  __int64 v81; // [rsp+0h] [rbp-1A0h]
  __int64 v82; // [rsp+10h] [rbp-190h]
  __int16 v83; // [rsp+1Ch] [rbp-184h]
  __int64 v84; // [rsp+28h] [rbp-178h]
  __int64 v85; // [rsp+28h] [rbp-178h]
  __int64 v86; // [rsp+28h] [rbp-178h]
  __int64 v87; // [rsp+28h] [rbp-178h]
  __int128 v88; // [rsp+30h] [rbp-170h]
  unsigned __int16 (__fastcall *v89)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+30h] [rbp-170h]
  _QWORD *v90; // [rsp+40h] [rbp-160h]
  __int64 v91; // [rsp+40h] [rbp-160h]
  __int64 v92; // [rsp+40h] [rbp-160h]
  int v94; // [rsp+68h] [rbp-138h]
  int v95; // [rsp+88h] [rbp-118h]
  __int128 v96; // [rsp+90h] [rbp-110h] BYREF
  __int128 v97; // [rsp+A0h] [rbp-100h] BYREF
  __int128 v98; // [rsp+B0h] [rbp-F0h] BYREF
  __int128 v99; // [rsp+C0h] [rbp-E0h] BYREF
  __int128 v100; // [rsp+D0h] [rbp-D0h] BYREF
  __int128 v101; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+F0h] [rbp-B0h] BYREF
  int v103; // [rsp+F8h] [rbp-A8h]
  __m128i v104; // [rsp+100h] [rbp-A0h] BYREF
  unsigned __int8 *v105; // [rsp+110h] [rbp-90h] BYREF
  __int64 v106; // [rsp+118h] [rbp-88h]
  __m128i v107; // [rsp+120h] [rbp-80h] BYREF
  __m128i v108; // [rsp+130h] [rbp-70h] BYREF
  __int64 v109; // [rsp+140h] [rbp-60h] BYREF
  __int64 v110; // [rsp+148h] [rbp-58h]
  __m128i v111; // [rsp+150h] [rbp-50h] BYREF
  __m128i v112; // [rsp+160h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v96 = 0;
  DWORD2(v96) = 0;
  *(_QWORD *)&v97 = 0;
  DWORD2(v97) = 0;
  *(_QWORD *)&v98 = 0;
  DWORD2(v98) = 0;
  *(_QWORD *)&v99 = 0;
  DWORD2(v99) = 0;
  *(_QWORD *)&v100 = 0;
  DWORD2(v100) = 0;
  *(_QWORD *)&v101 = 0;
  DWORD2(v101) = 0;
  v102 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v102, v7, 1);
  v8 = *(_DWORD *)(a2 + 24);
  v103 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(v9 + 40);
  v11 = *(_QWORD *)(v9 + 48);
  v12 = *(_QWORD *)(v10 + 48) + 16LL * *(unsigned int *)(v9 + 48);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v111.m128i_i16[0] = v13;
  v111.m128i_i64[1] = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v13 - 2) <= 7u || (unsigned __int16)(v13 - 176) <= 0x1Fu )
        goto LABEL_6;
LABEL_9:
      sub_375E6F0((__int64)a1, v10, v11, (__int64)&v96, (__int64)&v97);
      goto LABEL_10;
    }
  }
  else
  {
    v84 = v11;
    v15 = sub_30070B0((__int64)&v111);
    v11 = v84;
    if ( !v15 )
    {
      v16 = sub_3007070((__int64)&v111);
      v11 = v84;
      if ( v16 )
      {
LABEL_6:
        sub_375E510((__int64)a1, v10, v11, (__int64)&v96, (__int64)&v97);
        goto LABEL_10;
      }
      goto LABEL_9;
    }
  }
  sub_375E8D0((__int64)a1, v10, v11, (__int64)&v96, (__int64)&v97);
LABEL_10:
  v17 = *(_QWORD *)(a2 + 40);
  v18 = *(_QWORD *)(v17 + 80);
  v19 = *(_QWORD *)(v17 + 88);
  v20 = *(_QWORD *)(v18 + 48) + 16LL * *(unsigned int *)(v17 + 88);
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  v111.m128i_i16[0] = v21;
  v111.m128i_i64[1] = v22;
  if ( v21 )
  {
    if ( (unsigned __int16)(v21 - 17) > 0xD3u )
    {
      if ( (unsigned __int16)(v21 - 2) <= 7u || (unsigned __int16)(v21 - 176) <= 0x1Fu )
        goto LABEL_13;
LABEL_29:
      sub_375E6F0((__int64)a1, v18, v19, (__int64)&v98, (__int64)&v99);
      goto LABEL_14;
    }
  }
  else
  {
    v85 = v19;
    v54 = sub_30070B0((__int64)&v111);
    v19 = v85;
    if ( !v54 )
    {
      v55 = sub_3007070((__int64)&v111);
      v19 = v85;
      if ( v55 )
      {
LABEL_13:
        sub_375E510((__int64)a1, v18, v19, (__int64)&v98, (__int64)&v99);
        goto LABEL_14;
      }
      goto LABEL_29;
    }
  }
  sub_375E8D0((__int64)a1, v18, v19, (__int64)&v98, (__int64)&v99);
LABEL_14:
  v24 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v104 = v24;
  DWORD2(v101) = v24.m128i_i32[2];
  DWORD2(v100) = v24.m128i_i32[2];
  v25 = *(_QWORD *)(v24.m128i_i64[0] + 48) + 16LL * v24.m128i_u32[2];
  *(_QWORD *)&v101 = v24.m128i_i64[0];
  *(_QWORD *)&v100 = v24.m128i_i64[0];
  v26 = *(_WORD *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  v111.m128i_i16[0] = v26;
  v111.m128i_i64[1] = v27;
  if ( v26 )
  {
    if ( (unsigned __int16)(v26 - 17) > 0xD3u )
      goto LABEL_16;
  }
  else if ( !sub_30070B0((__int64)&v111) )
  {
    goto LABEL_16;
  }
  v38 = sub_3791F80(a1, a2);
  v105 = v38;
  v106 = v39;
  if ( v38 )
  {
    v40 = (_QWORD *)a1[1];
    v107.m128i_i16[0] = 0;
    v107.m128i_i64[1] = 0;
    v108.m128i_i16[0] = 0;
    v108.m128i_i64[1] = 0;
    v41 = *((_QWORD *)v38 + 6) + 16LL * (unsigned int)v106;
    v42 = *(_WORD *)v41;
    v43 = *(_QWORD *)(v41 + 8);
    LOWORD(v109) = v42;
    v110 = v43;
    sub_33D0340((__int64)&v111, (__int64)v40, &v109);
    v44 = _mm_loadu_si128(&v112);
    v107 = _mm_loadu_si128(&v111);
    v45 = (__int128 *)&v105;
    v108 = v44;
    goto LABEL_24;
  }
  v56 = (unsigned __int16 *)(*(_QWORD *)(v104.m128i_i64[0] + 48) + 16LL * v104.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v111, *a1, *(_QWORD *)(a1[1] + 64), *v56, *((_QWORD *)v56 + 1));
  if ( v111.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v104.m128i_u64[0], v104.m128i_i64[1], (__int64)&v100, (__int64)&v101);
    goto LABEL_16;
  }
  v57 = (unsigned __int16 *)(*(_QWORD *)(v104.m128i_i64[0] + 48) + 16LL * v104.m128i_u32[2]);
  if ( *(_DWORD *)(v104.m128i_i64[0] + 24) == 208 )
  {
    v62 = *(unsigned int **)(v104.m128i_i64[0] + 40);
    v63 = *(_QWORD *)v62;
    v64 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v62 + 48LL) + 16LL * v62[2]);
    v65 = *v64;
    v66 = *((_QWORD *)v64 + 1);
    v67 = *v57;
    v68 = *((_QWORD *)v57 + 1);
    LOWORD(v109) = v67;
    v110 = v68;
    if ( (_WORD)v67 )
    {
      v69 = word_4456580[(unsigned __int16)v67 - 1];
    }
    else
    {
      v91 = v66;
      v69 = sub_3009970((__int64)&v109, v65, v67, v63, v66);
      v66 = v91;
    }
    if ( v69 == 2 )
    {
      v86 = v66;
      sub_2FE6CC0((__int64)&v111, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v65, v66);
      if ( !v111.m128i_i8[0] )
      {
        v82 = v86;
        v87 = *a1;
        v70 = *(_QWORD *)(v104.m128i_i64[0] + 48) + 16LL * v104.m128i_u32[2];
        v71 = *(_WORD *)v70;
        v81 = *(_QWORD *)(v70 + 8);
        v72 = a1[1];
        v83 = v71;
        v89 = *(unsigned __int16 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 528LL);
        v92 = *(_QWORD *)(v72 + 64);
        v73 = sub_2E79000(*(__int64 **)(v72 + 40));
        if ( v83 == v89(v87, v73, v92, (unsigned __int16)v65, v82) && (v83 || v81 == v74) )
        {
          v75 = a1[1];
          v107.m128i_i16[0] = 0;
          v107.m128i_i64[1] = 0;
          v108.m128i_i16[0] = 0;
          v108.m128i_i64[1] = 0;
          v76 = *(_QWORD *)(v104.m128i_i64[0] + 48) + 16LL * v104.m128i_u32[2];
          v77 = *(_WORD *)v76;
          v78 = *(_QWORD *)(v76 + 8);
          v90 = (_QWORD *)v75;
          LOWORD(v109) = v77;
          v110 = v78;
          sub_33D0340((__int64)&v111, v75, &v109);
          v79 = _mm_loadu_si128(&v112);
          v107 = _mm_loadu_si128(&v111);
          v108 = v79;
          goto LABEL_38;
        }
      }
    }
    sub_377EF80(a1, v104.m128i_i64[0], (__int64)&v100, (__int64)&v101, v24);
LABEL_16:
    v28 = (_QWORD *)a1[1];
    if ( v8 - 488 <= 1 )
      goto LABEL_17;
    goto LABEL_25;
  }
  v107.m128i_i64[1] = 0;
  v58 = a1[1];
  v107.m128i_i16[0] = 0;
  v108.m128i_i16[0] = 0;
  v108.m128i_i64[1] = 0;
  v59 = *v57;
  v60 = *((_QWORD *)v57 + 1);
  v90 = (_QWORD *)v58;
  LOWORD(v109) = v59;
  v110 = v60;
  sub_33D0340((__int64)&v111, v58, &v109);
  v61 = _mm_loadu_si128(&v112);
  v107 = _mm_loadu_si128(&v111);
  v108 = v61;
LABEL_38:
  v40 = v90;
  v45 = (__int128 *)&v104;
LABEL_24:
  sub_3408290((__int64)&v111, v40, v45, (__int64)&v102, (unsigned int *)&v107, (unsigned int *)&v108, v24);
  v28 = (_QWORD *)a1[1];
  *(_QWORD *)&v100 = v111.m128i_i64[0];
  DWORD2(v100) = v111.m128i_i32[2];
  *(_QWORD *)&v101 = v112.m128i_i64[0];
  DWORD2(v101) = v112.m128i_i32[2];
  if ( v8 - 488 <= 1 )
  {
LABEL_17:
    sub_3408380(
      &v111,
      v28,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 128LL),
      **(unsigned __int16 **)(a2 + 48),
      *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
      v24,
      (__int64)&v102);
    *(_QWORD *)&v88 = v112.m128i_i64[0];
    *((_QWORD *)&v88 + 1) = v112.m128i_u32[2];
    v29 = (unsigned __int16 *)(*(_QWORD *)(v96 + 48) + 16LL * DWORD2(v96));
    *((_QWORD *)&v80 + 1) = v111.m128i_u32[2];
    *(_QWORD *)&v80 = v111.m128i_i64[0];
    v30 = sub_33FC130(
            (_QWORD *)a1[1],
            v8,
            (__int64)&v102,
            *v29,
            *((_QWORD *)v29 + 1),
            v111.m128i_u32[2],
            v100,
            v96,
            v98,
            v80);
    v94 = v31;
    v32 = v97;
    *(_QWORD *)a3 = v30;
    *(_DWORD *)(a3 + 8) = v94;
    v33 = (unsigned __int16 *)(*(_QWORD *)(v32 + 48) + 16LL * DWORD2(v97));
    v35 = sub_33FC130((_QWORD *)a1[1], v8, (__int64)&v102, *v33, *((_QWORD *)v33 + 1), v34, v101, v97, v99, v88);
    v36 = v102;
    *(_QWORD *)a4 = v35;
    *(_DWORD *)(a4 + 8) = v37;
    if ( !v36 )
      return;
    goto LABEL_18;
  }
LABEL_25:
  v46 = (unsigned __int16 *)(*(_QWORD *)(v96 + 48) + 16LL * DWORD2(v96));
  v47 = sub_340F900(v28, v8, (__int64)&v102, *v46, *((_QWORD *)v46 + 1), v23, v100, v96, v98);
  v95 = v48;
  v49 = v97;
  *(_QWORD *)a3 = v47;
  *(_DWORD *)(a3 + 8) = v95;
  v50 = (unsigned __int16 *)(*(_QWORD *)(v49 + 48) + 16LL * DWORD2(v97));
  v52 = sub_340F900((_QWORD *)a1[1], v8, (__int64)&v102, *v50, *((_QWORD *)v50 + 1), v51, v101, v97, v99);
  v36 = v102;
  *(_QWORD *)a4 = v52;
  *(_DWORD *)(a4 + 8) = v53;
  if ( v36 )
LABEL_18:
    sub_B91220((__int64)&v102, v36);
}
