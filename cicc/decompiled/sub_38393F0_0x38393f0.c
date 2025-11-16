// Function: sub_38393F0
// Address: 0x38393f0
//
unsigned __int8 *__fastcall sub_38393F0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // r13
  __int64 v10; // rsi
  unsigned __int16 *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r13d
  unsigned __int16 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int16 *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int16 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r11
  __int64 v25; // rdx
  __int128 v26; // rax
  unsigned __int8 *v27; // rax
  unsigned int v28; // r8d
  unsigned int v29; // edx
  int v30; // eax
  __int128 v31; // rax
  __int64 v32; // r15
  __int64 v33; // r14
  __int64 v34; // r9
  __int64 v35; // r9
  unsigned int v36; // edx
  unsigned __int8 *v37; // r14
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // r8
  unsigned __int8 *v45; // rax
  unsigned int v46; // edx
  __int128 v47; // rax
  __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // r9
  unsigned int v51; // edx
  unsigned int v52; // edx
  __int64 v53; // r9
  __int128 v54; // rax
  __int64 v55; // r9
  _QWORD *v56; // rdi
  unsigned __int8 *v57; // rax
  unsigned int v58; // edx
  __int64 v59; // r9
  unsigned __int8 *v60; // rax
  unsigned int v61; // edx
  __int16 v62; // ax
  __int64 v63; // r8
  __int128 v64; // [rsp-40h] [rbp-1C0h]
  __int128 v65; // [rsp-30h] [rbp-1B0h]
  __int128 v66; // [rsp-30h] [rbp-1B0h]
  __int128 v67; // [rsp-30h] [rbp-1B0h]
  __int128 v68; // [rsp-30h] [rbp-1B0h]
  unsigned int v69; // [rsp+8h] [rbp-178h]
  _QWORD *v70; // [rsp+10h] [rbp-170h]
  int v71; // [rsp+18h] [rbp-168h]
  unsigned int v72; // [rsp+1Ch] [rbp-164h]
  unsigned int v73; // [rsp+1Ch] [rbp-164h]
  __int128 v74; // [rsp+20h] [rbp-160h]
  __int64 v75; // [rsp+30h] [rbp-150h]
  unsigned int v76; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v77; // [rsp+40h] [rbp-140h]
  unsigned __int8 *v78; // [rsp+40h] [rbp-140h]
  __int64 v79; // [rsp+48h] [rbp-138h]
  __int128 v80; // [rsp+50h] [rbp-130h]
  __m128i v81; // [rsp+60h] [rbp-120h]
  __int128 v82; // [rsp+70h] [rbp-110h]
  __int128 v84; // [rsp+80h] [rbp-100h]
  unsigned __int64 v85; // [rsp+88h] [rbp-F8h]
  __int64 v86; // [rsp+88h] [rbp-F8h]
  __int64 v87; // [rsp+E0h] [rbp-A0h] BYREF
  int v88; // [rsp+E8h] [rbp-98h]
  __int64 v89; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v90; // [rsp+F8h] [rbp-88h]
  unsigned int v91; // [rsp+100h] [rbp-80h] BYREF
  __int64 v92; // [rsp+108h] [rbp-78h]
  unsigned __int16 v93; // [rsp+110h] [rbp-70h] BYREF
  __int64 v94; // [rsp+118h] [rbp-68h]
  __int64 v95; // [rsp+120h] [rbp-60h]
  __int64 v96; // [rsp+128h] [rbp-58h]
  __int64 v97; // [rsp+130h] [rbp-50h] BYREF
  __int64 v98; // [rsp+138h] [rbp-48h]

  *(_QWORD *)&v74 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v74 + 1) = v4;
  *(_QWORD *)&v82 = sub_37AE0F0(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v6 = (unsigned int)v5;
  v7 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v82 + 1) = v5;
  v8 = _mm_loadu_si128((const __m128i *)(v7 + 80));
  v72 = *(_DWORD *)(v7 + 88);
  v9 = 16LL * v72;
  v77 = *(unsigned __int8 **)(v7 + 80);
  v81 = _mm_loadu_si128((const __m128i *)(v7 + 120));
  v80 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 160));
  sub_2FE6CC0(
    (__int64)&v97,
    *a1,
    *(_QWORD *)(a1[1] + 64),
    *(unsigned __int16 *)(v9 + *((_QWORD *)v77 + 6)),
    *(_QWORD *)(v9 + *((_QWORD *)v77 + 6) + 8));
  if ( (_BYTE)v97 == 1 )
  {
    v45 = sub_3838540(
            (__int64)a1,
            v8.m128i_u64[0],
            v8.m128i_i64[1],
            v81.m128i_i64[0],
            v81.m128i_i64[1],
            v8,
            v81.m128i_i64[0],
            v80);
    v72 = v46;
    v77 = v45;
    v9 = 16LL * v46;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = (unsigned __int16 *)(v9 + *((_QWORD *)v77 + 6));
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v87 = v10;
  v79 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v87, v10, 1);
  v88 = *(_DWORD *)(a2 + 72);
  v14 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  LODWORD(v15) = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v90 = v16;
  v17 = (unsigned __int16 *)(*(_QWORD *)(v82 + 48) + 16 * v6);
  LOWORD(v89) = v15;
  v18 = *v17;
  v92 = *((_QWORD *)v17 + 1);
  LODWORD(v14) = *(_DWORD *)(a2 + 24);
  LOWORD(v91) = v18;
  v76 = (unsigned int)v14;
  if ( (_WORD)v15 )
  {
    if ( (unsigned __int16)(v15 - 17) > 0xD3u )
    {
      LOWORD(v97) = v15;
      v98 = v16;
      goto LABEL_8;
    }
    LOWORD(v15) = word_4456580[(int)v15 - 1];
    v20 = 0;
  }
  else
  {
    v75 = v16;
    if ( !sub_30070B0((__int64)&v89) )
    {
      v98 = v75;
      LOWORD(v97) = 0;
      goto LABEL_13;
    }
    v62 = sub_3009970((__int64)&v89, v18, v43, v75, v44);
    v63 = v15;
    LOWORD(v15) = v62;
    v20 = v63;
  }
  LOWORD(v97) = v15;
  v98 = v20;
  if ( !(_WORD)v15 )
  {
LABEL_13:
    v95 = sub_3007260((__int64)&v97);
    LODWORD(v19) = v95;
    v96 = v21;
    goto LABEL_14;
  }
LABEL_8:
  if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_42;
  v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
LABEL_14:
  v22 = v91;
  if ( !(_WORD)v91 )
  {
    if ( sub_30070B0((__int64)&v91) )
    {
      v22 = sub_3009970((__int64)&v91, v18, v39, v40, v41);
      v94 = v42;
      v93 = v22;
      if ( !v22 )
        goto LABEL_18;
      goto LABEL_31;
    }
    goto LABEL_16;
  }
  if ( (unsigned __int16)(v91 - 17) > 0xD3u )
  {
LABEL_16:
    v23 = v92;
    goto LABEL_17;
  }
  v23 = 0;
  v22 = word_4456580[(unsigned __int16)v91 - 1];
LABEL_17:
  v93 = v22;
  v94 = v23;
  if ( v22 )
  {
LABEL_31:
    if ( v22 != 1 && (unsigned __int16)(v22 - 504) > 7u )
    {
      v24 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
      goto LABEL_19;
    }
LABEL_42:
    BUG();
  }
LABEL_18:
  v97 = sub_3007260((__int64)&v93);
  LODWORD(v24) = v97;
  v98 = v25;
LABEL_19:
  v71 = v24;
  v69 = v24;
  v70 = (_QWORD *)a1[1];
  *(_QWORD *)&v26 = sub_3400BD0((__int64)v70, (unsigned int)v19, (__int64)&v87, v13, v79, 0, v8, 0);
  v85 = v72 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v27 = sub_33FC130(
          v70,
          406,
          (__int64)&v87,
          v13,
          v79,
          *((__int64 *)&v26 + 1),
          __PAIR128__(v85, (unsigned __int64)v77),
          v26,
          *(_OWORD *)&v81,
          v80);
  v73 = v29;
  v78 = v27;
  *(_QWORD *)&v84 = v27;
  *((_QWORD *)&v84 + 1) = v29 | v85 & 0xFFFFFFFF00000000LL;
  if ( 2 * (int)v19 > v69
    || (v30 = *((_DWORD *)v27 + 6), v30 == 35)
    || v30 == 11
    || (unsigned __int8)sub_3813820(*a1, v76, v91, 0, v28) )
  {
    *(_QWORD *)&v31 = sub_3400BD0(a1[1], (unsigned int)(v71 - v19), (__int64)&v87, v13, v79, 0, v8, 0);
    v32 = *((_QWORD *)&v31 + 1);
    v33 = v31;
    *(_QWORD *)&v82 = sub_33FC130((_QWORD *)a1[1], 402, (__int64)&v87, v91, v92, v34, v82, v31, *(_OWORD *)&v81, v80);
    *((_QWORD *)&v82 + 1) = v36 | *((_QWORD *)&v82 + 1) & 0xFFFFFFFF00000000LL;
    if ( v76 == 423 )
    {
      *((_QWORD *)&v68 + 1) = v32;
      *(_QWORD *)&v68 = v33;
      v78 = sub_33FC130((_QWORD *)a1[1], 395, (__int64)&v87, v13, v79, v35, v84, v68, *(_OWORD *)&v81, v80);
      v73 = v61;
    }
    v37 = sub_33FC1D0(
            (_QWORD *)a1[1],
            v76,
            (__int64)&v87,
            v91,
            v92,
            v35,
            v74,
            v82,
            __PAIR128__(v73 | *((_QWORD *)&v84 + 1) & 0xFFFFFFFF00000000LL, (unsigned __int64)v78),
            *(_OWORD *)&v81,
            v80);
  }
  else
  {
    *(_QWORD *)&v47 = sub_3400BD0(a1[1], (unsigned int)v19, (__int64)&v87, v91, v92, 0, v8, 0);
    v48 = *((_QWORD *)&v47 + 1);
    v49 = v47;
    *(_QWORD *)&v74 = sub_33FC130((_QWORD *)a1[1], 402, (__int64)&v87, v91, v92, v50, v74, v47, *(_OWORD *)&v81, v80);
    *((_QWORD *)&v74 + 1) = v51 | *((_QWORD *)&v74 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v82 = sub_3400810(
                        (_QWORD *)a1[1],
                        v82,
                        *((__int64 *)&v82 + 1),
                        v81.m128i_i64[0],
                        v81.m128i_i64[1],
                        (__int64)&v87,
                        v8,
                        v80,
                        v89,
                        v90);
    *((_QWORD *)&v65 + 1) = v52 | *((_QWORD *)&v82 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v65 = v82;
    *(_QWORD *)&v54 = sub_33FC130((_QWORD *)a1[1], 400, (__int64)&v87, v91, v92, v53, v74, v65, *(_OWORD *)&v81, v80);
    v56 = (_QWORD *)a1[1];
    if ( v76 == 423 )
    {
      v60 = sub_33FC130(v56, 398, (__int64)&v87, v91, v92, v55, v54, v84, *(_OWORD *)&v81, v80);
    }
    else
    {
      v66 = v84;
      v86 = *((_QWORD *)&v54 + 1);
      v57 = sub_33FC130(v56, 402, (__int64)&v87, v91, v92, v55, v54, v66, *(_OWORD *)&v81, v80);
      *((_QWORD *)&v67 + 1) = v48;
      *(_QWORD *)&v67 = v49;
      *(_QWORD *)&v64 = v57;
      *((_QWORD *)&v64 + 1) = v58 | v86 & 0xFFFFFFFF00000000LL;
      v60 = sub_33FC130((_QWORD *)a1[1], 398, (__int64)&v87, v91, v92, v59, v64, v67, *(_OWORD *)&v81, v80);
    }
    v37 = v60;
  }
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
  return v37;
}
