// Function: sub_34524D0
// Address: 0x34524d0
//
__int64 __fastcall sub_34524D0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 *v9; // rdi
  __int64 v10; // rax
  __m128i v11; // xmm2
  unsigned __int16 *v12; // rax
  int v13; // r14d
  __int64 v14; // rax
  __int64 (__fastcall *v15)(_QWORD *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r11d
  unsigned __int8 *v24; // rax
  __int64 v25; // r12
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rbx
  __int128 v34; // rax
  __int64 v35; // r9
  __int64 v36; // rdx
  char v37; // al
  int v38; // r11d
  char v39; // al
  unsigned __int8 *v40; // rax
  unsigned int v41; // edx
  unsigned int v42; // edx
  unsigned int v43; // edx
  __int64 v44; // rsi
  __int128 v45; // rax
  __int64 v46; // r14
  __int64 v47; // rdx
  __int64 v48; // r15
  __int128 v49; // rax
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r15
  __int64 v54; // r14
  __int64 v55; // r9
  unsigned __int8 *v56; // rax
  __int64 v57; // rdx
  __int128 v58; // rax
  __int64 v59; // r9
  unsigned __int8 *v60; // rax
  __int64 v61; // rdx
  __int128 v62; // rax
  __int128 v63; // rax
  __int64 v64; // r9
  unsigned int v65; // edx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdi
  unsigned __int16 *v69; // rdx
  __int64 v70; // r9
  unsigned int v71; // edx
  __int128 v72; // [rsp-30h] [rbp-140h]
  __int128 v73; // [rsp-10h] [rbp-120h]
  __int128 v74; // [rsp+10h] [rbp-100h]
  _QWORD *v75; // [rsp+20h] [rbp-F0h]
  __int64 v76; // [rsp+28h] [rbp-E8h]
  unsigned int v77; // [rsp+30h] [rbp-E0h]
  __int64 v78; // [rsp+38h] [rbp-D8h]
  __int128 v79; // [rsp+40h] [rbp-D0h]
  int v80; // [rsp+50h] [rbp-C0h]
  int v81; // [rsp+50h] [rbp-C0h]
  unsigned int v82; // [rsp+50h] [rbp-C0h]
  __int128 v83; // [rsp+50h] [rbp-C0h]
  __m128i v84; // [rsp+60h] [rbp-B0h]
  __int128 v85; // [rsp+60h] [rbp-B0h]
  __int64 v86; // [rsp+C0h] [rbp-50h] BYREF
  int v87; // [rsp+C8h] [rbp-48h]
  unsigned int v88; // [rsp+D0h] [rbp-40h] BYREF
  __int64 v89; // [rsp+D8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v86 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v86, v7, 1);
  v8 = *(_QWORD *)(a3 + 64);
  v9 = *(__int64 **)(a3 + 40);
  v87 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 40);
  v78 = v8;
  v11 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v84 = _mm_loadu_si128((const __m128i *)v10);
  v80 = *(_DWORD *)(a2 + 24);
  v12 = *(unsigned __int16 **)(a2 + 48);
  v79 = (__int128)v11;
  v13 = *v12;
  v89 = *((_QWORD *)v12 + 1);
  v14 = *a1;
  LOWORD(v88) = v13;
  v15 = *(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64))(v14 + 528);
  v16 = sub_2E79000(v9);
  v17 = v15(a1, v16, v78, v88, v89);
  v20 = *(_DWORD *)(a2 + 28);
  v77 = v17;
  v21 = 1;
  v76 = v22;
  v23 = (v80 != 285) + 281;
  if ( (_WORD)v13 == 1 || (_WORD)v13 && (v21 = (unsigned __int16)v13, a1[(unsigned __int16)v13 + 14]) )
  {
    if ( (*((_BYTE *)a1 + 500 * v21 + v23 + 6414) & 0xFB) == 0 )
    {
      if ( (v20 & 0x20) == 0 )
      {
        v81 = (v80 != 285) + 281;
        v37 = sub_33CE830((_QWORD **)a3, v84.m128i_i64[0], v84.m128i_i64[1], 1u, 0);
        v38 = v81;
        if ( !v37 )
        {
          v84.m128i_i64[0] = (__int64)sub_33FA050(
                                        a3,
                                        154,
                                        (__int64)&v86,
                                        v88,
                                        v89,
                                        v20,
                                        (unsigned __int8 *)v84.m128i_i64[0],
                                        v84.m128i_i64[1]);
          v38 = v81;
          v84.m128i_i64[1] = v43 | v84.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        }
        v82 = v38;
        v39 = sub_33CE830((_QWORD **)a3, v11.m128i_i64[0], v11.m128i_i64[1], 1u, 0);
        v23 = v82;
        if ( !v39 )
        {
          v40 = sub_33FA050(
                  a3,
                  154,
                  (__int64)&v86,
                  v88,
                  v89,
                  v20,
                  (unsigned __int8 *)v11.m128i_i64[0],
                  v11.m128i_i64[1]);
          v23 = v82;
          *(_QWORD *)&v79 = v40;
          *((_QWORD *)&v79 + 1) = v41 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        }
      }
      v24 = sub_3405C90((_QWORD *)a3, v23, (__int64)&v86, v88, v89, v20, a4, *(_OWORD *)&v84, v79);
      goto LABEL_7;
    }
  }
  v75 = *(_QWORD **)a3;
  if ( (v20 & 0x20) != 0
    || (unsigned __int8)sub_33CE830((_QWORD **)a3, v84.m128i_i64[0], v84.m128i_i64[1], 0, 0)
    && (unsigned __int8)sub_33CE830((_QWORD **)a3, v11.m128i_i64[0], v11.m128i_i64[1], 0, 0) )
  {
    v27 = (v80 != 285) + 283;
    if ( (_WORD)v13 == 1 )
    {
      v28 = 1;
    }
    else
    {
      if ( !(_WORD)v13 )
        goto LABEL_20;
      v28 = (unsigned __int16)v13;
      if ( !a1[(unsigned __int16)v13 + 14] )
        goto LABEL_20;
    }
    if ( (*((_BYTE *)a1 + 500 * v28 + v27 + 6414) & 0xFB) == 0 )
      goto LABEL_17;
LABEL_20:
    if ( (v20 & 0x20) != 0 )
      goto LABEL_21;
  }
  if ( !(unsigned __int8)sub_33CE830((_QWORD **)a3, v84.m128i_i64[0], v84.m128i_i64[1], 1u, 0)
    || !(unsigned __int8)sub_33CE830((_QWORD **)a3, v11.m128i_i64[0], v11.m128i_i64[1], 1u, 0) )
  {
    goto LABEL_29;
  }
LABEL_21:
  if ( (v20 & 0x80u) != 0
    || (unsigned __int8)sub_33CEB60(a3, v84.m128i_i64[0], v84.m128i_i64[1])
    || (unsigned __int8)sub_33CEB60(a3, v11.m128i_i64[0], v11.m128i_i64[1]) )
  {
    v27 = (v80 != 285) + 279;
    if ( (_WORD)v13 == 1 )
    {
      v29 = 1;
    }
    else
    {
      if ( !(_WORD)v13 )
        goto LABEL_45;
      v29 = (unsigned __int16)v13;
      if ( !a1[(unsigned __int16)v13 + 14] )
        goto LABEL_30;
    }
    if ( (*((_BYTE *)a1 + 500 * v29 + v27 + 6414) & 0xFB) == 0 )
    {
LABEL_17:
      v24 = sub_3405C90((_QWORD *)a3, v27, (__int64)&v86, v88, v89, v20, a4, *(_OWORD *)&v84, *(_OWORD *)&v11);
LABEL_7:
      v25 = (__int64)v24;
      goto LABEL_8;
    }
  }
LABEL_29:
  if ( (_WORD)v13 )
  {
LABEL_30:
    v30 = (unsigned int)(v13 - 17);
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
      goto LABEL_33;
    goto LABEL_31;
  }
LABEL_45:
  if ( !sub_30070B0((__int64)&v88) )
    goto LABEL_33;
LABEL_31:
  v31 = 1;
  if ( (_WORD)v88 != 1 && (!(_WORD)v88 || (v31 = (unsigned __int16)v88, !a1[(unsigned __int16)v88 + 14]))
    || (*((_BYTE *)a1 + 500 * v31 + 6620) & 0xFB) != 0 )
  {
    v25 = (__int64)sub_3412A00((_QWORD *)a3, a2, 0, v18, v19, v30, a4);
    goto LABEL_8;
  }
LABEL_33:
  if ( (v20 & 0x20) != 0 )
  {
    v32 = v84.m128i_i64[0];
    v33 = 16LL * v84.m128i_u32[2];
  }
  else
  {
    if ( !(unsigned __int8)sub_33CE830((_QWORD **)a3, v84.m128i_i64[0], v84.m128i_i64[1], 0, 0) )
    {
      v66 = sub_33ED040((_QWORD *)a3, 8u);
      v68 = v67;
      v69 = (unsigned __int16 *)(*(_QWORD *)(v11.m128i_i64[0] + 48) + 16LL * v11.m128i_u32[2]);
      *((_QWORD *)&v73 + 1) = v68;
      *(_QWORD *)&v73 = v66;
      v84.m128i_i64[0] = (__int64)sub_33FC1D0(
                                    (_QWORD *)a3,
                                    207,
                                    (__int64)&v86,
                                    *v69,
                                    *((_QWORD *)v69 + 1),
                                    v70,
                                    *(_OWORD *)&v84,
                                    *(_OWORD *)&v84,
                                    *(_OWORD *)&v11,
                                    *(_OWORD *)&v84,
                                    v73);
      v84.m128i_i64[1] = v71 | v84.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    v32 = v84.m128i_i64[0];
    if ( (unsigned __int8)sub_33CE830((_QWORD **)a3, v11.m128i_i64[0], v11.m128i_i64[1], 0, 0) )
    {
      v33 = 16LL * v84.m128i_u32[2];
    }
    else
    {
      *(_QWORD *)&v63 = sub_33ED040((_QWORD *)a3, 8u);
      v33 = 16LL * v84.m128i_u32[2];
      *(_QWORD *)&v79 = sub_33FC1D0(
                          (_QWORD *)a3,
                          207,
                          (__int64)&v86,
                          *(unsigned __int16 *)(v33 + *(_QWORD *)(v84.m128i_i64[0] + 48)),
                          *(_QWORD *)(v33 + *(_QWORD *)(v84.m128i_i64[0] + 48) + 8),
                          v64,
                          *(_OWORD *)&v11,
                          *(_OWORD *)&v11,
                          *(_OWORD *)&v84,
                          *(_OWORD *)&v11,
                          v63);
      *((_QWORD *)&v79 + 1) = v65 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
  }
  *(_QWORD *)&v34 = sub_33ED040((_QWORD *)a3, 2 * (unsigned int)(v80 != 286) + 18);
  *(_QWORD *)&v74 = sub_33FC1D0(
                      (_QWORD *)a3,
                      207,
                      (__int64)&v86,
                      *(unsigned __int16 *)(*(_QWORD *)(v32 + 48) + v33),
                      *(_QWORD *)(*(_QWORD *)(v32 + 48) + v33 + 8),
                      v35,
                      *(_OWORD *)&v84,
                      v79,
                      *(_OWORD *)&v84,
                      v79,
                      v34);
  *((_QWORD *)&v74 + 1) = v36;
  if ( (v20 & 0x20) == 0
    && !(unsigned __int8)sub_33CE830((_QWORD **)a3, v84.m128i_i64[0], v84.m128i_i64[1], 0, 0)
    && !(unsigned __int8)sub_33CE830((_QWORD **)a3, v79, *((__int64 *)&v79 + 1), 0, 0) )
  {
    *(_QWORD *)&v74 = sub_33FA050(a3, 154, (__int64)&v86, v88, v89, v20, (unsigned __int8 *)v74, *((__int64 *)&v74 + 1));
    *((_QWORD *)&v74 + 1) = v42 | *((_QWORD *)&v74 + 1) & 0xFFFFFFFF00000000LL;
  }
  if ( (v75[108] & 0x10) != 0
    || (v20 & 0x80u) != 0
    || (unsigned __int8)sub_33CEB60(a3, v84.m128i_i64[0], v84.m128i_i64[1])
    || (unsigned __int8)sub_33CEB60(a3, v79, *((_QWORD *)&v79 + 1)) )
  {
    v25 = v74;
  }
  else
  {
    v44 = 64;
    if ( v80 != 286 )
      v44 = 32;
    *(_QWORD *)&v45 = sub_3400BD0(a3, v44, (__int64)&v86, 7, 0, 1u, a4, 0);
    v83 = v45;
    v46 = sub_33FE730(a3, (__int64)&v86, v88, v89, 0, (__m128i)0LL);
    v48 = v47;
    *(_QWORD *)&v49 = sub_33ED040((_QWORD *)a3, 0x11u);
    *((_QWORD *)&v72 + 1) = v48;
    *(_QWORD *)&v72 = v46;
    v51 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v86, v77, v76, v50, v74, v72, v49);
    v53 = v52;
    v54 = v51;
    v56 = sub_3406EB0((_QWORD *)a3, 0x9Bu, (__int64)&v86, v77, v76, v55, *(_OWORD *)&v84, v83);
    *(_QWORD *)&v58 = sub_3288B20(a3, (int)&v86, v88, v89, (__int64)v56, v57, *(_OWORD *)&v84, v74, v20);
    v85 = v58;
    v60 = sub_3406EB0((_QWORD *)a3, 0x9Bu, (__int64)&v86, v77, v76, v59, v79, v83);
    *(_QWORD *)&v62 = sub_3288B20(a3, (int)&v86, v88, v89, (__int64)v60, v61, v79, v85, v20);
    v25 = sub_3288B20(a3, (int)&v86, v88, v89, v54, v53, v62, v74, v20);
  }
LABEL_8:
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v25;
}
