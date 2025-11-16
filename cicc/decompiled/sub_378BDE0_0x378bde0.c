// Function: sub_378BDE0
// Address: 0x378bde0
//
unsigned __int8 *__fastcall sub_378BDE0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v3; // r13
  __int16 *v6; // rax
  __int64 v7; // rsi
  __int16 v8; // dx
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  unsigned __int16 *v16; // rdx
  __int64 v17; // r8
  unsigned __int64 v18; // rsi
  int v19; // eax
  unsigned __int16 v20; // ax
  unsigned int v21; // r15d
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r10
  signed int v27; // esi
  const __m128i *v28; // rax
  __int128 *v29; // rcx
  unsigned __int8 *v30; // rax
  _QWORD *v31; // rdi
  __int32 v32; // edx
  __int64 v33; // r9
  __int64 v34; // r9
  __int32 v35; // edx
  unsigned __int8 *v36; // r14
  __int64 v38; // r14
  __int64 v39; // r15
  __int64 v40; // rax
  _QWORD *v41; // rdi
  __int32 v42; // edx
  __int64 v43; // r9
  __int32 v44; // edx
  __int64 v45; // rdx
  _QWORD *v46; // rdi
  __m128i v47; // xmm1
  __m128i v48; // xmm2
  unsigned __int8 *v49; // rax
  __m128i v50; // xmm4
  _QWORD *v51; // rdi
  __int32 v52; // edx
  const __m128i *v53; // rax
  __m128i v54; // xmm3
  __m128i v55; // xmm5
  unsigned int v56; // esi
  unsigned __int8 *v57; // rax
  _QWORD *v58; // r9
  __int32 v59; // edx
  unsigned __int8 *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int128 v63; // [rsp-20h] [rbp-190h]
  __int128 v64; // [rsp-20h] [rbp-190h]
  __int128 v65; // [rsp-20h] [rbp-190h]
  __int128 v66; // [rsp-10h] [rbp-180h]
  __int128 v67; // [rsp-10h] [rbp-180h]
  __int128 v68; // [rsp-10h] [rbp-180h]
  char v69; // [rsp+7h] [rbp-169h]
  __int64 v70; // [rsp+8h] [rbp-168h]
  __int128 v71; // [rsp+10h] [rbp-160h]
  char v72; // [rsp+10h] [rbp-160h]
  __int64 v73; // [rsp+20h] [rbp-150h]
  __int64 v74; // [rsp+20h] [rbp-150h]
  __int128 v75; // [rsp+20h] [rbp-150h]
  __int64 v76; // [rsp+20h] [rbp-150h]
  unsigned int v77; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+A8h] [rbp-C8h]
  __m128i v79; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v80; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+D0h] [rbp-A0h] BYREF
  int v82; // [rsp+D8h] [rbp-98h]
  __int16 v83; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v84; // [rsp+E8h] [rbp-88h]
  __int64 v85; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+F8h] [rbp-78h]
  __int16 v87; // [rsp+100h] [rbp-70h]
  __int64 v88; // [rsp+108h] [rbp-68h]
  __m128i v89; // [rsp+110h] [rbp-60h] BYREF
  __m128i v90; // [rsp+120h] [rbp-50h]
  __m128i v91; // [rsp+130h] [rbp-40h]

  v6 = *(__int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v81 = v7;
  v79.m128i_i64[0] = 0;
  LOWORD(v77) = v8;
  v78 = v9;
  v79.m128i_i32[2] = 0;
  v80.m128i_i64[0] = 0;
  v80.m128i_i32[2] = 0;
  if ( v7 )
    sub_B96E90((__int64)&v81, v7, 1);
  v10 = *(_DWORD *)(a2 + 24);
  v82 = *(_DWORD *)(a2 + 72);
  if ( v10 > 239 )
  {
    v11 = (unsigned int)(v10 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v11 = 40;
    if ( v10 <= 237 )
      v11 = (unsigned int)(v10 - 101) < 0x30 ? 0x28 : 0;
  }
  sub_375E8D0(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + v11),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + v11 + 8),
    (__int64)&v79,
    (__int64)&v80);
  v13 = (unsigned __int16 *)(*(_QWORD *)(v79.m128i_i64[0] + 48) + 16LL * v79.m128i_u32[2]);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v83 = v14;
  v84 = v15;
  if ( (_WORD)v14 )
  {
    v16 = word_4456340;
    LOBYTE(v12) = (unsigned __int16)(v14 - 176) <= 0x34u;
    v17 = (unsigned int)v12;
    v18 = word_4456340[v14 - 1];
    v19 = (unsigned __int16)v77;
    if ( (_WORD)v77 )
    {
LABEL_8:
      v73 = 0;
      v20 = word_4456580[v19 - 1];
      goto LABEL_9;
    }
  }
  else
  {
    v18 = sub_3007240((__int64)&v83);
    v19 = (unsigned __int16)v77;
    v17 = HIDWORD(v18);
    v12 = HIDWORD(v18);
    if ( (_WORD)v77 )
      goto LABEL_8;
  }
  v69 = v17;
  v72 = v12;
  v20 = sub_3009970((__int64)&v77, v18, (__int64)v16, v12, v17);
  LOBYTE(v17) = v69;
  v73 = v45;
  LOBYTE(v12) = v72;
LABEL_9:
  v21 = v20;
  v22 = *(__int64 **)(a1[1] + 64);
  v89.m128i_i32[0] = v18;
  v89.m128i_i8[4] = v17;
  if ( (_BYTE)v12 )
  {
    LOWORD(v23) = sub_2D43AD0(v20, v18);
    v26 = 0;
    if ( (_WORD)v23 )
      goto LABEL_11;
  }
  else
  {
    LOWORD(v23) = sub_2D43050(v20, v18);
    v26 = 0;
    if ( (_WORD)v23 )
      goto LABEL_11;
  }
  v23 = sub_3009450(v22, v21, v73, v89.m128i_i64[0], v24, v25);
  v3 = v23;
  v26 = v62;
LABEL_11:
  v27 = *(_DWORD *)(a2 + 24);
  LOWORD(v3) = v23;
  v28 = *(const __m128i **)(a2 + 40);
  if ( v27 <= 239 )
  {
    if ( v27 <= 237 )
    {
      v29 = (__int128 *)&v28[2].m128i_u64[1];
      if ( (unsigned int)(v27 - 101) > 0x2F )
        goto LABEL_14;
    }
    goto LABEL_24;
  }
  if ( (unsigned int)(v27 - 242) <= 1 )
  {
LABEL_24:
    v46 = (_QWORD *)a1[1];
    v47 = _mm_loadu_si128(&v79);
    v89 = _mm_loadu_si128(v28);
    v90 = v47;
    v48 = _mm_loadu_si128(v28 + 5);
    *((_QWORD *)&v67 + 1) = 3;
    *(_QWORD *)&v67 = &v89;
    v87 = 1;
    v91 = v48;
    v86 = v26;
    v76 = v26;
    v85 = v3;
    v88 = 0;
    v49 = sub_3411BE0(v46, v27, (__int64)&v81, (unsigned __int16 *)&v85, 2, (__int64)&v89, v67);
    v50 = _mm_loadu_si128(&v80);
    v51 = (_QWORD *)a1[1];
    v79.m128i_i64[0] = (__int64)v49;
    v79.m128i_i32[2] = v52;
    v53 = *(const __m128i **)(a2 + 40);
    v54 = _mm_loadu_si128(v53);
    v90 = v50;
    v89 = v54;
    v55 = _mm_loadu_si128(v53 + 5);
    v87 = 1;
    v56 = *(_DWORD *)(a2 + 24);
    *((_QWORD *)&v64 + 1) = 3;
    *(_QWORD *)&v64 = &v89;
    v86 = v76;
    v91 = v55;
    v85 = v3;
    v88 = 0;
    v57 = sub_3411BE0(v51, v56, (__int64)&v81, (unsigned __int16 *)&v85, 2, (__int64)&v89, v64);
    v58 = (_QWORD *)a1[1];
    v80.m128i_i64[0] = (__int64)v57;
    v80.m128i_i32[2] = v59;
    *((_QWORD *)&v68 + 1) = 1;
    *(_QWORD *)&v68 = v57;
    *((_QWORD *)&v65 + 1) = 1;
    *(_QWORD *)&v65 = v79.m128i_i64[0];
    v60 = sub_3406EB0(v58, 2u, (__int64)&v81, 1, 0, (__int64)v58, v65, v68);
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v60, v61);
    goto LABEL_15;
  }
  v29 = (__int128 *)&v28[2].m128i_u64[1];
  if ( v27 != 456 )
  {
LABEL_14:
    v74 = v26;
    v30 = sub_3406EB0((_QWORD *)a1[1], 0xE6u, (__int64)&v81, (unsigned int)v3, v26, v25, *(_OWORD *)&v79, *v29);
    v31 = (_QWORD *)a1[1];
    v79.m128i_i64[0] = (__int64)v30;
    v79.m128i_i32[2] = v32;
    v80.m128i_i64[0] = (__int64)sub_3406EB0(
                                  v31,
                                  0xE6u,
                                  (__int64)&v81,
                                  (unsigned int)v3,
                                  v74,
                                  v33,
                                  *(_OWORD *)&v80,
                                  *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
    v80.m128i_i32[2] = v35;
    goto LABEL_15;
  }
  v70 = v26;
  sub_3777990(&v89, a1, v28[2].m128i_u64[1], v28[3].m128i_u64[0], a3);
  *(_QWORD *)&v75 = v89.m128i_i64[0];
  *((_QWORD *)&v75 + 1) = v89.m128i_u32[2];
  *(_QWORD *)&v71 = v90.m128i_i64[0];
  *((_QWORD *)&v71 + 1) = v90.m128i_u32[2];
  sub_3408380(
    &v89,
    (_QWORD *)a1[1],
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
    **(unsigned __int16 **)(a2 + 48),
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
    a3,
    (__int64)&v81);
  v38 = v90.m128i_i64[0];
  *((_QWORD *)&v63 + 1) = v89.m128i_u32[2];
  *(_QWORD *)&v63 = v89.m128i_i64[0];
  v39 = v90.m128i_u32[2];
  v40 = sub_340F900((_QWORD *)a1[1], 0x1C8u, (__int64)&v81, v3, v70, a1[1], *(_OWORD *)&v79, v75, v63);
  v41 = (_QWORD *)a1[1];
  *((_QWORD *)&v66 + 1) = v39;
  *(_QWORD *)&v66 = v38;
  v79.m128i_i64[0] = v40;
  v79.m128i_i32[2] = v42;
  v80.m128i_i64[0] = sub_340F900(v41, 0x1C8u, (__int64)&v81, v3, v70, v43, *(_OWORD *)&v80, v71, v66);
  v80.m128i_i32[2] = v44;
LABEL_15:
  v36 = sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v81, v77, v78, v34, *(_OWORD *)&v79, *(_OWORD *)&v80);
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  return v36;
}
