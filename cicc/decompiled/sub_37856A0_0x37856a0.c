// Function: sub_37856A0
// Address: 0x37856a0
//
unsigned __int8 *__fastcall sub_37856A0(__int64 *a1, unsigned __int64 a2, __m128i a3)
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
  __int64 v26; // r11
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int32 v31; // edx
  int v32; // r9d
  __int32 v33; // edx
  __int64 v34; // r9
  _QWORD *v35; // rdi
  __m128i v36; // xmm1
  __m128i v37; // xmm0
  unsigned __int8 *v38; // rax
  _QWORD *v39; // rdi
  __m128i v40; // xmm3
  __int32 v41; // edx
  __m128i v42; // xmm2
  unsigned int v43; // esi
  unsigned __int8 *v44; // rax
  _QWORD *v45; // r9
  __int32 v46; // edx
  unsigned __int8 *v47; // rax
  __int64 v48; // rdx
  unsigned __int8 *v49; // r14
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rdi
  unsigned int v55; // esi
  __int32 v56; // edx
  __int64 v57; // r9
  __int128 v58; // [rsp-20h] [rbp-190h]
  __int128 v59; // [rsp-20h] [rbp-190h]
  __int128 v60; // [rsp-20h] [rbp-190h]
  __int128 v61; // [rsp-10h] [rbp-180h]
  __int128 v62; // [rsp-10h] [rbp-180h]
  char v63; // [rsp+8h] [rbp-168h]
  __int64 v64; // [rsp+8h] [rbp-168h]
  __int128 v65; // [rsp+10h] [rbp-160h]
  __int64 v66; // [rsp+20h] [rbp-150h]
  __int128 v67; // [rsp+20h] [rbp-150h]
  __int64 v68; // [rsp+30h] [rbp-140h]
  __int64 v69; // [rsp+30h] [rbp-140h]
  char v70; // [rsp+30h] [rbp-140h]
  __int128 v71; // [rsp+30h] [rbp-140h]
  unsigned int v72; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-B8h]
  __m128i v74; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v75; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+E0h] [rbp-90h] BYREF
  int v77; // [rsp+E8h] [rbp-88h]
  __int16 v78; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v79; // [rsp+F8h] [rbp-78h]
  __int64 v80; // [rsp+100h] [rbp-70h] BYREF
  __int64 v81; // [rsp+108h] [rbp-68h]
  __int16 v82; // [rsp+110h] [rbp-60h]
  __int64 v83; // [rsp+118h] [rbp-58h]
  __m128i v84; // [rsp+120h] [rbp-50h] BYREF
  __m128i v85; // [rsp+130h] [rbp-40h]

  v6 = *(__int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v76 = v7;
  v74.m128i_i64[0] = 0;
  LOWORD(v72) = v8;
  v73 = v9;
  v74.m128i_i32[2] = 0;
  v75.m128i_i64[0] = 0;
  v75.m128i_i32[2] = 0;
  if ( v7 )
    sub_B96E90((__int64)&v76, v7, 1);
  v10 = *(_DWORD *)(a2 + 24);
  v77 = *(_DWORD *)(a2 + 72);
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
    (__int64)&v74,
    (__int64)&v75);
  v13 = (unsigned __int16 *)(*(_QWORD *)(v74.m128i_i64[0] + 48) + 16LL * v74.m128i_u32[2]);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v78 = v14;
  v79 = v15;
  if ( (_WORD)v14 )
  {
    v16 = word_4456340;
    LOBYTE(v12) = (unsigned __int16)(v14 - 176) <= 0x34u;
    v17 = (unsigned int)v12;
    v18 = word_4456340[v14 - 1];
    v19 = (unsigned __int16)v72;
    if ( (_WORD)v72 )
    {
LABEL_8:
      v66 = 0;
      v20 = word_4456580[v19 - 1];
      goto LABEL_9;
    }
  }
  else
  {
    v18 = sub_3007240((__int64)&v78);
    v19 = (unsigned __int16)v72;
    v17 = HIDWORD(v18);
    v12 = HIDWORD(v18);
    if ( (_WORD)v72 )
      goto LABEL_8;
  }
  v63 = v17;
  v70 = v12;
  v20 = sub_3009970((__int64)&v72, v18, (__int64)v16, v12, v17);
  LOBYTE(v17) = v63;
  v66 = v52;
  LOBYTE(v12) = v70;
LABEL_9:
  v21 = v20;
  v22 = *(__int64 **)(a1[1] + 64);
  v84.m128i_i32[0] = v18;
  v84.m128i_i8[4] = v17;
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
  v23 = sub_3009450(v22, v21, v66, v84.m128i_i64[0], v24, v25);
  v3 = v23;
  v26 = v51;
LABEL_11:
  v27 = *(unsigned int *)(a2 + 24);
  LOWORD(v3) = v23;
  if ( (int)v27 > 239 )
  {
    if ( (unsigned int)(v27 - 242) > 1 )
      goto LABEL_14;
  }
  else if ( (int)v27 <= 237 && (unsigned int)(v27 - 101) > 0x2F )
  {
LABEL_14:
    if ( *(_DWORD *)(a2 + 64) == 3 )
    {
      v64 = v26;
      sub_3777990(&v84, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a3);
      *(_QWORD *)&v71 = v84.m128i_i64[0];
      *((_QWORD *)&v71 + 1) = v84.m128i_u32[2];
      *(_QWORD *)&v67 = v85.m128i_i64[0];
      *((_QWORD *)&v67 + 1) = v85.m128i_u32[2];
      sub_3408380(
        &v84,
        (_QWORD *)a1[1],
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
        **(unsigned __int16 **)(a2 + 48),
        *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
        a3,
        (__int64)&v76);
      *(_QWORD *)&v65 = v85.m128i_i64[0];
      *((_QWORD *)&v60 + 1) = v84.m128i_u32[2];
      *(_QWORD *)&v60 = v84.m128i_i64[0];
      *((_QWORD *)&v65 + 1) = v85.m128i_u32[2];
      v53 = sub_340F900(
              (_QWORD *)a1[1],
              *(_DWORD *)(a2 + 24),
              (__int64)&v76,
              v3,
              v64,
              v84.m128i_u32[2],
              *(_OWORD *)&v74,
              v71,
              v60);
      v54 = (_QWORD *)a1[1];
      v55 = *(_DWORD *)(a2 + 24);
      v74.m128i_i64[0] = v53;
      v74.m128i_i32[2] = v56;
      v75.m128i_i64[0] = sub_340F900(v54, v55, (__int64)&v76, v3, v64, v57, *(_OWORD *)&v75, v67, v65);
    }
    else
    {
      v68 = v26;
      v28 = sub_33FAF80(a1[1], v27, (__int64)&v76, (unsigned int)v3, v26, v25, a3);
      v29 = a1[1];
      v30 = *(unsigned int *)(a2 + 24);
      v74.m128i_i64[0] = (__int64)v28;
      v74.m128i_i32[2] = v31;
      v75.m128i_i64[0] = (__int64)sub_33FAF80(v29, v30, (__int64)&v76, (unsigned int)v3, v68, v32, a3);
    }
    v75.m128i_i32[2] = v33;
    goto LABEL_20;
  }
  v35 = (_QWORD *)a1[1];
  v36 = _mm_loadu_si128(&v74);
  v37 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  *((_QWORD *)&v61 + 1) = 2;
  *(_QWORD *)&v61 = &v84;
  v82 = 1;
  v84 = v37;
  v85 = v36;
  v81 = v26;
  v69 = v26;
  v80 = v3;
  v83 = 0;
  v38 = sub_3411BE0(v35, v27, (__int64)&v76, (unsigned __int16 *)&v80, 2, (__int64)&v84, v61);
  v39 = (_QWORD *)a1[1];
  v40 = _mm_loadu_si128(&v75);
  v74.m128i_i64[0] = (__int64)v38;
  v74.m128i_i32[2] = v41;
  v42 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v82 = 1;
  v43 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v58 + 1) = 2;
  *(_QWORD *)&v58 = &v84;
  v81 = v69;
  v84 = v42;
  v85 = v40;
  v80 = v3;
  v83 = 0;
  v44 = sub_3411BE0(v39, v43, (__int64)&v76, (unsigned __int16 *)&v80, 2, (__int64)&v84, v58);
  v45 = (_QWORD *)a1[1];
  v75.m128i_i64[0] = (__int64)v44;
  v75.m128i_i32[2] = v46;
  *((_QWORD *)&v62 + 1) = 1;
  *(_QWORD *)&v62 = v44;
  *((_QWORD *)&v59 + 1) = 1;
  *(_QWORD *)&v59 = v74.m128i_i64[0];
  v47 = sub_3406EB0(v45, 2u, (__int64)&v76, 1, 0, (__int64)v45, v59, v62);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v47, v48);
LABEL_20:
  v49 = sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v76, v72, v73, v34, *(_OWORD *)&v74, *(_OWORD *)&v75);
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
  return v49;
}
