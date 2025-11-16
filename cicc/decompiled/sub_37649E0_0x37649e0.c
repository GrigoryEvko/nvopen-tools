// Function: sub_37649E0
// Address: 0x37649e0
//
unsigned __int8 *__fastcall sub_37649E0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rcx
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  unsigned __int16 v10; // r12
  __int64 v11; // r15
  unsigned int v12; // r10d
  _DWORD *v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // r9
  unsigned __int16 *v16; // r8
  __int64 v17; // rsi
  __int64 v18; // r11
  bool v19; // al
  int v20; // eax
  unsigned __int8 *v21; // r13
  unsigned __int16 v23; // di
  __int16 v24; // ax
  unsigned __int16 v25; // ax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int8 *v29; // r12
  unsigned int v30; // edx
  __int64 v31; // r14
  int v32; // r9d
  unsigned int v33; // edx
  __int64 v34; // rdx
  __int64 v35; // r9
  unsigned __int8 *v36; // r12
  unsigned int v37; // edx
  __int64 v38; // r14
  __int64 v39; // r9
  unsigned int v40; // edx
  __int64 v41; // r9
  int v42; // r9d
  __int64 v43; // [rsp+10h] [rbp-F0h]
  __int64 v44; // [rsp+18h] [rbp-E8h]
  char v45; // [rsp+23h] [rbp-DDh]
  unsigned __int16 *v46; // [rsp+28h] [rbp-D8h]
  _DWORD *v47; // [rsp+30h] [rbp-D0h]
  unsigned int v48; // [rsp+30h] [rbp-D0h]
  __int128 v49; // [rsp+30h] [rbp-D0h]
  __int64 v50; // [rsp+30h] [rbp-D0h]
  __int64 v51; // [rsp+40h] [rbp-C0h]
  unsigned int v52; // [rsp+40h] [rbp-C0h]
  unsigned int v53; // [rsp+40h] [rbp-C0h]
  unsigned int v54; // [rsp+48h] [rbp-B8h]
  __int64 v55; // [rsp+48h] [rbp-B8h]
  __int128 v56; // [rsp+60h] [rbp-A0h]
  __m128i v57; // [rsp+70h] [rbp-90h]
  unsigned __int64 v58; // [rsp+78h] [rbp-88h]
  __int64 v59; // [rsp+A0h] [rbp-60h] BYREF
  int v60; // [rsp+A8h] [rbp-58h]
  unsigned __int16 v61; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-48h]
  __int64 v63; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v59 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v59, v4, 1);
  v60 = *(_DWORD *)(a2 + 72);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = _mm_loadu_si128((const __m128i *)v5);
  v7 = *(_QWORD *)(v5 + 40);
  v8 = *(unsigned int *)(v5 + 48);
  v57 = _mm_loadu_si128((const __m128i *)(v5 + 40));
  v56 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 80));
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * *(unsigned int *)(v5 + 8));
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = *v9;
  if ( !*v9 )
    goto LABEL_13;
  v13 = (_DWORD *)a1[1];
  v14 = &v13[125 * v10];
  if ( v14[6600] == 2 || v14[6602] == 2 || v14[6601] == 2 )
    goto LABEL_13;
  v15 = 16 * v8;
  v16 = (unsigned __int16 *)(16 * v8 + *(_QWORD *)(v7 + 48));
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v63) = v17;
  v64 = v18;
  if ( !(_WORD)v17 )
  {
    v43 = v7;
    v44 = v18;
    v46 = v16;
    v51 = v15;
    v47 = v13;
    v45 = sub_3007030((__int64)&v63);
    v19 = sub_30070B0((__int64)&v63);
    v13 = v47;
    v15 = v51;
    v16 = v46;
    v17 = (unsigned int)v17;
    v18 = v44;
    v7 = v43;
    v12 = v10;
    if ( !v19 )
    {
      if ( !v45 )
      {
LABEL_10:
        v20 = v13[15];
        goto LABEL_11;
      }
      goto LABEL_22;
    }
    goto LABEL_20;
  }
  v23 = v17 - 17;
  if ( (unsigned __int16)(v17 - 10) > 6u && (unsigned __int16)(v17 - 126) > 0x31u )
  {
    if ( v23 > 0xD3u )
      goto LABEL_10;
    goto LABEL_20;
  }
  if ( v23 <= 0xD3u )
  {
LABEL_20:
    v20 = v13[17];
    goto LABEL_11;
  }
LABEL_22:
  v20 = v13[16];
LABEL_11:
  if ( v20 != 2 )
  {
    if ( v20 != 1 )
      goto LABEL_13;
    LOWORD(v63) = v17;
    v64 = v18;
    if ( (_WORD)v17 )
    {
      v24 = word_4456580[(unsigned __int16)v17 - 1];
    }
    else
    {
      v53 = v12;
      v50 = v7;
      v55 = v15;
      v24 = sub_3009970((__int64)&v63, v17, (__int64)v13, v7, (__int64)v16);
      v12 = v53;
      v7 = v50;
      v15 = v55;
    }
    if ( v24 != 2 )
      goto LABEL_13;
    v15 += *(_QWORD *)(v7 + 48);
    v16 = (unsigned __int16 *)v15;
  }
  v25 = *v16;
  v26 = *((_QWORD *)v16 + 1);
  v61 = v25;
  v62 = v26;
  if ( v25 )
  {
    if ( v25 == 1 || (unsigned __int16)(v25 - 504) <= 7u )
LABEL_38:
      BUG();
    v28 = 16LL * (v25 - 1) + 71615648;
    v27 = *(_QWORD *)&byte_444C4A0[16 * v25 - 16];
    LOBYTE(v28) = *(_BYTE *)(v28 + 8);
  }
  else
  {
    v54 = v12;
    v27 = sub_3007260((__int64)&v61);
    v12 = v54;
    v63 = v27;
    v64 = v28;
  }
  if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    goto LABEL_38;
  if ( v27 != *(_QWORD *)&byte_444C4A0[16 * v10 - 16] || byte_444C4A0[16 * v10 - 8] != (_BYTE)v28 )
  {
LABEL_13:
    v21 = 0;
    goto LABEL_14;
  }
  v48 = v12;
  v29 = sub_33FAF80(*a1, 234, (__int64)&v59, v12, v11, v15, v6);
  v31 = v30;
  *(_QWORD *)&v56 = sub_33FAF80(*a1, 234, (__int64)&v59, v48, v11, v32, v6);
  v52 = v48;
  *((_QWORD *)&v56 + 1) = v33 | *((_QWORD *)&v56 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v49 = sub_34074A0((_QWORD *)*a1, (__int64)&v59, v6.m128i_i64[0], v6.m128i_i64[1], v48, v11, v6);
  v58 = v31 | v57.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v49 + 1) = v34;
  v36 = sub_3406EB0(
          (_QWORD *)*a1,
          0xBAu,
          (__int64)&v59,
          v52,
          v11,
          v35,
          __PAIR128__(v58, (unsigned __int64)v29),
          *(_OWORD *)&v6);
  v38 = v37;
  *(_QWORD *)&v56 = sub_3406EB0((_QWORD *)*a1, 0xBAu, (__int64)&v59, v52, v11, v39, v56, v49);
  *((_QWORD *)&v56 + 1) = v40 | *((_QWORD *)&v56 + 1) & 0xFFFFFFFF00000000LL;
  sub_3406EB0(
    (_QWORD *)*a1,
    0xBBu,
    (__int64)&v59,
    v52,
    v11,
    v41,
    __PAIR128__(v38 | v58 & 0xFFFFFFFF00000000LL, (unsigned __int64)v36),
    v56);
  v21 = sub_33FAF80(
          *a1,
          234,
          (__int64)&v59,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v42,
          v6);
LABEL_14:
  if ( v59 )
    sub_B91220((__int64)&v59, v59);
  return v21;
}
