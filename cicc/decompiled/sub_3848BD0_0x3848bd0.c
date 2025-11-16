// Function: sub_3848BD0
// Address: 0x3848bd0
//
unsigned __int8 *__fastcall sub_3848BD0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int16 v9; // r14
  __int64 v10; // r8
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  int v12; // eax
  const __m128i *v13; // rdx
  unsigned __int64 v14; // rcx
  __m128i v15; // xmm1
  unsigned __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int16 v25; // cx
  __int64 v26; // rax
  __m128i *v27; // rax
  _QWORD *v28; // rdi
  __int32 v29; // edx
  unsigned __int8 *v30; // rax
  _QWORD *v31; // rdi
  unsigned int v32; // edx
  __int64 v33; // rax
  __int16 v34; // si
  unsigned __int8 v35; // cl
  unsigned __int64 v36; // r8
  int v37; // edx
  __int64 v38; // r13
  __m128i *v39; // rax
  _QWORD *v40; // rdi
  int v41; // edx
  __int64 v42; // r9
  unsigned __int8 *v43; // r12
  __int64 v45; // r13
  char v46; // r9
  bool v47; // al
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // [rsp+0h] [rbp-130h]
  unsigned __int64 v51; // [rsp+10h] [rbp-120h]
  unsigned __int64 v52; // [rsp+18h] [rbp-118h]
  unsigned __int64 v53; // [rsp+20h] [rbp-110h]
  unsigned __int64 v54; // [rsp+28h] [rbp-108h]
  __int64 v55; // [rsp+70h] [rbp-C0h] BYREF
  int v56; // [rsp+78h] [rbp-B8h]
  int v57; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+88h] [rbp-A8h]
  __m128i v59; // [rsp+90h] [rbp-A0h] BYREF
  __m128i *v60; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v61; // [rsp+A8h] [rbp-88h]
  __int64 v62; // [rsp+B0h] [rbp-80h]
  __int64 v63; // [rsp+B8h] [rbp-78h]
  __int128 v64; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v65; // [rsp+D0h] [rbp-60h]
  __m128i v66; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v67; // [rsp+F0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 80);
  v55 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v55, v5, 1);
  v6 = *a1;
  v56 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL);
  v8 = a1[1];
  v9 = *(_WORD *)v7;
  v10 = *(_QWORD *)(v7 + 8);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v66, v6, *(_QWORD *)(v8 + 64), v9, v10);
    LOWORD(v12) = v66.m128i_i16[4];
    LOWORD(v57) = v66.m128i_i16[4];
    v58 = v67.m128i_i64[0];
  }
  else
  {
    v12 = v11(v6, *(_QWORD *)(v8 + 64), v9, v10);
    v57 = v12;
    v58 = v48;
  }
  v13 = *(const __m128i **)(a2 + 40);
  v14 = v13->m128i_u64[1];
  v15 = _mm_loadu_si128(v13 + 5);
  v16 = v13->m128i_i64[0];
  v17 = *(_QWORD *)(a2 + 112);
  v52 = v14;
  v66 = _mm_loadu_si128((const __m128i *)(v17 + 40));
  v67 = _mm_loadu_si128((const __m128i *)(v17 + 56));
  if ( (_WORD)v12 )
  {
    if ( (_WORD)v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      BUG();
    v19 = 16LL * ((unsigned __int16)v12 - 1);
    v18 = *(_QWORD *)&byte_444C4A0[v19];
    LOBYTE(v19) = byte_444C4A0[v19 + 8];
  }
  else
  {
    v18 = sub_3007260((__int64)&v57);
    v62 = v18;
    v63 = v19;
  }
  BYTE8(v64) = v19;
  *(_QWORD *)&v64 = v18;
  v20 = sub_CA1930(&v64);
  v59.m128i_i32[2] = 0;
  LODWORD(v61) = 0;
  v51 = v20 >> 3;
  v21 = *(_QWORD *)(a2 + 40);
  v59.m128i_i64[0] = 0;
  v22 = *(_QWORD *)(v21 + 40);
  v23 = *(_QWORD *)(v21 + 48);
  v60 = 0;
  v24 = *(_QWORD *)(v22 + 48) + 16LL * *(unsigned int *)(v21 + 48);
  v25 = *(_WORD *)v24;
  v26 = *(_QWORD *)(v24 + 8);
  LOWORD(v64) = v25;
  *((_QWORD *)&v64 + 1) = v26;
  if ( v25 )
  {
    if ( (unsigned __int16)(v25 - 2) <= 7u
      || (unsigned __int16)(v25 - 17) <= 0x6Cu
      || (unsigned __int16)(v25 - 176) <= 0x1Fu )
    {
      goto LABEL_11;
    }
LABEL_23:
    sub_375E6F0((__int64)a1, v22, v23, (__int64)&v59, (__int64)&v60);
    goto LABEL_12;
  }
  v50 = v23;
  v47 = sub_3007070((__int64)&v64);
  v23 = v50;
  if ( !v47 )
    goto LABEL_23;
LABEL_11:
  sub_375E510((__int64)a1, v22, v23, (__int64)&v59, (__int64)&v60);
LABEL_12:
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) == 1 || v9 == 16 )
  {
    a3 = _mm_loadu_si128(&v59);
    v59.m128i_i64[0] = (__int64)v60;
    v59.m128i_i32[2] = v61;
    v60 = (__m128i *)a3.m128i_i64[0];
    LODWORD(v61) = a3.m128i_i32[2];
  }
  v27 = sub_33F4560(
          (_QWORD *)a1[1],
          v16,
          v52,
          (__int64)&v55,
          v59.m128i_u64[0],
          v59.m128i_u64[1],
          v15.m128i_u64[0],
          v15.m128i_u64[1],
          *(_OWORD *)*(_QWORD *)(a2 + 112),
          *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
          *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
          *(_WORD *)(*(_QWORD *)(a2 + 112) + 32LL),
          (__int64)&v66);
  BYTE8(v64) = 0;
  v28 = (_QWORD *)a1[1];
  v59.m128i_i64[0] = (__int64)v27;
  *(_QWORD *)&v64 = (unsigned int)v51;
  v59.m128i_i32[2] = v29;
  v30 = sub_3409320(v28, v15.m128i_i64[0], v15.m128i_i64[1], (unsigned int)v51, 0, (__int64)&v55, a3, 1);
  v31 = (_QWORD *)a1[1];
  v53 = (unsigned __int64)v30;
  v54 = v32 | v15.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v33 = *(_QWORD *)(a2 + 112);
  v34 = *(_WORD *)(v33 + 32);
  v35 = *(_BYTE *)(v33 + 34);
  v36 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v36 )
  {
    v45 = *(_QWORD *)(v33 + 8) + (unsigned int)v51;
    v46 = *(_BYTE *)(v33 + 20);
    if ( (*(_QWORD *)v33 & 4) != 0 )
    {
      *((_QWORD *)&v64 + 1) = *(_QWORD *)(v33 + 8) + (unsigned int)v51;
      BYTE4(v65) = v46;
      *(_QWORD *)&v64 = v36 | 4;
      LODWORD(v65) = *(_DWORD *)(v36 + 12);
    }
    else
    {
      *(_QWORD *)&v64 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v64 + 1) = v45;
      BYTE4(v65) = v46;
      v49 = *(_QWORD *)(v36 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 <= 1 )
        v49 = **(_QWORD **)(v49 + 16);
      LODWORD(v65) = *(_DWORD *)(v49 + 8) >> 8;
    }
  }
  else
  {
    v37 = *(_DWORD *)(v33 + 16);
    v38 = *(_QWORD *)(v33 + 8) + (unsigned int)v51;
    *(_QWORD *)&v64 = 0;
    *((_QWORD *)&v64 + 1) = v38;
    LODWORD(v65) = v37;
    BYTE4(v65) = 0;
  }
  v39 = sub_33F4560(
          v31,
          v16,
          v52,
          (__int64)&v55,
          (unsigned __int64)v60,
          v61,
          v53,
          v54,
          v64,
          v65,
          v35,
          v34,
          (__int64)&v66);
  v40 = (_QWORD *)a1[1];
  v60 = v39;
  LODWORD(v61) = v41;
  v43 = sub_3406EB0(v40, 2u, (__int64)&v55, 1, 0, v42, *(_OWORD *)&v59, __PAIR128__(v61, (unsigned __int64)v39));
  if ( v55 )
    sub_B91220((__int64)&v55, v55);
  return v43;
}
