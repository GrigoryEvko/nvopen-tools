// Function: sub_2BEE440
// Address: 0x2bee440
//
__int64 __fastcall sub_2BEE440(__int64 a1)
{
  int v2; // eax
  unsigned int v3; // r12d
  bool v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // r15
  unsigned __int64 *v11; // r8
  __m128i *v12; // rsi
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  unsigned __int64 *v22; // r15
  __m128i v23; // xmm3
  __m128i v24; // xmm0
  __m128i v25; // xmm2
  __m128i *v26; // rsi
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rax
  _QWORD *v35; // rdx
  __int64 *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rsi
  unsigned __int64 *v39; // rdi
  unsigned __int64 v40; // r14
  __int64 v41; // rax
  unsigned __int64 *v42; // rdi
  bool v43; // zf
  __m128i v44; // xmm5
  __m128i v45; // xmm4
  __int64 v46; // [rsp+8h] [rbp-D8h]
  __int64 v47; // [rsp+8h] [rbp-D8h]
  __int64 v48; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v49; // [rsp+10h] [rbp-D0h]
  __int64 v50; // [rsp+10h] [rbp-D0h]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  __int64 v52; // [rsp+18h] [rbp-C8h]
  unsigned __int64 *v53; // [rsp+18h] [rbp-C8h]
  __m128i v54; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v55; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v57; // [rsp+50h] [rbp-90h] BYREF
  __m128i v58; // [rsp+60h] [rbp-80h] BYREF
  __m128i v59; // [rsp+70h] [rbp-70h] BYREF
  __m128i v60; // [rsp+80h] [rbp-60h] BYREF
  __m128i v61; // [rsp+90h] [rbp-50h] BYREF
  __m128i v62; // [rsp+A0h] [rbp-40h] BYREF

  v2 = *(_DWORD *)(a1 + 152);
  if ( v2 == 22 )
  {
    v3 = sub_2BE0030(a1);
    if ( (_BYTE)v3 )
    {
      v60.m128i_i32[0] = 4;
      v39 = *(unsigned __int64 **)(a1 + 256);
      goto LABEL_41;
    }
    v2 = *(_DWORD *)(a1 + 152);
  }
  if ( v2 != 23 )
    goto LABEL_3;
  v3 = sub_2BE0030(a1);
  if ( (_BYTE)v3 )
  {
    v60.m128i_i32[0] = 5;
    v39 = *(unsigned __int64 **)(a1 + 256);
LABEL_41:
    v60.m128i_i64[1] = -1;
    v40 = sub_2BE03F0(v39, &v60);
    if ( v60.m128i_i32[0] == 11 )
      sub_A17130((__int64)&v61);
    goto LABEL_43;
  }
  v2 = *(_DWORD *)(a1 + 152);
LABEL_3:
  if ( v2 == 24 )
  {
    v3 = sub_2BE0030(a1);
    if ( !(_BYTE)v3 )
    {
      v2 = *(_DWORD *)(a1 + 152);
      goto LABEL_4;
    }
    v42 = *(unsigned __int64 **)(a1 + 256);
    v43 = **(_BYTE **)(a1 + 272) == 110;
    v62 = _mm_loadu_si128(&v59);
    v58.m128i_i8[8] = v43;
    v44 = _mm_loadu_si128(&v58);
    v57.m128i_i32[0] = 6;
    v57.m128i_i64[1] = -1;
    v45 = _mm_loadu_si128(&v57);
    v61 = v44;
    v60 = v45;
    v40 = sub_2BE03F0(v42, &v60);
    if ( v60.m128i_i32[0] == 11 )
      sub_A17130((__int64)&v61);
    if ( v57.m128i_i32[0] == 11 )
      sub_A17130((__int64)&v58);
LABEL_43:
    v41 = *(_QWORD *)(a1 + 256);
    v60.m128i_i64[1] = v40;
    v61.m128i_i64[0] = v40;
    v60.m128i_i64[0] = v41;
    sub_2BE3490((unsigned __int64 *)(a1 + 304), &v60);
    return v3;
  }
LABEL_4:
  if ( v2 == 7 && (unsigned __int8)sub_2BE0030(a1) )
  {
    v5 = **(_BYTE **)(a1 + 272) == 110;
    sub_2BECC80(a1);
    if ( *(_DWORD *)(a1 + 152) != 8 )
      goto LABEL_51;
    v3 = sub_2BE0030(a1);
    if ( !(_BYTE)v3 )
      goto LABEL_51;
    v6 = *(_QWORD *)(a1 + 352);
    if ( v6 == *(_QWORD *)(a1 + 360) )
    {
      v35 = *(_QWORD **)(*(_QWORD *)(a1 + 376) - 8LL);
      v10 = v35[62];
      v48 = v35[60];
      v51 = v35[61];
      j_j___libc_free_0(v6);
      v7 = v48;
      v8 = v51;
      v36 = (__int64 *)(*(_QWORD *)(a1 + 376) - 8LL);
      *(_QWORD *)(a1 + 376) = v36;
      v37 = *v36;
      v38 = *v36 + 504;
      *(_QWORD *)(a1 + 360) = v37;
      *(_QWORD *)(a1 + 368) = v38;
      *(_QWORD *)(a1 + 352) = v37 + 480;
    }
    else
    {
      v7 = *(_QWORD *)(v6 - 24);
      v8 = *(_QWORD *)(v6 - 16);
      v9 = v6 - 24;
      v10 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(a1 + 352) = v9;
    }
    v11 = *(unsigned __int64 **)(a1 + 256);
    v60.m128i_i32[0] = 12;
    v60.m128i_i64[1] = -1;
    v12 = (__m128i *)v11[8];
    if ( v12 == (__m128i *)v11[9] )
    {
      v47 = v8;
      v50 = v7;
      v53 = v11;
      sub_2BE00E0(v11 + 7, v12, &v60);
      v11 = v53;
      v8 = v47;
      v7 = v50;
      v18 = v53[8];
    }
    else
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(&v60);
        v13 = _mm_loadu_si128(&v61);
        v12[1] = v13;
        v12[2] = _mm_loadu_si128(&v62);
        if ( v60.m128i_i32[0] == 11 )
        {
          v12[2].m128i_i64[0] = 0;
          v14 = _mm_loadu_si128(&v61);
          v61 = v13;
          v12[1] = v14;
          v15 = v62.m128i_i64[0];
          v62.m128i_i64[0] = 0;
          v16 = v12[2].m128i_i64[1];
          v12[2].m128i_i64[0] = v15;
          v17 = v62.m128i_i64[1];
          v62.m128i_i64[1] = v16;
          v12[2].m128i_i64[1] = v17;
        }
        v12 = (__m128i *)v11[8];
      }
      v18 = (unsigned __int64)&v12[3];
      v11[8] = v18;
    }
    v19 = v18 - v11[7];
    v20 = 0xAAAAAAAAAAAAAAABLL * (v19 >> 4);
    if ( (unsigned __int64)v19 > 0x493E00 )
      goto LABEL_51;
    v21 = v20 - 1;
    if ( v60.m128i_i32[0] == 11 )
    {
      v46 = v8;
      v49 = v20 - 1;
      v52 = v7;
      sub_A17130((__int64)&v61);
      v8 = v46;
      v21 = v49;
      v7 = v52;
    }
    v54.m128i_i32[0] = 7;
    *(_QWORD *)(*(_QWORD *)(v7 + 56) + 48 * v10 + 8) = v21;
    v22 = *(unsigned __int64 **)(a1 + 256);
    v55.m128i_i64[0] = v8;
    v23 = _mm_loadu_si128(&v56);
    v54.m128i_i64[1] = -1;
    v24 = _mm_loadu_si128(&v54);
    v55.m128i_i8[8] = v5;
    v25 = _mm_loadu_si128(&v55);
    v57 = v24;
    v58 = v25;
    v59 = v23;
    v26 = (__m128i *)v22[8];
    if ( v26 == (__m128i *)v22[9] )
    {
      sub_2BE00E0(v22 + 7, v26, &v57);
      v32 = v22[8];
    }
    else
    {
      if ( v26 )
      {
        *v26 = v24;
        v27 = _mm_loadu_si128(&v58);
        v26[1] = v27;
        v26[2] = _mm_loadu_si128(&v59);
        if ( v57.m128i_i32[0] == 11 )
        {
          v26[2].m128i_i64[0] = 0;
          v28 = _mm_loadu_si128(&v58);
          v58 = v27;
          v26[1] = v28;
          v29 = v59.m128i_i64[0];
          v59.m128i_i64[0] = 0;
          v30 = v26[2].m128i_i64[1];
          v26[2].m128i_i64[0] = v29;
          v31 = v59.m128i_i64[1];
          v59.m128i_i64[1] = v30;
          v26[2].m128i_i64[1] = v31;
        }
        v26 = (__m128i *)v22[8];
      }
      v32 = (unsigned __int64)&v26[3];
      v22[8] = v32;
    }
    v33 = v32 - v22[7];
    if ( (unsigned __int64)v33 > 0x493E00 )
LABEL_51:
      abort();
    if ( v57.m128i_i32[0] == 11 )
      sub_A17130((__int64)&v58);
    if ( v54.m128i_i32[0] == 11 )
      sub_A17130((__int64)&v55);
    v34 = *(_QWORD *)(a1 + 256);
    v60.m128i_i64[1] = 0xAAAAAAAAAAAAAAABLL * (v33 >> 4) - 1;
    v61.m128i_i64[0] = v60.m128i_i64[1];
    v60.m128i_i64[0] = v34;
    sub_2BE3490((unsigned __int64 *)(a1 + 304), &v60);
  }
  else
  {
    return 0;
  }
  return v3;
}
