// Function: sub_36E6D40
// Address: 0x36e6d40
//
void __fastcall sub_36E6D40(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r12
  __int64 v5; // rsi
  unsigned __int16 v6; // bx
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __int32 v11; // edx
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned __int8 *v18; // rax
  const __m128i *v19; // rcx
  unsigned __int64 *v20; // r14
  __int32 v21; // edx
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __m128i v24; // xmm0
  __m128i si128; // xmm4
  __m128i v26; // xmm1
  __int64 v27; // rax
  unsigned __int64 v28; // rsi
  unsigned int v29; // edi
  __int64 v30; // rdx
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // r9
  __int64 v33; // r8
  _OWORD *v34; // r9
  unsigned __int16 v35; // r14
  __int64 v36; // rbx
  __m128i v37; // xmm0
  __m128i v38; // xmm0
  _QWORD *v39; // rdi
  unsigned __int64 v40; // r9
  unsigned int v41; // eax
  __int64 v42; // r8
  int v43; // esi
  __int64 v44; // r15
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int32 v48; // [rsp+8h] [rbp-2F8h]
  unsigned __int8 *v49; // [rsp+10h] [rbp-2F0h]
  __int32 v50; // [rsp+1Ch] [rbp-2E4h]
  __m128i v51; // [rsp+20h] [rbp-2E0h] BYREF
  __m128i v52; // [rsp+30h] [rbp-2D0h] BYREF
  _OWORD *v53; // [rsp+40h] [rbp-2C0h]
  __int64 *v54; // [rsp+48h] [rbp-2B8h]
  __int64 v55; // [rsp+50h] [rbp-2B0h] BYREF
  int v56; // [rsp+58h] [rbp-2A8h]
  __int64 v57; // [rsp+60h] [rbp-2A0h] BYREF
  int v58; // [rsp+68h] [rbp-298h]
  __m128i v59; // [rsp+70h] [rbp-290h] BYREF
  __m128i v60; // [rsp+80h] [rbp-280h] BYREF
  __m128i v61; // [rsp+90h] [rbp-270h] BYREF
  __m128i v62[2]; // [rsp+A0h] [rbp-260h] BYREF
  unsigned __int64 *v63; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-238h]
  _OWORD v65[35]; // [rsp+D0h] [rbp-230h] BYREF

  v4 = a1;
  v5 = *(_QWORD *)(a2 + 80);
  v54 = &v55;
  v55 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v55, v5, 1);
  v6 = *(_WORD *)(a2 + 96);
  v7 = *(_QWORD *)(a1 + 64);
  v56 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v49 = sub_3400BD0(v7, (__int64)v9, (__int64)v54, 8, 0, 1u, a3, 0);
  v10 = *(_QWORD *)(a2 + 40);
  v48 = v11;
  v12 = *(_QWORD *)(v10 + 120);
  v13 = *(_QWORD *)(v10 + 128);
  v59.m128i_i64[0] = 0;
  v59.m128i_i32[2] = 0;
  v63 = 0;
  LODWORD(v64) = 0;
  sub_36DF750(v4, v12, v13, (__int64)&v59, (__int64)&v63, a3);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = v59.m128i_i64[0];
  LODWORD(v53) = v59.m128i_i32[2];
  v57 = v14;
  v51.m128i_i64[0] = (__int64)v63;
  v50 = v64;
  v52.m128i_i64[0] = (__int64)&v57;
  if ( v14 )
    sub_B96E90((__int64)&v57, v14, 1);
  v16 = *(_QWORD *)(a2 + 112);
  v58 = *(_DWORD *)(a2 + 72);
  v17 = sub_36D7800(v16);
  v18 = sub_3400BD0(*(_QWORD *)(v4 + 64), v17, v52.m128i_i64[0], 7, 0, 1u, a3, 0);
  v19 = *(const __m128i **)(a2 + 40);
  v61.m128i_i64[0] = v15;
  v60.m128i_i64[0] = (__int64)v18;
  v20 = (unsigned __int64 *)v65;
  v60.m128i_i32[2] = v21;
  v61.m128i_i32[2] = (int)v53;
  v22 = _mm_load_si128(&v60);
  v23 = _mm_load_si128(&v61);
  v62[0].m128i_i64[0] = v51.m128i_i64[0];
  v62[0].m128i_i32[2] = v50;
  v24 = _mm_loadu_si128(v19 + 10);
  si128 = _mm_load_si128(v62);
  v63 = (unsigned __int64 *)v65;
  v59.m128i_i64[0] = (__int64)v49;
  v62[1] = v24;
  v59.m128i_i32[2] = v48;
  v26 = _mm_load_si128(&v59);
  v64 = 0x2000000005LL;
  v65[0] = v26;
  v65[1] = v22;
  v65[2] = v23;
  v65[3] = si128;
  v65[4] = v24;
  if ( !v57 )
  {
    v29 = *(_DWORD *)(a2 + 64);
    if ( v29 <= 6 )
    {
      v38 = _mm_loadu_si128(v19);
      v30 = 5;
      goto LABEL_17;
    }
    v27 = 5;
    v28 = 32;
    goto LABEL_9;
  }
  sub_B91220(v52.m128i_i64[0], v57);
  v27 = (unsigned int)v64;
  v28 = HIDWORD(v64);
  v29 = *(_DWORD *)(a2 + 64);
  v19 = *(const __m128i **)(a2 + 40);
  v30 = (unsigned int)v64;
  v31 = (unsigned int)v64 + 1LL;
  v32 = HIDWORD(v64);
  if ( v29 > 6 )
  {
LABEL_9:
    v33 = 240;
    v52.m128i_i64[0] = v4;
    v34 = v65;
    v35 = v6;
    v36 = 240;
    while ( 1 )
    {
      v37 = _mm_loadu_si128((const __m128i *)((char *)v19 + v36));
      if ( v27 + 1 > v28 )
      {
        v53 = v34;
        v51 = v37;
        sub_C8D5F0((__int64)&v63, v34, v27 + 1, 0x10u, v33, (__int64)v34);
        v27 = (unsigned int)v64;
        v37 = _mm_load_si128(&v51);
        v34 = v53;
      }
      v36 += 40;
      *(__m128i *)&v63[2 * v27] = v37;
      v27 = (unsigned int)(v64 + 1);
      LODWORD(v64) = v64 + 1;
      if ( 40LL * (v29 - 7) + 280 == v36 )
        break;
      v19 = *(const __m128i **)(a2 + 40);
      v28 = HIDWORD(v64);
    }
    v6 = v35;
    v30 = (unsigned int)v27;
    v20 = (unsigned __int64 *)v34;
    v4 = v52.m128i_i64[0];
    v19 = *(const __m128i **)(a2 + 40);
    v32 = HIDWORD(v64);
    v31 = (unsigned int)v27 + 1LL;
  }
  v38 = _mm_loadu_si128(v19);
  if ( v31 > v32 )
  {
    v52 = v38;
    sub_C8D5F0((__int64)&v63, v20, v31, 0x10u, v31, v32);
    v30 = (unsigned int)v64;
    v38 = _mm_load_si128(&v52);
  }
LABEL_17:
  *(__m128i *)&v63[2 * v30] = v38;
  v39 = *(_QWORD **)(v4 + 64);
  v40 = *(_QWORD *)(a2 + 48);
  v41 = v64 + 1;
  v42 = *(unsigned int *)(a2 + 68);
  LODWORD(v64) = v64 + 1;
  if ( v6 == 64 )
  {
    v43 = 2897;
  }
  else if ( v6 > 0x40u )
  {
    switch ( v6 )
    {
      case 0x99u:
        v43 = 2889;
        break;
      case 0xA7u:
        v43 = 2891;
        break;
      case 0x95u:
        v43 = 2888;
        break;
      default:
        goto LABEL_43;
    }
  }
  else if ( v6 == 58 )
  {
    v43 = 2895;
  }
  else
  {
    if ( v6 <= 0x3Au )
    {
      if ( v6 == 7 )
      {
        v43 = 2894;
        goto LABEL_23;
      }
      if ( v6 == 13 )
      {
        v43 = 2890;
        goto LABEL_23;
      }
LABEL_43:
      BUG();
    }
    if ( v6 != 60 )
      goto LABEL_43;
    v43 = 2896;
  }
LABEL_23:
  v44 = sub_33E66D0(v39, v43, (__int64)v54, v40, v42, v40, v63, v41);
  sub_34158F0(*(_QWORD *)(v4 + 64), a2, v44, v45, v46, v47);
  sub_3421DB0(v44);
  sub_33ECEA0(*(const __m128i **)(v4 + 64), a2);
  if ( v63 != v20 )
    _libc_free((unsigned __int64)v63);
  if ( v55 )
    sub_B91220((__int64)v54, v55);
}
