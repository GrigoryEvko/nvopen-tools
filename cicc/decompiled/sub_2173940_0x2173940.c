// Function: sub_2173940
// Address: 0x2173940
//
__int64 __fastcall sub_2173940(__int64 a1, _QWORD *a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rdx
  __int8 v8; // al
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  int v11; // eax
  __int64 v12; // rsi
  __int64 *v13; // rcx
  __m128i v14; // xmm1
  __int64 v15; // rsi
  __m128i v16; // xmm0
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rax
  __m128i v20; // xmm3
  __int64 v21; // rax
  __int32 v22; // edx
  __m128i v23; // xmm2
  int v24; // r8d
  int v25; // r9d
  __m128i *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r11
  unsigned __int8 v31; // r8
  __int64 v32; // r9
  int v33; // r15d
  __int64 v34; // rcx
  __int64 v35; // r14
  __int64 v37; // [rsp+8h] [rbp-138h]
  unsigned __int8 v38; // [rsp+10h] [rbp-130h]
  __int64 v39; // [rsp+18h] [rbp-128h]
  __int64 v40; // [rsp+20h] [rbp-120h]
  unsigned __int16 v41; // [rsp+2Ch] [rbp-114h]
  __int64 *v42; // [rsp+30h] [rbp-110h]
  __int64 v43; // [rsp+38h] [rbp-108h]
  __int64 v44; // [rsp+40h] [rbp-100h] BYREF
  int v45; // [rsp+48h] [rbp-F8h]
  __m128i v46; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v47; // [rsp+60h] [rbp-E0h] BYREF
  __int64 *v48; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-C8h]
  _BYTE v50[64]; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v51; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v52; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v53; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v54; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v55[4]; // [rsp+100h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_BYTE *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v51.m128i_i8[0] = v8;
  v51.m128i_i64[1] = v9;
  if ( v8 )
  {
    v10 = v8 - 14;
    if ( v10 <= 0x5Fu )
    {
      v11 = word_432BB60[v10];
LABEL_6:
      v41 = (v11 != 2) + 679;
      goto LABEL_7;
    }
    v41 = 678;
  }
  else
  {
    if ( sub_1F58D20((__int64)&v51) )
    {
      v11 = sub_1F58D30((__int64)&v51);
      goto LABEL_6;
    }
    v41 = 678;
  }
LABEL_7:
  v12 = *(_QWORD *)(a1 + 72);
  v51.m128i_i64[0] = v12;
  if ( v12 )
    sub_1623A60((__int64)&v51, v12, 2);
  v13 = *(__int64 **)(a1 + 32);
  v51.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  sub_2170130((__int64)&v46, v13[5], v13[10], v13[11], v13[20], v13[21], a3, a4, a5, (__int64)&v51, a2);
  if ( v51.m128i_i64[0] )
    sub_161E7C0((__int64)&v51, v51.m128i_i64[0]);
  v14 = _mm_loadu_si128(&v46);
  v15 = *(_QWORD *)(a1 + 72);
  v16 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 32));
  v52 = v14;
  v44 = v15;
  v51 = v16;
  if ( v15 )
    sub_1623A60((__int64)&v44, v15, 2);
  v17 = *(_QWORD *)(a1 + 104);
  v45 = *(_DWORD *)(a1 + 64);
  v18 = sub_1E340A0(v17);
  v19 = sub_1D38BB0((__int64)a2, v18, (__int64)&v44, 3, 0, 1, v16, *(double *)v14.m128i_i64, a5, 0);
  v20 = _mm_loadu_si128(&v47);
  v53.m128i_i64[0] = v19;
  v21 = *(_QWORD *)(a1 + 32);
  v53.m128i_i32[2] = v22;
  v23 = _mm_loadu_si128((const __m128i *)(v21 + 120));
  v48 = (__int64 *)v50;
  v49 = 0x400000000LL;
  v54 = v23;
  v55[0] = v20;
  sub_16CD150((__int64)&v48, v50, 5u, 16, v24, v25);
  v26 = (__m128i *)&v48[2 * (unsigned int)v49];
  *v26 = _mm_loadu_si128(&v51);
  v27 = v44;
  v26[1] = _mm_loadu_si128(&v52);
  v26[2] = _mm_loadu_si128(&v53);
  v26[3] = _mm_loadu_si128(&v54);
  v26[4] = _mm_loadu_si128(v55);
  v28 = (unsigned int)(v49 + 5);
  LODWORD(v49) = v49 + 5;
  if ( v27 )
  {
    sub_161E7C0((__int64)&v44, v27);
    v28 = (unsigned int)v49;
  }
  v29 = *(_QWORD *)(a1 + 72);
  v43 = v28;
  v30 = *(_QWORD *)(a1 + 104);
  v31 = *(_BYTE *)(a1 + 88);
  v32 = *(_QWORD *)(a1 + 96);
  v33 = *(_DWORD *)(a1 + 60);
  v42 = v48;
  v51.m128i_i64[0] = v29;
  v34 = *(_QWORD *)(a1 + 40);
  if ( v29 )
  {
    v37 = *(_QWORD *)(a1 + 40);
    v38 = v31;
    v39 = v32;
    v40 = v30;
    sub_1623A60((__int64)&v51, v29, 2);
    v34 = v37;
    v31 = v38;
    v32 = v39;
    v30 = v40;
  }
  v51.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  v35 = sub_1D24DC0(a2, v41, (__int64)&v51, v34, v33, v30, v42, v43, v31, v32);
  if ( v51.m128i_i64[0] )
    sub_161E7C0((__int64)&v51, v51.m128i_i64[0]);
  if ( v48 != (__int64 *)v50 )
    _libc_free((unsigned __int64)v48);
  return v35;
}
