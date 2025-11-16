// Function: sub_33A9E30
// Address: 0x33a9e30
//
__int64 __fastcall sub_33A9E30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned int v23; // r12d
  __int64 v25; // r12
  __m128i v26; // xmm0
  _QWORD *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __m128i si128; // xmm0
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-D0h]
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+18h] [rbp-B8h]
  __int64 v35; // [rsp+18h] [rbp-B8h]
  __int64 v36; // [rsp+20h] [rbp-B0h]
  __int64 v37; // [rsp+20h] [rbp-B0h]
  __int64 v38; // [rsp+28h] [rbp-A8h]
  __m128i v39; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v40; // [rsp+40h] [rbp-90h]
  __int64 v41; // [rsp+50h] [rbp-80h] BYREF
  int v42; // [rsp+58h] [rbp-78h]
  unsigned __int64 v43; // [rsp+60h] [rbp-70h] BYREF
  __int64 v44; // [rsp+68h] [rbp-68h]
  __int64 v45; // [rsp+70h] [rbp-60h]
  __m128i v46; // [rsp+80h] [rbp-50h] BYREF
  __m128i v47[4]; // [rsp+90h] [rbp-40h] BYREF

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v3));
  v6 = *(_QWORD *)(a2 + 32 * (2 - v3));
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 8LL);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 72LL);
  v44 = 0;
  BYTE4(v45) = 0;
  v39.m128i_i64[0] = v8;
  v43 = v4 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = 0;
  if ( v4 )
  {
    v10 = *(_QWORD *)(v4 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
      v10 = **(_QWORD **)(v10 + 16);
    v9 = *(_DWORD *)(v10 + 8) >> 8;
  }
  v34 = v4;
  v36 = v5;
  LODWORD(v45) = v9;
  v11 = sub_338B750(a1, v6);
  v13 = v12;
  v37 = sub_338B750(a1, v36);
  v38 = v14;
  v15 = sub_338B750(a1, v34);
  v16 = *(_QWORD *)(a1 + 864);
  v41 = 0;
  v17 = v15;
  v19 = v18;
  v20 = *(_QWORD *)a1;
  v42 = *(_DWORD *)(a1 + 848);
  v21 = v16;
  if ( v20 )
  {
    if ( &v41 != (__int64 *)(v20 + 48) )
    {
      v22 = *(_QWORD *)(v20 + 48);
      v41 = v22;
      if ( v22 )
      {
        v32 = v17;
        v33 = v19;
        v35 = v16;
        sub_B96E90((__int64)&v41, v22, 1);
        v21 = *(_QWORD *)(a1 + 864);
        v17 = v32;
        v19 = v33;
        v16 = v35;
      }
    }
  }
  if ( (_OWORD *(__fastcall *)(_OWORD *))v39.m128i_i64[0] == sub_3364F80 )
  {
    v23 = 0;
    if ( v41 )
      sub_B91220((__int64)&v41, v41);
  }
  else
  {
    ((void (__fastcall *)(__m128i *, __int64, __int64, __int64 *, _QWORD, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, unsigned __int64, __int64, __int64))v39.m128i_i64[0])(
      &v46,
      v7,
      v21,
      &v41,
      *(_QWORD *)(v16 + 384),
      *(_QWORD *)(v16 + 392),
      v17,
      v19,
      v37,
      v38,
      v11,
      v13,
      v43,
      v44,
      v45);
    v25 = v46.m128i_i64[0];
    if ( v41 )
      sub_B91220((__int64)&v41, v41);
    if ( v25 )
    {
      v26 = _mm_load_si128(&v46);
      v43 = a2;
      v39 = v26;
      v27 = sub_337DC20(a1 + 8, (__int64 *)&v43);
      v40 = _mm_load_si128(&v39);
      *v27 = v40.m128i_i64[0];
      si128 = _mm_load_si128(v47);
      *((_DWORD *)v27 + 2) = v40.m128i_i32[2];
      v31 = *(unsigned int *)(a1 + 136);
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        v39 = si128;
        sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v31 + 1, 0x10u, v28, v29);
        v31 = *(unsigned int *)(a1 + 136);
        si128 = _mm_load_si128(&v39);
      }
      v23 = 1;
      *(__m128i *)(*(_QWORD *)(a1 + 128) + 16 * v31) = si128;
      ++*(_DWORD *)(a1 + 136);
    }
    else
    {
      return 0;
    }
  }
  return v23;
}
