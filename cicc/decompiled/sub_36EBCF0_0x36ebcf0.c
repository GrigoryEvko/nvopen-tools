// Function: sub_36EBCF0
// Address: 0x36ebcf0
//
void __fastcall sub_36EBCF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __m128i v11; // xmm2
  __m128i v12; // xmm1
  __m128i v13; // xmm0
  _BOOL4 v14; // r13d
  __m128i v15; // xmm3
  int v16; // r13d
  unsigned __int64 *v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rax
  _QWORD *v20; // r9
  unsigned __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __m128i v27; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+10h] [rbp-C0h] BYREF
  int v29; // [rsp+18h] [rbp-B8h]
  __m128i v30; // [rsp+20h] [rbp-B0h]
  __m128i v31; // [rsp+30h] [rbp-A0h]
  __m128i v32; // [rsp+40h] [rbp-90h]
  unsigned __int64 *v33; // [rsp+50h] [rbp-80h] BYREF
  __int64 v34; // [rsp+58h] [rbp-78h]
  _OWORD v35[7]; // [rsp+60h] [rbp-70h] BYREF

  v3 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v6 = sub_AE2980(v3, 3u)[1];
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 80) + 96LL);
  if ( *(_DWORD *)(v8 + 32) <= 0x40u )
    v9 = *(_QWORD *)(v8 + 24);
  else
    v9 = **(_QWORD **)(v8 + 24);
  v10 = *(_QWORD *)(a2 + 80);
  v28 = v10;
  if ( v10 )
  {
    v27.m128i_i64[0] = v9;
    sub_B96E90((__int64)&v28, v10, 1);
    v7 = *(_QWORD *)(a2 + 40);
    LOBYTE(v9) = v27.m128i_i8[0];
  }
  v11 = _mm_loadu_si128((const __m128i *)(v7 + 120));
  v12 = _mm_loadu_si128((const __m128i *)(v7 + 160));
  v13 = _mm_loadu_si128((const __m128i *)(v7 + 200));
  v29 = *(_DWORD *)(a2 + 72);
  v14 = v6 != 64;
  v33 = (unsigned __int64 *)v35;
  v34 = 0x400000003LL;
  v30 = v11;
  v31 = v12;
  v32 = v13;
  v35[0] = v11;
  v35[1] = v12;
  v35[2] = v13;
  if ( (v9 & 1) != 0 )
  {
    v15 = _mm_loadu_si128((const __m128i *)(v7 + 240));
    LODWORD(v34) = 4;
    v16 = 2 * v14 + 436;
    v35[3] = v15;
    v27 = _mm_loadu_si128((const __m128i *)v7);
    sub_C8D5F0((__int64)&v33, v35, 5u, 0x10u, v4, v5);
    v17 = v33;
    v18 = _mm_load_si128(&v27);
    v19 = 2LL * (unsigned int)v34;
  }
  else
  {
    v18 = _mm_loadu_si128((const __m128i *)v7);
    v16 = 2 * v14 + 435;
    v19 = 6;
    v17 = (unsigned __int64 *)v35;
  }
  *(__m128i *)&v17[v19] = v18;
  v20 = *(_QWORD **)(a1 + 64);
  v21 = *(_QWORD *)(a2 + 48);
  v22 = *(unsigned int *)(a2 + 68);
  LODWORD(v34) = v34 + 1;
  v23 = sub_33E66D0(v20, v16, (__int64)&v28, v21, v22, (__int64)v20, v33, (unsigned int)v34);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v23, v24, v25, v26);
  sub_3421DB0(v23);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v33 != (unsigned __int64 *)v35 )
    _libc_free((unsigned __int64)v33);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
}
