// Function: sub_36EB980
// Address: 0x36eb980
//
void __fastcall sub_36EB980(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _DWORD *v4; // rax
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int8 v11; // cl
  char v12; // r15
  int v13; // eax
  __m128i v14; // xmm3
  __m128i v15; // xmm2
  __m128i v16; // xmm1
  __m128i v17; // xmm0
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r15d
  __m128i v22; // xmm0
  unsigned __int64 v23; // rdx
  _QWORD *v24; // r9
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r15
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __m128i v31; // xmm5
  unsigned int v32; // [rsp+8h] [rbp-128h]
  unsigned __int8 v33; // [rsp+Fh] [rbp-121h]
  __m128i v34; // [rsp+10h] [rbp-120h] BYREF
  __int64 v35; // [rsp+20h] [rbp-110h] BYREF
  int v36; // [rsp+28h] [rbp-108h]
  __m128i v37; // [rsp+30h] [rbp-100h]
  __m128i v38; // [rsp+40h] [rbp-F0h]
  __m128i v39; // [rsp+50h] [rbp-E0h]
  __m128i v40; // [rsp+60h] [rbp-D0h]
  unsigned __int64 *v41; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+78h] [rbp-B8h]
  _OWORD v43[11]; // [rsp+80h] [rbp-B0h] BYREF

  v3 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v4 = sub_AE2980(v3, 3u);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = (unsigned int)v4[1];
  v8 = *(_QWORD *)(*(_QWORD *)(v6 + 80) + 96LL);
  if ( *(_DWORD *)(v8 + 32) <= 0x40u )
    v9 = *(_QWORD *)(v8 + 24);
  else
    v9 = **(_QWORD **)(v8 + 24);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = v9 & 2;
  LOBYTE(v5) = (v9 & 2) != 0;
  v35 = v10;
  v12 = v9 & 1;
  if ( v10 )
  {
    v32 = v7;
    v33 = v5;
    v34.m128i_i8[0] = v11;
    sub_B96E90((__int64)&v35, v10, 1);
    v6 = *(_QWORD *)(a2 + 40);
    v7 = v32;
    v5 = v33;
    v11 = v34.m128i_i8[0];
  }
  v13 = *(_DWORD *)(a2 + 72);
  v14 = _mm_loadu_si128((const __m128i *)(v6 + 120));
  v15 = _mm_loadu_si128((const __m128i *)(v6 + 160));
  v16 = _mm_loadu_si128((const __m128i *)(v6 + 200));
  v41 = (unsigned __int64 *)v43;
  v17 = _mm_loadu_si128((const __m128i *)(v6 + 240));
  v36 = v13;
  v42 = 0x800000004LL;
  v37 = v14;
  v38 = v15;
  v39 = v16;
  v40 = v17;
  v43[0] = v14;
  v43[1] = v15;
  v43[2] = v16;
  v43[3] = v17;
  if ( v11 )
  {
    v31 = _mm_loadu_si128((const __m128i *)(v6 + 280));
    LODWORD(v42) = 5;
    v43[4] = v31;
    if ( !v12 )
    {
      v19 = 8;
      v20 = 5;
LABEL_20:
      v21 = 4 * ((_DWORD)v7 != 64) + 421;
      goto LABEL_9;
    }
    v18 = 5;
LABEL_7:
    v43[v18] = _mm_loadu_si128((const __m128i *)(v6 + 320));
    v19 = HIDWORD(v42);
    v20 = (unsigned int)(v42 + 1);
    LODWORD(v42) = v42 + 1;
    if ( (_BYTE)v5 )
    {
      v21 = 4 * ((_DWORD)v7 != 64) + 422;
      goto LABEL_9;
    }
    if ( !v11 )
    {
      v22 = _mm_loadu_si128((const __m128i *)v6);
      v23 = v20 + 1;
      v21 = 4 * ((_DWORD)v7 != 64) + 420;
      if ( HIDWORD(v42) >= (unsigned __int64)(v20 + 1) )
        goto LABEL_10;
      goto LABEL_17;
    }
    goto LABEL_20;
  }
  v18 = 4;
  if ( v12 )
    goto LABEL_7;
  if ( (_DWORD)v7 != 64 )
  {
    v19 = 8;
    v20 = 4;
    v21 = 423;
LABEL_9:
    v22 = _mm_loadu_si128((const __m128i *)v6);
    v23 = v20 + 1;
    if ( v19 >= v20 + 1 )
      goto LABEL_10;
LABEL_17:
    v34 = v22;
    sub_C8D5F0((__int64)&v41, v43, v23, 0x10u, v7, v5);
    v20 = (unsigned int)v42;
    v22 = _mm_load_si128(&v34);
    goto LABEL_10;
  }
  v22 = _mm_loadu_si128((const __m128i *)v6);
  v21 = 419;
  v20 = 4;
LABEL_10:
  *(__m128i *)&v41[2 * v20] = v22;
  v24 = *(_QWORD **)(a1 + 64);
  v25 = *(_QWORD *)(a2 + 48);
  v26 = *(unsigned int *)(a2 + 68);
  LODWORD(v42) = v42 + 1;
  v27 = sub_33E66D0(v24, v21, (__int64)&v35, v25, v26, (__int64)v24, v41, (unsigned int)v42);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v27, v28, v29, v30);
  sub_3421DB0(v27);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v41 != (unsigned __int64 *)v43 )
    _libc_free((unsigned __int64)v41);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
}
