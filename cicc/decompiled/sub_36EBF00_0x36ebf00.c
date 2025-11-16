// Function: sub_36EBF00
// Address: 0x36ebf00
//
void __fastcall sub_36EBF00(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  _DWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rsi
  int v14; // eax
  __m128i v15; // xmm1
  __m128i v16; // xmm0
  __int64 v17; // rbx
  __m128i v18; // xmm0
  unsigned __int64 **v19; // rdi
  unsigned __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r12
  const __m128i *v23; // rcx
  unsigned __int64 v24; // r8
  __m128i v25; // xmm0
  char v26; // r11
  unsigned int v27; // eax
  int v28; // r11d
  __int64 v29; // r12
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __m128i v33; // xmm0
  unsigned __int64 *v34; // rax
  __int64 v35; // rdx
  const __m128i *v36; // rcx
  __m128i v37; // [rsp+10h] [rbp-F0h] BYREF
  __m128i v38; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v39; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 **v40; // [rsp+40h] [rbp-C0h]
  int v41; // [rsp+48h] [rbp-B8h]
  char v42; // [rsp+4Fh] [rbp-B1h]
  __int64 v43; // [rsp+50h] [rbp-B0h] BYREF
  int v44; // [rsp+58h] [rbp-A8h]
  __m128i v45; // [rsp+60h] [rbp-A0h]
  __m128i v46; // [rsp+70h] [rbp-90h]
  unsigned __int64 *v47; // [rsp+80h] [rbp-80h] BYREF
  __int64 v48; // [rsp+88h] [rbp-78h]
  _OWORD v49[7]; // [rsp+90h] [rbp-70h] BYREF

  v3 = a1;
  v4 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v5 = sub_AE2980(v4, 3u);
  v6 = *(_QWORD *)(a2 + 40);
  v41 = v5[1];
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 96LL);
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v8 = *(_QWORD *)(v7 + 24);
  else
    v8 = **(_QWORD **)(v7 + 24);
  v10 = sub_36D7030(v8);
  v11 = *(_QWORD *)(*(_QWORD *)(v9 + 80) + 96LL);
  if ( *(_DWORD *)(v11 + 32) <= 0x40u )
    v12 = *(_QWORD *)(v11 + 24);
  else
    v12 = **(_QWORD **)(v11 + 24);
  v13 = *(_QWORD *)(a2 + 80);
  v39.m128i_i8[0] = v12 & 1;
  v43 = v13;
  v42 = v12 & 0x1C;
  if ( v13 )
  {
    v38.m128i_i64[0] = v12;
    sub_B96E90((__int64)&v43, v13, 1);
    v9 = *(_QWORD *)(a2 + 40);
    v12 = v38.m128i_i64[0];
  }
  v14 = *(_DWORD *)(a2 + 72);
  v15 = _mm_loadu_si128((const __m128i *)(v9 + 120));
  v16 = _mm_loadu_si128((const __m128i *)(v9 + 160));
  v47 = (unsigned __int64 *)v49;
  v44 = v14;
  v48 = 0x400000002LL;
  v45 = v15;
  v46 = v16;
  v49[0] = v15;
  v49[1] = v16;
  if ( v10 )
  {
    v38.m128i_i64[0] = v3;
    v17 = v12;
    v18 = _mm_loadu_si128((const __m128i *)(v9 + 200));
    v19 = &v47;
    v20 = (unsigned __int64 *)v49;
    v21 = 2;
    v22 = 0;
    while ( 1 )
    {
      ++v22;
      *(__m128i *)&v20[2 * v21] = v18;
      v21 = (unsigned int)(v48 + 1);
      LODWORD(v48) = v48 + 1;
      if ( v10 == v22 )
        break;
      v18 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)(v22 + 5)));
      if ( v21 + 1 > (unsigned __int64)HIDWORD(v48) )
      {
        v40 = v19;
        v37 = v18;
        sub_C8D5F0((__int64)v19, v49, v21 + 1, 0x10u, v21 + 1, v12);
        v21 = (unsigned int)v48;
        v18 = _mm_load_si128(&v37);
        v19 = v40;
      }
      v20 = v47;
    }
    v12 = v17;
    v23 = *(const __m128i **)(a2 + 40);
    v24 = v21 + 1;
    v3 = v38.m128i_i64[0];
    if ( !v39.m128i_i8[0] )
    {
      v25 = _mm_loadu_si128(v23);
      if ( HIDWORD(v48) >= v24 )
        goto LABEL_15;
LABEL_34:
      v39.m128i_i64[0] = v12;
      v38 = v25;
      sub_C8D5F0((__int64)&v47, v49, v24, 0x10u, v24, v12);
      v21 = (unsigned int)v48;
      v25 = _mm_load_si128(&v38);
      v12 = v39.m128i_i64[0];
      goto LABEL_15;
    }
    v33 = _mm_loadu_si128((const __m128i *)((char *)v23 + 40 * (unsigned int)(v10 + 5)));
    if ( HIDWORD(v48) < v24 )
    {
      v38.m128i_i64[0] = v12;
      v39 = v33;
      sub_C8D5F0((__int64)&v47, v49, v21 + 1, 0x10u, v24, v12);
      v34 = v47;
      v12 = v38.m128i_i64[0];
      v33 = _mm_load_si128(&v39);
      v35 = 2LL * (unsigned int)v48;
    }
    else
    {
      v34 = v47;
      v35 = 2 * v21;
    }
LABEL_33:
    *(__m128i *)&v34[v35] = v33;
    v36 = *(const __m128i **)(a2 + 40);
    v21 = (unsigned int)(v48 + 1);
    v24 = v21 + 1;
    LODWORD(v48) = v48 + 1;
    v25 = _mm_loadu_si128(v36);
    if ( HIDWORD(v48) >= (unsigned __int64)(v21 + 1) )
      goto LABEL_15;
    goto LABEL_34;
  }
  if ( v39.m128i_i8[0] )
  {
    v33 = _mm_loadu_si128((const __m128i *)(v9 + 200));
    v35 = 4;
    v34 = (unsigned __int64 *)v49;
    goto LABEL_33;
  }
  v25 = _mm_loadu_si128((const __m128i *)v9);
  v21 = 2;
LABEL_15:
  *(__m128i *)&v47[2 * v21] = v25;
  v26 = v12 & 1;
  v27 = v48 + 1;
  LODWORD(v48) = v48 + 1;
  if ( v42 != 4 )
  {
    switch ( v10 )
    {
      case 1LL:
        if ( v26 )
          v28 = 2 * (v41 != 32) + 968;
        else
          v28 = 2 * (v41 != 32) + 967;
        goto LABEL_23;
      case 2LL:
        if ( v26 )
          v28 = 2 * (v41 != 32) + 972;
        else
          v28 = 2 * (v41 != 32) + 971;
        goto LABEL_23;
      case 3LL:
        if ( v26 )
          v28 = 2 * (v41 != 32) + 980;
        else
          v28 = 2 * (v41 != 32) + 979;
        goto LABEL_23;
      case 4LL:
        if ( v26 )
          v28 = 2 * (v41 != 32) + 988;
        else
          v28 = 2 * (v41 != 32) + 987;
        goto LABEL_23;
      case 5LL:
        if ( v26 )
          v28 = 2 * (v41 != 32) + 996;
        else
          v28 = 2 * (v41 != 32) + 995;
        goto LABEL_23;
      default:
        goto LABEL_59;
    }
  }
  switch ( v10 )
  {
    case 4LL:
      if ( v26 )
        v28 = 2 * (v41 == 32) + 984;
      else
        v28 = 2 * (v41 == 32) + 983;
      break;
    case 5LL:
      if ( v26 )
        v28 = 2 * (v41 == 32) + 992;
      else
        v28 = 2 * (v41 == 32) + 991;
      break;
    case 3LL:
      if ( v26 )
        v28 = 2 * (v41 == 32) + 976;
      else
        v28 = 2 * (v41 == 32) + 975;
      break;
    default:
LABEL_59:
      BUG();
  }
LABEL_23:
  v29 = sub_33E66D0(
          *(_QWORD **)(v3 + 64),
          v28,
          (__int64)&v43,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v12,
          v47,
          v27);
  sub_34158F0(*(_QWORD *)(v3 + 64), a2, v29, v30, v31, v32);
  sub_3421DB0(v29);
  sub_33ECEA0(*(const __m128i **)(v3 + 64), a2);
  if ( v47 != (unsigned __int64 *)v49 )
    _libc_free((unsigned __int64)v47);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
}
