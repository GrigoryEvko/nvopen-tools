// Function: sub_3804970
// Address: 0x3804970
//
unsigned __int8 *__fastcall sub_3804970(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v2; // r12d
  __int16 *v4; // rdx
  unsigned __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int8 v9; // al
  unsigned int v10; // eax
  __int64 v11; // r13
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r11
  bool v15; // zf
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r11
  unsigned __int64 v21; // r8
  __int64 v22; // r9
  __int64 *v23; // r12
  __int64 v24; // rdx
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int, __int64); // rax
  int v26; // r9d
  __int64 v27; // r8
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rsi
  unsigned __int8 *v32; // r12
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // [rsp+0h] [rbp-C0h]
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  const __m128i *v39; // [rsp+20h] [rbp-A0h]
  __int64 v40; // [rsp+20h] [rbp-A0h]
  unsigned __int16 v41; // [rsp+2Eh] [rbp-92h]
  __int64 v42; // [rsp+30h] [rbp-90h]
  __int64 v43; // [rsp+40h] [rbp-80h] BYREF
  __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+50h] [rbp-70h] BYREF
  int v46; // [rsp+58h] [rbp-68h]
  __int64 v47; // [rsp+60h] [rbp-60h]
  __int64 v48; // [rsp+68h] [rbp-58h]
  __m128i v49; // [rsp+70h] [rbp-50h] BYREF
  __m128i v50; // [rsp+80h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v43) = v5;
  v44 = v6;
  if ( v5 )
  {
    if ( v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
      BUG();
    v34 = 16LL * (v5 - 1);
    v8 = *(_QWORD *)&byte_444C4A0[v34];
    v9 = byte_444C4A0[v34 + 8];
  }
  else
  {
    v47 = sub_3007260((__int64)&v43);
    v48 = v7;
    v8 = v47;
    v9 = v48;
  }
  v49.m128i_i64[0] = v8;
  v49.m128i_i8[8] = v9;
  v10 = sub_CA1930(&v49);
  v11 = a1[1];
  switch ( v10 )
  {
    case 1u:
      v41 = 2;
      break;
    case 2u:
      v41 = 3;
      break;
    case 4u:
      v41 = 4;
      break;
    case 8u:
      v41 = 5;
      break;
    case 0x10u:
      v41 = 6;
      break;
    case 0x20u:
      v41 = 7;
      break;
    case 0x40u:
      v41 = 8;
      break;
    case 0x80u:
      v41 = 9;
      break;
    default:
      v12 = sub_3007020(*(_QWORD **)(v11 + 64), v10);
      v11 = a1[1];
      HIWORD(v2) = HIWORD(v12);
      v41 = v12;
      v14 = v13;
      goto LABEL_14;
  }
  v14 = 0;
LABEL_14:
  v15 = *(_DWORD *)(a2 + 24) == 339;
  LOWORD(v2) = v41;
  v39 = *(const __m128i **)(a2 + 112);
  v16 = *(_QWORD *)(a2 + 40);
  v17 = _mm_loadu_si128((const __m128i *)v16);
  v49 = v17;
  if ( v15 )
    v50 = _mm_loadu_si128((const __m128i *)(v16 + 80));
  else
    v50 = _mm_loadu_si128((const __m128i *)(v16 + 40));
  v38 = v14;
  v18 = sub_33E5110((__int64 *)v11, v2, v14, 1, 0);
  v20 = v38;
  v21 = v18;
  v22 = v19;
  v45 = *(_QWORD *)(a2 + 80);
  if ( v45 )
  {
    v37 = v19;
    v36 = v18;
    sub_B96E90((__int64)&v45, v45, 1);
    v21 = v36;
    v22 = v37;
    v20 = v38;
  }
  v46 = *(_DWORD *)(a2 + 72);
  v23 = sub_33E6BC0((_QWORD *)v11, 338, (__int64)&v45, v41, v20, v39, v21, v22, (unsigned __int64 *)&v49, 2);
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v23, 1);
  v24 = a1[1];
  v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v25 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v49, *a1, *(_QWORD *)(v24 + 64), v43, v44);
    v27 = v50.m128i_i64[0];
    v28 = v49.m128i_u16[4];
  }
  else
  {
    v28 = v25(*a1, *(_QWORD *)(v24 + 64), v43, v44);
    v27 = v35;
  }
  v29 = *(_QWORD *)(a2 + 80);
  v30 = a1[1];
  v49.m128i_i64[0] = v29;
  if ( v29 )
  {
    v40 = v27;
    v42 = v28;
    sub_B96E90((__int64)&v49, v29, 1);
    v27 = v40;
    v28 = v42;
  }
  v49.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v43 == 11 )
  {
    v31 = 236;
  }
  else if ( v41 == 11 )
  {
    v31 = 237;
  }
  else if ( (_WORD)v43 == 10 )
  {
    v31 = 240;
  }
  else
  {
    if ( v41 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v31 = 241;
  }
  v32 = sub_33FAF80(v30, v31, (__int64)&v49, v28, v27, v26, v17);
  if ( v49.m128i_i64[0] )
    sub_B91220((__int64)&v49, v49.m128i_i64[0]);
  return v32;
}
