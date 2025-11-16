// Function: sub_3804540
// Address: 0x3804540
//
unsigned __int8 *__fastcall sub_3804540(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r12
  __int16 *v5; // rdx
  unsigned __int16 v6; // ax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r11
  __int64 v17; // r10
  __int64 v18; // r12
  __m128i v19; // xmm0
  __int16 v20; // cx
  unsigned __int16 v21; // si
  char v22; // dl
  __m128i *v23; // r12
  __int64 v24; // rdx
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int, __int64); // rax
  int v26; // r9d
  __int64 v27; // r8
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rsi
  unsigned __int8 *v32; // r12
  __int64 v34; // rdx
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  __int16 v37; // [rsp+14h] [rbp-9Ch]
  __int64 v38; // [rsp+18h] [rbp-98h]
  unsigned __int8 v39; // [rsp+20h] [rbp-90h]
  __int64 v40; // [rsp+20h] [rbp-90h]
  __int64 *v41; // [rsp+28h] [rbp-88h]
  int v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+30h] [rbp-80h] BYREF
  __int64 v44; // [rsp+38h] [rbp-78h]
  __int64 v45; // [rsp+40h] [rbp-70h] BYREF
  int v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+50h] [rbp-60h]
  __int64 v48; // [rsp+58h] [rbp-58h]
  __m128i v49; // [rsp+60h] [rbp-50h] BYREF
  __m128i v50; // [rsp+70h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  LOWORD(v43) = v6;
  v44 = v7;
  if ( v6 )
  {
    if ( v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
      BUG();
    v9 = 16LL * (v6 - 1);
    v8 = *(_QWORD *)&byte_444C4A0[v9];
    LOBYTE(v9) = byte_444C4A0[v9 + 8];
  }
  else
  {
    v8 = sub_3007260((__int64)&v43);
    v47 = v8;
    v48 = v9;
  }
  v49.m128i_i64[0] = v8;
  v49.m128i_i8[8] = v9;
  v10 = sub_CA1930(&v49);
  v11 = a1[1];
  v41 = (__int64 *)a1[1];
  switch ( v10 )
  {
    case 1u:
      LOWORD(v12) = 2;
      break;
    case 2u:
      LOWORD(v12) = 3;
      break;
    case 4u:
      LOWORD(v12) = 4;
      break;
    case 8u:
      LOWORD(v12) = 5;
      break;
    case 0x10u:
      LOWORD(v12) = 6;
      break;
    case 0x20u:
      LOWORD(v12) = 7;
      break;
    case 0x40u:
      LOWORD(v12) = 8;
      break;
    case 0x80u:
      LOWORD(v12) = 9;
      break;
    default:
      v12 = sub_3007020(*(_QWORD **)(v11 + 64), v10);
      v2 = v12;
      v41 = (__int64 *)a1[1];
      goto LABEL_14;
  }
  v13 = 0;
LABEL_14:
  LOWORD(v2) = v12;
  v14 = *(_QWORD *)(a2 + 80);
  v15 = *(_QWORD *)(a2 + 40);
  v16 = v13;
  v17 = v2;
  v18 = *(_QWORD *)(a2 + 112);
  v45 = v14;
  v19 = _mm_loadu_si128((const __m128i *)(v18 + 40));
  v49 = v19;
  v50 = _mm_loadu_si128((const __m128i *)(v18 + 56));
  v20 = *(_WORD *)(v18 + 32);
  v39 = *(_BYTE *)(v18 + 34);
  if ( v14 )
  {
    v36 = v13;
    v35 = v17;
    v37 = *(_WORD *)(v18 + 32);
    sub_B96E90((__int64)&v45, v14, 1);
    v17 = v35;
    v16 = v36;
    v20 = v37;
  }
  v21 = *(_WORD *)(a2 + 32);
  v22 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v46 = *(_DWORD *)(a2 + 72);
  v23 = sub_33EA290(
          v41,
          (v21 >> 7) & 7,
          v22,
          v17,
          v16,
          (__int64)&v45,
          *(_OWORD *)v15,
          *(_QWORD *)(v15 + 40),
          *(_QWORD *)(v15 + 48),
          *(_OWORD *)(v15 + 80),
          *(_OWORD *)v18,
          *(_QWORD *)(v18 + 16),
          v17,
          v16,
          v39,
          v20,
          (__int64)&v49,
          0);
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v23, 1);
  v24 = a1[1];
  v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v25 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v49, *a1, *(_QWORD *)(v24 + 64), v43, v44);
    v26 = v49.m128i_u16[4];
    v27 = v50.m128i_i64[0];
    v28 = v49.m128i_u16[4];
  }
  else
  {
    v28 = v25(*a1, *(_QWORD *)(v24 + 64), v43, v44);
    v27 = v34;
    v26 = v28;
  }
  v29 = *(_QWORD *)(a2 + 80);
  v30 = a1[1];
  v49.m128i_i64[0] = v29;
  if ( v29 )
  {
    v38 = v27;
    v40 = v28;
    v42 = v26;
    sub_B96E90((__int64)&v49, v29, 1);
    v27 = v38;
    v28 = v40;
    v26 = v42;
  }
  v49.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v43 == 11 )
  {
    v31 = 236;
  }
  else if ( (_WORD)v26 == 11 )
  {
    v31 = 237;
  }
  else if ( (_WORD)v43 == 10 )
  {
    v31 = 240;
  }
  else
  {
    if ( (_WORD)v26 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v31 = 241;
  }
  LOWORD(v28) = v26;
  v32 = sub_33FAF80(v30, v31, (__int64)&v49, v28, v27, v26, v19);
  if ( v49.m128i_i64[0] )
    sub_B91220((__int64)&v49, v49.m128i_i64[0]);
  return v32;
}
