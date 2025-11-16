// Function: sub_38128A0
// Address: 0x38128a0
//
unsigned __int8 *__fastcall sub_38128A0(__int64 *a1, __int64 a2)
{
  const __m128i *v3; // rax
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __int64 v6; // rcx
  unsigned __int32 v7; // r15d
  unsigned __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r11
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int16 v13; // r15
  __int64 v14; // r8
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int, __int64); // rax
  __int16 v16; // cx
  __int64 v17; // r8
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned __int64 v20; // r13
  __int64 v21; // rsi
  unsigned int v22; // eax
  unsigned int v23; // edx
  unsigned __int8 *v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // edx
  unsigned __int8 *v27; // r12
  _QWORD *v28; // r10
  __int128 *v29; // r14
  __int64 v30; // r8
  __int64 v31; // rcx
  unsigned __int64 v32; // r13
  unsigned __int8 *v33; // r12
  __int64 v35; // rdx
  __int128 v36; // [rsp-40h] [rbp-110h]
  __int64 v37; // [rsp+0h] [rbp-D0h]
  __int64 v38; // [rsp+8h] [rbp-C8h]
  _QWORD *v39; // [rsp+8h] [rbp-C8h]
  __int64 v40; // [rsp+10h] [rbp-C0h]
  __int16 v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+10h] [rbp-C0h]
  int v43; // [rsp+18h] [rbp-B8h]
  unsigned int v44; // [rsp+18h] [rbp-B8h]
  __int128 v45; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+70h] [rbp-60h] BYREF
  int v48; // [rsp+78h] [rbp-58h]
  __int64 v49; // [rsp+80h] [rbp-50h] BYREF
  int v50; // [rsp+88h] [rbp-48h]
  __int64 v51; // [rsp+90h] [rbp-40h]

  v3 = *(const __m128i **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = _mm_loadu_si128(v3);
  v6 = v3->m128i_i64[0];
  v47 = v4;
  v7 = v3->m128i_u32[2];
  v8 = v3[2].m128i_u64[1];
  v9 = v3[3].m128i_i64[0];
  if ( v4 )
  {
    v40 = v6;
    sub_B96E90((__int64)&v47, v4, 1);
    v6 = v40;
  }
  v10 = *a1;
  v11 = a1[1];
  v48 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(v6 + 48) + 16LL * v7;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
  if ( v15 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v49, v10, *(_QWORD *)(v11 + 64), v13, v14);
    v16 = v50;
    v17 = v51;
  }
  else
  {
    v43 = v15(v10, *(_QWORD *)(v11 + 64), v13, v14);
    v17 = v35;
    v16 = v43;
  }
  v38 = v17;
  v41 = v16;
  sub_380F170((__int64)a1, v5.m128i_u64[0], v5.m128i_i64[1]);
  v46 = v18 | v5.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  sub_380F170((__int64)a1, v8, v9);
  v20 = v19 | v9 & 0xFFFFFFFF00000000LL;
  if ( v13 == 11 )
  {
    v21 = 236;
  }
  else if ( v41 == 11 )
  {
    v21 = 237;
  }
  else if ( v13 == 10 )
  {
    v21 = 240;
  }
  else
  {
    if ( v41 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v21 = 241;
  }
  HIWORD(v22) = HIWORD(v43);
  LOWORD(v22) = v41;
  v44 = v22;
  *(_QWORD *)&v45 = sub_33FAF80(a1[1], v21, (__int64)&v47, v22, v38, 0, v5);
  *((_QWORD *)&v45 + 1) = v23 | v46 & 0xFFFFFFFF00000000LL;
  v24 = sub_33FAF80(a1[1], (unsigned int)v21, (__int64)&v47, v44, v38, 0, v5);
  v25 = *(_QWORD *)(a2 + 80);
  v27 = v24;
  v28 = (_QWORD *)a1[1];
  v29 = *(__int128 **)(a2 + 40);
  v30 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v31 = **(unsigned __int16 **)(a2 + 48);
  v32 = v26 | v20 & 0xFFFFFFFF00000000LL;
  v49 = v25;
  if ( v25 )
  {
    v37 = v31;
    v39 = v28;
    v42 = v30;
    sub_B96E90((__int64)&v49, v25, 1);
    v31 = v37;
    v28 = v39;
    v30 = v42;
  }
  v50 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v36 + 1) = v32;
  *(_QWORD *)&v36 = v27;
  v33 = sub_33FC1D0(
          v28,
          207,
          (__int64)&v49,
          v31,
          v30,
          (__int64)&v49,
          v45,
          v36,
          v29[5],
          *(__int128 *)((char *)v29 + 120),
          v29[10]);
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  if ( v47 )
    sub_B91220((__int64)&v47, v47);
  return v33;
}
