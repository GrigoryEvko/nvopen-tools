// Function: sub_37A24F0
// Address: 0x37a24f0
//
__m128i *__fastcall sub_37A24F0(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  const __m128i *v10; // rax
  __int64 v11; // rsi
  __int128 v12; // xmm0
  unsigned __int64 v13; // r14
  __int64 v14; // r15
  char v15; // r13
  __int64 v16; // rax
  __int64 v17; // r11
  unsigned int v18; // edx
  __m128i *v19; // r13
  __int64 v21; // rdx
  __int128 v22; // [rsp-40h] [rbp-D0h]
  __int64 v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+20h] [rbp-70h]
  __int64 v26; // [rsp+20h] [rbp-70h]
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  int v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v27, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    v8 = v29;
    v9 = (unsigned __int16)v28;
  }
  else
  {
    v9 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v8 = v21;
  }
  v10 = *(const __m128i **)(a2 + 40);
  v11 = *(_QWORD *)(a2 + 80);
  v12 = (__int128)_mm_loadu_si128(v10 + 10);
  v13 = v10[7].m128i_u64[1];
  v14 = v10[8].m128i_i64[0];
  v15 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v27 = v11;
  if ( v11 )
  {
    v23 = v8;
    v25 = v9;
    sub_B96E90((__int64)&v27, v11, 1);
    v8 = v23;
    v9 = v25;
  }
  v24 = v8;
  v26 = v9;
  v28 = *(_DWORD *)(a2 + 72);
  v16 = sub_379AB60((__int64)a1, v13, v14);
  v17 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v22 + 1) = v18 | v14 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v22 = v16;
  v19 = sub_33E9660(
          (__int64 *)a1[1],
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          v15,
          v26,
          v24,
          (__int64)&v27,
          *(_OWORD *)v17,
          *(_QWORD *)(v17 + 40),
          *(_QWORD *)(v17 + 48),
          *(_OWORD *)(v17 + 80),
          v22,
          v12,
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v19, 1);
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  return v19;
}
