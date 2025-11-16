// Function: sub_382B110
// Address: 0x382b110
//
unsigned __int8 *__fastcall sub_382B110(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r10
  __int64 v5; // r9
  const __m128i *v6; // rax
  __int64 v7; // r11
  __int128 v8; // xmm0
  __int64 v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r9
  __int128 v16; // rax
  unsigned __int8 *v17; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  int v23; // [rsp+28h] [rbp-58h]
  _BYTE v24[8]; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int16 v25; // [rsp+38h] [rbp-48h]
  __int64 v26; // [rsp+40h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v22 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v22, v3, 1);
  v4 = *a1;
  v5 = a1[1];
  v23 = *(_DWORD *)(a2 + 72);
  v6 = *(const __m128i **)(a2 + 40);
  v21 = v5;
  v7 = *(_QWORD *)(v5 + 64);
  v8 = (__int128)_mm_loadu_si128(v6);
  v9 = *(_QWORD *)(v6[2].m128i_i64[1] + 48) + 16LL * v6[3].m128i_u32[0];
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v24, v4, v7, v10, v11);
    v13 = v26;
    v14 = v25;
    v15 = v21;
  }
  else
  {
    v19 = v12(v4, v7, v10, v11);
    v15 = v21;
    v14 = v19;
    v13 = v20;
  }
  *(_QWORD *)&v16 = sub_33FAF80(v15, 215, (__int64)&v22, v14, v13, v15, (__m128i)v8);
  v17 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v22,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a1[1],
          v8,
          v16);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v17;
}
