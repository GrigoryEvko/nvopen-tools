// Function: sub_37FCF70
// Address: 0x37fcf70
//
__m128i *__fastcall sub_37FCF70(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v3; // rcx
  __int16 *v6; // rax
  __int64 v7; // r10
  __int64 v8; // rdx
  __int16 v9; // di
  __int64 v10; // rbx
  __int64 v11; // r11
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // r11
  __int64 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int16 v17; // si
  __int64 *v18; // rdi
  __int64 v19; // r8
  unsigned __int8 v20; // r9
  int v21; // esi
  __int16 v22; // r14
  __m128i *v23; // r14
  __int64 v25; // r11
  __int64 v26; // rax
  unsigned int v27; // eax
  int v28; // r9d
  unsigned __int8 *v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rdx
  __int64 v32; // [rsp+0h] [rbp-A0h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int16 v34; // [rsp+12h] [rbp-8Eh]
  __int16 v35; // [rsp+1Eh] [rbp-82h]
  __m128i *v36; // [rsp+30h] [rbp-70h]
  __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  int v38; // [rsp+48h] [rbp-58h]
  __m128i v39; // [rsp+50h] [rbp-50h] BYREF
  __m128i v40; // [rsp+60h] [rbp-40h]

  v3 = 0;
  v6 = *(__int16 **)(a2 + 48);
  v7 = *a1;
  v8 = a1[1];
  v9 = *v6;
  v10 = *((_QWORD *)v6 + 1);
  v11 = *(_QWORD *)(v8 + 64);
  v35 = *v6;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    LOWORD(v3) = v9;
    sub_2FE6CC0((__int64)&v39, v7, v11, v3, v10);
    v13 = v40.m128i_i64[0];
    v14 = v39.m128i_u16[4];
  }
  else
  {
    LOWORD(v3) = v35;
    v34 = 0;
    v14 = v12(v7, v11, v3, v10);
    v13 = v31;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v37 = v15;
  if ( v15 )
  {
    v32 = v14;
    v33 = v13;
    sub_B96E90((__int64)&v37, v15, 1);
    v14 = v32;
    v13 = v33;
  }
  v16 = *(_QWORD *)(a2 + 112);
  v17 = *(_WORD *)(a2 + 32);
  v18 = (__int64 *)a1[1];
  v19 = *(_QWORD *)(a2 + 40);
  v20 = *(_BYTE *)(v16 + 34);
  v38 = *(_DWORD *)(a2 + 72);
  v21 = (v17 >> 7) & 7;
  v22 = *(_WORD *)(v16 + 32) & 0x3CF;
  if ( (*(_BYTE *)(a2 + 33) & 0xC) != 0 )
  {
    v25 = *(_QWORD *)(a2 + 104);
    v26 = *(unsigned __int16 *)(a2 + 96);
    v39 = _mm_loadu_si128((const __m128i *)(v16 + 40));
    v40 = _mm_loadu_si128((const __m128i *)(v16 + 56));
    v36 = sub_33EA290(
            v18,
            v21,
            0,
            (unsigned __int16)v26,
            v25,
            (__int64)&v37,
            *(_OWORD *)v19,
            *(_QWORD *)(v19 + 40),
            *(_QWORD *)(v19 + 48),
            *(_OWORD *)(v19 + 80),
            *(_OWORD *)v16,
            *(_QWORD *)(v16 + 16),
            v26,
            v25,
            v20,
            v22,
            (__int64)&v39,
            0);
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v36, 1);
    HIWORD(v27) = v34;
    LOWORD(v27) = v35;
    v29 = sub_33FAF80(a1[1], 233, (__int64)&v37, v27, v10, v28, a3);
    v23 = (__m128i *)sub_375A6A0((__int64)a1, (__int64)v29, v30, a3);
  }
  else
  {
    v39 = _mm_loadu_si128((const __m128i *)(v16 + 40));
    v40 = _mm_loadu_si128((const __m128i *)(v16 + 56));
    v23 = sub_33EA290(
            v18,
            v21,
            0,
            v14,
            v13,
            (__int64)&v37,
            *(_OWORD *)v19,
            *(_QWORD *)(v19 + 40),
            *(_QWORD *)(v19 + 48),
            *(_OWORD *)(v19 + 80),
            *(_OWORD *)v16,
            *(_QWORD *)(v16 + 16),
            v14,
            v13,
            v20,
            v22,
            (__int64)&v39,
            0);
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v23, 1);
  }
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  return v23;
}
