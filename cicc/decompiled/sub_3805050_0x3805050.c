// Function: sub_3805050
// Address: 0x3805050
//
unsigned __int8 *__fastcall sub_3805050(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // rsi
  unsigned __int16 v7; // bx
  __int64 v8; // rax
  __int64 v9; // rdx
  bool v10; // zf
  __int64 *v11; // rax
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rdi
  const __m128i *v15; // r9
  unsigned __int8 *v16; // rcx
  unsigned __int16 *v17; // rax
  __int64 v18; // r11
  unsigned int v19; // r10d
  __int64 v20; // rax
  __m128i v21; // xmm0
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  __int64 v26; // rdx
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int, __int64); // rax
  int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // [rsp+8h] [rbp-C8h]
  const __m128i *v35; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v36; // [rsp+18h] [rbp-B8h]
  __int64 v37; // [rsp+20h] [rbp-B0h]
  __int64 *v38; // [rsp+30h] [rbp-A0h]
  __int64 v39; // [rsp+40h] [rbp-90h]
  __int64 v40; // [rsp+60h] [rbp-70h] BYREF
  int v41; // [rsp+68h] [rbp-68h]
  __m128i v42; // [rsp+70h] [rbp-60h] BYREF
  __m128i v43; // [rsp+80h] [rbp-50h]
  unsigned __int8 *v44; // [rsp+90h] [rbp-40h]
  __int64 v45; // [rsp+98h] [rbp-38h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v40 = v6;
  v39 = v8;
  if ( v6 )
    sub_B96E90((__int64)&v40, v6, 1);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_DWORD *)(a2 + 24) == 339;
  v41 = *(_DWORD *)(a2 + 72);
  v11 = (__int64 *)(v9 + 80);
  if ( v10 )
    v11 = (__int64 *)(v9 + 40);
  v12 = sub_375A6A0((__int64)a1, *v11, v11[1], a3);
  v14 = (__int64 *)a1[1];
  v15 = *(const __m128i **)(a2 + 112);
  v16 = v12;
  v17 = (unsigned __int16 *)(*((_QWORD *)v12 + 6) + 16LL * (unsigned int)v13);
  v10 = *(_DWORD *)(a2 + 24) == 339;
  v18 = *((_QWORD *)v17 + 1);
  v19 = *v17;
  v20 = *(_QWORD *)(a2 + 40);
  v21 = _mm_loadu_si128((const __m128i *)v20);
  v42 = v21;
  if ( v10 )
    v43 = _mm_loadu_si128((const __m128i *)(v20 + 80));
  else
    v43 = _mm_loadu_si128((const __m128i *)(v20 + 40));
  v44 = v16;
  v45 = v13;
  v35 = v15;
  v36 = v19;
  v37 = v18;
  v22 = sub_33E5110(v14, v19, v18, 1, 0);
  v38 = sub_33E6BC0(v14, 342, (__int64)&v40, v36, v37, v35, v22, v23, (unsigned __int64 *)&v42, 3);
  v24 = (unsigned __int8 *)v38;
  sub_2FE6CC0((__int64)&v42, *a1, *(_QWORD *)(a1[1] + 64), v7, v39);
  if ( v42.m128i_i8[0] == 8 )
  {
    v26 = a1[1];
    v27 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v27 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v42, *a1, *(_QWORD *)(v26 + 64), v7, v39);
      LOWORD(v29) = v42.m128i_i16[4];
      v30 = v43.m128i_i64[0];
    }
    else
    {
      v29 = v27(*a1, *(_QWORD *)(v26 + 64), v7, v39);
      v34 = v29;
      v30 = v33;
    }
    if ( v7 == 11 )
    {
      v31 = 236;
    }
    else if ( (_WORD)v29 == 11 )
    {
      v31 = 237;
    }
    else if ( v7 == 10 )
    {
      v31 = 240;
    }
    else
    {
      if ( (_WORD)v29 != 10 )
        sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
      v31 = 241;
    }
    v32 = v34;
    LOWORD(v32) = v29;
    v24 = sub_33FAF80(a1[1], v31, (__int64)&v40, v32, v30, v28, v21);
  }
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v38, 1);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v24;
}
