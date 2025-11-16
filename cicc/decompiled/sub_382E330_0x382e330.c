// Function: sub_382E330
// Address: 0x382e330
//
__m128i *__fastcall sub_382E330(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r10
  unsigned int v14; // r9d
  __int64 v15; // r11
  char v16; // bl
  int v17; // ebx
  unsigned __int64 v18; // rdx
  const __m128i *v19; // rcx
  __int64 *v20; // rdi
  __int64 v21; // rax
  __m128i v22; // xmm0
  unsigned __int16 v23; // r14
  __m128i v24; // xmm4
  unsigned __int64 v25; // rax
  __int32 v26; // edx
  __m128i *v27; // r14
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // [rsp+18h] [rbp-C8h]
  __int16 v32; // [rsp+24h] [rbp-BCh]
  unsigned int v33; // [rsp+28h] [rbp-B8h]
  const __m128i *v34; // [rsp+28h] [rbp-B8h]
  unsigned int v35; // [rsp+30h] [rbp-B0h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v37; // [rsp+30h] [rbp-B0h]
  __int64 v38; // [rsp+38h] [rbp-A8h]
  __int64 v39; // [rsp+40h] [rbp-A0h] BYREF
  int v40; // [rsp+48h] [rbp-98h]
  __m128i v41; // [rsp+50h] [rbp-90h] BYREF
  __int64 v42; // [rsp+60h] [rbp-80h]
  __int64 v43; // [rsp+68h] [rbp-78h]
  __m128i v44; // [rsp+70h] [rbp-70h]
  __m128i v45; // [rsp+80h] [rbp-60h]
  __m128i v46; // [rsp+90h] [rbp-50h]
  __m128i v47; // [rsp+A0h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v41, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = v41.m128i_u16[4];
    v31 = v42;
  }
  else
  {
    v29 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v31 = v30;
    v9 = v29;
  }
  v35 = v9;
  v10 = sub_37AE0F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v12 = *(_QWORD *)(a2 + 80);
  v13 = v10;
  v14 = v35;
  v15 = v11;
  v16 = *(_BYTE *)(a2 + 33) >> 2;
  v39 = v12;
  v17 = v16 & 3;
  if ( !v17 )
    LOBYTE(v17) = 1;
  if ( v12 )
  {
    v38 = v11;
    v33 = v35;
    v36 = v10;
    sub_B96E90((__int64)&v39, v12, 1);
    v14 = v33;
    v13 = v36;
    v15 = v38;
  }
  v18 = *(_QWORD *)(a2 + 104);
  v19 = *(const __m128i **)(a2 + 112);
  v20 = (__int64 *)a1[1];
  v40 = *(_DWORD *)(a2 + 72);
  v21 = *(_QWORD *)(a2 + 40);
  v37 = v18;
  v22 = _mm_loadu_si128((const __m128i *)v21);
  v42 = v13;
  v43 = v15;
  v23 = *(_WORD *)(a2 + 96);
  v41 = v22;
  v34 = v19;
  v44 = _mm_loadu_si128((const __m128i *)(v21 + 80));
  v45 = _mm_loadu_si128((const __m128i *)(v21 + 120));
  v46 = _mm_loadu_si128((const __m128i *)(v21 + 160));
  v24 = _mm_loadu_si128((const __m128i *)(v21 + 200));
  LOWORD(v21) = *(_WORD *)(a2 + 32) >> 7;
  v47 = v24;
  v32 = v21 & 7;
  v25 = sub_33E5110(v20, v14, v31, 1, 0);
  v27 = sub_33E8420(v20, v25, v26, v23, v37, (__int64)&v39, (unsigned __int64 *)&v41, 6, v34, v32, v17);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v27, 1);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v27;
}
