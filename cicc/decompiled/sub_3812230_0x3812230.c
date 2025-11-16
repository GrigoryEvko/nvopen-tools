// Function: sub_3812230
// Address: 0x3812230
//
unsigned __int8 *__fastcall sub_3812230(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  unsigned __int16 v6; // di
  int v7; // eax
  unsigned __int16 v8; // r12
  unsigned int v9; // edx
  __int64 v10; // rax
  bool v11; // r14
  __int64 v12; // rsi
  const __m128i *v13; // roff
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int16 v17; // cx
  __int64 v18; // r8
  unsigned __int16 v19; // r13
  __int64 v20; // r11
  __int64 v21; // rdx
  __int64 (__fastcall *v22)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // r9
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  __m128i v29; // xmm2
  __m128i v30; // xmm0
  unsigned int v31; // esi
  unsigned __int8 *v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rdx
  unsigned int v35; // esi
  __m128i v36; // xmm3
  __int64 v37; // rdx
  unsigned __int64 v38; // r13
  __int64 v39; // r8
  unsigned __int8 *v40; // r13
  unsigned int v41; // edx
  __int64 v42; // rsi
  int v43; // r9d
  __int64 v45; // rdx
  __int128 v46; // [rsp-20h] [rbp-100h]
  __int128 v47; // [rsp-10h] [rbp-F0h]
  __int64 v48; // [rsp+8h] [rbp-D8h]
  __int64 v49; // [rsp+10h] [rbp-D0h]
  __int64 v50; // [rsp+18h] [rbp-C8h]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  __m128i v52; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int8 *v53; // [rsp+30h] [rbp-B0h]
  __int64 v54; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v55; // [rsp+40h] [rbp-A0h]
  __int64 v56; // [rsp+48h] [rbp-98h]
  __int64 v57; // [rsp+50h] [rbp-90h]
  __int64 v58; // [rsp+58h] [rbp-88h]
  __int64 v59; // [rsp+60h] [rbp-80h] BYREF
  int v60; // [rsp+68h] [rbp-78h]
  __int64 v61; // [rsp+70h] [rbp-70h] BYREF
  __int64 v62; // [rsp+78h] [rbp-68h]
  __int16 v63; // [rsp+80h] [rbp-60h]
  __int64 v64; // [rsp+88h] [rbp-58h]
  __m128i v65; // [rsp+90h] [rbp-50h] BYREF
  __m128i v66; // [rsp+A0h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v48 = *((_QWORD *)v5 + 1);
  v7 = *(_DWORD *)(a2 + 24);
  v8 = v6;
  if ( v7 > 239 )
  {
    v41 = v7 - 242;
    v10 = (unsigned int)(v7 - 242) < 2 ? 0x28 : 0;
    v11 = v41 < 2;
  }
  else if ( v7 > 237 )
  {
    v10 = 40;
    v11 = 1;
  }
  else
  {
    v9 = v7 - 101;
    v10 = (unsigned int)(v7 - 101) < 0x30 ? 0x28 : 0;
    v11 = v9 < 0x30;
  }
  v12 = *(_QWORD *)(a2 + 80);
  v13 = (const __m128i *)(*(_QWORD *)(a2 + 40) + v10);
  v14 = v13->m128i_i64[0];
  v15 = v13->m128i_u32[2];
  v52 = _mm_loadu_si128(v13);
  v16 = *(_QWORD *)(v14 + 48) + 16 * v15;
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v59 = v12;
  v19 = v17;
  if ( v12 )
  {
    v50 = v18;
    sub_B96E90((__int64)&v59, v12, 1);
    v18 = v50;
  }
  v20 = *a1;
  v21 = a1[1];
  v60 = *(_DWORD *)(a2 + 72);
  v22 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v20 + 592LL);
  if ( v22 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v65, v20, *(_QWORD *)(v21 + 64), v19, v18);
    v23 = v66.m128i_i64[0];
    v24 = v65.m128i_u16[4];
  }
  else
  {
    v24 = v22(v20, *(_QWORD *)(v21 + 64), v19, v18);
    v23 = v45;
  }
  v49 = v23;
  v51 = v24;
  v25 = sub_380F170((__int64)a1, v52.m128i_u64[0], v52.m128i_i64[1]);
  v27 = (_QWORD *)a1[1];
  v57 = v25;
  v52.m128i_i64[0] = v25;
  v58 = v28;
  v52.m128i_i64[1] = (unsigned int)v28 | v52.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  if ( v11 )
  {
    v29 = _mm_load_si128(&v52);
    v30 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v61 = v51;
    v62 = v49;
    v63 = 1;
    v64 = 0;
    v65 = v30;
    v66 = v29;
    if ( v19 == 11 )
    {
      v31 = 238;
LABEL_11:
      *((_QWORD *)&v47 + 1) = 2;
      *(_QWORD *)&v47 = &v65;
      v32 = sub_3411BE0(v27, v31, (__int64)&v59, (unsigned __int16 *)&v61, 2, v26, v47);
      v33 = (_QWORD *)a1[1];
      v56 = v34;
      v65.m128i_i64[0] = (__int64)v32;
      v62 = v48;
      *((_QWORD *)&v46 + 1) = 2;
      *(_QWORD *)&v46 = &v65;
      v52.m128i_i64[1] = (unsigned int)v34 | v52.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v35 = *(_DWORD *)(a2 + 24);
      v52.m128i_i64[0] = (__int64)v32;
      v36 = _mm_load_si128(&v52);
      v63 = 1;
      v66 = v36;
      LOWORD(v61) = v8;
      v55 = v32;
      v65.m128i_i32[2] = 1;
      v64 = 0;
      v53 = sub_3411BE0(v33, v35, (__int64)&v59, (unsigned __int16 *)&v61, 2, 0xFFFFFFFF00000000LL, v46);
      v52.m128i_i64[0] = (__int64)v53;
      v54 = v37;
      v38 = (unsigned int)v37 | v52.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v53, 1);
      v39 = v38;
      v40 = 0;
      sub_3760E70((__int64)a1, a2, 0, v52.m128i_u64[0], v39);
      goto LABEL_16;
    }
    if ( v8 == 11 )
    {
      v31 = 239;
      goto LABEL_11;
    }
    if ( v19 == 10 )
    {
      v31 = 242;
      goto LABEL_11;
    }
    if ( v8 == 10 )
    {
      v31 = 243;
      goto LABEL_11;
    }
LABEL_22:
    sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
  if ( v19 == 11 )
  {
    v42 = 236;
  }
  else if ( v8 == 11 )
  {
    v42 = 237;
  }
  else if ( v19 == 10 )
  {
    v42 = 240;
  }
  else
  {
    if ( v8 != 10 )
      goto LABEL_22;
    v42 = 241;
  }
  sub_33FAF80((__int64)v27, v42, (__int64)&v59, v51, v49, v26, a3);
  v40 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v59, v8, v48, v43, a3);
LABEL_16:
  if ( v59 )
    sub_B91220((__int64)&v59, v59);
  return v40;
}
