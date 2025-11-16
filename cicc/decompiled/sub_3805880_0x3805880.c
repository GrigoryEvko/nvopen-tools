// Function: sub_3805880
// Address: 0x3805880
//
unsigned __int8 *__fastcall sub_3805880(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // r9
  unsigned __int16 v7; // bx
  __int64 v8; // r8
  __int64 v9; // r10
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // r9
  __int16 v14; // r14
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rsi
  _QWORD *v18; // rdi
  __int64 v19; // rax
  int v20; // r9d
  __int64 v21; // rsi
  unsigned __int8 *v22; // r14
  __m128i v23; // xmm1
  __m128i v24; // rax
  __int64 v25; // r9
  _QWORD *v26; // rdi
  unsigned int v27; // esi
  __int64 v29; // rdx
  __int128 v30; // [rsp-10h] [rbp-B0h]
  __int64 v31; // [rsp-10h] [rbp-B0h]
  __int128 v32; // [rsp-10h] [rbp-B0h]
  __int64 v33; // [rsp+0h] [rbp-A0h]
  unsigned __int16 v34; // [rsp+8h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-80h] BYREF
  int v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+38h] [rbp-68h]
  __int16 v39; // [rsp+40h] [rbp-60h]
  __int64 v40; // [rsp+48h] [rbp-58h]
  __m128i v41; // [rsp+50h] [rbp-50h] BYREF
  __m128i v42; // [rsp+60h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *a1;
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_QWORD *)(a1[1] + 64);
  v34 = *v5;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v10 == sub_2D56A50 )
  {
    v11 = v7;
    v12 = 0;
    sub_2FE6CC0((__int64)&v41, v6, v9, v11, v8);
    v14 = v41.m128i_i16[4];
    v15 = v42.m128i_i64[0];
  }
  else
  {
    v12 = v10(v6, v9, v34, v8);
    v15 = v29;
    v14 = v12;
  }
  v16 = *(_QWORD *)(a2 + 80);
  v35 = v16;
  if ( v16 )
  {
    v33 = v15;
    sub_B96E90((__int64)&v35, v16, 1);
    v15 = v33;
  }
  v17 = *(unsigned int *)(a2 + 24);
  v18 = (_QWORD *)a1[1];
  v36 = *(_DWORD *)(a2 + 72);
  v19 = *(_QWORD *)(a2 + 40);
  if ( (int)v17 > 239 )
  {
    if ( (unsigned int)(v17 - 242) > 1 )
    {
LABEL_8:
      LOWORD(v12) = v14;
      sub_33FAF80((__int64)v18, v17, (__int64)&v35, (unsigned int)v12, v15, v13, a3);
      if ( v14 == 11 )
      {
        v21 = 236;
LABEL_10:
        v22 = sub_33FAF80(a1[1], v21, (__int64)&v35, 6, 0, v20, a3);
        goto LABEL_15;
      }
      if ( v34 == 11 )
      {
        v21 = 237;
        goto LABEL_10;
      }
      if ( v14 == 10 )
      {
        v21 = 240;
        goto LABEL_10;
      }
      if ( v34 == 10 )
      {
        v21 = 241;
        goto LABEL_10;
      }
LABEL_21:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    }
  }
  else if ( (int)v17 <= 237 && (unsigned int)(v17 - 101) > 0x2F )
  {
    goto LABEL_8;
  }
  LOWORD(v12) = v14;
  v41 = _mm_loadu_si128((const __m128i *)v19);
  v23 = _mm_loadu_si128((const __m128i *)(v19 + 40));
  *((_QWORD *)&v30 + 1) = 2;
  *(_QWORD *)&v30 = &v41;
  v37 = v12;
  v38 = v15;
  v39 = 1;
  v40 = 0;
  v42 = v23;
  v24.m128i_i64[0] = (__int64)sub_3411BE0(v18, v17, (__int64)&v35, (unsigned __int16 *)&v37, 2, v13, v30);
  v25 = v31;
  v26 = (_QWORD *)a1[1];
  v41.m128i_i64[0] = v24.m128i_i64[0];
  v42 = v24;
  v41.m128i_i32[2] = 1;
  LOWORD(v37) = 6;
  v38 = 0;
  v39 = 1;
  v40 = 0;
  if ( v14 == 11 )
  {
    v27 = 238;
  }
  else if ( v34 == 11 )
  {
    v27 = 239;
  }
  else if ( v14 == 10 )
  {
    v27 = 242;
  }
  else
  {
    v27 = 243;
    if ( v34 != 10 )
      goto LABEL_21;
  }
  *((_QWORD *)&v32 + 1) = 2;
  *(_QWORD *)&v32 = &v41;
  v22 = sub_3411BE0(v26, v27, (__int64)&v35, (unsigned __int16 *)&v37, 2, v25, v32);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v22, 1);
LABEL_15:
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v22;
}
