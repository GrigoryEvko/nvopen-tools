// Function: sub_380A560
// Address: 0x380a560
//
unsigned __int8 *__fastcall sub_380A560(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int16 v4; // r14
  int v5; // eax
  unsigned int v6; // eax
  __int64 v7; // rdx
  bool v8; // bl
  __int64 v9; // rax
  __int64 v10; // rsi
  __m128i v11; // xmm0
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r12
  _QWORD *v17; // r9
  __int64 v18; // rsi
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  unsigned int v21; // esi
  unsigned __int8 *v22; // r12
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdi
  int v27; // eax
  unsigned __int64 v28; // rax
  __int64 v29; // rsi
  int v30; // ecx
  _WORD *v31; // r12
  unsigned int v32; // r9d
  __int32 v33; // edx
  __int64 v34; // r10
  __int64 v35; // rsi
  __int64 v36; // r12
  __int128 v37; // [rsp-10h] [rbp-110h]
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-F8h]
  int v39; // [rsp+10h] [rbp-F0h]
  _QWORD *v40; // [rsp+20h] [rbp-E0h]
  _QWORD *v41; // [rsp+20h] [rbp-E0h]
  __m128i v42; // [rsp+20h] [rbp-E0h]
  __m128i v43; // [rsp+40h] [rbp-C0h] BYREF
  int v44; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+58h] [rbp-A8h]
  __int64 v46; // [rsp+60h] [rbp-A0h] BYREF
  int v47; // [rsp+68h] [rbp-98h]
  __int64 v48; // [rsp+70h] [rbp-90h] BYREF
  int v49; // [rsp+78h] [rbp-88h]
  _QWORD v50[2]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v51; // [rsp+90h] [rbp-70h]
  __int64 v52; // [rsp+98h] [rbp-68h]
  __m128i v53; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v54; // [rsp+B0h] [rbp-50h]
  __int64 v55; // [rsp+C0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 48);
  v4 = *(_WORD *)v3;
  v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v3 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 > 239 )
  {
    v24 = v5 - 242;
    v7 = v24 < 2 ? 0x28 : 0;
    v8 = v24 < 2;
  }
  else if ( v5 > 237 )
  {
    v7 = 40;
    v8 = 1;
  }
  else
  {
    v6 = v5 - 101;
    v7 = v6 < 0x30 ? 0x28 : 0;
    v8 = v6 < 0x30;
  }
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *a1;
  v11 = _mm_loadu_si128((const __m128i *)(v9 + v7));
  v43 = v11;
  v12 = *(_QWORD *)(v11.m128i_i64[0] + 48) + 16LL * v11.m128i_u32[2];
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v15 = a1[1];
  LOWORD(v44) = v13;
  v16 = *(_QWORD *)(v15 + 64);
  v45 = v14;
  sub_2FE6CC0((__int64)&v53, v10, v16, v13, v14);
  if ( v53.m128i_i8[0] == 3 )
  {
    v27 = sub_2FE5850(v44, v45, v4);
    if ( v8 )
      v42 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    else
      v42 = 0u;
    v39 = v27;
    v28 = sub_3805E70((__int64)a1, v43.m128i_u64[0], v43.m128i_i64[1]);
    v29 = *(_QWORD *)(a2 + 80);
    v54.m128i_i16[0] = v4;
    v30 = v39;
    v31 = (_WORD *)*a1;
    v43.m128i_i64[0] = v28;
    v53.m128i_i64[1] = 1;
    v32 = v4;
    v43.m128i_i32[2] = v33;
    v53.m128i_i64[0] = (__int64)&v44;
    LOBYTE(v55) = 20;
    v54.m128i_i64[1] = (__int64)v38;
    v48 = v29;
    if ( v29 )
    {
      sub_B96E90((__int64)&v48, v29, 1);
      v32 = v4;
      v30 = v39;
    }
    v34 = a1[1];
    v49 = *(_DWORD *)(a2 + 72);
    sub_3494590(
      (__int64)v50,
      v31,
      v34,
      v30,
      v32,
      v38,
      (__int64)&v43,
      1u,
      v53.m128i_i64[0],
      v53.m128i_i32[2],
      v54.m128i_u32[0],
      v54.m128i_i64[1],
      v55,
      (__int64)&v48,
      v42.m128i_i64[0],
      v42.m128i_i64[1]);
    if ( v48 )
      sub_B91220((__int64)&v48, v48);
    if ( v8 )
      sub_3760E70((__int64)a1, a2, 1, v51, v52);
    v35 = *(_QWORD *)(a2 + 80);
    v36 = a1[1];
    v48 = v35;
    if ( v35 )
      sub_B96E90((__int64)&v48, v35, 1);
    v25 = 234;
    v26 = v36;
    v49 = *(_DWORD *)(a2 + 72);
  }
  else
  {
    v17 = (_QWORD *)a1[1];
    v18 = *(_QWORD *)(a2 + 80);
    if ( v8 )
    {
      v19 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      v20 = _mm_loadu_si128(&v43);
      LOWORD(v50[0]) = 6;
      v50[1] = 0;
      LOWORD(v51) = 1;
      v52 = 0;
      v46 = v18;
      v53 = v19;
      v54 = v20;
      if ( v18 )
      {
        v40 = v17;
        sub_B96E90((__int64)&v46, v18, 1);
        v17 = v40;
      }
      v47 = *(_DWORD *)(a2 + 72);
      if ( (_WORD)v44 == 11 )
      {
        v21 = 238;
LABEL_10:
        *((_QWORD *)&v37 + 1) = 2;
        *(_QWORD *)&v37 = &v53;
        v22 = sub_3411BE0(v17, v21, (__int64)&v46, (unsigned __int16 *)v50, 2, (__int64)v17, v37);
        if ( v46 )
          sub_B91220((__int64)&v46, v46);
        sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v22, 1);
        return v22;
      }
      if ( v4 == 11 )
      {
        v21 = 239;
        goto LABEL_10;
      }
      if ( (_WORD)v44 == 10 )
      {
        v21 = 242;
        goto LABEL_10;
      }
      if ( v4 == 10 )
      {
        v21 = 243;
        goto LABEL_10;
      }
LABEL_41:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    }
    v48 = *(_QWORD *)(a2 + 80);
    if ( v18 )
    {
      v41 = v17;
      sub_B96E90((__int64)&v48, v18, 1);
      v17 = v41;
    }
    v49 = *(_DWORD *)(a2 + 72);
    if ( (_WORD)v44 == 11 )
    {
      v25 = 236;
    }
    else if ( v4 == 11 )
    {
      v25 = 237;
    }
    else if ( (_WORD)v44 == 10 )
    {
      v25 = 240;
    }
    else
    {
      v25 = 241;
      if ( v4 != 10 )
        goto LABEL_41;
    }
    v26 = (__int64)v17;
  }
  v22 = sub_33FAF80(v26, v25, (__int64)&v48, 6, 0, (_DWORD)v17, v11);
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v22;
}
