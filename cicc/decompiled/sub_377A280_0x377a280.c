// Function: sub_377A280
// Address: 0x377a280
//
void __fastcall sub_377A280(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int128 v9; // xmm0
  unsigned __int16 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r11
  __int64 v13; // rsi
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  __m128i v19; // xmm3
  __int16 *v20; // rdx
  __int64 v21; // rsi
  __int16 v22; // cx
  __m128i v23; // kr00_16
  int v24; // edx
  unsigned __int8 *v25; // rax
  __int64 v26; // rsi
  int v27; // edx
  __int64 v28; // [rsp+0h] [rbp-120h]
  __int64 v29; // [rsp+8h] [rbp-118h]
  _QWORD *v30; // [rsp+8h] [rbp-118h]
  __int64 v33; // [rsp+50h] [rbp-D0h] BYREF
  int v34; // [rsp+58h] [rbp-C8h]
  __int128 v35; // [rsp+60h] [rbp-C0h] BYREF
  __int128 v36; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+90h] [rbp-90h] BYREF
  int v39; // [rsp+98h] [rbp-88h]
  __m128i v40; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v41; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v42; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+C8h] [rbp-58h]
  __m128i v44; // [rsp+D0h] [rbp-50h] BYREF
  __m128i v45; // [rsp+E0h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v33 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v33, v5, 1);
  v6 = *(_DWORD *)(a2 + 72);
  v7 = *a1;
  DWORD2(v35) = 0;
  v34 = v6;
  v8 = *(_QWORD *)(a2 + 40);
  DWORD2(v36) = 0;
  *(_QWORD *)&v35 = 0;
  v9 = (__int128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  *(_QWORD *)&v36 = 0;
  v37 = _mm_loadu_si128((const __m128i *)v8);
  v10 = (unsigned __int16 *)(*(_QWORD *)(v37.m128i_i64[0] + 48) + 16LL * v37.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v44, v7, *(_QWORD *)(a1[1] + 64), *v10, *((_QWORD *)v10 + 1));
  if ( v44.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v37.m128i_u64[0], v37.m128i_i64[1], (__int64)&v35, (__int64)&v36);
  }
  else
  {
    v11 = v37.m128i_i64[0];
    v12 = a1[1];
    v13 = *(_QWORD *)(v37.m128i_i64[0] + 80);
    v38 = v13;
    if ( v13 )
    {
      v28 = v37.m128i_i64[0];
      v29 = v12;
      sub_B96E90((__int64)&v38, v13, 1);
      v14 = v37.m128i_i64[0];
      v12 = v29;
      v11 = v28;
    }
    else
    {
      v14 = v37.m128i_i64[0];
    }
    v15 = *(_DWORD *)(v11 + 72);
    v40.m128i_i64[1] = 0;
    v41.m128i_i16[0] = 0;
    v39 = v15;
    v40.m128i_i16[0] = 0;
    v41.m128i_i64[1] = 0;
    v16 = *(_QWORD *)(v14 + 48) + 16LL * v37.m128i_u32[2];
    v17 = *(_WORD *)v16;
    v18 = *(_QWORD *)(v16 + 8);
    v30 = (_QWORD *)v12;
    LOWORD(v42) = v17;
    v43 = v18;
    sub_33D0340((__int64)&v44, v12, &v42);
    v19 = _mm_loadu_si128(&v45);
    v40 = _mm_loadu_si128(&v44);
    v41 = v19;
    sub_3408290(
      (__int64)&v44,
      v30,
      (__int128 *)v37.m128i_i8,
      (__int64)&v38,
      (unsigned int *)&v40,
      (unsigned int *)&v41,
      (__m128i)v9);
    *(_QWORD *)&v35 = v44.m128i_i64[0];
    DWORD2(v35) = v44.m128i_i32[2];
    *(_QWORD *)&v36 = v45.m128i_i64[0];
    DWORD2(v36) = v45.m128i_i32[2];
    if ( v38 )
      sub_B91220((__int64)&v38, v38);
  }
  v20 = *(__int16 **)(a2 + 48);
  v21 = a1[1];
  v22 = *v20;
  v43 = *((_QWORD *)v20 + 1);
  LOWORD(v42) = v22;
  sub_33D0340((__int64)&v44, v21, &v42);
  v23 = v45;
  *(_QWORD *)a3 = sub_3405C90(
                    (_QWORD *)a1[1],
                    0x9Bu,
                    (__int64)&v33,
                    v44.m128i_i64[0],
                    v44.m128i_i64[1],
                    *(_DWORD *)(a2 + 28),
                    (__m128i)v9,
                    v35,
                    v9);
  *(_DWORD *)(a3 + 8) = v24;
  v25 = sub_3405C90(
          (_QWORD *)a1[1],
          0x9Bu,
          (__int64)&v33,
          v23.m128i_i64[0],
          v23.m128i_i64[1],
          *(_DWORD *)(a2 + 28),
          (__m128i)v9,
          v36,
          v9);
  v26 = v33;
  *(_QWORD *)a4 = v25;
  *(_DWORD *)(a4 + 8) = v27;
  if ( v26 )
    sub_B91220((__int64)&v33, v26);
}
