// Function: sub_15A56E0
// Address: 0x15a56e0
//
__int64 __fastcall sub_15A56E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        const __m128i *a7)
{
  __int64 v7; // rbx
  __int64 v8; // r10
  __int32 v9; // r11d
  __int8 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r14
  __m128i v16; // xmm3
  __int64 v17; // rdx
  __m128i v18; // xmm4
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __m128i v23; // xmm2
  int v24; // ebx
  int v25; // esi
  __int64 v27; // [rsp+0h] [rbp-100h]
  __int64 v28; // [rsp+8h] [rbp-F8h]
  __int32 v29; // [rsp+8h] [rbp-F8h]
  __int32 v30; // [rsp+10h] [rbp-F0h]
  __int64 v31; // [rsp+10h] [rbp-F0h]
  __int8 v32; // [rsp+1Fh] [rbp-E1h]
  char v33; // [rsp+1Fh] [rbp-E1h]
  __int64 v34; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-D8h]
  __m128i v36; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v37; // [rsp+50h] [rbp-B0h]
  __int32 v38; // [rsp+70h] [rbp-90h] BYREF
  __int64 v39; // [rsp+78h] [rbp-88h]
  char v40; // [rsp+80h] [rbp-80h]
  __m128i v41[2]; // [rsp+90h] [rbp-70h] BYREF
  __m128i v42; // [rsp+B0h] [rbp-50h]
  __int64 v43; // [rsp+C0h] [rbp-40h]

  v13 = a7[1].m128i_i8[0];
  if ( v13 )
  {
    v7 = a7->m128i_i64[1];
    v36 = _mm_loadu_si128(a7);
  }
  if ( a6[1].m128i_i8[8] )
  {
    v14 = a6[1].m128i_i64[0];
    v9 = a6->m128i_i32[0];
    v15 = *(_QWORD *)(a1 + 8);
    v41[0] = _mm_loadu_si128(a6);
    if ( v13 )
    {
      v36.m128i_i64[1] = v7;
      v16 = _mm_loadu_si128(&v36);
      v17 = v14;
      v8 = 0;
      v41[0].m128i_i32[0] = v9;
      v18 = _mm_loadu_si128(v41);
      v43 = v14;
      v42 = v18;
      v19 = v18.m128i_i64[1];
      v37 = v16;
      if ( !v14 )
        goto LABEL_7;
    }
    else
    {
      v41[0].m128i_i32[0] = v9;
      v23 = _mm_loadu_si128(v41);
      v17 = v14;
      v43 = v14;
      v42 = v23;
      v19 = v23.m128i_i64[1];
      if ( !v14 )
      {
        v8 = 0;
        goto LABEL_13;
      }
    }
    v28 = a4;
    v30 = v9;
    v32 = v13;
    v20 = sub_161FF10(v15, v19, v17);
    a4 = v28;
    v9 = v30;
    v13 = v32;
    v8 = v20;
LABEL_7:
    if ( v13 )
    {
      v37.m128i_i64[1] = v7;
      v21 = v37.m128i_i64[0];
      v22 = v7;
      if ( v7 )
      {
LABEL_9:
        v31 = a4;
        v33 = v13;
        v27 = v8;
        v29 = v9;
        LOBYTE(v35) = 1;
        v34 = sub_161FF10(v15, v21, v22);
        a4 = v31;
        v40 = v33;
        if ( !v33 )
          goto LABEL_17;
        v9 = v29;
        v8 = v27;
        goto LABEL_14;
      }
      v34 = 0;
      v35 = 1;
      v40 = 1;
LABEL_14:
      v38 = v9;
      v39 = v8;
      goto LABEL_17;
    }
LABEL_13:
    LOBYTE(v35) = 0;
    v40 = 1;
    goto LABEL_14;
  }
  v15 = *(_QWORD *)(a1 + 8);
  if ( v13 )
  {
    v36.m128i_i64[1] = v7;
    v22 = v7;
    v13 = 0;
    v37.m128i_i64[0] = _mm_loadu_si128(&v36).m128i_u64[0];
    v21 = v37.m128i_i64[0];
    v37.m128i_i64[1] = v7;
    if ( v7 )
      goto LABEL_9;
    v34 = 0;
    v35 = 1;
    v40 = 0;
  }
  else
  {
    LOBYTE(v35) = 0;
    v40 = 0;
  }
LABEL_17:
  v24 = 0;
  if ( a5 )
    v24 = sub_161FF10(v15, a4, a5);
  v25 = 0;
  if ( a3 )
    v25 = sub_161FF10(v15, a2, a3);
  return sub_15BF650(v15, v25, v24, (unsigned int)&v38, (unsigned int)&v34, 0, 1);
}
