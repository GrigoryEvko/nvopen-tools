// Function: sub_3809850
// Address: 0x3809850
//
__int64 *__fastcall sub_3809850(_QWORD *a1, unsigned __int64 a2)
{
  int v3; // eax
  __int64 *v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int16 v10; // bx
  __m128i v11; // rax
  __m128i v12; // rax
  __int64 v13; // rsi
  _WORD *v14; // r10
  __int64 v15; // rsi
  _QWORD *v16; // rbx
  __int128 v17; // rax
  __int128 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rsi
  __int128 v22; // kr00_16
  __int64 v23; // r8
  unsigned int v24; // r15d
  __int32 v25; // edx
  _WORD *v26; // [rsp+0h] [rbp-C0h]
  char v27; // [rsp+Fh] [rbp-B1h]
  __m128i v28; // [rsp+10h] [rbp-B0h]
  __int128 v29; // [rsp+10h] [rbp-B0h]
  __m128i v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  unsigned int v32; // [rsp+4Ch] [rbp-74h] BYREF
  __m128i v33; // [rsp+50h] [rbp-70h] BYREF
  __m128i v34; // [rsp+60h] [rbp-60h] BYREF
  __m128i v35; // [rsp+70h] [rbp-50h] BYREF
  __int64 v36; // [rsp+80h] [rbp-40h] BYREF
  int v37; // [rsp+88h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(__int64 **)(a2 + 40);
  if ( v3 > 239 )
  {
    if ( (unsigned int)(v3 - 242) > 1 )
    {
LABEL_4:
      v5 = *v4;
      v27 = 0;
      v6 = 10;
      v7 = *((unsigned int *)v4 + 2);
      v33.m128i_i64[0] = 0;
      v33.m128i_i32[2] = 0;
      v30 = _mm_loadu_si128((const __m128i *)v4);
      v28 = _mm_loadu_si128((const __m128i *)(v4 + 5));
      goto LABEL_5;
    }
  }
  else if ( v3 <= 237 && (unsigned int)(v3 - 101) > 0x2F )
  {
    goto LABEL_4;
  }
  v6 = 15;
  v27 = 1;
  v5 = v4[5];
  v7 = *((unsigned int *)v4 + 12);
  v30 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v28 = _mm_loadu_si128((const __m128i *)v4 + 5);
  v33 = _mm_loadu_si128((const __m128i *)v4);
LABEL_5:
  v8 = *(_QWORD *)(v5 + 48) + 16 * v7;
  v9 = *(_QWORD *)(v8 + 8);
  v32 = *(_DWORD *)(v4[v6] + 96);
  v10 = *(_WORD *)v8;
  v11.m128i_i64[0] = sub_3805E70((__int64)a1, v30.m128i_u64[0], v30.m128i_i64[1]);
  v34 = v11;
  v12.m128i_i64[0] = sub_3805E70((__int64)a1, v28.m128i_u64[0], v28.m128i_i64[1]);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = (_WORD *)*a1;
  v35 = v12;
  v36 = v13;
  if ( v13 )
  {
    v26 = v14;
    sub_B96E90((__int64)&v36, v13, 1);
    v14 = v26;
  }
  v15 = a1[1];
  v37 = *(_DWORD *)(a2 + 72);
  sub_3494EA0(
    v14,
    v15,
    v10,
    v9,
    &v34,
    &v35,
    &v32,
    (__int64)&v36,
    v30.m128i_i64[0],
    v30.m128i_i64[1],
    v28.m128i_i64[0],
    v28.m128i_i64[1],
    v33.m128i_i64);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  if ( v35.m128i_i64[0] )
  {
    v16 = (_QWORD *)a1[1];
    if ( !v27 )
    {
      *(_QWORD *)&v17 = sub_33ED040(v16, v32);
      return sub_33EC3B0(
               v16,
               (__int64 *)a2,
               v34.m128i_i64[0],
               v34.m128i_i64[1],
               v35.m128i_i64[0],
               v35.m128i_i64[1],
               v17);
    }
    *(_QWORD *)&v19 = sub_33ED040(v16, v32);
    v21 = *(_QWORD *)(a2 + 80);
    v22 = v19;
    v23 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v24 = **(unsigned __int16 **)(a2 + 48);
    v36 = v21;
    if ( v21 )
    {
      v29 = v19;
      v31 = v23;
      sub_B96E90((__int64)&v36, v21, 1);
      v23 = v31;
      v22 = v29;
    }
    v37 = *(_DWORD *)(a2 + 72);
    v34.m128i_i64[0] = sub_340F900(v16, 0xD0u, (__int64)&v36, v24, v23, v20, *(_OWORD *)&v34, *(_OWORD *)&v35, v22);
    v34.m128i_i32[2] = v25;
    if ( v36 )
      sub_B91220((__int64)&v36, v36);
    goto LABEL_18;
  }
  if ( v27 )
  {
LABEL_18:
    sub_3760E70((__int64)a1, a2, 0, v34.m128i_u64[0], v34.m128i_i64[1]);
    sub_3760E70((__int64)a1, a2, 1, v33.m128i_u64[0], v33.m128i_i64[1]);
    return 0;
  }
  return (__int64 *)v34.m128i_i64[0];
}
