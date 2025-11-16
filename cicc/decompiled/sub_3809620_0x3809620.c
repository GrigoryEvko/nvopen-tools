// Function: sub_3809620
// Address: 0x3809620
//
__int64 *__fastcall sub_3809620(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  __int16 v10; // r15
  __int32 v11; // edx
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rcx
  _WORD *v15; // r10
  __int32 v16; // edx
  __int64 v17; // rsi
  unsigned int v18; // esi
  _QWORD *v19; // r13
  __int128 v20; // rax
  __int64 v22; // rsi
  __int64 v23; // r14
  __int64 v24; // r8
  unsigned int v25; // r15d
  __int32 v26; // edx
  __int64 *v27; // [rsp+0h] [rbp-B0h]
  _WORD *v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  unsigned int v30; // [rsp+4Ch] [rbp-64h] BYREF
  __m128i v31; // [rsp+50h] [rbp-60h] BYREF
  __m128i v32; // [rsp+60h] [rbp-50h] BYREF
  __int64 v33; // [rsp+70h] [rbp-40h] BYREF
  int v34; // [rsp+78h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)v3;
  v5 = _mm_loadu_si128((const __m128i *)v3);
  v6 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v7 = *(_QWORD *)(v3 + 160);
  v31 = v5;
  LODWORD(v7) = *(_DWORD *)(v7 + 96);
  v32 = v6;
  v30 = v7;
  v8 = *(_QWORD *)(v4 + 48) + 16LL * v5.m128i_u32[2];
  v9 = *(_QWORD *)(v8 + 8);
  v10 = *(_WORD *)v8;
  v31.m128i_i64[0] = sub_3805E70((__int64)a1, v4, v5.m128i_i64[1]);
  v31.m128i_i32[2] = v11;
  v12 = sub_3805E70((__int64)a1, v6.m128i_u64[0], v6.m128i_i64[1]);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(__int64 **)(a2 + 40);
  v15 = (_WORD *)*a1;
  v32.m128i_i64[0] = v12;
  v33 = v13;
  v32.m128i_i32[2] = v16;
  if ( v13 )
  {
    v27 = v14;
    v28 = v15;
    sub_B96E90((__int64)&v33, v13, 1);
    v14 = v27;
    v15 = v28;
  }
  v17 = a1[1];
  v34 = *(_DWORD *)(a2 + 72);
  sub_3495B30(v15, v17, v10, v9, &v31, &v32, &v30, (__int64)&v33, *v14, v14[1], v14[5], v14[6]);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  if ( v32.m128i_i64[0] )
  {
    v18 = v30;
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 80);
    v23 = a1[1];
    v24 = *(_QWORD *)(*(_QWORD *)(v31.m128i_i64[0] + 48) + 16LL * v31.m128i_u32[2] + 8);
    v25 = *(unsigned __int16 *)(*(_QWORD *)(v31.m128i_i64[0] + 48) + 16LL * v31.m128i_u32[2]);
    v33 = v22;
    if ( v22 )
    {
      v29 = v24;
      sub_B96E90((__int64)&v33, v22, 1);
      v24 = v29;
    }
    v34 = *(_DWORD *)(a2 + 72);
    v32.m128i_i64[0] = (__int64)sub_3400BD0(v23, 0, (__int64)&v33, v25, v24, 0, v5, 0);
    v32.m128i_i32[2] = v26;
    if ( v33 )
      sub_B91220((__int64)&v33, v33);
    v30 = 22;
    v18 = 22;
  }
  v19 = (_QWORD *)a1[1];
  *(_QWORD *)&v20 = sub_33ED040(v19, v18);
  return sub_33EC430(
           v19,
           (__int64 *)a2,
           v31.m128i_i64[0],
           v31.m128i_i64[1],
           v32.m128i_i64[0],
           v32.m128i_i64[1],
           *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
           *(_OWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
           v20);
}
