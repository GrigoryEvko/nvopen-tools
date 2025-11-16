// Function: sub_38090D0
// Address: 0x38090d0
//
__int64 *__fastcall sub_38090D0(_QWORD *a1, __int64 a2)
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
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v24; // rsi
  __int64 v25; // r14
  __int64 v26; // r8
  unsigned int v27; // r15d
  __int32 v28; // edx
  __int64 *v29; // [rsp+0h] [rbp-B0h]
  _WORD *v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+8h] [rbp-A8h]
  unsigned int v32; // [rsp+4Ch] [rbp-64h] BYREF
  __m128i v33; // [rsp+50h] [rbp-60h] BYREF
  __m128i v34; // [rsp+60h] [rbp-50h] BYREF
  __int64 v35; // [rsp+70h] [rbp-40h] BYREF
  int v36; // [rsp+78h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 80);
  v5 = _mm_loadu_si128((const __m128i *)(v3 + 80));
  v6 = _mm_loadu_si128((const __m128i *)(v3 + 120));
  v7 = *(_QWORD *)(v3 + 40);
  v33 = v5;
  LODWORD(v7) = *(_DWORD *)(v7 + 96);
  v34 = v6;
  v32 = v7;
  v8 = *(_QWORD *)(v4 + 48) + 16LL * v5.m128i_u32[2];
  v9 = *(_QWORD *)(v8 + 8);
  v10 = *(_WORD *)v8;
  v33.m128i_i64[0] = sub_3805E70((__int64)a1, v4, v5.m128i_i64[1]);
  v33.m128i_i32[2] = v11;
  v12 = sub_3805E70((__int64)a1, v6.m128i_u64[0], v6.m128i_i64[1]);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(__int64 **)(a2 + 40);
  v15 = (_WORD *)*a1;
  v34.m128i_i64[0] = v12;
  v35 = v13;
  v34.m128i_i32[2] = v16;
  if ( v13 )
  {
    v29 = v14;
    v30 = v15;
    sub_B96E90((__int64)&v35, v13, 1);
    v14 = v29;
    v15 = v30;
  }
  v17 = a1[1];
  v36 = *(_DWORD *)(a2 + 72);
  sub_3495B30(v15, v17, v10, v9, &v33, &v34, &v32, (__int64)&v35, v14[10], v14[11], v14[15], v14[16]);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  if ( v34.m128i_i64[0] )
  {
    v18 = v32;
  }
  else
  {
    v24 = *(_QWORD *)(a2 + 80);
    v25 = a1[1];
    v26 = *(_QWORD *)(*(_QWORD *)(v33.m128i_i64[0] + 48) + 16LL * v33.m128i_u32[2] + 8);
    v27 = *(unsigned __int16 *)(*(_QWORD *)(v33.m128i_i64[0] + 48) + 16LL * v33.m128i_u32[2]);
    v35 = v24;
    if ( v24 )
    {
      v31 = v26;
      sub_B96E90((__int64)&v35, v24, 1);
      v26 = v31;
    }
    v36 = *(_DWORD *)(a2 + 72);
    v34.m128i_i64[0] = (__int64)sub_3400BD0(v25, 0, (__int64)&v35, v27, v26, 0, v5, 0);
    v34.m128i_i32[2] = v28;
    if ( v35 )
      sub_B91220((__int64)&v35, v35);
    v32 = 22;
    v18 = 22;
  }
  v19 = (_QWORD *)a1[1];
  v20 = *(_QWORD *)(a2 + 40);
  v21 = sub_33ED040(v19, v18);
  return sub_33EC430(
           v19,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           v21,
           v22,
           *(_OWORD *)&v33,
           *(_OWORD *)&v34,
           *(_OWORD *)(v20 + 160));
}
