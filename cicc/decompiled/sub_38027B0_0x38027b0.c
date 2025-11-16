// Function: sub_38027B0
// Address: 0x38027b0
//
__int64 *__fastcall sub_38027B0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int64 v7; // rax
  unsigned int v8; // esi
  _QWORD *v9; // r13
  __int128 v10; // rax
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r8
  unsigned int v15; // r15d
  __int32 v16; // edx
  __int64 v17; // [rsp+8h] [rbp-98h]
  unsigned int v18; // [rsp+2Ch] [rbp-74h] BYREF
  __m128i v19; // [rsp+30h] [rbp-70h] BYREF
  __m128i v20; // [rsp+40h] [rbp-60h] BYREF
  __int64 v21; // [rsp+50h] [rbp-50h] BYREF
  int v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h] BYREF
  int v24; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = _mm_loadu_si128((const __m128i *)v3);
  v6 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v21 = 0;
  v7 = *(_QWORD *)(v3 + 160);
  v22 = 0;
  v19 = v5;
  LODWORD(v7) = *(_DWORD *)(v7 + 96);
  v20 = v6;
  v23 = v4;
  v18 = v7;
  if ( v4 )
    sub_B96E90((__int64)&v23, v4, 1);
  v24 = *(_DWORD *)(a2 + 72);
  sub_38014E0(a1, (unsigned __int64 *)&v19, (__int64)&v20, &v18, (__int64)&v23, (__int64)&v21, 0);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  if ( v20.m128i_i64[0] )
  {
    v8 = v18;
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 80);
    v13 = a1[1];
    v14 = *(_QWORD *)(*(_QWORD *)(v19.m128i_i64[0] + 48) + 16LL * v19.m128i_u32[2] + 8);
    v15 = *(unsigned __int16 *)(*(_QWORD *)(v19.m128i_i64[0] + 48) + 16LL * v19.m128i_u32[2]);
    v23 = v12;
    if ( v12 )
    {
      v17 = v14;
      sub_B96E90((__int64)&v23, v12, 1);
      v14 = v17;
    }
    v24 = *(_DWORD *)(a2 + 72);
    v20.m128i_i64[0] = (__int64)sub_3400BD0(v13, 0, (__int64)&v23, v15, v14, 0, v5, 0);
    v20.m128i_i32[2] = v16;
    if ( v23 )
      sub_B91220((__int64)&v23, v23);
    v18 = 22;
    v8 = 22;
  }
  v9 = (_QWORD *)a1[1];
  *(_QWORD *)&v10 = sub_33ED040(v9, v8);
  return sub_33EC430(
           v9,
           (__int64 *)a2,
           v19.m128i_i64[0],
           v19.m128i_i64[1],
           v20.m128i_i64[0],
           v20.m128i_i64[1],
           *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
           *(_OWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
           v10);
}
