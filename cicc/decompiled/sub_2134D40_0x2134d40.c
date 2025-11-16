// Function: sub_2134D40
// Address: 0x2134d40
//
__int64 *__fastcall sub_2134D40(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // esi
  _QWORD *v16; // r13
  __int128 v17; // rax
  __int64 v19; // rsi
  __int64 v20; // r14
  const void **v21; // r8
  unsigned int v22; // r15d
  __int32 v23; // edx
  __int64 v24; // [rsp-8h] [rbp-98h]
  const void **v25; // [rsp+8h] [rbp-88h]
  unsigned int v26; // [rsp+2Ch] [rbp-64h] BYREF
  __m128i v27; // [rsp+30h] [rbp-60h] BYREF
  __m128i v28; // [rsp+40h] [rbp-50h] BYREF
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF
  int v30; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v10 = *(_QWORD *)(v6 + 160);
  v27 = v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v28 = v9;
  v29 = v7;
  v26 = v10;
  if ( v7 )
    sub_1623A60((__int64)&v29, v7, 2);
  v30 = *(_DWORD *)(a2 + 64);
  sub_2133C10(a1, (unsigned __int64 *)&v27, (__int64)&v28, &v26, (__int64)&v29, v8, v9, a5);
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  if ( v28.m128i_i64[0] )
  {
    v15 = v26;
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 72);
    v20 = *(_QWORD *)(a1 + 8);
    v21 = *(const void ***)(*(_QWORD *)(v27.m128i_i64[0] + 40) + 16LL * v27.m128i_u32[2] + 8);
    v22 = *(unsigned __int8 *)(*(_QWORD *)(v27.m128i_i64[0] + 40) + 16LL * v27.m128i_u32[2]);
    v29 = v19;
    if ( v19 )
    {
      v25 = v21;
      sub_1623A60((__int64)&v29, v19, 2);
      v21 = v25;
    }
    v30 = *(_DWORD *)(a2 + 64);
    v28.m128i_i64[0] = sub_1D38BB0(v20, 0, (__int64)&v29, v22, v21, 0, v8, *(double *)v9.m128i_i64, a5, 0);
    v28.m128i_i32[2] = v23;
    v11 = v24;
    if ( v29 )
      sub_161E7C0((__int64)&v29, v29);
    v26 = 22;
    v15 = 22;
  }
  v16 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v17 = sub_1D28D50(v16, v15, v11, v12, v13, v14);
  return sub_1D2E370(
           v16,
           (__int64 *)a2,
           v27.m128i_i64[0],
           v27.m128i_i64[1],
           v28.m128i_i64[0],
           v28.m128i_i64[1],
           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
           v17);
}
