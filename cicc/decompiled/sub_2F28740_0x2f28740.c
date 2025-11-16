// Function: sub_2F28740
// Address: 0x2f28740
//
__int64 __fastcall sub_2F28740(__int64 a1)
{
  _QWORD *v1; // r13
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 (*v4)(); // rax
  __int64 v5; // rsi
  _QWORD *v6; // r14
  __int64 *v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 (*v12)(); // rax
  __int64 v13; // rsi
  _QWORD *v14; // r14
  __int64 *v15; // r9
  _QWORD *v16; // rax
  __int64 *v17; // r9
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // [rsp+8h] [rbp-98h]
  __int64 *v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int8 *v24; // [rsp+20h] [rbp-80h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  __int64 v26; // [rsp+30h] [rbp-70h]
  __m128i v27; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]

  v1 = *(_QWORD **)(a1 + 328);
  v2 = sub_B2D620(*(_QWORD *)a1, "patchable-function-entry", 0x18u);
  if ( (_BYTE)v2 )
  {
    v3 = v2;
    v4 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 128LL);
    if ( v4 == sub_2DAC790 )
      BUG();
    v5 = *(_QWORD *)(v4() + 8);
    v27 = 0u;
    v23 = 0;
    v28 = 0;
    v6 = (_QWORD *)v1[4];
    v7 = (__int64 *)v1[7];
    v24 = 0;
    v8 = (__int64)sub_2E7B380(v6, v5 - 1440, &v24, 0);
    if ( v24 )
      sub_B91220((__int64)&v24, (__int64)v24);
    sub_2E31040(v1 + 5, v8);
    v9 = *v7;
    v10 = *(_QWORD *)v8;
    *(_QWORD *)(v8 + 8) = v7;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v8 = v9 | v10 & 7;
    *(_QWORD *)(v9 + 8) = v8;
    *v7 = v8 | *v7 & 7;
    if ( v27.m128i_i64[1] )
      sub_2E882B0(v8, (__int64)v6, v27.m128i_i64[1]);
    if ( v28 )
      sub_2E88680(v8, (__int64)v6, v28);
    if ( v27.m128i_i64[0] )
      sub_B91220((__int64)&v27, v27.m128i_i64[0]);
    if ( v23 )
      sub_B91220((__int64)&v23, v23);
  }
  else
  {
    v3 = sub_B2D620(*(_QWORD *)a1, "patchable-function", 0x12u);
    if ( (_BYTE)v3 )
    {
      v12 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 128LL);
      if ( v12 == sub_2DAC790 )
        BUG();
      v13 = *(_QWORD *)(v12() + 8);
      v24 = 0;
      v23 = 0;
      v25 = 0;
      v26 = 0;
      v14 = (_QWORD *)v1[4];
      v15 = (__int64 *)v1[7];
      v27.m128i_i64[0] = 0;
      v21 = v15;
      v16 = sub_2E7B380(v14, v13 - 1400, (unsigned __int8 **)&v27, 0);
      v17 = v21;
      v18 = (__int64)v16;
      if ( v27.m128i_i64[0] )
      {
        sub_B91220((__int64)&v27, v27.m128i_i64[0]);
        v17 = v21;
      }
      v22 = v17;
      sub_2E31040(v1 + 5, v18);
      v19 = *v22;
      v20 = *(_QWORD *)v18 & 7LL;
      *(_QWORD *)(v18 + 8) = v22;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v18 = v19 | v20;
      *(_QWORD *)(v19 + 8) = v18;
      *v22 = v18 | *v22 & 7;
      if ( v25 )
        sub_2E882B0(v18, (__int64)v14, v25);
      if ( v26 )
        sub_2E88680(v18, (__int64)v14, v26);
      v27.m128i_i64[0] = 1;
      v28 = 0;
      v29 = 2;
      sub_2E8EAD0(v18, (__int64)v14, &v27);
      if ( v24 )
        sub_B91220((__int64)&v24, (__int64)v24);
      if ( v23 )
        sub_B91220((__int64)&v23, v23);
      if ( *(_BYTE *)(a1 + 340) <= 3u )
        *(_BYTE *)(a1 + 340) = 4;
    }
  }
  return v3;
}
