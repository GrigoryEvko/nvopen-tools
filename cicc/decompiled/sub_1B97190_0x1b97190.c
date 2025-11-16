// Function: sub_1B97190
// Address: 0x1b97190
//
__int64 __fastcall sub_1B97190(__int64 a1, __int64 a2, __m128i a3, __m128i a4, double a5)
{
  __int64 v5; // r12
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // [rsp+8h] [rbp-C8h]
  __int64 v24[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v25; // [rsp+20h] [rbp-B0h]
  char v26; // [rsp+21h] [rbp-AFh]
  _BYTE v27[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v28; // [rsp+40h] [rbp-90h]
  __int64 v29; // [rsp+50h] [rbp-80h] BYREF
  __int64 v30; // [rsp+58h] [rbp-78h]
  __int64 *v31; // [rsp+60h] [rbp-70h]
  __int64 v32; // [rsp+68h] [rbp-68h]
  __int64 v33; // [rsp+70h] [rbp-60h]
  int v34; // [rsp+78h] [rbp-58h]
  __int64 v35; // [rsp+80h] [rbp-50h]
  __int64 v36; // [rsp+88h] [rbp-48h]

  v5 = *(_QWORD *)(a1 + 440);
  if ( !v5 )
  {
    v7 = sub_1B91F20((_QWORD *)a1, a2, a3, a4);
    v8 = sub_13FC520(a2);
    v9 = sub_157EBA0(v8);
    v29 = 0;
    v32 = sub_16498A0(v9);
    v31 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v30 = 0;
    sub_17050D0(&v29, v9);
    v10 = sub_15A0680(*(_QWORD *)v7, (unsigned int)(*(_DWORD *)(a1 + 92) * *(_DWORD *)(a1 + 88)), 0);
    v26 = 1;
    v11 = v10;
    v25 = 3;
    v24[0] = (__int64)"n.mod.vf";
    if ( *(_BYTE *)(v7 + 16) > 0x10u
      || *(_BYTE *)(v10 + 16) > 0x10u
      || (v12 = sub_15A2A30(
                  (__int64 *)0x14,
                  (__int64 *)v7,
                  v10,
                  0,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  a5)) == 0 )
    {
      v28 = 257;
      v14 = sub_15FB440(20, (__int64 *)v7, v11, (__int64)v27, 0);
      v12 = v14;
      if ( v30 )
      {
        v23 = v31;
        sub_157E9D0(v30 + 40, v14);
        v15 = *v23;
        v16 = *(_QWORD *)(v12 + 24) & 7LL;
        *(_QWORD *)(v12 + 32) = v23;
        v15 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v15 | v16;
        *(_QWORD *)(v15 + 8) = v12 + 24;
        *v23 = *v23 & 7 | (v12 + 24);
      }
      sub_164B780(v12, v24);
      sub_12A86E0(&v29, v12);
    }
    if ( *(_DWORD *)(a1 + 88) > 1u && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 456) + 384LL) + 40LL) )
    {
      v28 = 257;
      v21 = sub_15A0680(*(_QWORD *)v12, 0, 0);
      v22 = sub_12AA0C0(&v29, 0x20u, (_BYTE *)v12, v21, (__int64)v27);
      v28 = 257;
      v12 = sub_156B790(&v29, v22, v11, v12, (__int64)v27, 0);
    }
    v26 = 1;
    v24[0] = (__int64)"n.vec";
    v25 = 3;
    if ( *(_BYTE *)(v7 + 16) > 0x10u || *(_BYTE *)(v12 + 16) > 0x10u )
    {
      v28 = 257;
      v17 = sub_15FB440(13, (__int64 *)v7, v12, (__int64)v27, 0);
      v5 = v17;
      if ( v30 )
      {
        v18 = v31;
        sub_157E9D0(v30 + 40, v17);
        v19 = *(_QWORD *)(v5 + 24);
        v20 = *v18;
        *(_QWORD *)(v5 + 32) = v18;
        v20 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v5 + 24) = v20 | v19 & 7;
        *(_QWORD *)(v20 + 8) = v5 + 24;
        *v18 = *v18 & 7 | (v5 + 24);
      }
      sub_164B780(v5, v24);
      sub_12A86E0(&v29, v5);
    }
    else
    {
      v5 = sub_15A2B60((__int64 *)v7, v12, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
    }
    v13 = v29;
    *(_QWORD *)(a1 + 440) = v5;
    if ( v13 )
      sub_161E7C0((__int64)&v29, v13);
  }
  return v5;
}
