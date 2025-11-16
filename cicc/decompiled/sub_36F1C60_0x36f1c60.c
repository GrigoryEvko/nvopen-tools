// Function: sub_36F1C60
// Address: 0x36f1c60
//
void __fastcall sub_36F1C60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rdx
  _QWORD *v5; // rax
  __int64 v6; // r15
  __int16 v7; // ax
  unsigned __int8 v8; // bl
  unsigned __int8 v9; // al
  __int64 *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // r11
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  unsigned __int8 v17; // bl
  __int64 v18; // rax
  __int64 v19; // r8
  unsigned int v20; // eax
  unsigned int v21; // ecx
  unsigned int v22; // [rsp-ECh] [rbp-ECh]
  char v23; // [rsp-ECh] [rbp-ECh]
  __int64 *v24; // [rsp-E8h] [rbp-E8h]
  _QWORD *v25; // [rsp-E8h] [rbp-E8h]
  __int64 v26; // [rsp-E8h] [rbp-E8h]
  __int64 v27; // [rsp-E0h] [rbp-E0h]
  __m128i v28; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v29; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v30; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v31; // [rsp-A8h] [rbp-A8h]
  _QWORD *v32; // [rsp-80h] [rbp-80h]
  void *v33; // [rsp-48h] [rbp-48h]

  v2 = *(_QWORD *)(a1 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  v24 = (__int64 *)sub_B2BD20(a2);
  v27 = sub_B2BEC0(a1);
  v22 = *(_DWORD *)(v27 + 4);
  v29.m128i_i64[0] = (__int64)sub_BD5D20(a2);
  v31 = 261;
  v29.m128i_i64[1] = v4;
  v5 = sub_BD2C40(80, 1u);
  v6 = (__int64)v5;
  if ( v5 )
    sub_B4CE50((__int64)v5, v24, v22, (__int64)&v29, v3, 1);
  v7 = sub_A74840((_QWORD *)(a1 + 120), *(_DWORD *)(a2 + 32));
  v8 = v7;
  v23 = HIBYTE(v7);
  v9 = sub_AE5260(v27, (__int64)v24);
  if ( !v23 )
    v8 = v9;
  *(_WORD *)(v6 + 2) = *(_WORD *)(v6 + 2) & 0xFFC0 | v8;
  sub_BD84D0(a2, v6);
  v10 = (__int64 *)sub_BD5C60(a2);
  v11 = sub_BCE3C0(v10, 101);
  v29.m128i_i64[0] = (__int64)sub_BD5D20(a2);
  v31 = 261;
  v29.m128i_i64[1] = v12;
  v13 = sub_BD2C40(72, 1u);
  v14 = v13;
  if ( v13 )
  {
    v25 = v13;
    sub_B51C90((__int64)v13, a2, v11, (__int64)&v29, v3, 1u);
    v14 = v25;
  }
  v26 = (__int64)v14;
  sub_B4CED0((__int64)&v29, v6, v27);
  if ( v3 )
    v3 -= 24;
  v28 = _mm_loadu_si128(&v29);
  sub_23D0AB0((__int64)&v29, v3, 0, 0, 0);
  v15 = sub_CA1930(&v28);
  _BitScanReverse64(&v16, 1LL << *(_WORD *)(v6 + 2));
  v17 = 63 - (v16 ^ 0x3F);
  v18 = sub_BCB2E0(v32);
  v19 = sub_ACD640(v18, v15, 0);
  v20 = 256;
  v21 = v17;
  LOBYTE(v20) = v17;
  BYTE1(v21) = 1;
  sub_B343C0((__int64)&v29, 0xEEu, v6, v21, v26, v20, v19, 0, 0, 0, 0, 0);
  nullsub_61();
  v33 = &unk_49DA100;
  nullsub_63();
  if ( (__int64 *)v29.m128i_i64[0] != &v30 )
    _libc_free(v29.m128i_u64[0]);
}
