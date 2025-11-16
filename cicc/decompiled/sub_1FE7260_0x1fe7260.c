// Function: sub_1FE7260
// Address: 0x1fe7260
//
__int64 __fastcall sub_1FE7260(__int64 a1, int a2, unsigned int a3, unsigned __int8 a4, __int64 *a5, int a6)
{
  __int64 v7; // r14
  unsigned __int32 v9; // r12d
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 (__fastcall *v14)(__int64, __int64); // rax
  __int64 v16; // r10
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, __int64); // r8
  __int64 (__fastcall *v19)(__int64, unsigned __int8); // rax
  __int64 v20; // rsi
  __int32 v21; // eax
  __int64 v22; // r9
  __int64 v23; // r14
  __int64 *v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-70h]
  __int64 (__fastcall *v31)(__int64, __int64); // [rsp+0h] [rbp-70h]
  __int32 v32; // [rsp+8h] [rbp-68h]
  unsigned __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __m128i v35; // [rsp+10h] [rbp-60h] BYREF
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h]

  v7 = a4;
  v9 = a2;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v14 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 112LL);
  if ( v14 != sub_1E15B90 )
  {
    v33 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v28 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v14)(v12, v13, a3);
    if ( !v28 || v33 == v28 )
      v13 = v28;
    else
      v13 = sub_1E69410(*(__int64 **)(a1 + 8), a2, v28, 4u);
  }
  if ( !v13 )
  {
    v16 = *(_QWORD *)(a1 + 24);
    v17 = *(_QWORD *)(a1 + 32);
    v18 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 112LL);
    v19 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v17 + 288LL);
    if ( v19 == sub_1D45FB0 )
    {
      v20 = *(_QWORD *)(v17 + 8 * v7 + 120);
    }
    else
    {
      v31 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 112LL);
      v34 = *(_QWORD *)(a1 + 24);
      v29 = v19(v17, v7);
      v18 = v31;
      v16 = v34;
      v20 = v29;
    }
    if ( v18 != sub_1E15B90 )
      v20 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v18)(v16, v20, a3);
    v21 = sub_1E6B9A0(*(_QWORD *)(a1 + 8), v20, (unsigned __int8 *)byte_3F871B3, 0, (__int64)v18, a6);
    v22 = *(_QWORD *)(a1 + 40);
    v32 = v21;
    v23 = *(_QWORD *)(v22 + 56);
    v24 = *(__int64 **)(a1 + 48);
    v30 = v22;
    v25 = (__int64)sub_1E0B640(v23, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) + 960LL, a5, 0);
    sub_1DD5BA0((__int64 *)(v30 + 16), v25);
    v26 = *v24;
    v27 = *(_QWORD *)v25;
    *(_QWORD *)(v25 + 8) = v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v25 = v26 | v27 & 7;
    *(_QWORD *)(v26 + 8) = v25;
    *v24 = v25 | *v24 & 7;
    v35.m128i_i64[0] = 0x10000000;
    v36 = 0;
    v35.m128i_i32[2] = v32;
    v37 = 0;
    v38 = 0;
    sub_1E1A9C0(v25, v23, &v35);
    v35.m128i_i64[0] = 0;
    v35.m128i_i32[2] = v9;
    v9 = v32;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    sub_1E1A9C0(v25, v23, &v35);
  }
  return v9;
}
