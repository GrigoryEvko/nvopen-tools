// Function: sub_37FE9C0
// Address: 0x37fe9c0
//
void __fastcall sub_37FE9C0(__int64 a1, __int64 a2, __m128i *a3, __m128i *a4, __m128i a5)
{
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // rax
  unsigned __int16 *v11; // rax
  int v12; // r9d
  __int32 v13; // edx
  _QWORD *v14; // r14
  unsigned __int16 *v15; // rax
  int v16; // r9d
  __int128 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r12
  __int128 v20; // rax
  __int64 v21; // r9
  unsigned __int8 *v22; // rax
  __int64 v23; // rsi
  __int32 v24; // edx
  __int128 v25; // [rsp+0h] [rbp-B0h]
  __int128 v26; // [rsp+10h] [rbp-A0h]
  __int128 v27; // [rsp+20h] [rbp-90h]
  __int128 v28; // [rsp+30h] [rbp-80h]
  __int64 v29; // [rsp+60h] [rbp-50h] BYREF
  int v30; // [rsp+68h] [rbp-48h]
  __m128i v31; // [rsp+70h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v29 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v29, v8, 1);
  v9 = *(_DWORD *)(a2 + 72);
  v31.m128i_i32[2] = 0;
  v30 = v9;
  v10 = *(_QWORD *)(a2 + 40);
  v31.m128i_i64[0] = 0;
  sub_375E6F0(a1, *(_QWORD *)v10, *(_QWORD *)(v10 + 8), (__int64)a3, (__int64)&v31);
  v11 = (unsigned __int16 *)(*(_QWORD *)(v31.m128i_i64[0] + 48) + 16LL * v31.m128i_u32[2]);
  a4->m128i_i64[0] = (__int64)sub_33FAF80(*(_QWORD *)(a1 + 8), 245, (__int64)&v29, *v11, *((_QWORD *)v11 + 1), v12, a5);
  a4->m128i_i32[2] = v13;
  v14 = *(_QWORD **)(a1 + 8);
  v15 = (unsigned __int16 *)(*(_QWORD *)(a3->m128i_i64[0] + 48) + 16LL * a3->m128i_u32[2]);
  *(_QWORD *)&v17 = sub_33FAF80((__int64)v14, 244, (__int64)&v29, *v15, *((_QWORD *)v15 + 1), v16, a5);
  v18 = a3->m128i_i64[0];
  v26 = (__int128)_mm_loadu_si128(&v31);
  v27 = (__int128)_mm_loadu_si128(a4);
  v28 = (__int128)_mm_loadu_si128(a3);
  v19 = 16LL * a3->m128i_u32[2];
  v25 = v17;
  *(_QWORD *)&v20 = sub_33ED040(v14, 0x11u);
  v22 = sub_33FC1D0(
          v14,
          207,
          (__int64)&v29,
          *(unsigned __int16 *)(*(_QWORD *)(v18 + 48) + v19),
          *(_QWORD *)(*(_QWORD *)(v18 + 48) + v19 + 8),
          v21,
          v26,
          v27,
          v28,
          v25,
          v20);
  v23 = v29;
  a3->m128i_i64[0] = (__int64)v22;
  a3->m128i_i32[2] = v24;
  if ( v23 )
    sub_B91220((__int64)&v29, v23);
}
