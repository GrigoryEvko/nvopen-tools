// Function: sub_211BF80
// Address: 0x211bf80
//
unsigned __int64 __fastcall sub_211BF80(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        __m128i *a4,
        double a5,
        double a6,
        double a7)
{
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rax
  const void ***v13; // rax
  __int32 v14; // edx
  __int64 *v15; // r14
  const void ***v16; // rax
  __int128 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int128 v23; // rax
  __int64 v24; // r9
  __int64 *v25; // rax
  __int64 v26; // rsi
  unsigned __int32 v27; // edx
  unsigned __int64 result; // rax
  __int128 v29; // [rsp+0h] [rbp-B0h]
  __int128 v30; // [rsp+10h] [rbp-A0h]
  __int128 v31; // [rsp+20h] [rbp-90h]
  __int128 v32; // [rsp+30h] [rbp-80h]
  __int64 v33; // [rsp+60h] [rbp-50h] BYREF
  int v34; // [rsp+68h] [rbp-48h]
  __m128i v35; // [rsp+70h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v33 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v33, v10, 2);
  v11 = *(_DWORD *)(a2 + 64);
  v35.m128i_i32[2] = 0;
  v34 = v11;
  v12 = *(_QWORD *)(a2 + 32);
  v35.m128i_i64[0] = 0;
  sub_2016B80(a1, *(_QWORD *)v12, *(_QWORD *)(v12 + 8), a3, &v35);
  v13 = (const void ***)(*(_QWORD *)(v35.m128i_i64[0] + 40) + 16LL * v35.m128i_u32[2]);
  a4->m128i_i64[0] = sub_1D309E0(
                       *(__int64 **)(a1 + 8),
                       163,
                       (__int64)&v33,
                       *(unsigned __int8 *)v13,
                       v13[1],
                       0,
                       a5,
                       a6,
                       a7,
                       *(_OWORD *)&v35);
  a4->m128i_i32[2] = v14;
  v15 = *(__int64 **)(a1 + 8);
  v16 = (const void ***)(*(_QWORD *)(a3->m128i_i64[0] + 40) + 16LL * a3->m128i_u32[2]);
  *(_QWORD *)&v17 = sub_1D309E0(v15, 162, (__int64)&v33, *(unsigned __int8 *)v16, v16[1], 0, a5, a6, a7, (__int128)*a3);
  v18 = a3->m128i_i64[0];
  v30 = (__int128)_mm_loadu_si128(&v35);
  v31 = (__int128)_mm_loadu_si128(a4);
  v32 = (__int128)_mm_loadu_si128(a3);
  v19 = 16LL * a3->m128i_u32[2];
  v29 = v17;
  *(_QWORD *)&v23 = sub_1D28D50(v15, 0x11u, *((__int64 *)&v17 + 1), v20, v21, v22);
  v25 = sub_1D36A20(
          v15,
          136,
          (__int64)&v33,
          *(unsigned __int8 *)(*(_QWORD *)(v18 + 40) + v19),
          *(const void ***)(*(_QWORD *)(v18 + 40) + v19 + 8),
          v24,
          v30,
          v31,
          v32,
          v29,
          v23);
  v26 = v33;
  a3->m128i_i64[0] = (__int64)v25;
  result = v27;
  a3->m128i_i32[2] = v27;
  if ( v26 )
    return sub_161E7C0((__int64)&v33, v26);
  return result;
}
