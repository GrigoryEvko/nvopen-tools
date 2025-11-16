// Function: sub_21D6060
// Address: 0x21d6060
//
__int64 *__fastcall sub_21D6060(double a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // eax
  unsigned __int16 v12; // r15
  int v13; // eax
  __int128 v14; // rax
  const __m128i *v15; // rax
  __int64 v16; // rdx
  int v17; // r8d
  __int64 *v18; // r14
  __int64 v20; // [rsp+0h] [rbp-60h] BYREF
  int v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h] BYREF
  __int64 v23; // [rsp+18h] [rbp-48h]
  __m128i v24; // [rsp+20h] [rbp-40h]

  v9 = *(_QWORD *)(a5 + 72);
  v20 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v20, v9, 2);
  v10 = *(_QWORD *)(a5 + 104);
  v11 = *(_DWORD *)(a5 + 64);
  v22 = 0;
  v23 = 0;
  v24.m128i_i64[0] = 0;
  v12 = *(_WORD *)(v10 + 32);
  v21 = v11;
  v13 = sub_1E34390(v10);
  *(_QWORD *)&v14 = sub_1D2B810(
                      a7,
                      3u,
                      (__int64)&v20,
                      4u,
                      0,
                      v13,
                      *(_OWORD *)*(_QWORD *)(a5 + 32),
                      *(_QWORD *)(*(_QWORD *)(a5 + 32) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a5 + 32) + 48LL),
                      *(_OWORD *)*(_QWORD *)(a5 + 104),
                      *(_QWORD *)(*(_QWORD *)(a5 + 104) + 16LL),
                      3,
                      0,
                      v12,
                      (__int64)&v22);
  v22 = sub_1D309E0(a7, 145, (__int64)&v20, 2, 0, 0, a1, a2, *(double *)a3.m128i_i64, v14);
  v15 = *(const __m128i **)(a5 + 32);
  v23 = v16;
  v24 = _mm_loadu_si128(v15);
  v18 = sub_1D37190((__int64)a7, (__int64)&v22, 2u, (__int64)&v20, v17, *(double *)v24.m128i_i64, a2, a3);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v18;
}
