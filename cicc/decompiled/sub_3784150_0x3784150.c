// Function: sub_3784150
// Address: 0x3784150
//
void __fastcall sub_3784150(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v10; // rsi
  __int64 v11; // rdi
  _QWORD *v12; // rdx
  __int128 v13; // rax
  _QWORD *v14; // r15
  __m128i v15; // xmm1
  __int64 v16; // rsi
  __int64 v17; // [rsp+0h] [rbp-A0h] BYREF
  int v18; // [rsp+8h] [rbp-98h]
  __int128 v19; // [rsp+10h] [rbp-90h] BYREF
  __m128i v20; // [rsp+20h] [rbp-80h] BYREF
  __m128i v21; // [rsp+30h] [rbp-70h] BYREF
  __int64 v22[2]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v23; // [rsp+50h] [rbp-50h] BYREF
  __m128i v24; // [rsp+60h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 80);
  v17 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v17, v10, 1);
  v11 = *a1;
  v12 = (_QWORD *)a1[1];
  v18 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v13 = sub_346F960(a6, v11, a2, v12, a4, a5);
  v14 = (_QWORD *)a1[1];
  v20.m128i_i64[1] = 0;
  v19 = v13;
  v20.m128i_i16[0] = 0;
  v21.m128i_i64[1] = 0;
  *(_QWORD *)&v13 = *(_QWORD *)(v13 + 48) + 16LL * DWORD2(v13);
  v21.m128i_i16[0] = 0;
  WORD4(v13) = *(_WORD *)v13;
  *(_QWORD *)&v13 = *(_QWORD *)(v13 + 8);
  LOWORD(v22[0]) = WORD4(v13);
  v22[1] = v13;
  sub_33D0340((__int64)&v23, (__int64)v14, v22);
  v15 = _mm_loadu_si128(&v24);
  v20 = _mm_loadu_si128(&v23);
  v21 = v15;
  sub_3408290((__int64)&v23, v14, &v19, (__int64)&v17, (unsigned int *)&v20, (unsigned int *)&v21, v20);
  v16 = v17;
  *(_QWORD *)a3 = v23.m128i_i64[0];
  *(_DWORD *)(a3 + 8) = v23.m128i_i32[2];
  *(_QWORD *)a4 = v24.m128i_i64[0];
  *(_DWORD *)(a4 + 8) = v24.m128i_i32[2];
  if ( v16 )
    sub_B91220((__int64)&v17, v16);
}
