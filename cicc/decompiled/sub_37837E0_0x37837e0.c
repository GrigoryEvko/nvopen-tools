// Function: sub_37837E0
// Address: 0x37837e0
//
void __fastcall sub_37837E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 *v10; // rdx
  __int128 v11; // rax
  _QWORD *v12; // r15
  __m128i v13; // xmm1
  __int64 v14; // rsi
  __int64 v15; // [rsp+0h] [rbp-A0h] BYREF
  int v16; // [rsp+8h] [rbp-98h]
  __int128 v17; // [rsp+10h] [rbp-90h] BYREF
  __m128i v18; // [rsp+20h] [rbp-80h] BYREF
  __m128i v19; // [rsp+30h] [rbp-70h] BYREF
  __int64 v20[2]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v21; // [rsp+50h] [rbp-50h] BYREF
  __m128i v22; // [rsp+60h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v15 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v15, v8, 1);
  v9 = *a1;
  v10 = (__int64 *)a1[1];
  v16 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v11 = sub_346DAE0(v9, a2, v10);
  v12 = (_QWORD *)a1[1];
  v18.m128i_i64[1] = 0;
  v17 = v11;
  v18.m128i_i16[0] = 0;
  v19.m128i_i64[1] = 0;
  *(_QWORD *)&v11 = *(_QWORD *)(v11 + 48) + 16LL * DWORD2(v11);
  v19.m128i_i16[0] = 0;
  WORD4(v11) = *(_WORD *)v11;
  *(_QWORD *)&v11 = *(_QWORD *)(v11 + 8);
  LOWORD(v20[0]) = WORD4(v11);
  v20[1] = v11;
  sub_33D0340((__int64)&v21, (__int64)v12, v20);
  v13 = _mm_loadu_si128(&v22);
  v18 = _mm_loadu_si128(&v21);
  v19 = v13;
  sub_3408290((__int64)&v21, v12, &v17, (__int64)&v15, (unsigned int *)&v18, (unsigned int *)&v19, v18);
  v14 = v15;
  *(_QWORD *)a3 = v21.m128i_i64[0];
  *(_DWORD *)(a3 + 8) = v21.m128i_i32[2];
  *(_QWORD *)a4 = v22.m128i_i64[0];
  *(_DWORD *)(a4 + 8) = v22.m128i_i32[2];
  if ( v14 )
    sub_B91220((__int64)&v15, v14);
}
