// Function: sub_1F19C20
// Address: 0x1f19c20
//
unsigned __int64 __fastcall sub_1F19C20(
        _QWORD *a1,
        __int32 a2,
        __int32 a3,
        __int64 a4,
        unsigned __int64 *a5,
        unsigned int a6,
        __int64 a7,
        char a8,
        unsigned __int64 a9)
{
  unsigned __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  bool v19; // zf
  unsigned __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  unsigned __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  __int64 v30; // [rsp+38h] [rbp-68h] BYREF
  __m128i v31; // [rsp+40h] [rbp-60h] BYREF
  __int64 (__fastcall *v32)(const __m128i **, const __m128i *, int); // [rsp+50h] [rbp-50h]
  __int64 (__fastcall *v33)(__int64, __int64 *); // [rsp+58h] [rbp-48h]
  __int64 v34; // [rsp+60h] [rbp-40h]

  v11 = a9;
  v12 = *(_QWORD *)(a4 + 56);
  v13 = *(_QWORD *)(a1[6] + 8LL) + 960LL;
  v28 = a9 & 0xFFFFFFFFFFFFFFF8LL;
  v30 = 0;
  v14 = (__int64)sub_1E0B640(v12, v13, &v30, 0);
  sub_1DD5BA0((__int64 *)(a4 + 16), v14);
  v15 = *(_QWORD *)v14;
  v16 = *a5;
  *(_QWORD *)(v14 + 8) = a5;
  v16 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v14 = v16 | v15 & 7;
  *(_QWORD *)(v16 + 8) = v14;
  v17 = *a5;
  v31.m128i_i64[0] = 0x10000000;
  v31.m128i_i32[2] = a3;
  v32 = 0;
  *a5 = v14 | v17 & 7;
  v31.m128i_i16[1] = 4096;
  v33 = 0;
  v34 = 0;
  v31.m128i_i8[4] = (v28 == 0) | (-2 * (v28 == 0) + 2) & 2;
  v31.m128i_i32[0] = ((a6 & 0xFFF) << 8) | v31.m128i_i32[0] & 0xFFF000FF;
  sub_1E1A9C0(v14, v12, &v31);
  v32 = 0;
  v31.m128i_i32[2] = a2;
  v33 = 0;
  v34 = 0;
  v31.m128i_i64[0] = (unsigned __int16)(a6 & 0xFFF) << 8;
  sub_1E1A9C0(v14, v12, &v31);
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  v18 = a1[2];
  v19 = v28 == 0;
  v29 = v18 + 296;
  if ( v19 )
  {
    v20 = sub_1DC1550(*(_QWORD *)(v18 + 272), v14, a8);
    v21 = v29;
    v11 = v20 & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
  else
  {
    sub_1E163F0((__int64 *)v14);
    v21 = v29;
  }
  v22 = *(unsigned int *)(*(_QWORD *)(a1[7] + 248LL) + 4LL * a6);
  v31.m128i_i64[0] = v11;
  v33 = sub_1F13500;
  v32 = sub_1F13520;
  v31.m128i_i64[1] = v21;
  sub_1DB5D80(a7, v21, v22, (__int64)&v31);
  if ( v32 )
    v32((const __m128i **)&v31, &v31, 3);
  return v11;
}
