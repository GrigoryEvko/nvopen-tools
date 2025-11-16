// Function: sub_2128280
// Address: 0x2128280
//
__int64 *__fastcall sub_2128280(__int64 *a1, unsigned __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  unsigned int v8; // edx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // r15
  __int64 v13; // rax
  int v14; // edx
  __int64 *v15; // r9
  const void ***v16; // rcx
  int v17; // r8d
  __int64 v18; // rsi
  const __m128i *v19; // r9
  __int64 *v20; // r14
  __m128i *v21; // rdx
  __m128i *v22; // r15
  __int128 v24; // [rsp-10h] [rbp-C0h]
  int v25; // [rsp+8h] [rbp-A8h]
  const void ***v26; // [rsp+10h] [rbp-A0h]
  __int64 v27; // [rsp+18h] [rbp-98h]
  __int64 v28; // [rsp+20h] [rbp-90h] BYREF
  int v29; // [rsp+28h] [rbp-88h]
  unsigned __int8 v30[8]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v31; // [rsp+38h] [rbp-78h]
  __int8 v32; // [rsp+40h] [rbp-70h]
  __int64 v33; // [rsp+48h] [rbp-68h]
  __m128i v34; // [rsp+50h] [rbp-60h] BYREF
  __m128i v35; // [rsp+60h] [rbp-50h]
  __int64 v36; // [rsp+70h] [rbp-40h]
  int v37; // [rsp+78h] [rbp-38h]

  sub_1F40D10(
    (__int64)&v34,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL));
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_BYTE *)v6;
  v31 = *(_QWORD *)(v6 + 8);
  v30[0] = v7;
  v8 = *(_DWORD *)(a2 + 56);
  v32 = v34.m128i_i8[8];
  v33 = v35.m128i_i64[0];
  v9 = *(_QWORD *)(a2 + 32);
  v10 = _mm_loadu_si128((const __m128i *)v9);
  v34 = v10;
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v36 = 0;
  v37 = 0;
  v35 = v11;
  if ( v8 == 3 )
  {
    v36 = *(_QWORD *)(v9 + 80);
    v37 = *(_DWORD *)(v9 + 88);
  }
  v12 = v8;
  v27 = a1[1];
  v13 = sub_1D25C30(v27, v30, 2);
  v15 = (__int64 *)v27;
  v16 = (const void ***)v13;
  v17 = v14;
  v28 = *(_QWORD *)(a2 + 72);
  if ( v28 )
  {
    v25 = v14;
    v26 = (const void ***)v13;
    sub_1623A60((__int64)&v28, v28, 2);
    v17 = v25;
    v16 = v26;
    v15 = (__int64 *)v27;
  }
  *((_QWORD *)&v24 + 1) = v12;
  *(_QWORD *)&v24 = &v34;
  v18 = *(unsigned __int16 *)(a2 + 24);
  v29 = *(_DWORD *)(a2 + 64);
  v20 = sub_1D36D80(
          v15,
          v18,
          (__int64)&v28,
          v16,
          v17,
          *(double *)v10.m128i_i64,
          *(double *)v11.m128i_i64,
          a5,
          (__int64)v15,
          v24);
  v22 = v21;
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
  sub_2013400((__int64)a1, a2, 0, (__int64)v20, v22, v19);
  return v20;
}
