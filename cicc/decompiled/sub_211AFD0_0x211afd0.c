// Function: sub_211AFD0
// Address: 0x211afd0
//
__int64 __fastcall sub_211AFD0(__int64 *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  __m128i *v6; // r15
  __int64 v7; // rsi
  __int64 v8; // r9
  __m128i v9; // xmm0
  __int64 v10; // r8
  __int64 v11; // r10
  __m128i v12; // xmm1
  unsigned __int8 *v13; // rax
  unsigned int v14; // r15d
  int v15; // eax
  __m128i *v16; // r10
  __int64 v17; // r9
  int v18; // ecx
  __int64 v19; // r11
  __int64 v20; // r14
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  int v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  __m128i *v27; // [rsp+18h] [rbp-88h]
  __m128i v28; // [rsp+20h] [rbp-80h] BYREF
  __m128i v29; // [rsp+30h] [rbp-70h] BYREF
  __int64 v30; // [rsp+40h] [rbp-60h] BYREF
  int v31; // [rsp+48h] [rbp-58h]
  __m128i v32; // [rsp+50h] [rbp-50h] BYREF
  __int64 v33; // [rsp+60h] [rbp-40h]

  sub_1F40D10((__int64)&v32, *a1, *(_QWORD *)(a1[1] + 48), 9, 0);
  v6 = (__m128i *)*a1;
  v7 = *(_QWORD *)(a2 + 72);
  v8 = v33;
  v9 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v10 = v32.m128i_u8[8];
  v30 = v7;
  v28 = v9;
  if ( v7 )
  {
    v23 = v32.m128i_u8[8];
    v25 = v33;
    sub_1623A60((__int64)&v30, v7, 2);
    v10 = v23;
    v8 = v25;
  }
  v11 = a1[1];
  v31 = *(_DWORD *)(a2 + 64);
  sub_20BE530((__int64)&v32, v6, v11, 220, v10, v8, v9, a4, a5, (__int64)&v28, 1u, 0, (__int64)&v30, 0, 1);
  v12 = _mm_loadu_si128(&v32);
  v29 = v12;
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  v13 = *(unsigned __int8 **)(a2 + 40);
  if ( *v13 == 9 )
    return v29.m128i_i64[0];
  sub_1F40D10((__int64)&v32, *a1, *(_QWORD *)(a1[1] + 48), *v13, *((_QWORD *)v13 + 1));
  v14 = v32.m128i_u8[8];
  v26 = v33;
  v15 = sub_1F3FE80(9, 0, **(_BYTE **)(a2 + 40));
  v16 = (__m128i *)*a1;
  v17 = v26;
  v18 = v15;
  v30 = *(_QWORD *)(a2 + 72);
  if ( v30 )
  {
    v22 = v26;
    v27 = v16;
    v24 = v15;
    sub_1623A60((__int64)&v30, v30, 2);
    v17 = v22;
    v18 = v24;
    v16 = v27;
  }
  v19 = a1[1];
  v31 = *(_DWORD *)(a2 + 64);
  sub_20BE530((__int64)&v32, v16, v19, v18, v14, v17, v9, v12, a5, (__int64)&v29, 1u, 0, (__int64)&v30, 0, 1);
  v20 = v32.m128i_i64[0];
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  return v20;
}
