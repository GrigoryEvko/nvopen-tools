// Function: sub_213B7D0
// Address: 0x213b7d0
//
__int64 *__fastcall sub_213B7D0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r9d
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r15
  _QWORD *v8; // rdi
  __int64 v9; // r11
  __int64 v10; // rax
  __m128i v11; // xmm0
  unsigned __int8 v12; // r10
  __m128i v13; // xmm4
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // edx
  __int64 *v17; // r14
  const __m128i *v18; // r9
  __int64 v20; // [rsp+0h] [rbp-C0h]
  unsigned __int8 v21; // [rsp+8h] [rbp-B8h]
  unsigned int v22; // [rsp+10h] [rbp-B0h]
  __int64 v23; // [rsp+10h] [rbp-B0h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+20h] [rbp-A0h] BYREF
  int v26; // [rsp+28h] [rbp-98h]
  __m128i v27; // [rsp+30h] [rbp-90h] BYREF
  __int64 v28; // [rsp+40h] [rbp-80h]
  __int64 v29; // [rsp+48h] [rbp-78h]
  __m128i v30; // [rsp+50h] [rbp-70h]
  __m128i v31; // [rsp+60h] [rbp-60h]
  __m128i v32; // [rsp+70h] [rbp-50h]
  __m128i v33; // [rsp+80h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v27,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v24 = v28;
  v22 = v27.m128i_u8[8];
  v3 = sub_2138AD0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v4 = v22;
  v5 = v3;
  v7 = v6;
  v25 = *(_QWORD *)(a2 + 72);
  if ( v25 )
  {
    sub_1623A60((__int64)&v25, v25, 2);
    v4 = v22;
  }
  v8 = (_QWORD *)a1[1];
  v9 = *(_QWORD *)(a2 + 96);
  v26 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(a2 + 32);
  v20 = v9;
  v11 = _mm_loadu_si128((const __m128i *)v10);
  v28 = v5;
  v29 = v7;
  v12 = *(_BYTE *)(a2 + 88);
  v27 = v11;
  v21 = v12;
  v30 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v31 = _mm_loadu_si128((const __m128i *)(v10 + 120));
  v32 = _mm_loadu_si128((const __m128i *)(v10 + 160));
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 200));
  v14 = *(_QWORD *)(a2 + 104);
  v33 = v13;
  v23 = v14;
  v15 = sub_1D252B0((__int64)v8, v4, v24, 1, 0);
  v17 = sub_1D24AE0(v8, v15, v16, v21, v20, (__int64)&v25, v27.m128i_i64, 6, v23);
  sub_2013400((__int64)a1, a2, 1, (__int64)v17, (__m128i *)1, v18);
  if ( v25 )
    sub_161E7C0((__int64)&v25, v25);
  return v17;
}
