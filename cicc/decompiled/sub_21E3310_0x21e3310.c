// Function: sub_21E3310
// Address: 0x21e3310
//
__int64 __fastcall sub_21E3310(__int64 a1, __int64 a2, int a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // rax
  __int16 v11; // r12
  int v12; // ebx
  __int64 v13; // rax
  bool v14; // cc
  unsigned __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // edx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 *v21; // rax
  __int32 v22; // r11d
  __int64 v23; // r11
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r8
  const __m128i *v27; // r9
  __m128i v28; // xmm0
  __int64 v29; // r8
  __int64 v30; // r12
  __int64 v32; // [rsp+8h] [rbp-478h]
  __int64 v33; // [rsp+8h] [rbp-478h]
  __m128i v34; // [rsp+10h] [rbp-470h] BYREF
  const __m128i *v35; // [rsp+20h] [rbp-460h]
  _QWORD *v36; // [rsp+28h] [rbp-458h]
  __int64 v37; // [rsp+30h] [rbp-450h] BYREF
  int v38; // [rsp+38h] [rbp-448h]
  __int64 *v39; // [rsp+40h] [rbp-440h] BYREF
  __int64 v40; // [rsp+48h] [rbp-438h]
  _BYTE v41[1072]; // [rsp+50h] [rbp-430h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v36 = *(_QWORD **)(a1 - 176);
  v37 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v37, v8, 2);
  v9 = *(_DWORD *)(a2 + 64);
  v39 = (__int64 *)v41;
  v38 = v9;
  v40 = 0x4000000000LL;
  v10 = (unsigned int)(a3 - 5304);
  v11 = a3 - 833;
  v12 = byte_435EDA0[v10];
  v34.m128i_i32[0] = byte_435ED00[v10];
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 88LL);
  v14 = *(_DWORD *)(v13 + 32) <= 0x40u;
  v15 = *(_QWORD *)(v13 + 24);
  if ( !v14 )
    v15 = *(_QWORD *)v15;
  v16 = (v15 >> 3) & 1;
  if ( v12 != 1 )
    LOBYTE(v16) = 0;
  v18 = sub_1D38BB0((__int64)v36, v15 & 0xFFFFFFFFFFFFFFF7LL | (8 * (v16 & 1)), (__int64)&v37, 6, 0, 1, a4, a5, a6, 0);
  v19 = v17;
  v20 = (unsigned int)v40;
  if ( (unsigned int)v40 >= HIDWORD(v40) )
  {
    v33 = v18;
    v35 = (const __m128i *)v17;
    sub_16CD150((__int64)&v39, v41, 0, 16, v18, v17);
    v20 = (unsigned int)v40;
    v18 = v33;
    v19 = (__int64)v35;
  }
  v21 = &v39[2 * v20];
  v22 = v34.m128i_i32[0];
  *v21 = v18;
  v21[1] = v19;
  v23 = (unsigned int)(v12 + v22);
  v24 = 120;
  v25 = (unsigned int)(v40 + 1);
  v26 = 40 * v23 + 160;
  LODWORD(v40) = v40 + 1;
  do
  {
    v27 = (const __m128i *)(v24 + *(_QWORD *)(a2 + 32));
    if ( HIDWORD(v40) <= (unsigned int)v25 )
    {
      v32 = v26;
      v35 = (const __m128i *)(v24 + *(_QWORD *)(a2 + 32));
      v34.m128i_i64[0] = (__int64)&v39;
      sub_16CD150((__int64)&v39, v41, 0, 16, v26, (int)v27);
      v25 = (unsigned int)v40;
      v26 = v32;
      v27 = v35;
    }
    v24 += 40;
    *(__m128i *)&v39[2 * v25] = _mm_loadu_si128(v27);
    v25 = (unsigned int)(v40 + 1);
    LODWORD(v40) = v40 + 1;
  }
  while ( v26 != v24 );
  v28 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  if ( (unsigned int)v25 >= HIDWORD(v40) )
  {
    v34 = v28;
    sub_16CD150((__int64)&v39, v41, 0, 16, v26, (int)v27);
    v25 = (unsigned int)v40;
    v28 = _mm_load_si128(&v34);
  }
  *(__m128i *)&v39[2 * v25] = v28;
  v29 = *(_QWORD *)(a2 + 40);
  LODWORD(v40) = v40 + 1;
  v30 = sub_1D23DE0(v36, v11, (__int64)&v37, v29, *(_DWORD *)(a2 + 60), (__int64)v27, v39, (unsigned int)v40);
  if ( v39 != (__int64 *)v41 )
    _libc_free((unsigned __int64)v39);
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
  return v30;
}
