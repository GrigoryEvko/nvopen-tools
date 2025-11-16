// Function: sub_211B1D0
// Address: 0x211b1d0
//
__int64 __fastcall sub_211B1D0(__int64 *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  char *v6; // rdx
  const void **v7; // r14
  unsigned int v8; // r15d
  __m128i v9; // xmm0
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rsi
  __m128i *v13; // r11
  int v14; // ecx
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v18; // rsi
  __int64 *v19; // r12
  int v20; // [rsp+Ch] [rbp-84h]
  __m128i *v21; // [rsp+10h] [rbp-80h]
  __int128 v22; // [rsp+20h] [rbp-70h] BYREF
  __int64 v23; // [rsp+30h] [rbp-60h] BYREF
  int v24; // [rsp+38h] [rbp-58h]
  __int64 v25; // [rsp+40h] [rbp-50h] BYREF
  __int64 v26; // [rsp+48h] [rbp-48h]
  const void **v27; // [rsp+50h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v25,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(char **)(a2 + 40);
  v7 = v27;
  v8 = (unsigned __int8)v26;
  v9 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v22 = (__int128)v9;
  if ( *v6 == 8 )
  {
    v18 = *(_QWORD *)(a2 + 72);
    v19 = (__int64 *)a1[1];
    v25 = v18;
    if ( v18 )
      sub_1623A60((__int64)&v25, v18, 2);
    LODWORD(v26) = *(_DWORD *)(a2 + 64);
    v16 = sub_1D309E0(
            v19,
            161,
            (__int64)&v25,
            v8,
            v7,
            0,
            *(double *)v9.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            v22);
    if ( v25 )
      sub_161E7C0((__int64)&v25, v25);
  }
  else
  {
    v10 = *(_QWORD *)(v22 + 40) + 16LL * DWORD2(v22);
    v11 = sub_1F3FF10(*(_BYTE *)v10, *(_QWORD *)(v10 + 8), *v6);
    v12 = *(_QWORD *)(a2 + 72);
    v13 = (__m128i *)*a1;
    v14 = v11;
    v23 = v12;
    if ( v12 )
    {
      v21 = v13;
      v20 = v11;
      sub_1623A60((__int64)&v23, v12, 2);
      v14 = v20;
      v13 = v21;
    }
    v15 = a1[1];
    v24 = *(_DWORD *)(a2 + 64);
    sub_20BE530((__int64)&v25, v13, v15, v14, v8, (__int64)v7, v9, a4, a5, (__int64)&v22, 1u, 0, (__int64)&v23, 0, 1);
    v16 = v25;
    if ( v23 )
      sub_161E7C0((__int64)&v23, v23);
  }
  return v16;
}
