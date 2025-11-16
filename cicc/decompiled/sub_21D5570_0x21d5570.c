// Function: sub_21D5570
// Address: 0x21d5570
//
__int64 *__fastcall sub_21D5570(double a1, double a2, __m128i a3, __int64 a4, __int64 a5, unsigned int a6, __int64 *a7)
{
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int8 v10; // r13
  const void **v11; // rbx
  int v12; // eax
  const __m128i *v13; // rax
  __m128i v14; // xmm0
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rax
  char v18; // dl
  const void **v19; // rax
  __int64 *v20; // r13
  int v22; // r12d
  int v23; // eax
  __int128 v24; // [rsp-10h] [rbp-90h]
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  int v27; // [rsp+28h] [rbp-58h]
  char v28[8]; // [rsp+30h] [rbp-50h] BYREF
  const void **v29; // [rsp+38h] [rbp-48h]
  char v30[8]; // [rsp+40h] [rbp-40h] BYREF
  const void **v31; // [rsp+48h] [rbp-38h]

  v8 = *(_QWORD *)(a5 + 40) + 16LL * a6;
  v9 = *(_QWORD *)(a5 + 72);
  v10 = *(_BYTE *)v8;
  v11 = *(const void ***)(v8 + 8);
  v26 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v26, v9, 2);
  v12 = *(_DWORD *)(a5 + 64);
  v30[0] = v10;
  v31 = v11;
  v27 = v12;
  v13 = *(const __m128i **)(a5 + 32);
  v14 = _mm_loadu_si128(v13);
  v15 = v13[2].m128i_i64[1];
  v16 = v13[3].m128i_i64[0];
  v17 = *(_QWORD *)(v15 + 40) + 16LL * v13[3].m128i_u32[0];
  v18 = *(_BYTE *)v17;
  v19 = *(const void ***)(v17 + 8);
  v28[0] = v18;
  v29 = v19;
  if ( v18 == v10 )
  {
    if ( v10 || v19 == v11 )
      goto LABEL_5;
LABEL_15:
    v22 = sub_1F58D40((__int64)v28);
    if ( !v10 )
      goto LABEL_16;
LABEL_11:
    v23 = sub_1F3E310(v30);
    goto LABEL_12;
  }
  if ( !v18 )
    goto LABEL_15;
  v22 = sub_1F3E310(v28);
  if ( v10 )
    goto LABEL_11;
LABEL_16:
  v23 = sub_1F58D40((__int64)v30);
LABEL_12:
  if ( v23 == v22 )
  {
LABEL_5:
    *((_QWORD *)&v24 + 1) = v16;
    *(_QWORD *)&v24 = v15;
    v20 = sub_1D332F0(
            a7,
            286,
            (__int64)&v26,
            v10,
            v11,
            0,
            *(double *)v14.m128i_i64,
            a2,
            a3,
            v14.m128i_i64[0],
            v14.m128i_u64[1],
            v24);
    goto LABEL_6;
  }
  v20 = 0;
LABEL_6:
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v20;
}
