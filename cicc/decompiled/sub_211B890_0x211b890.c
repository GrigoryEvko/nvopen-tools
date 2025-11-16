// Function: sub_211B890
// Address: 0x211b890
//
__int64 __fastcall sub_211B890(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // rax
  char v6; // bl
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int8 v9; // di
  __int64 v10; // r15
  unsigned int v11; // r15d
  int v12; // r14d
  int v13; // eax
  bool v14; // cl
  unsigned int v15; // r13d
  unsigned int v16; // edx
  __int64 v17; // rax
  __m128i *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v25; // [rsp+28h] [rbp-88h]
  __int16 v26; // [rsp+36h] [rbp-7Ah]
  __int64 v27; // [rsp+38h] [rbp-78h]
  int v28; // [rsp+38h] [rbp-78h]
  __int64 v29; // [rsp+40h] [rbp-70h] BYREF
  int v30; // [rsp+48h] [rbp-68h]
  _QWORD v31[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v32; // [rsp+60h] [rbp-50h] BYREF
  __int64 v33; // [rsp+68h] [rbp-48h]
  __int64 v34; // [rsp+70h] [rbp-40h]

  v26 = *(_WORD *)(a2 + 24);
  v5 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v6 = *(_BYTE *)v5;
  v27 = *(_QWORD *)(v5 + 8);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_BYTE *)v7;
  v10 = *(_QWORD *)(v7 + 8);
  v29 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v29, v8, 2);
  v25 = v10;
  v11 = 0;
  v12 = 2;
  v30 = *(_DWORD *)(a2 + 64);
  while ( 1 )
  {
    LOBYTE(v32) = v6;
    LOBYTE(v11) = v12;
    v33 = v27;
    if ( v6 == (_BYTE)v12 )
      break;
    v15 = sub_211A7A0(v12);
    v16 = v6 ? sub_211A7A0(v6) : sub_1F58D40((__int64)&v32);
    v14 = 0;
    v13 = 462;
    if ( v16 <= v15 )
      break;
    if ( (unsigned int)++v12 > 7 )
      goto LABEL_13;
LABEL_7:
    if ( v14 )
      goto LABEL_13;
  }
  if ( v26 == 146 )
    v13 = sub_1F40200(v12, 0, v9);
  else
    v13 = sub_1F402E0(v12, 0, v9);
  v14 = v13 != 462;
  if ( (unsigned int)++v12 <= 7 )
    goto LABEL_7;
LABEL_13:
  v28 = v13;
  v17 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          (unsigned int)(v26 != 146) + 142,
          (__int64)&v29,
          v11,
          0,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  v18 = *(__m128i **)a1;
  v31[0] = v17;
  v19 = *(_QWORD *)(a1 + 8);
  v31[1] = v20;
  sub_1F40D10((__int64)&v32, (__int64)v18, *(_QWORD *)(v19 + 48), v9, v25);
  sub_20BE530(
    (__int64)&v32,
    v18,
    *(_QWORD *)(a1 + 8),
    v28,
    (unsigned __int8)v33,
    v34,
    a3,
    a4,
    a5,
    (__int64)v31,
    1u,
    v26 == 146,
    (__int64)&v29,
    0,
    1);
  v21 = v32;
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  return v21;
}
