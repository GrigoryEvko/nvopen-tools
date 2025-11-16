// Function: sub_2124080
// Address: 0x2124080
//
__int64 __fastcall sub_2124080(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  char v7; // si
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int8 v11; // bl
  const void **v12; // r13
  int v13; // eax
  unsigned int v14; // r14d
  int v15; // r15d
  int v16; // eax
  bool v17; // cl
  unsigned int v18; // r12d
  unsigned int v19; // edx
  unsigned __int64 v20; // rax
  __int64 v21; // r10
  __m128i *v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v26; // [rsp+8h] [rbp-A8h]
  char v27; // [rsp+28h] [rbp-88h]
  __int64 v28; // [rsp+30h] [rbp-80h]
  __int16 v29; // [rsp+3Ch] [rbp-74h]
  int v30; // [rsp+3Ch] [rbp-74h]
  __int64 v31; // [rsp+40h] [rbp-70h] BYREF
  int v32; // [rsp+48h] [rbp-68h]
  _QWORD v33[2]; // [rsp+50h] [rbp-60h] BYREF
  __int128 v34[5]; // [rsp+60h] [rbp-50h] BYREF

  v29 = *(_WORD *)(a2 + 24);
  v6 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v7 = *(_BYTE *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  v9 = *(_QWORD *)(a2 + 40);
  v27 = v7;
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *(_BYTE *)v9;
  v12 = *(const void ***)(v9 + 8);
  v28 = v8;
  v31 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v31, v10, 2);
  v13 = *(_DWORD *)(a2 + 64);
  v26 = a2;
  v14 = 0;
  v15 = 2;
  v32 = v13;
  while ( 1 )
  {
    LOBYTE(v34[0]) = v11;
    LOBYTE(v14) = v15;
    *((_QWORD *)&v34[0] + 1) = v12;
    if ( v11 == (_BYTE)v15 )
      break;
    v18 = sub_211A7A0(v15);
    v19 = v11 ? sub_211A7A0(v11) : sub_1F58D40((__int64)v34);
    v17 = 0;
    v16 = 462;
    if ( v19 <= v18 )
      break;
    if ( (unsigned int)++v15 > 7 )
      goto LABEL_13;
LABEL_7:
    if ( v17 )
      goto LABEL_13;
  }
  if ( v29 == 152 )
    v16 = sub_1F40000(v27, v28, v15);
  else
    v16 = sub_1F40100(v27, v28, v15);
  v17 = v16 != 462;
  if ( (unsigned int)++v15 <= 7 )
    goto LABEL_7;
LABEL_13:
  v30 = v16;
  v20 = sub_2120330(a1, **(_QWORD **)(v26 + 32), *(_QWORD *)(*(_QWORD *)(v26 + 32) + 8LL));
  v21 = *(_QWORD *)(a1 + 8);
  v22 = *(__m128i **)a1;
  v33[0] = v20;
  v33[1] = v23;
  sub_20BE530((__int64)v34, v22, v21, v30, v14, 0, a3, a4, a5, (__int64)v33, 1u, 0, (__int64)&v31, 0, 1);
  v24 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          145,
          (__int64)&v31,
          v11,
          v12,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          v34[0]);
  if ( v31 )
    sub_161E7C0((__int64)&v31, v31);
  return v24;
}
