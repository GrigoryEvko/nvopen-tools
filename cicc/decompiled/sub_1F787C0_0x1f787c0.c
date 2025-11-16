// Function: sub_1F787C0
// Address: 0x1f787c0
//
__int64 *__fastcall sub_1F787C0(__int64 a1, __int64 *a2, __m128i a3, double a4, __m128i a5)
{
  int v6; // r15d
  __int64 *v7; // rax
  __int64 v8; // r14
  unsigned int v9; // ebx
  int v10; // ecx
  __int64 v12; // rax
  unsigned int *v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int8 *v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r15
  __int64 v24; // r14
  __int64 v25; // rsi
  unsigned __int64 v26; // rdx
  __int128 v27; // [rsp+10h] [rbp-80h]
  __int64 *v28; // [rsp+20h] [rbp-70h]
  const void **v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  int v31; // [rsp+38h] [rbp-58h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  int v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h] BYREF
  int v35; // [rsp+58h] [rbp-38h]

  v6 = *(unsigned __int16 *)(a1 + 24);
  v7 = *(__int64 **)(a1 + 32);
  if ( v6 == 52 )
  {
    v8 = v7[5];
    v9 = *((_DWORD *)v7 + 12);
  }
  else
  {
    v8 = *v7;
    v9 = *((_DWORD *)v7 + 2);
    v7 += 5;
  }
  v10 = *(unsigned __int16 *)(v8 + 24);
  if ( v10 != 32 && v10 != 10 )
    return 0;
  v12 = *v7;
  if ( *(_WORD *)(v12 + 24) != 143 )
    return 0;
  v13 = *(unsigned int **)(v12 + 32);
  v14 = *(_QWORD *)v13;
  if ( *(_WORD *)(*(_QWORD *)v13 + 24LL) != 137 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(v14 + 40) + 16LL * v13[2]) != 2 )
    return 0;
  v15 = *(_QWORD *)(v14 + 32);
  if ( *(_DWORD *)(*(_QWORD *)(v15 + 80) + 84LL) != 17 )
    return 0;
  if ( !sub_1D185B0(*(_QWORD *)(v15 + 40)) )
    return 0;
  v16 = **(_QWORD **)(v14 + 32);
  if ( *(_WORD *)(v16 + 24) != 118 || !sub_1D18910(*(_QWORD *)(*(_QWORD *)(v16 + 32) + 40LL)) )
    return 0;
  v17 = a1;
  v18 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16LL * v9);
  v29 = (const void **)*((_QWORD *)v18 + 1);
  v19 = *v18;
  v30 = *(_QWORD *)(a1 + 72);
  if ( v30 )
  {
    sub_1F6CA20(&v30);
    v17 = a1;
  }
  v31 = *(_DWORD *)(v17 + 64);
  *(_QWORD *)&v27 = sub_1D323C0(
                      a2,
                      **(_QWORD **)(v14 + 32),
                      *(_QWORD *)(*(_QWORD *)(v14 + 32) + 8LL),
                      (__int64)&v30,
                      v19,
                      v29,
                      *(double *)a3.m128i_i64,
                      a4,
                      *(double *)a5.m128i_i64);
  *((_QWORD *)&v27 + 1) = v20;
  sub_13A38D0((__int64)&v32, *(_QWORD *)(v8 + 88) + 24LL);
  if ( v6 == 52 )
  {
    sub_16A7490((__int64)&v32, 1);
    v35 = v33;
    v34 = v32;
    v33 = 0;
    v24 = sub_1D38970((__int64)a2, (__int64)&v34, (__int64)&v30, v19, v29, 0, a3, a4, a5, 0);
    v23 = v26;
    sub_135E100(&v34);
    sub_135E100(&v32);
    v25 = 53;
  }
  else
  {
    sub_16A7800((__int64)&v32, 1u);
    v35 = v33;
    v34 = v32;
    v33 = 0;
    v21 = sub_1D38970((__int64)a2, (__int64)&v34, (__int64)&v30, v19, v29, 0, a3, a4, a5, 0);
    v23 = v22;
    v24 = v21;
    sub_135E100(&v34);
    sub_135E100(&v32);
    v25 = 52;
  }
  v28 = sub_1D332F0(a2, v25, (__int64)&v30, v19, v29, 0, *(double *)a3.m128i_i64, a4, a5, v24, v23, v27);
  sub_17CD270(&v30);
  return v28;
}
