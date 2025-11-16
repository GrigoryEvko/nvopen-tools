// Function: sub_12B24C0
// Address: 0x12b24c0
//
_QWORD *__fastcall sub_12B24C0(__int64 a1, int a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __m128i *v12; // r14
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  int *v18; // rdx
  _BOOL4 v19; // r9d
  __int64 v20; // rax
  __m128i v21; // xmm1
  __int64 v22; // rax
  int v23; // edx
  int *v27; // [rsp+18h] [rbp-B8h]
  _BOOL4 v28; // [rsp+24h] [rbp-ACh]
  __int64 v29; // [rsp+28h] [rbp-A8h]
  __int64 v30; // [rsp+30h] [rbp-A0h] BYREF
  int v31; // [rsp+38h] [rbp-98h] BYREF
  __int64 v32; // [rsp+40h] [rbp-90h]
  int *v33; // [rsp+48h] [rbp-88h]
  int *v34; // [rsp+50h] [rbp-80h]
  __int64 v35; // [rsp+58h] [rbp-78h]
  int v36; // [rsp+60h] [rbp-70h] BYREF
  __int64 v37; // [rsp+68h] [rbp-68h]
  __int64 v38; // [rsp+70h] [rbp-60h]
  __int64 v39; // [rsp+78h] [rbp-58h]
  int v40; // [rsp+80h] [rbp-50h]
  __int64 v41; // [rsp+88h] [rbp-48h]
  __int64 v42; // [rsp+90h] [rbp-40h]
  __int64 v43; // [rsp+98h] [rbp-38h]
  char v44; // [rsp+A0h] [rbp-30h] BYREF

  v29 = a1 + 640;
  if ( !*(_QWORD *)(a1 + 672) )
  {
    v12 = (__m128i *)&v36;
    v13 = sub_16432A0(*(_QWORD *)(a1 + 360));
    v14 = sub_16432B0(*(_QWORD *)(a1 + 360));
    v15 = sub_16463B0(v14, 2);
    v16 = sub_16463B0(v13, 8);
    v39 = v15;
    v36 = 751;
    v37 = 1;
    v38 = 9;
    v40 = 752;
    v41 = 25;
    v42 = 10;
    v43 = v16;
    v31 = 0;
    v32 = 0;
    v33 = &v31;
    v34 = &v31;
    v35 = 0;
    do
    {
      v17 = sub_12B23C0(&v30, (__int64)&v31, v12->m128i_i32);
      if ( v18 )
      {
        v19 = v17 || v18 == &v31 || v12->m128i_i32[0] < v18[8];
        v27 = v18;
        v28 = v19;
        v20 = sub_22077B0(64);
        v21 = _mm_loadu_si128(v12 + 1);
        *(__m128i *)(v20 + 32) = _mm_loadu_si128(v12);
        *(__m128i *)(v20 + 48) = v21;
        sub_220F040(v28, v20, v27, &v31);
        ++v35;
      }
      v12 += 2;
    }
    while ( v12 != (__m128i *)&v44 );
    sub_12A7A00(*(_QWORD *)(a1 + 648));
    v22 = v32;
    *(_QWORD *)(a1 + 648) = 0;
    *(_QWORD *)(a1 + 672) = 0;
    *(_QWORD *)(a1 + 656) = v29;
    *(_QWORD *)(a1 + 664) = v29;
    if ( v22 )
    {
      v23 = v31;
      *(_QWORD *)(a1 + 648) = v22;
      *(_DWORD *)(a1 + 640) = v23;
      *(_QWORD *)(a1 + 656) = v33;
      *(_QWORD *)(a1 + 664) = v34;
      *(_QWORD *)(v22 + 8) = v29;
      v32 = 0;
      *(_QWORD *)(a1 + 672) = v35;
      v33 = &v31;
      v34 = &v31;
      v35 = 0;
    }
    sub_12A7A00(0);
  }
  v6 = *(_QWORD *)(a1 + 648);
  if ( !v6 )
    goto LABEL_9;
  v7 = a1 + 640;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v6 + 16);
      v9 = *(_QWORD *)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) >= a2 )
        break;
      v6 = *(_QWORD *)(v6 + 24);
      if ( !v9 )
        goto LABEL_7;
    }
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
  }
  while ( v8 );
LABEL_7:
  if ( v29 == v7 || a2 < *(_DWORD *)(v7 + 32) )
LABEL_9:
    sub_127B630("unexpected overloaded mma store intrinsic call!", 0);
  v10 = *(_QWORD *)(v7 + 56);
  *a4 = a3 | (*(_QWORD *)(v7 + 40) << 32) | (16LL * *(_QWORD *)(v7 + 48)) | 6;
  *a5 = v10;
  return a5;
}
