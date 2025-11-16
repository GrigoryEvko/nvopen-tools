// Function: sub_2140210
// Address: 0x2140210
//
__int64 *__fastcall sub_2140210(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // r10
  unsigned int *v6; // rax
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // r14
  unsigned int v13; // r15d
  unsigned int v14; // edx
  unsigned int v15; // ebx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int128 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // r10
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rbx
  __int64 v24; // rsi
  unsigned __int64 v25; // r15
  unsigned __int8 *v26; // rbx
  __int64 *v27; // r12
  __int64 v28; // r14
  const void **v29; // r8
  __int64 v30; // rcx
  __int64 v31; // r9
  __int64 *v32; // r14
  __int64 v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 *v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  unsigned __int64 v42; // [rsp+20h] [rbp-50h]
  __int64 *v43; // [rsp+20h] [rbp-50h]
  const void **v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+28h] [rbp-48h]
  __int64 v46; // [rsp+30h] [rbp-40h] BYREF
  int v47; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v6 = *(unsigned int **)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)v6;
  v9 = *((_QWORD *)v6 + 1);
  v10 = *(_QWORD *)(*(_QWORD *)v6 + 72LL);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]);
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v46 = v10;
  if ( v10 )
  {
    v38 = v5;
    v42 = v8;
    v45 = v9;
    sub_1623A60((__int64)&v46, v10, 2);
    v5 = v38;
    v8 = v42;
    v9 = v45;
  }
  v36 = v5;
  v35 = v9;
  v47 = *(_DWORD *)(v7 + 64);
  v39 = sub_2138AD0(a1, v8, v9);
  v15 = v14;
  v43 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v19 = sub_1D2EF30(v43, v13, v12, v16, v17, v18);
  v20 = sub_1D332F0(
          v43,
          148,
          (__int64)&v46,
          *(unsigned __int8 *)(*(_QWORD *)(v39 + 40) + 16LL * v15),
          *(const void ***)(*(_QWORD *)(v39 + 40) + 16LL * v15 + 8),
          0,
          a3,
          a4,
          a5,
          v39,
          v15 | v35 & 0xFFFFFFFF00000000LL,
          v19);
  v21 = v36;
  v23 = v22;
  if ( v46 )
  {
    v40 = v20;
    sub_161E7C0((__int64)&v46, v46);
    v20 = v40;
    v21 = v36;
  }
  v24 = *(_QWORD *)(v21 + 72);
  v25 = v23;
  v26 = (unsigned __int8 *)(v20[5] + 16LL * (unsigned int)v23);
  v27 = *(__int64 **)(a1 + 8);
  v28 = (__int64)v20;
  v29 = (const void **)*((_QWORD *)v26 + 1);
  v30 = *v26;
  v46 = v24;
  v31 = *(_QWORD *)(v21 + 32);
  if ( v24 )
  {
    v37 = v30;
    v34 = *(_QWORD *)(v21 + 32);
    v41 = v21;
    v44 = v29;
    sub_1623A60((__int64)&v46, v24, 2);
    v30 = v37;
    v31 = v34;
    v21 = v41;
    v29 = v44;
  }
  v47 = *(_DWORD *)(v21 + 64);
  v32 = sub_1D332F0(v27, 3, (__int64)&v46, v30, v29, 0, a3, a4, a5, v28, v25, *(_OWORD *)(v31 + 40));
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  return v32;
}
