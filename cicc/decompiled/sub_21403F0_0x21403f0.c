// Function: sub_21403F0
// Address: 0x21403f0
//
__int64 *__fastcall sub_21403F0(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // r11
  unsigned int *v7; // rax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rsi
  unsigned __int8 *v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned int v17; // ebx
  __int64 v18; // r8
  __int64 v19; // r9
  __int128 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r11
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // r11
  __int64 v29; // rsi
  __int64 *v30; // r10
  unsigned __int8 *v31; // rdx
  const void **v32; // r13
  unsigned int v33; // ebx
  __int64 *v34; // r14
  __int64 *v36; // rax
  unsigned int v37; // edx
  __int128 v38; // [rsp-10h] [rbp-90h]
  __int64 v39; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  unsigned __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  __int64 *v48; // [rsp+18h] [rbp-68h]
  __int64 *v49; // [rsp+18h] [rbp-68h]
  __int64 *v50; // [rsp+20h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  unsigned __int64 v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  int v54; // [rsp+38h] [rbp-48h]

  v5 = a2;
  v7 = *(unsigned int **)(a2 + 32);
  v8 = *(_QWORD *)v7;
  v9 = *(_QWORD *)v7;
  v10 = *((_QWORD *)v7 + 1);
  v11 = *(_QWORD *)(*(_QWORD *)v7 + 72LL);
  v12 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2]);
  v13 = *((_QWORD *)v12 + 1);
  v14 = *v12;
  v53 = v11;
  if ( v11 )
  {
    v43 = v14;
    v46 = v5;
    sub_1623A60((__int64)&v53, v11, 2);
    v14 = v43;
    v5 = v46;
  }
  v44 = v5;
  v41 = v14;
  v54 = *(_DWORD *)(v8 + 64);
  v15 = sub_2138AD0((__int64)a1, v9, v10);
  v17 = v16;
  v47 = v15;
  v50 = (__int64 *)a1[1];
  *(_QWORD *)&v20 = sub_1D2EF30(v50, (unsigned int)v41, v13, v41, v18, v19);
  v21 = sub_1D332F0(
          v50,
          148,
          (__int64)&v53,
          *(unsigned __int8 *)(*(_QWORD *)(v47 + 40) + 16LL * v17),
          *(const void ***)(*(_QWORD *)(v47 + 40) + 16LL * v17 + 8),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v47,
          v17 | v10 & 0xFFFFFFFF00000000LL,
          v20);
  v22 = v44;
  v48 = v21;
  v24 = v23;
  if ( v53 )
  {
    sub_161E7C0((__int64)&v53, v53);
    v22 = v44;
  }
  v39 = v22;
  v52 = v24;
  v51 = (__int64)v48;
  v25 = *(_QWORD *)(v22 + 32);
  v26 = *(_QWORD *)(v25 + 40);
  v27 = *(_QWORD *)(v25 + 48);
  v45 = v26;
  v42 = *(unsigned int *)(v25 + 48);
  sub_1F40D10(
    (__int64)&v53,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    *(unsigned __int8 *)(*(_QWORD *)(v45 + 40) + 16 * v42),
    *(_QWORD *)(*(_QWORD *)(v45 + 40) + 16 * v42 + 8));
  v28 = v39;
  if ( (_BYTE)v53 == 1 )
  {
    v36 = sub_2139210((__int64)a1, v26, v27, a3, a4, a5);
    v28 = v39;
    v45 = (unsigned __int64)v36;
    v42 = v37;
  }
  v29 = *(_QWORD *)(v28 + 72);
  v30 = (__int64 *)a1[1];
  v31 = (unsigned __int8 *)(v48[5] + 16LL * (unsigned int)v24);
  v32 = (const void **)*((_QWORD *)v31 + 1);
  v33 = *v31;
  v53 = v29;
  if ( v29 )
  {
    v40 = v28;
    v49 = v30;
    sub_1623A60((__int64)&v53, v29, 2);
    v28 = v40;
    v30 = v49;
  }
  v54 = *(_DWORD *)(v28 + 64);
  *((_QWORD *)&v38 + 1) = v27 & 0xFFFFFFFF00000000LL | v42;
  *(_QWORD *)&v38 = v45;
  v34 = sub_1D332F0(v30, 123, (__int64)&v53, v33, v32, 0, *(double *)a3.m128i_i64, a4, a5, v51, v52, v38);
  if ( v53 )
    sub_161E7C0((__int64)&v53, v53);
  return v34;
}
