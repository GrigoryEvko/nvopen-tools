// Function: sub_3810F90
// Address: 0x3810f90
//
unsigned __int8 *__fastcall sub_3810F90(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int64 v5; // r10
  unsigned __int16 v6; // r15
  __int64 v7; // r8
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v9; // r14d
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rdi
  int v15; // r9d
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int8 *v18; // r12
  int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-A8h]
  __int64 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+8h] [rbp-A8h]
  __int64 v25; // [rsp+10h] [rbp-A0h]
  __int64 v26; // [rsp+10h] [rbp-A0h]
  __int64 v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+10h] [rbp-A0h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  __int16 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+60h] [rbp-50h] BYREF
  int v34; // [rsp+68h] [rbp-48h]
  __int64 v35; // [rsp+70h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *a1;
  v6 = *v4;
  v7 = *((_QWORD *)v4 + 1);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v33, v5, *(_QWORD *)(a1[1] + 64), v6, v7);
    v10 = v35;
    v11 = a2;
    v30 = v34;
  }
  else
  {
    v20 = v8(v5, *(_QWORD *)(a1[1] + 64), v6, v7);
    v11 = a2;
    HIWORD(v9) = HIWORD(v20);
    v30 = v20;
    v10 = v21;
  }
  v22 = v10;
  v25 = v11;
  sub_380F170((__int64)a1, **(_QWORD **)(v11 + 40), *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8LL));
  v12 = v25;
  v13 = v22;
  v33 = *(_QWORD *)(v25 + 80);
  if ( v33 )
  {
    sub_B96E90((__int64)&v33, v33, 1);
    v13 = v22;
    v12 = v25;
  }
  v14 = a1[1];
  v34 = *(_DWORD *)(v12 + 72);
  if ( v6 == 11 )
  {
    LOWORD(v9) = v30;
    v23 = v12;
    v27 = v13;
    sub_33FAF80(v14, 236, (__int64)&v33, v9, v13, v12, a3);
    sub_33FAF80(a1[1], *(unsigned int *)(v23 + 24), (__int64)&v33, v9, v27, v23, a3);
    v16 = a1[1];
    v17 = (unsigned int)(v30 != 11) + 236;
  }
  else if ( v30 == 11 )
  {
    LOWORD(v9) = 11;
    v28 = v12;
    v32 = v13;
    sub_33FAF80(v14, 237, (__int64)&v33, v9, v13, v12, a3);
    sub_33FAF80(a1[1], *(unsigned int *)(v28 + 24), (__int64)&v33, v9, v32, v28, a3);
    v16 = a1[1];
    v17 = 236;
  }
  else if ( v6 == 10 )
  {
    LOWORD(v9) = v30;
    v24 = v12;
    v29 = v13;
    sub_33FAF80(v14, 240, (__int64)&v33, v9, v13, v12, a3);
    sub_33FAF80(a1[1], *(unsigned int *)(v24 + 24), (__int64)&v33, v9, v29, v24, a3);
    v16 = a1[1];
    v17 = (unsigned int)(v30 != 10) + 240;
  }
  else
  {
    if ( v30 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    LOWORD(v9) = 10;
    v26 = v12;
    v31 = v13;
    sub_33FAF80(v14, 241, (__int64)&v33, v9, v13, v12, a3);
    sub_33FAF80(a1[1], *(unsigned int *)(v26 + 24), (__int64)&v33, v9, v31, v26, a3);
    v16 = a1[1];
    v17 = 240;
  }
  v18 = sub_33FAF80(v16, v17, (__int64)&v33, 6, 0, v15, a3);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v18;
}
