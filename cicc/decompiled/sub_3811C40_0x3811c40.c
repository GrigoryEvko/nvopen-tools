// Function: sub_3811C40
// Address: 0x3811c40
//
unsigned __int8 *__fastcall sub_3811C40(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int16 v15; // r15
  __int64 v16; // r10
  unsigned __int16 v17; // di
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int, __int64); // r11
  __int16 v21; // cx
  __int64 v22; // r8
  int v23; // r9d
  unsigned int v24; // r9d
  unsigned int v25; // edx
  unsigned __int64 v26; // r13
  __int64 v27; // rsi
  unsigned __int8 *v28; // r12
  unsigned int v29; // edx
  __int64 v30; // r9
  unsigned __int8 *v31; // r12
  __int64 v33; // rdx
  __int128 v34; // [rsp-20h] [rbp-C0h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int16 v36; // [rsp+Ah] [rbp-96h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  __int16 v38; // [rsp+12h] [rbp-8Eh]
  __int16 v39; // [rsp+18h] [rbp-88h]
  __int16 v40; // [rsp+1Ah] [rbp-86h]
  __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  int v42; // [rsp+48h] [rbp-58h]
  char v43[8]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_QWORD *)(v9 + 48);
  v13 = 16LL * *(unsigned int *)(v9 + 48);
  v14 = *(_QWORD *)(v11 + 48) + v13;
  v37 = v13;
  v15 = *(_WORD *)v14;
  v41 = v10;
  if ( v10 )
  {
    v36 = HIWORD(a7);
    sub_B96E90((__int64)&v41, v10, 1);
    HIWORD(a7) = v36;
    v14 = *(_QWORD *)(v11 + 48) + v37;
  }
  v16 = *a1;
  v40 = HIWORD(a7);
  v42 = *(_DWORD *)(a2 + 72);
  v17 = *(_WORD *)v14;
  v18 = *(_QWORD *)(v14 + 8);
  v19 = a1[1];
  v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v16 + 592LL);
  if ( v20 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v43, v16, *(_QWORD *)(v19 + 64), v17, v18);
    v21 = v44;
    v22 = v45;
    HIWORD(v23) = v40;
  }
  else
  {
    v23 = v20(v16, *(_QWORD *)(v19 + 64), v17, v18);
    v22 = v33;
    v21 = v23;
  }
  v35 = v22;
  v38 = HIWORD(v23);
  v39 = v21;
  sub_380F170((__int64)a1, v11, v12);
  HIWORD(v24) = v38;
  v26 = v25 | v12 & 0xFFFFFFFF00000000LL;
  if ( v15 == 11 )
  {
    v27 = 236;
  }
  else if ( v39 == 11 )
  {
    v27 = 237;
  }
  else if ( v15 == 10 )
  {
    v27 = 240;
  }
  else
  {
    if ( v39 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v27 = 241;
  }
  LOWORD(v24) = v39;
  v28 = sub_33FAF80(a1[1], v27, (__int64)&v41, v24, v35, v24, a3);
  *((_QWORD *)&v34 + 1) = v29 | v26 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v34 = v28;
  v31 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v41,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v30,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          v34);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v31;
}
