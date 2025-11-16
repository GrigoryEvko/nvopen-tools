// Function: sub_3811390
// Address: 0x3811390
//
unsigned __int8 *__fastcall sub_3811390(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  unsigned __int16 v8; // di
  __int64 v9; // r8
  __int64 v10; // r10
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v12; // r13d
  __int64 v13; // r14
  __int64 v14; // rdx
  int v15; // r9d
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned int v19; // edx
  int v20; // r9d
  unsigned int v21; // edx
  __int64 v22; // r9
  int v23; // r9d
  __int64 v24; // rsi
  unsigned __int8 *v25; // r12
  int v27; // kr00_4
  __int64 v28; // rdx
  __int128 v29; // [rsp-10h] [rbp-B0h]
  unsigned __int16 v30; // [rsp+Ch] [rbp-94h]
  __int16 v31; // [rsp+Eh] [rbp-92h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int128 v33; // [rsp+20h] [rbp-80h]
  __int64 v34; // [rsp+28h] [rbp-78h]
  unsigned __int8 *v35; // [rsp+30h] [rbp-70h]
  __int64 v36; // [rsp+50h] [rbp-50h] BYREF
  int v37; // [rsp+58h] [rbp-48h]
  __int64 v38; // [rsp+60h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *a1;
  v7 = a1[1];
  v8 = *v5;
  v9 = *((_QWORD *)v5 + 1);
  v10 = *(_QWORD *)(v7 + 64);
  v30 = *v5;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    HIWORD(v12) = 0;
    sub_2FE6CC0((__int64)&v36, v6, v10, v8, v9);
    v13 = v38;
    v31 = v37;
  }
  else
  {
    v27 = v11(v6, v10, v30, v9);
    HIWORD(v12) = HIWORD(v27);
    v31 = v27;
    v13 = v28;
  }
  sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v34 = v14;
  sub_380F170((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v16 = *(_QWORD *)(a2 + 80);
  v32 = v17;
  v36 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v36, v16, 1);
  v37 = *(_DWORD *)(a2 + 72);
  if ( v30 == 11 )
  {
    v18 = 236;
  }
  else if ( v31 == 11 )
  {
    v18 = 237;
  }
  else if ( v30 == 10 )
  {
    v18 = 240;
  }
  else
  {
    if ( v31 != 10 )
      goto LABEL_19;
    v18 = 241;
  }
  LOWORD(v12) = v31;
  *(_QWORD *)&v33 = sub_33FAF80(a1[1], v18, (__int64)&v36, v12, v13, v15, a3);
  *((_QWORD *)&v33 + 1) = v19 | v34 & 0xFFFFFFFF00000000LL;
  v35 = sub_33FAF80(a1[1], (unsigned int)v18, (__int64)&v36, v12, v13, v20, a3);
  *((_QWORD *)&v29 + 1) = v21 | v32 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v29 = v35;
  sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v36, v12, v13, v22, v33, v29);
  if ( v31 == 11 )
  {
    v24 = 236;
    goto LABEL_13;
  }
  if ( v30 == 11 )
  {
    v24 = 237;
  }
  else if ( v31 == 10 )
  {
    v24 = 240;
  }
  else
  {
    v24 = 241;
    if ( v30 != 10 )
LABEL_19:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
LABEL_13:
  v25 = sub_33FAF80(a1[1], v24, (__int64)&v36, 6, 0, v23, a3);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v25;
}
