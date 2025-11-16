// Function: sub_380FD60
// Address: 0x380fd60
//
unsigned __int8 *__fastcall sub_380FD60(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  unsigned __int16 v8; // di
  __int64 v9; // r8
  __int64 v10; // r10
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v12; // r15d
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // r9d
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // edx
  int v21; // r9d
  unsigned int v22; // edx
  int v23; // r9d
  unsigned int v24; // edx
  __int64 v25; // r9
  int v26; // r9d
  __int64 v27; // rsi
  unsigned __int8 *v28; // r12
  int v30; // kr00_4
  __int64 v31; // rdx
  __int128 v32; // [rsp-20h] [rbp-E0h]
  unsigned __int16 v33; // [rsp+Ch] [rbp-B4h]
  __int16 v34; // [rsp+Eh] [rbp-B2h]
  unsigned __int8 *v35; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  __int128 v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  __int128 v39; // [rsp+30h] [rbp-90h]
  __int64 v40; // [rsp+38h] [rbp-88h]
  __int64 v41; // [rsp+70h] [rbp-50h] BYREF
  int v42; // [rsp+78h] [rbp-48h]
  __int64 v43; // [rsp+80h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *a1;
  v7 = a1[1];
  v8 = *v5;
  v9 = *((_QWORD *)v5 + 1);
  v10 = *(_QWORD *)(v7 + 64);
  v33 = *v5;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    HIWORD(v12) = 0;
    sub_2FE6CC0((__int64)&v41, v6, v10, v8, v9);
    v13 = v43;
    v34 = v42;
  }
  else
  {
    v30 = v11(v6, v10, v33, v9);
    HIWORD(v12) = HIWORD(v30);
    v34 = v30;
    v13 = v31;
  }
  sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v40 = v14;
  sub_380F170((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v38 = v15;
  sub_380F170((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v17 = *(_QWORD *)(a2 + 80);
  v36 = v18;
  v41 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v41, v17, 1);
  v42 = *(_DWORD *)(a2 + 72);
  if ( v33 == 11 )
  {
    v19 = 236;
  }
  else if ( v34 == 11 )
  {
    v19 = 237;
  }
  else if ( v33 == 10 )
  {
    v19 = 240;
  }
  else
  {
    if ( v34 != 10 )
      goto LABEL_19;
    v19 = 241;
  }
  LOWORD(v12) = v34;
  *(_QWORD *)&v39 = sub_33FAF80(a1[1], v19, (__int64)&v41, v12, v13, v16, a3);
  *((_QWORD *)&v39 + 1) = v20 | v40 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v37 = sub_33FAF80(a1[1], (unsigned int)v19, (__int64)&v41, v12, v13, v21, a3);
  *((_QWORD *)&v37 + 1) = v22 | v38 & 0xFFFFFFFF00000000LL;
  v35 = sub_33FAF80(a1[1], (unsigned int)v19, (__int64)&v41, v12, v13, v23, a3);
  *((_QWORD *)&v32 + 1) = v24 | v36 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v32 = v35;
  sub_340F900((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v41, v12, v13, v25, v39, v37, v32);
  if ( v34 == 11 )
  {
    v27 = 236;
    goto LABEL_13;
  }
  if ( v33 == 11 )
  {
    v27 = 237;
  }
  else if ( v34 == 10 )
  {
    v27 = 240;
  }
  else
  {
    v27 = 241;
    if ( v33 != 10 )
LABEL_19:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
LABEL_13:
  v28 = sub_33FAF80(a1[1], v27, (__int64)&v41, 6, 0, v26, a3);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v28;
}
