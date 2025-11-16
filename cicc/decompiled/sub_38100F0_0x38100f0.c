// Function: sub_38100F0
// Address: 0x38100f0
//
unsigned __int8 *__fastcall sub_38100F0(__int64 *a1, __int64 a2)
{
  __int16 *v3; // rax
  unsigned __int16 v4; // r15
  __int64 v5; // r8
  __int64 v6; // r11
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // r8
  int v9; // r10d
  int v10; // r9d
  unsigned int v11; // r10d
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r13
  __int128 v15; // xmm0
  __int64 v16; // rdi
  unsigned __int8 *v17; // rax
  unsigned int v18; // edx
  __int64 v19; // r9
  int v20; // r9d
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rax
  unsigned int v24; // edx
  __int64 v25; // r9
  unsigned __int8 *v26; // r12
  unsigned __int8 *v28; // rax
  unsigned int v29; // edx
  __int64 v30; // r9
  int v31; // kr00_4
  __int64 v32; // rdx
  unsigned __int8 *v33; // rax
  unsigned int v34; // edx
  __int64 v35; // r9
  __int128 v36; // [rsp-30h] [rbp-F0h]
  __int128 v37; // [rsp-30h] [rbp-F0h]
  __int128 v38; // [rsp-30h] [rbp-F0h]
  __int128 v39; // [rsp-30h] [rbp-F0h]
  __int64 v40; // [rsp+18h] [rbp-A8h]
  unsigned int v41; // [rsp+18h] [rbp-A8h]
  unsigned int v42; // [rsp+18h] [rbp-A8h]
  unsigned int v43; // [rsp+20h] [rbp-A0h]
  __int64 v44; // [rsp+20h] [rbp-A0h]
  unsigned int v45; // [rsp+20h] [rbp-A0h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int16 v47; // [rsp+22h] [rbp-9Eh]
  __int16 v48; // [rsp+28h] [rbp-98h]
  __int64 v49; // [rsp+28h] [rbp-98h]
  __int64 v50; // [rsp+28h] [rbp-98h]
  __int64 v51; // [rsp+70h] [rbp-50h] BYREF
  int v52; // [rsp+78h] [rbp-48h]
  __int64 v53; // [rsp+80h] [rbp-40h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(_QWORD *)(a1[1] + 64);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v51, *a1, v6, v4, v5);
    v8 = v53;
    v48 = v52;
    v9 = (unsigned __int16)v52;
  }
  else
  {
    v31 = v7(*a1, v6, v4, v5);
    HIWORD(v9) = HIWORD(v31);
    v48 = v31;
    v8 = v32;
  }
  v40 = v8;
  v47 = HIWORD(v9);
  sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  HIWORD(v11) = v47;
  v12 = v40;
  v14 = v13;
  v15 = (__int128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v51 = *(_QWORD *)(a2 + 80);
  if ( v51 )
  {
    sub_B96E90((__int64)&v51, v51, 1);
    v12 = v40;
    HIWORD(v11) = v47;
  }
  v16 = a1[1];
  v52 = *(_DWORD *)(a2 + 72);
  if ( v4 == 11 )
  {
    LOWORD(v11) = v48;
    v41 = v11;
    v44 = v12;
    v23 = sub_33FAF80(v16, 236, (__int64)&v51, v11, v12, v10, (__m128i)v15);
    *((_QWORD *)&v37 + 1) = v24 | v14 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v37 = v23;
    sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v51, v41, v44, v25, v37, v15);
    v21 = a1[1];
    v22 = (unsigned int)(v48 != 11) + 236;
  }
  else if ( v48 == 11 )
  {
    LOWORD(v11) = 11;
    v45 = v11;
    v50 = v12;
    v28 = sub_33FAF80(v16, 237, (__int64)&v51, v11, v12, v10, (__m128i)v15);
    *((_QWORD *)&v38 + 1) = v29 | v14 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v38 = v28;
    sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v51, v45, v50, v30, v38, v15);
    v21 = a1[1];
    v22 = 236;
  }
  else if ( v4 == 10 )
  {
    LOWORD(v11) = v48;
    v42 = v11;
    v46 = v12;
    v33 = sub_33FAF80(v16, 240, (__int64)&v51, v11, v12, v10, (__m128i)v15);
    *((_QWORD *)&v39 + 1) = v34 | v14 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v39 = v33;
    sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v51, v42, v46, v35, v39, v15);
    v21 = a1[1];
    v22 = (unsigned int)(v48 != 10) + 240;
  }
  else
  {
    if ( v48 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    LOWORD(v11) = 10;
    v43 = v11;
    v49 = v12;
    v17 = sub_33FAF80(v16, 241, (__int64)&v51, v11, v12, v10, (__m128i)v15);
    *((_QWORD *)&v36 + 1) = v18 | v14 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v36 = v17;
    sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v51, v43, v49, v19, v36, v15);
    v21 = a1[1];
    v22 = 240;
  }
  v26 = sub_33FAF80(v21, v22, (__int64)&v51, 6, 0, v20, (__m128i)v15);
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  return v26;
}
