// Function: sub_3810510
// Address: 0x3810510
//
unsigned __int8 *__fastcall sub_3810510(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  unsigned __int16 v8; // di
  __int64 v9; // r8
  __int64 v10; // r10
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  int v12; // r10d
  int v13; // r9d
  unsigned int v14; // r10d
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int8 *v19; // r12
  unsigned int v20; // edx
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // r13
  unsigned int *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  int v28; // r9d
  __int64 v29; // rsi
  unsigned __int8 *v30; // r12
  int v32; // eax
  __int64 v33; // rdx
  __int128 v34; // [rsp-20h] [rbp-A0h]
  unsigned int v35; // [rsp+8h] [rbp-78h]
  __int16 v36; // [rsp+Ah] [rbp-76h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 *v38; // [rsp+10h] [rbp-70h]
  unsigned __int16 v39; // [rsp+1Ch] [rbp-64h]
  __int16 v40; // [rsp+1Eh] [rbp-62h]
  __int64 v41; // [rsp+30h] [rbp-50h] BYREF
  int v42; // [rsp+38h] [rbp-48h]
  __int64 v43; // [rsp+40h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *a1;
  v7 = a1[1];
  v8 = *v5;
  v9 = *((_QWORD *)v5 + 1);
  v10 = *(_QWORD *)(v7 + 64);
  v39 = *v5;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v41, v6, v10, v8, v9);
    v40 = v42;
    v12 = (unsigned __int16)v42;
    v37 = v43;
  }
  else
  {
    v32 = v11(v6, v10, v39, v9);
    v37 = v33;
    HIWORD(v12) = HIWORD(v32);
    v40 = v32;
  }
  v36 = HIWORD(v12);
  sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  HIWORD(v14) = v36;
  v16 = v15;
  v41 = *(_QWORD *)(a2 + 80);
  if ( v41 )
  {
    sub_B96E90((__int64)&v41, v41, 1);
    HIWORD(v14) = v36;
  }
  v17 = a1[1];
  v42 = *(_DWORD *)(a2 + 72);
  if ( v39 == 11 )
  {
    v18 = 236;
  }
  else if ( v40 == 11 )
  {
    v18 = 237;
  }
  else if ( v39 == 10 )
  {
    v18 = 240;
  }
  else
  {
    if ( v40 != 10 )
      goto LABEL_19;
    v18 = 241;
  }
  LOWORD(v14) = v40;
  v35 = v14;
  v19 = sub_33FAF80(v17, v18, (__int64)&v41, v14, v37, v13, a3);
  v21 = v20 | v16 & 0xFFFFFFFF00000000LL;
  v22 = v37;
  v23 = v21;
  v38 = (__int64 *)a1[1];
  v24 = (unsigned int *)sub_33E5110(
                          v38,
                          v35,
                          v22,
                          *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL));
  *((_QWORD *)&v34 + 1) = v23;
  *(_QWORD *)&v34 = v19;
  v27 = sub_3411EF0(v38, *(unsigned int *)(a2 + 24), (__int64)&v41, v24, v25, v26, v34);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v27, 1);
  if ( v40 == 11 )
  {
    v29 = 236;
    goto LABEL_13;
  }
  if ( v39 == 11 )
  {
    v29 = 237;
  }
  else if ( v40 == 10 )
  {
    v29 = 240;
  }
  else
  {
    v29 = 241;
    if ( v39 != 10 )
LABEL_19:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
LABEL_13:
  v30 = sub_33FAF80(a1[1], v29, (__int64)&v41, 6, 0, v28, a3);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v30;
}
