// Function: sub_38107B0
// Address: 0x38107b0
//
__int64 __fastcall sub_38107B0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  unsigned __int16 v6; // r13
  __int64 v7; // r8
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v9; // r15d
  __int64 v10; // r8
  int v11; // r9d
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned __int8 *v16; // rax
  __int64 *v17; // r13
  unsigned int v18; // edx
  unsigned int *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  int v22; // r9d
  unsigned __int8 *v23; // rax
  __int64 *v24; // r13
  unsigned int v25; // edx
  unsigned int *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r9
  unsigned int v29; // r15d
  unsigned __int64 v30; // r13
  __int64 v31; // r14
  unsigned __int8 *v32; // rax
  __int64 v33; // rdx
  unsigned __int8 *v35; // rax
  __int64 *v36; // r13
  unsigned int v37; // edx
  unsigned int *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r9
  int v41; // kr00_4
  __int64 v42; // rdx
  unsigned __int8 *v43; // rax
  __int64 *v44; // r13
  unsigned int v45; // edx
  unsigned int *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r9
  unsigned __int64 v49; // [rsp+8h] [rbp-C8h]
  int v50; // [rsp+1Ch] [rbp-B4h]
  __int64 v51; // [rsp+20h] [rbp-B0h]
  __int64 v52; // [rsp+20h] [rbp-B0h]
  __int64 v53; // [rsp+20h] [rbp-B0h]
  __int16 v54; // [rsp+28h] [rbp-A8h]
  __int64 v55; // [rsp+28h] [rbp-A8h]
  unsigned int v56; // [rsp+28h] [rbp-A8h]
  __int64 v57; // [rsp+28h] [rbp-A8h]
  __int128 v58; // [rsp+30h] [rbp-A0h]
  __int128 v59; // [rsp+30h] [rbp-A0h]
  __int128 v60; // [rsp+30h] [rbp-A0h]
  __int128 v61; // [rsp+30h] [rbp-A0h]
  __int64 v62; // [rsp+38h] [rbp-98h]
  __int64 v63; // [rsp+80h] [rbp-50h] BYREF
  int v64; // [rsp+88h] [rbp-48h]
  __int64 v65; // [rsp+90h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v8 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v63, *a1, *(_QWORD *)(a1[1] + 64), v6, v7);
    v10 = v65;
    v54 = v64;
  }
  else
  {
    v41 = v8(*a1, *(_QWORD *)(a1[1] + 64), v6, v7);
    HIWORD(v9) = HIWORD(v41);
    v54 = v41;
    v10 = v42;
  }
  v51 = v10;
  sub_380F170((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v12 = *(_QWORD *)(a2 + 80);
  v13 = v51;
  v62 = v14;
  v63 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v63, v12, 1);
    v13 = v51;
  }
  v15 = a1[1];
  v64 = *(_DWORD *)(a2 + 72);
  if ( v6 == 11 )
  {
    LOWORD(v9) = v54;
    v52 = v13;
    v23 = sub_33FAF80(v15, 236, (__int64)&v63, v9, v13, v11, a3);
    v24 = (__int64 *)a1[1];
    *(_QWORD *)&v59 = v23;
    *((_QWORD *)&v59 + 1) = v25 | v62 & 0xFFFFFFFF00000000LL;
    v26 = (unsigned int *)sub_33E5110(v24, v9, v52, v9, v52);
    sub_3411EF0(v24, *(unsigned int *)(a2 + 24), (__int64)&v63, v26, v27, v28, v59);
    v56 = (v54 != 11) + 236;
  }
  else if ( v54 == 11 )
  {
    LOWORD(v9) = 11;
    v57 = v13;
    v35 = sub_33FAF80(v15, 237, (__int64)&v63, v9, v13, v11, a3);
    v36 = (__int64 *)a1[1];
    *(_QWORD *)&v60 = v35;
    *((_QWORD *)&v60 + 1) = v37 | v62 & 0xFFFFFFFF00000000LL;
    v38 = (unsigned int *)sub_33E5110(v36, v9, v57, v9, v57);
    v56 = 236;
    sub_3411EF0(v36, *(unsigned int *)(a2 + 24), (__int64)&v63, v38, v39, v40, v60);
  }
  else if ( v6 == 10 )
  {
    LOWORD(v9) = v54;
    v53 = v13;
    v43 = sub_33FAF80(v15, 240, (__int64)&v63, v9, v13, v11, a3);
    v44 = (__int64 *)a1[1];
    *(_QWORD *)&v61 = v43;
    *((_QWORD *)&v61 + 1) = v45 | v62 & 0xFFFFFFFF00000000LL;
    v46 = (unsigned int *)sub_33E5110(v44, v9, v53, v9, v53);
    sub_3411EF0(v44, *(unsigned int *)(a2 + 24), (__int64)&v63, v46, v47, v48, v61);
    v56 = (v54 != 10) + 240;
  }
  else
  {
    if ( v54 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    LOWORD(v9) = 10;
    v55 = v13;
    v16 = sub_33FAF80(v15, 241, (__int64)&v63, v9, v13, v11, a3);
    v17 = (__int64 *)a1[1];
    *(_QWORD *)&v58 = v16;
    *((_QWORD *)&v58 + 1) = v18 | v62 & 0xFFFFFFFF00000000LL;
    v19 = (unsigned int *)sub_33E5110(v17, v9, v55, v9, v55);
    v56 = 240;
    sub_3411EF0(v17, *(unsigned int *)(a2 + 24), (__int64)&v63, v19, v20, v21, v58);
  }
  v29 = 0;
  v50 = *(_DWORD *)(a2 + 68);
  if ( v50 )
  {
    v30 = v49;
    do
    {
      v31 = v29++;
      v30 = v31 | v30 & 0xFFFFFFFF00000000LL;
      v32 = sub_33FAF80(a1[1], v56, (__int64)&v63, 6, 0, v22, a3);
      sub_375F970((__int64)a1, a2, v31, (unsigned __int64)v32, v33);
    }
    while ( v50 != v29 );
  }
  if ( v63 )
    sub_B91220((__int64)&v63, v63);
  return 0;
}
