// Function: sub_38040F0
// Address: 0x38040f0
//
unsigned __int8 *__fastcall sub_38040F0(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // rax
  __int128 v8; // xmm0
  __int128 v9; // xmm1
  __int64 v10; // rdx
  __int16 *v11; // rax
  unsigned __int16 v12; // si
  __int16 *v13; // rdx
  __int64 v14; // r8
  __int16 v15; // r15
  unsigned __int16 v16; // si
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int, __int64); // r10
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned int *v27; // rcx
  __int64 v28; // rsi
  __int128 v29; // rax
  __int64 v30; // r14
  __int64 *v31; // rdi
  __int64 v32; // rdx
  unsigned int *v33; // rcx
  __int64 v34; // rsi
  unsigned __int8 *v35; // r14
  int v37; // eax
  __int64 v38; // rdx
  __int128 v39; // [rsp-20h] [rbp-E0h]
  __int64 v40; // [rsp+8h] [rbp-B8h]
  __int128 v41; // [rsp+20h] [rbp-A0h]
  __int16 v42; // [rsp+38h] [rbp-88h]
  __int64 v43; // [rsp+40h] [rbp-80h] BYREF
  int v44; // [rsp+48h] [rbp-78h]
  unsigned __int16 v45; // [rsp+50h] [rbp-70h] BYREF
  __int64 v46; // [rsp+58h] [rbp-68h]
  __int64 v47; // [rsp+60h] [rbp-60h] BYREF
  char v48; // [rsp+68h] [rbp-58h]
  __int64 v49; // [rsp+70h] [rbp-50h] BYREF
  __int64 v50; // [rsp+78h] [rbp-48h]
  __int64 v51; // [rsp+80h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 80);
  v43 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v43, v5, 1);
  v6 = *a1;
  v44 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = (__int128)_mm_loadu_si128((const __m128i *)v7);
  v9 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(_QWORD *)(v7 + 40);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *(__int16 **)(v10 + 48);
  v46 = *((_QWORD *)v11 + 1);
  v14 = *((_QWORD *)v11 + 1);
  v45 = v12;
  v15 = *v13;
  v16 = *v11;
  v17 = a1[1];
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v18 == sub_2D56A50 )
  {
    HIWORD(v19) = 0;
    sub_2FE6CC0((__int64)&v49, v6, *(_QWORD *)(v17 + 64), v16, v14);
    v42 = v50;
    v40 = v51;
  }
  else
  {
    v37 = v18(v6, *(_QWORD *)(v17 + 64), v16, v14);
    v40 = v38;
    HIWORD(v19) = HIWORD(v37);
    v42 = v37;
  }
  if ( v45 )
  {
    if ( v45 == 1 || (unsigned __int16)(v45 - 504) <= 7u )
      BUG();
    v21 = 16LL * (v45 - 1);
    v20 = *(_QWORD *)&byte_444C4A0[v21];
    LOBYTE(v21) = byte_444C4A0[v21 + 8];
  }
  else
  {
    v20 = sub_3007260((__int64)&v45);
    v49 = v20;
    v50 = v21;
  }
  v47 = v20;
  v48 = v21;
  v22 = sub_CA1930(&v47);
  v23 = a1[1];
  switch ( v22 )
  {
    case 1u:
      LOWORD(v24) = 2;
      break;
    case 2u:
      LOWORD(v24) = 3;
      break;
    case 4u:
      LOWORD(v24) = 4;
      break;
    case 8u:
      LOWORD(v24) = 5;
      break;
    case 0x10u:
      LOWORD(v24) = 6;
      break;
    case 0x20u:
      LOWORD(v24) = 7;
      break;
    case 0x40u:
      LOWORD(v24) = 8;
      break;
    case 0x80u:
      LOWORD(v24) = 9;
      break;
    default:
      v24 = sub_3007020(*(_QWORD **)(v23 + 64), v22);
      v23 = a1[1];
      HIWORD(v2) = HIWORD(v24);
      goto LABEL_18;
  }
  v25 = 0;
LABEL_18:
  LOWORD(v2) = v24;
  v27 = (unsigned int *)sub_33E5110((__int64 *)v23, v2, v25, 1, 0);
  if ( v15 == 11 )
  {
    v28 = 238;
  }
  else if ( v45 == 11 )
  {
    v28 = 239;
  }
  else if ( v15 == 10 )
  {
    v28 = 242;
  }
  else
  {
    if ( v45 != 10 )
      goto LABEL_28;
    v28 = 243;
  }
  *(_QWORD *)&v29 = sub_3411F20((_QWORD *)v23, v28, (__int64)&v43, v27, v26, (__int64)&v43, v8, v9);
  v41 = v29;
  LOWORD(v19) = v42;
  v30 = v29;
  v31 = (__int64 *)a1[1];
  v33 = (unsigned int *)sub_33E5110(v31, v19, v40, 1, 0);
  if ( v45 == 11 )
  {
    v34 = 238;
  }
  else if ( v42 == 11 )
  {
    v34 = 239;
  }
  else if ( v45 == 10 )
  {
    v34 = 242;
  }
  else
  {
    v34 = 243;
    if ( v42 != 10 )
LABEL_28:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
  *((_QWORD *)&v39 + 1) = 1;
  *(_QWORD *)&v39 = v30;
  v35 = sub_3411F20(v31, v34, (__int64)&v43, v33, v32, (__int64)&v43, v39, v41);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v35, 1);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v35;
}
