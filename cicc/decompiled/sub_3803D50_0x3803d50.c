// Function: sub_3803D50
// Address: 0x3803d50
//
unsigned __int8 *__fastcall sub_3803D50(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r14d
  __int64 v6; // rsi
  __int64 v7; // r9
  __int16 *v8; // rax
  __int16 *v9; // rdx
  __int64 v10; // rcx
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 (__fastcall *v14)(__int64, __int64, unsigned int, __int64); // r10
  unsigned int v15; // r15d
  __int64 v16; // rdx
  __int64 v17; // rdx
  char v18; // al
  unsigned int v19; // eax
  int v20; // r9d
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rsi
  int v26; // r9d
  unsigned int v27; // r12d
  unsigned __int8 *v28; // r12
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int16 v34; // [rsp+10h] [rbp-90h]
  __int16 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+20h] [rbp-80h] BYREF
  int v37; // [rsp+28h] [rbp-78h]
  unsigned __int16 v38; // [rsp+30h] [rbp-70h] BYREF
  __int64 v39; // [rsp+38h] [rbp-68h]
  __int64 v40; // [rsp+40h] [rbp-60h] BYREF
  char v41; // [rsp+48h] [rbp-58h]
  __int64 v42; // [rsp+50h] [rbp-50h] BYREF
  __int64 v43; // [rsp+58h] [rbp-48h]
  __int64 v44; // [rsp+60h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v36 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v36, v6, 1);
  v7 = *a1;
  v37 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *(__int16 **)(**(_QWORD **)(a2 + 40) + 48LL);
  v10 = *((_QWORD *)v8 + 1);
  v38 = *v8;
  v39 = v10;
  v11 = *v8;
  v12 = *((_QWORD *)v8 + 1);
  v13 = a1[1];
  v34 = *v9;
  v14 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  if ( v14 == sub_2D56A50 )
  {
    HIWORD(v15) = 0;
    sub_2FE6CC0((__int64)&v42, v7, *(_QWORD *)(v13 + 64), v11, v12);
    v35 = v43;
    v33 = v44;
  }
  else
  {
    v31 = v14(v7, *(_QWORD *)(v13 + 64), v11, v12);
    v33 = v32;
    HIWORD(v15) = HIWORD(v31);
    v35 = v31;
  }
  if ( v38 )
  {
    if ( v38 == 1 || (unsigned __int16)(v38 - 504) <= 7u )
      BUG();
    v30 = 16LL * (v38 - 1);
    v17 = *(_QWORD *)&byte_444C4A0[v30];
    v18 = byte_444C4A0[v30 + 8];
  }
  else
  {
    v42 = sub_3007260((__int64)&v38);
    v43 = v16;
    v17 = v42;
    v18 = v43;
  }
  v40 = v17;
  v41 = v18;
  v19 = sub_CA1930(&v40);
  v21 = a1[1];
  switch ( v19 )
  {
    case 1u:
      LOWORD(v22) = 2;
      break;
    case 2u:
      LOWORD(v22) = 3;
      break;
    case 4u:
      LOWORD(v22) = 4;
      break;
    case 8u:
      LOWORD(v22) = 5;
      break;
    case 0x10u:
      LOWORD(v22) = 6;
      break;
    case 0x20u:
      LOWORD(v22) = 7;
      break;
    case 0x40u:
      LOWORD(v22) = 8;
      break;
    case 0x80u:
      LOWORD(v22) = 9;
      break;
    default:
      v22 = sub_3007020(*(_QWORD **)(v21 + 64), v19);
      v21 = a1[1];
      HIWORD(v3) = HIWORD(v22);
      v24 = v23;
      goto LABEL_18;
  }
  v24 = 0;
LABEL_18:
  LOWORD(v3) = v22;
  if ( v34 == 11 )
  {
    v25 = 236;
  }
  else if ( v38 == 11 )
  {
    v25 = 237;
  }
  else if ( v34 == 10 )
  {
    v25 = 240;
  }
  else
  {
    if ( v38 != 10 )
      goto LABEL_28;
    v25 = 241;
  }
  sub_33FAF80(v21, v25, (__int64)&v36, v3, v24, v20, a3);
  if ( v38 == 11 )
  {
    v27 = 236;
  }
  else if ( v35 == 11 )
  {
    v27 = 237;
  }
  else if ( v38 == 10 )
  {
    v27 = 240;
  }
  else
  {
    v27 = 241;
    if ( v35 != 10 )
LABEL_28:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
  }
  LOWORD(v15) = v35;
  v28 = sub_33FAF80(a1[1], v27, (__int64)&v36, v15, v33, v26, a3);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v28;
}
