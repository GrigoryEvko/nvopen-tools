// Function: sub_38039F0
// Address: 0x38039f0
//
unsigned __int8 *__fastcall sub_38039F0(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r14d
  __int16 *v5; // rdx
  __int64 v6; // rsi
  unsigned __int16 v7; // ax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int16 v12; // ax
  __int64 v13; // rdx
  __int16 v14; // r15
  __int64 *v15; // rsi
  __int64 v16; // rdx
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  int v18; // r9d
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // rsi
  unsigned __int8 *v22; // r14
  __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  __int64 v27; // [rsp+20h] [rbp-80h] BYREF
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h] BYREF
  int v30; // [rsp+38h] [rbp-68h]
  __int64 v31; // [rsp+40h] [rbp-60h]
  __int64 v32; // [rsp+48h] [rbp-58h]
  unsigned __int64 v33; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+58h] [rbp-48h]
  __int64 v35; // [rsp+60h] [rbp-40h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v29 = v6;
  LOWORD(v27) = v7;
  v28 = v8;
  if ( v6 )
  {
    sub_B96E90((__int64)&v29, v6, 1);
    v7 = v27;
  }
  v30 = *(_DWORD *)(a2 + 72);
  if ( v7 )
  {
    if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v10 = 16LL * (v7 - 1);
    v9 = *(_QWORD *)&byte_444C4A0[v10];
    LOBYTE(v10) = byte_444C4A0[v10 + 8];
  }
  else
  {
    v9 = sub_3007260((__int64)&v27);
    v31 = v9;
    v32 = v10;
  }
  v33 = v9;
  LOBYTE(v34) = v10;
  v11 = sub_CA1930(&v33);
  v26 = a1[1];
  switch ( v11 )
  {
    case 1u:
      v12 = 2;
      break;
    case 2u:
      v12 = 3;
      break;
    case 4u:
      v12 = 4;
      break;
    case 8u:
      v12 = 5;
      break;
    case 0x10u:
      v12 = 6;
      break;
    case 0x20u:
      v12 = 7;
      break;
    case 0x40u:
      v12 = 8;
      break;
    case 0x80u:
      v12 = 9;
      break;
    default:
      v12 = sub_3007020(*(_QWORD **)(a1[1] + 64), v11);
      v25 = v13;
      v26 = a1[1];
      goto LABEL_16;
  }
  v25 = 0;
LABEL_16:
  v14 = v12;
  v15 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
  if ( (void *)*v15 == sub_C33340() )
    sub_C3E660((__int64)&v33, (__int64)v15);
  else
    sub_C3A850((__int64)&v33, v15);
  sub_34007B0(v26, (__int64)&v33, (__int64)&v29, v14, v25, 0, a3, 0);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  v16 = a1[1];
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v17 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v33, *a1, *(_QWORD *)(v16 + 64), v27, v28);
    LOWORD(v19) = v34;
    v20 = v35;
  }
  else
  {
    v19 = v17(*a1, *(_QWORD *)(v16 + 64), v27, v28);
    HIWORD(v3) = HIWORD(v19);
    v20 = v24;
  }
  if ( (_WORD)v27 == 11 )
  {
    v21 = 236;
  }
  else if ( (_WORD)v19 == 11 )
  {
    v21 = 237;
  }
  else if ( (_WORD)v27 == 10 )
  {
    v21 = 240;
  }
  else
  {
    if ( (_WORD)v19 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v21 = 241;
  }
  LOWORD(v3) = v19;
  v22 = sub_33FAF80(a1[1], v21, (__int64)&v29, v3, v20, v18, a3);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v22;
}
