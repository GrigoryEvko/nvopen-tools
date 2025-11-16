// Function: sub_380D0E0
// Address: 0x380d0e0
//
__m128i *__fastcall sub_380D0E0(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // r12
  unsigned int v11; // edx
  unsigned int v12; // r13d
  __int64 v13; // rax
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // eax
  int v19; // r9d
  __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int16 v24; // ax
  __int64 v25; // rsi
  unsigned int v26; // edx
  __m128i *v27; // r12
  unsigned __int8 *v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+20h] [rbp-70h] BYREF
  int v31; // [rsp+28h] [rbp-68h]
  unsigned __int16 v32; // [rsp+30h] [rbp-60h] BYREF
  __int64 v33; // [rsp+38h] [rbp-58h]
  __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  char v35; // [rsp+48h] [rbp-48h]
  __int64 v36; // [rsp+50h] [rbp-40h]
  __int64 v37; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD *)(v6 + 40);
  v9 = *(_QWORD *)(v6 + 48);
  v30 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v30, v7, 1);
  v31 = *(_DWORD *)(a2 + 72);
  v10 = sub_380AAE0(a1, v8, v9);
  v12 = v11;
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v32 = v14;
  v33 = v15;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      BUG();
    v17 = 16LL * (v14 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v17];
    LOBYTE(v17) = byte_444C4A0[v17 + 8];
  }
  else
  {
    v16 = sub_3007260((__int64)&v32);
    v36 = v16;
    v37 = v17;
  }
  v34 = v16;
  v35 = v17;
  v18 = sub_CA1930(&v34);
  v20 = *(_QWORD *)(a1 + 8);
  switch ( v18 )
  {
    case 1u:
      LOWORD(v21) = 2;
      break;
    case 2u:
      LOWORD(v21) = 3;
      break;
    case 4u:
      LOWORD(v21) = 4;
      break;
    case 8u:
      LOWORD(v21) = 5;
      break;
    case 0x10u:
      LOWORD(v21) = 6;
      break;
    case 0x20u:
      LOWORD(v21) = 7;
      break;
    case 0x40u:
      LOWORD(v21) = 8;
      break;
    case 0x80u:
      LOWORD(v21) = 9;
      break;
    default:
      v21 = sub_3007020(*(_QWORD **)(v20 + 64), v18);
      v20 = *(_QWORD *)(a1 + 8);
      HIWORD(v3) = HIWORD(v21);
      v23 = v22;
      goto LABEL_16;
  }
  v23 = 0;
LABEL_16:
  LOWORD(v3) = v21;
  v24 = *(_WORD *)(*(_QWORD *)(v10 + 48) + 16LL * v12);
  if ( v24 == 11 )
  {
    v25 = 236;
  }
  else if ( v32 == 11 )
  {
    v25 = 237;
  }
  else if ( v24 == 10 )
  {
    v25 = 240;
  }
  else
  {
    if ( v32 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v25 = 241;
  }
  v29 = sub_33FAF80(v20, v25, (__int64)&v30, v3, v23, v19, a3);
  v27 = sub_33F3F90(
          *(_QWORD **)(a1 + 8),
          **(_QWORD **)(a2 + 40),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
          (__int64)&v30,
          (unsigned __int64)v29,
          v26,
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
          *(const __m128i **)(a2 + 112));
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v27;
}
